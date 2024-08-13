from datasets import load_dataset
import os
from dotenv import load_dotenv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import openai
from openai import APIError
import os
import json
import re
import asyncio
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import pandas as pd
import torch
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from collections import defaultdict
# import pytrec_eval
import json
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import CustomInformationRetrievalEvaluator
import importlib
# from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
import warnings
import api_request_parallel_processor
warnings.filterwarnings('ignore')




def save_jobs(data_pair, filename, model = "text-embedding-3-small"):    
    jobs = [
                {
                    "model": model,
                    # "response_format": "json", # TODO check 
                    # "temperature": 0,
                    "metadata": {"id": indx},
                    "input": text
                }
                for indx, text in data_pair.items()
            ]
    with open(filename, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


def get_data_for_evaluation(dataset_name):
    loaded_table = pq.read_table(dataset_name)
    df = loaded_table.to_pandas()
    contexts = {}
    queries = {}
    relevant_docs = {}
    query_idx = 1
    for idx, row in df.iterrows():
        if idx >= 100:  # Break the loop after two iterations
            break
        contexts[idx] = row['context']
        for query in row['queries']:
            query = query.strip()
            queries[query_idx] = query
            if query_idx not in relevant_docs:
                relevant_docs[query_idx] = set()
            relevant_docs[query_idx].add(idx)
            query_idx += 1
    return queries, contexts, relevant_docs

def load_sentence_tranformer_from_transformer(model_name):
    model = AutoModel.from_pretrained(model_name)
    # Combine the model and pooling into a SentenceTransformer
    word_embedding_model = models.Transformer(model_name_or_path=model_name)
    pooling_model = models.Pooling(word_embedding_dimension=model.config.hidden_size, pooling_mode_mean_tokens=True)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def get_model_or_model_name(model_name, is_openAI):
    model = None
    if not is_openAI:
        model = load_sentence_tranformer_from_transformer(model_name)
        model_name = None
    return model_name,model

def evaluate(model_name, dataset_name, is_openAI):

    name = f"{model_name}-{dataset_name}-evlatuation"
    queries, contexts, relevant_docs = get_data_for_evaluation(dataset_name)

    ir_evaluator = CustomInformationRetrievalEvaluator.InformationRetrievalEvaluator(
    queries=queries,
    corpus=contexts,
    relevant_docs=relevant_docs,
    name=name,
    write_csv=True
    )

    model_name, model = get_model_or_model_name(model_name, is_openAI)
        
    results = ir_evaluator(model, openAI_model=model_name)
    # print(ir_evaluator.primary_metric)
    # print(results[ir_evaluator.primary_metric])
    return results


def save_contexts_query_jobs(dataset_name, model_name = "text-embedding-3-small"):
    queries, contexts, _ = get_data_for_evaluation(dataset_name)
    document_name = dataset_name.split("/")[1]
    name = document_name.split("_")[0]
    query_path = "datasets/queries"
    context_path = "datasets/contexts"
    save_jobs(queries, query_path)
    save_jobs(contexts, context_path)
    api_request_parallel_processor.helper(query_path, f"datasets/{model_name}-{name}-queries.jsonl", "https://api.openai.com/v1/embeddings", OPENAI_API_KEY)
    api_request_parallel_processor.helper(context_path, f"datasets/{model_name}-{name}-contexts.jsonl", "https://api.openai.com/v1/embeddings", OPENAI_API_KEY)



if __name__ == "__main__":
    # must for Custom scripts
    # importlib.reload(CustomInformationRetrievalEvaluator)
    # importlib.reload(api_request_parallel_processor)


    # Load the .env file
    load_dotenv()

    # Access the variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    save_contexts_query_jobs("processed_datasets/squad_processed.parquet")
    # python evaluation-pipetine-test/api_request_parallel_processor.py   --requests_filepath evaluation-pipetine-test/datasets/queries   --save_filepath evaluation-pipetine-test/datasets/text-embedding-3-small-squad-queries.jsonl   --request_url https://api.openai.com/v1/embeddings   --max_requests_per_minute 1500   --max_tokens_per_minute 6250000   --token_encoding_name cl100k_base   --max_attempts 5   --logging_level 20 
    # python evaluation-pipetine-test/api_request_parallel_processor.py   --requests_filepath evaluation-pipetine-test/datasets/contexts   --save_filepath evaluation-pipetine-test/datasets/text-embedding-3-small-squad-contexts.jsonl   --request_url https://api.openai.com/v1/embeddings   --max_requests_per_minute 1500   --max_tokens_per_minute 6250000   --token_encoding_name cl100k_base   --max_attempts 5   --logging_level 20 


    datasets = ["processed_datasets/squad_processed.parquet"]

    models_ = {
    "google-bert/bert-base-multilingual-cased": False,
    "datasets/text-embedding-3-small-squad" : True   
    }
    print(models_.keys())
    for dataset_name in datasets:
        for model_name in models_.keys():
            # print(dataset_name)
            # print(model_name)
            # print(models_[model_name])
            res = evaluate(model_name=model_name, dataset_name=dataset_name, is_openAI=models_[model_name])
            pprint(res)




