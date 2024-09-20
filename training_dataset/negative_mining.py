import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List
from datetime import datetime

import faiss
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append('..')
from api_request_parallel_processor import run_api_request_processor


def save_jobs(sentences: List[str], filename: Path, model: str = "text-embedding-3-small") -> None:    

    jobs = [
                {
                    "model": model,
                    "metadata": {"id": index},
                    "input": sentence
                }
                for index, sentence in enumerate(sentences)
            ]
    with open(filename, "w", encoding="UTF-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")

def load_dataset(dataset_name: str):
    path = f"datasets/{dataset_name}.parquet"
    dataset = pd.read_parquet(path)
    dataset["dataset"] = [dataset_name]*dataset.shape[0]

    return dataset[:20]

def make_embedding_list(file_path):
   returned_dict = {}
   with open(file_path, 'r') as file:
      for line in file:
        try:
            data = json.loads(line)
            indx = data[-1]['id']
            embedding = data[1]['data'][0]['embedding']
            returned_dict[indx] = embedding
        except Exception as e:
            returned_dict[indx] = [0.0]*1536
   
         
      result = [returned_dict[key] for key in sorted(returned_dict.keys())]

   return result

def save_combined_embeddings_jobs(sentences: List[str], dataset_name: str, model_name: str = "text-embedding-3-small") -> None:
   sentence_path = Path(f"datasets/{dataset_name}.jsonl")
   date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
   saved_path = Path(f"datasets/{dataset_name}_embeddings_{date}.jsonl")
   saved_path.parent.mkdir(parents=True, exist_ok=True)
   save_jobs(sentences, sentence_path)
   run_api_request_processor(sentence_path, saved_path, "https://api.openai.com/v1/embeddings")

   return make_embedding_list(saved_path)

def index_vectors(embeddings):
    """
    Indexes embeddings using FAISS for efficient similarity search.

    Args:
        embeddings: A list of numpy arrays representing the embeddings to be indexed.

    Returns:
        index: A FAISS index object containing the indexed embeddings.
    """

    index = faiss.IndexFlatIP(len(embeddings[0]))
    embedding_length = len(embeddings[0])

    if not all(len(embed) == embedding_length for embed in embeddings):
        raise ValueError("All embeddings must have the same length.")
    
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    index.add(embeddings)
    return index

def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 8):

    """
    Performs batch search on a vector database (index) for given queries.

    Args:
        index: The vector database index object.
        query: A list of query embeddings (numpy arrays).
        topk: The number of top results to retrieve for each query.
        batch_size: The size of each batch of queries to process.

    Returns:
        all_scores: A list of lists containing similarity scores for each query.
        all_inxs: A list of lists containing the indexes of the top-k results for each query.
    """
    
    all_scores, all_inxs = [], []

    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches"):
        batch_query = query[start_index:start_index + batch_size]

        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
        
    assert len(all_inxs) == len(query)

    return all_scores, all_inxs

def get_hard_negatives(data, all_scores, all_inxs, dataset_name, query_len, save_data = True) -> pd.DataFrame:

    data['neg'] = [[] for _ in range(len(data))]
    data['neg_scores'] = [[] for _ in range(len(data))]
    data = data.assign(pos_score=[0.0] * len(data))
    for i, _ in data.iterrows():
        filtered_inx = []
        filtered_scores = []
        relevant_indexes = all_inxs[i]
        relevant_scores = all_scores[i]
        for which, inx in enumerate(relevant_indexes):
            current_score = round(relevant_scores[which], 3)
            if inx == -1:
                break
            if inx != i:
                filtered_inx.append(inx)
                filtered_scores.append(current_score)
            else:
                data.loc[i, 'pos_score'] = float(current_score)   # Update 'pos_score' for the current index

        
        filtered_inx = random.sample(filtered_inx, min(5, len(filtered_inx)))  
        res = [data['context'][inx] for inx in filtered_inx]        
        data.at[i, 'neg'] = res
        data.at[i, 'neg_scores'] = filtered_scores
        assert len(res) == len(filtered_scores)
        if save_data:
            data = data[[query_len, "context", "neg", "pos_score", "neg_scores", "dataset"]]
            data.to_parquet(f"datasets/{dataset_name}_{query_len}_neg.parquet")


    return data

if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    for dataset_name in ["wiki", "news", "science", "literature"]:
        dataset = load_dataset(dataset_name)
        context_embedings_file_name = f'embeddings/{dataset_name}_embeddings_contexts.jsonl'
        context2embed = make_embedding_list(context_embedings_file_name)[:20]
        # for query_len in ["short_query", "medium_query", "long_query"]:
        keywords = [' '.join(x) for x in dataset['keywords']]
        query2embed = save_combined_embeddings_jobs(keywords, dataset_name=dataset_name)
        index = index_vectors(context2embed)
        scores, indices = batch_search(index, query2embed, topk=5, batch_size=10)
        hard_negatives = get_hard_negatives(dataset, scores, indices, dataset_name=dataset_name, query_len='keywords')


    



