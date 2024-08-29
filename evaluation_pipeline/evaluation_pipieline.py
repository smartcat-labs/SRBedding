import json
import os
import sys
import warnings
from pathlib import Path
from pprint import pprint
from typing import Dict

import CustomInformationRetrievalEvaluator
import openai
import pyarrow.parquet as pq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel

warnings.filterwarnings("ignore")

sys.path.append("..")
from api_request_parallel_processor import run_api_request_processor


def save_jobs(
    data_pair: Dict[int, str], filename: Path, model: str = "text-embedding-3-small"
) -> None:
    """
    Saves a collection of text embedding jobs (comands) to a file.

    This function takes a dictionary of data pairs, where each key-value pair consists of an identifier (key)
    and a text input (value). It formats these pairs into a list of jobs, each represented as a dictionary
    with metadata and input fields. The jobs are then serialized to JSON format and written to a specified
    file, with one job per line.

    Args:
        data_pair (Dict[int, str]): A dictionary where each key is an integer ID and each value is a string
                                    representing the text to be embedded.
        filename (Path): The file path where the jobs will be saved. The file will be created or overwritten
                         if it already exists.
        model (str, optional): The model name to be included in each job. Defaults to "text-embedding-3-small".

    Returns:
        None

    """
    jobs = [
        {"model": model, "metadata": {"id": indx}, "input": text}
        for indx, text in data_pair.items()
    ]
    with open(filename, "w", encoding="UTF-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")


def get_data_for_evaluation(dataset_name: Path) -> tuple:
    """
    Extracts data necessary for evaluating a question-answering system from a given dataset.

    This function reads a dataset from a Parquet file, converts it to a pandas DataFrame, and then processes it
    to extract contexts, queries, and relevant document mappings. It returns these as dictionaries, which are
    used for evaluation purposes.

    Args:
        dataset_name (Path): The file path of the Parquet dataset to be loaded and processed.

    Returns:
        tuple: A tuple containing three elements:
            - queries (Dict[int, str]): A dictionary where each key is a unique query identifier, and each value
                                        is a string representing a query.
            - contexts (Dict[int, str]): A dictionary where each key is a context identifier, and each value is
                                         a string representing a context passage.
            - relevant_docs (Dict[int, Set[int]]): A dictionary where each key is a query identifier, and each
                                                  value is a set of context identifiers that are relevant to the query.
    """
    loaded_table = pq.read_table(dataset_name)
    df = loaded_table.to_pandas()
    contexts = {}
    queries = {}
    relevant_docs = {}
    query_idx = 1
    embedded = 0
    for idx, row in df.iterrows():
        if embedded >= 8000:
            break
        contexts[idx] = row["context"]
        embedded += 1
        for query in row["queries"]:
            embedded += 1
            query = query.strip()
            queries[query_idx] = query
            if query_idx not in relevant_docs:
                relevant_docs[query_idx] = set()
            relevant_docs[query_idx].add(idx)
            query_idx += 1
    return queries, contexts, relevant_docs


def load_sentence_tranformer_from_transformer(model_name: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model using a specified transformer model name.

    This function loads a pre-trained transformer model and combines it with a pooling layer to create
    a SentenceTransformer model. The resulting model can be used for tasks like sentence embeddings,
    semantic similarity, and more.

    Args:
        model_name (str): The name of the pre-trained transformer model to load. This should be a model
                          available in the Hugging Face model hub.

    Returns:
        SentenceTransformer: A SentenceTransformer model that is a combination of the specified transformer
                             model and a pooling layer.
    """
    # model = AutoModel.from_pretrained(model_name)
    # # Combine the model and pooling into a SentenceTransformer
    # word_embedding_model = models.Transformer(model_name_or_path=model_name)
    # pooling_model = models.Pooling(
    #     word_embedding_dimension=model.config.hidden_size, pooling_mode_mean_tokens=True
    # )
    # return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return SentenceTransformer(model_name)


def evaluate(model_name: str, dataset_name: Path, is_openAI: bool) -> Dict[str, float]:
    """
    Evaluates a model's performance on an information retrieval task using a specified dataset.

    This function evaluates either a custom SentenceTransformer model or an OpenAI model on an information
    retrieval task. It processes the dataset to extract queries, contexts, and relevant documents, then
    performs the evaluation using the `CustomInformationRetrievalEvaluator`. The evaluation results are
    saved and returned.

    Args:
        model_name (str): The name of the model to evaluate. This can be a Hugging Face transformer model or
                          an OpenAI model, depending on the value of `is_openAI`.
        dataset_name (Path): The file path of the dataset to be used for evaluation. The dataset is expected
                             to be in a Parquet format.
        is_openAI (bool): A flag indicating whether the model is an OpenAI model. If `True`, the model is
                          treated as an OpenAI model; otherwise, it is treated as a Hugging Face transformer model.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics and their respective scores.
    """
    stem = dataset_name.stem
    name = f"{model_name}-{stem.split('_')[0]}-evlatuation"
    name = name.replace("/", "-")
    queries, contexts, relevant_docs = get_data_for_evaluation(dataset_name)

    ir_evaluator = CustomInformationRetrievalEvaluator.InformationRetrievalEvaluator(
        queries=queries,
        corpus=contexts,
        relevant_docs=relevant_docs,
        name=name,
        write_csv=True,
    )

    model = None
    if not is_openAI:
        model = load_sentence_tranformer_from_transformer(model_name)

    results = ir_evaluator(model, openAI_model=model_name, output_path="results/")
    return results


def save_contexts_query_jobs(
    dataset_name: Path, model_name: str = "text-embedding-3-small"
) -> None:
    """
    Saves contexts and queries from a dataset as jobs and processes them through an API for embeddings.

    This function extracts contexts and queries from a specified dataset and saves them as job files.
    It then sends these jobs to an API for generating embeddings, saving the resulting embeddings to
    specified output files.

    Args:
        dataset_name (Path): The file path of the dataset to be processed. The dataset is expected to
                             be in a Parquet format.
        model_name (str, optional): The name of the model to be used for generating embeddings. Defaults
                                    to "text-embedding-3-small".

    Returns:
        None
    """
    queries, contexts, _ = get_data_for_evaluation(dataset_name)
    document_name = dataset_name.stem
    name = document_name.split("_")[0]
    query_path = "datasets/queries"
    context_path = "datasets/contexts"
    save_jobs(queries, query_path)
    save_jobs(contexts, context_path)
    run_api_request_processor(
        query_path,
        f"datasets/{model_name}-{name}-queries.jsonl",
        "https://api.openai.com/v1/embeddings",
    )
    run_api_request_processor(
        context_path,
        f"datasets/{model_name}-{name}-contexts.jsonl",
        "https://api.openai.com/v1/embeddings",
    )


if __name__ == "__main__":
    # set the environment
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    datasets = [
        "squad", 
        "marco", 
        "naquestions"
        ]

    models_ = {
        "output/final_model": False,
        # "google-bert/bert-base-multilingual-cased": False, ## mtb rettrivsl na modelCard koje dt je imao u sebi
        # "mixedbread-ai/mxbai-embed-large-v1": False  ## v2??? medium, small
        # "datasets/text-embedding-3-small": True, ### open AI, sve u fajluuuuuuuuu
    }   # 

    for dataset_name in datasets:
        for model_name in models_.keys():
            dataset_path = Path(f"datasets/{dataset_name}_processed.parquet")
            name = model_name

            if models_[model_name]:
                # save_contexts_query_jobs(
                #     dataset_path, model_name=model_name.split("/")[1]
                # )
                model_name += "-" + dataset_name

            res = evaluate(
                model_name=model_name,
                dataset_name=dataset_path,
                is_openAI=models_[name],
            )
            pprint(res)
