import asyncio
import os
from typing import Dict
from dotenv import load_dotenv
import pyarrow.parquet as pq
import openai
import os
import json
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel
import json
import CustomInformationRetrievalEvaluator
from pprint import pprint
import warnings
from api_request_parallel_processor import process_api_requests_from_file
from pathlib import Path
warnings.filterwarnings('ignore')



def run_api_request_processor(
    requests_filepath: Path,
    save_filepath: Path,
    request_url: str,
    max_requests_per_minute: int = 500,
    max_tokens_per_minute: int = 1250000,
    token_encoding_name: str = 'cl100k_base',
    max_attempts: int = 5,
    logging_level: int = 20,
) -> None:
    """
    Processes API requests from a file and saves the responses.

    This function reads requests from a specified file, sends them to an API endpoint, and writes the 
    responses to an output file. It manages request limits and retries to handle API rate limits and 
    potential request failures. The function uses asynchronous processing to efficiently handle multiple 
    requests.

    Args:
        requests_filepath (Path): The file path of the input file containing the API requests to be processed.
        save_filepath (Path): The file path where the API responses will be saved.
        request_url (str): The URL of the API endpoint to send requests to.
        max_requests_per_minute (int, optional): The maximum number of requests to send per minute. Defaults to 1500.
        max_tokens_per_minute (int, optional): The maximum number of tokens to process per minute. Defaults to 6250000.
        token_encoding_name (str, optional): The name of the token encoding to use. Defaults to 'cl100k_base'.
        max_attempts (int, optional): The maximum number of attempts to retry a failed request. Defaults to 5.
        logging_level (int, optional): The logging level to use for the process. Defaults to 20 (INFO level).
    
    Returns:
        None
    """
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )


def save_jobs(data_pair: Dict[int, str], filename: Path, model: str = "text-embedding-3-small") -> None:    
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
                {
                    "model": model,
                    "metadata": {"id": indx},
                    "input": text
                }
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
    model = AutoModel.from_pretrained(model_name)
    # Combine the model and pooling into a SentenceTransformer
    word_embedding_model = models.Transformer(model_name_or_path=model_name)
    pooling_model = models.Pooling(word_embedding_dimension=model.config.hidden_size, pooling_mode_mean_tokens=True)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


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
    name = name.replace("/", '-')
    queries, contexts, relevant_docs = get_data_for_evaluation(dataset_name)

    ir_evaluator = CustomInformationRetrievalEvaluator.InformationRetrievalEvaluator(
    queries=queries,
    corpus=contexts,
    relevant_docs=relevant_docs,
    name=name,
    write_csv=True
    )
    
    model = None
    if not is_openAI:
        model = load_sentence_tranformer_from_transformer(model_name)
        
    results = ir_evaluator(model, openAI_model=model_name, output_path=f"results/")
    return results


def save_contexts_query_jobs(dataset_name: Path, model_name: str = "text-embedding-3-small") -> None:
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
    document_name = dataset_name.split("/")[1]
    name = document_name.split("_")[0]
    query_path = "datasets/queries"
    context_path = "datasets/contexts"
    save_jobs(queries, query_path)
    save_jobs(contexts, context_path)
    run_api_request_processor(query_path, f"datasets/{model_name}-{name}-queries.jsonl", "https://api.openai.com/v1/embeddings")
    run_api_request_processor(context_path, f"datasets/{model_name}-{name}-contexts.jsonl", "https://api.openai.com/v1/embeddings")



if __name__ == "__main__":

    # set the environment
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # save_contexts_query_jobs("datasets/squad_processed.parquet")

    datasets = [Path("datasets/squad_processed.parquet")]

    models_ = {
    "google-bert/bert-base-multilingual-cased": False,
    "datasets/text-embedding-3-small-squad" : True   
    }
    print(models_.keys())
    for dataset_name in datasets:
        for model_name in models_.keys():
            res = evaluate(model_name=model_name, dataset_name=dataset_name, is_openAI=models_[model_name])
            pprint(res)




