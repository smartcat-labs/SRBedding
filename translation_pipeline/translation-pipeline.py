import asyncio
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import openai
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sentence_transformers
import transformers
from api_request_parallel_processor import process_api_requests_from_file
from datasets import load_dataset
from openai import OpenAI
from prompts import SYSTEM_PROMPT


def run_api_request_processor(
    requests_filepath: Path,
    save_filepath: Path,
    request_url: str,
    max_requests_per_minute: int = 1500,
    max_tokens_per_minute: int = 6250000,
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

def make_jobs(model, prompt, filename, dataset):  # type hint
    jobs = [
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "metadata": {"id": sample["query_id"]},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(sample)},
            ],
        }
        for sample in dataset
    ]
    save_jobs(filename, jobs)

def save_jobs(filename, jobs):
    with open(filename, "w", encoding='utf-8') as f:  # Convert to method
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")

def load_data(
    dataset_name="microsoft/ms_marco",
):  # Make it a config variable + type hint
    datasets_dir = make_cache_dir()
    data = load_dataset(dataset_name, "v1.1", cache_dir=datasets_dir)
    data_test_split = data["test"]
    ms_marco = data_test_split.select_columns(["passages", "query", "query_id"])

    final_data = []
    for i in range(3):
        final_data.append(
            {
                "query_id": ms_marco["query_id"][i],
                "query": ms_marco["query"][i],
                "passage_text": ms_marco["passages"][i]["passage_text"],
            }
        )

    return final_data

def make_cache_dir():
    datasets_dir = Path("~/Datasets/").expanduser()
    #assert datasets_dir.exists(), "Datasets directory not found"
    datasets_dir = datasets_dir / "SRBedding_datasets/ms_marco_v1"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir

def make_dataset(file_path: Path): #file_path is a path to chat gpt translation results
        returned_dict = {
             "id": [],
             "query": [],
             "passage_text": [],
        }
        # Open and iterate through the .jsonl file
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                id_ = data[-1]['id']
                returned_data = data[1]['choices'][0]['message']['content']
                returned_data = json.loads(returned_data) # gpt message i.e. translation in this case
                tranlation = returned_data['translations'][0]
                returned_dict['id'].append(id_)
                returned_dict['query'].append(tranlation['query'])
                returned_dict['passage_text'].append(tranlation['passage_text'])
        return returned_dict

def save_in_file(path):
    data_for_df = make_dataset(path)
    dataset = pd.DataFrame(data_for_df)
    table = pa.Table.from_pandas(dataset)
    print(dataset.head())
    pq.write_table(table, 'datasets/train.parquet')

if __name__ == "__main__":

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-3.5-turbo-0125"

    final_data = load_data()
    path = "test.jsonl"
    saved_filepath = "translation_pipeline_test/test_results.jsonl"

    make_jobs(model=model, prompt=SYSTEM_PROMPT, filename=path, dataset=final_data)
    run_api_request_processor(requests_filepath=path, save_filepath=saved_filepath, request_url="https://api.openai.com/v1/chat/completions")
    save_in_file(saved_filepath)

