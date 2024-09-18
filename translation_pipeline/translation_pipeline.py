from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from prompts import SYSTEM_PROMPT

sys.path.append("..")
from api_request_parallel_processor import run_api_request_processor


def make_jobs(
    model: str, prompt: str, filename: Path, dataset: List[Dict[str, str | List[str]]]
) -> None:
    """
    Creates a list of jobs formatted for an LLM model and saves them to a file.

    Args:
        model (str): The model name to be used in each job.
        prompt (str): The system prompt content for the model.
        filename (Path): The file path where the jobs will be saved.
        dataset (List[Dict[str, str | List[str]]]): A list of dictionaries representing the dataset,
            where each dictionary contains the keys 'query_id' and associated data.

    Returns:
        None
    """
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


def save_jobs(filename: Path, jobs: List[Dict[str, Any]]) -> None:
    """
    Saves a list of job dictionaries to a file in JSONL format.

    Args:
        filename (Path): The file path where the jobs will be saved.
        jobs (List[Dict[str, Any]]): A list of job dictionaries to be saved.

    Returns:
        None
    """
    with open(filename, "w", encoding="utf-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")


def load_data_ms_marco(
    loading_size,
    dataset_name: str = "microsoft/ms_marco",
) -> List[Dict[str, Any]]:
    """
    Loads and processes a subset of the MS MARCO dataset.

    Args:
        dataset_name (str): The name of the dataset to load. Defaults to "microsoft/ms_marco".

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the query ID, query, and passage text.
    """
    datasets_dir = make_cache_dir()
    data = load_dataset(dataset_name, "v1.1", cache_dir=datasets_dir)
    data_test_split = data["test"]
    ms_marco = data_test_split.select_columns(["passages", "query", "query_id"])

    final_data = []
    for i in range(loading_size):
        final_data.append(
            {
                "query_id": str(ms_marco["query_id"][i]),
                "query": ms_marco["query"][i],
                "passage_text": ms_marco["passages"][i]["passage_text"],
            }
        )

    return final_data


def make_cache_dir() -> Path:
    """
    Creates and returns the cache directory path for storing datasets.

    Returns:
        Path: The expanded user path to the cache directory.
    """
    return Path("~/Datasets/SRBendding").expanduser()


def save_failed_ids(failed: List[Dict[str, str]], dataset_name: str) -> None:
    file_path = Path(f"failed/failed_{dataset_name}.json")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the IDs to a text file, one per line
    with open(file_path, "w") as f:
        # Convert exceptions to string because exceptions are not JSON serializable
        json.dump(failed, f, default=str, indent=4)


def make_dataset(
    processed_commands_path: Path, dataset_name: str
) -> Dict[str, str | List[str]]:
    returned_dict = {
        "id": [],
        "query": [],
        "passage_text": [],
    }
    failed = []
    # Open and iterate through the .jsonl file
    with open(processed_commands_path, "r") as file:
        for line in file:
            id_ = None
            try:
                data = json.loads(line)
                id_ = data[-1]["id"]
                returned_data = data[1]["choices"][0]["message"]["content"]
                tranlation = json.loads(
                    returned_data
                )  # gpt message i.e. translation in this case
                returned_dict["id"].append(id_)
                returned_dict["query"].append(tranlation["query"])
                returned_dict["passage_text"].append(tranlation["passage_text"])
            except Exception as e:
                failed.append({"id": id_, "exception": e})
    if failed:
        save_failed_ids(failed, dataset_name=dataset_name)
    return returned_dict


def save_in_file(processed_commands_path: Path, save_path: Path) -> None:
    data_for_df = make_dataset(processed_commands_path, save_path.stem)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.DataFrame(data_for_df)
    # dataset['id'] = dataset['id'].astype(str)
    # dataset['query'] = dataset['query'].astype(str)
    dataset["passage_text"] = dataset["passage_text"].apply(
        lambda x: np.array(x, dtype=str)
    )
    dataset.to_parquet(save_path, engine="pyarrow")


def load_data_natural(
    loading_size: int,
    dataset_name: str = "google-research-datasets/natural_questions",
) -> List[Dict[str, str | List[str]]]:
    """
    Loads and processes the Natural Questions dataset to extract query and passage data.

    Args:
        dataset_name (str): The name of the dataset to load. Default is "google-research-datasets/natural_questions".

    Returns:
        List[Dict[str, str | List[str]]]: A list of dictionaries, each containing a query ID, query text, and passage text.
    """
    dir = make_cache_dir()
    data = load_dataset(dataset_name, "dev", cache_dir=dir)
    validation_dataset = data["validation"]

    result = []
    i = 0
    while len(result) < loading_size and i < len(validation_dataset):
        record = validation_dataset[i]
        id = record["id"]
        start_byte, end_byte = get_start_and_end_byte(record)
        context = make_context(record, start_byte, end_byte)
        question = record["question"]["text"]
        current = {
            "query_id": id,
            "query": question,
            "passage_text": [context],
        }
        i += 1
        if context != "" and len(context) < 48875:
            result.append(current)

    return result


def make_context(record: Dict[str, Any], start_byte: int, end_byte: int) -> str:
    """
    Constructs the context string from a document's tokens within the specified byte range.

    Args:
        record (Dict[str, Any]): A dictionary containing the document with tokenized text.
        start_byte (int): The starting byte position for extracting the context.
        end_byte (int): The ending byte position for extracting the context.

    Returns:
        str: The context string constructed from tokens within the specified byte range.
    """
    context = ""
    token_element = record["document"]["tokens"]
    for j in range(len(token_element["token"])):
        if (
            not token_element["is_html"][j]
            and start_byte <= token_element["start_byte"][j] <= end_byte
        ):
            context += (token_element["token"][j]).strip() + " "
    context = context.strip()
    return context


def get_start_and_end_byte(record: Dict[str, Any]) -> Tuple[int, int]:
    """
    Extracts the start and end byte positions of the first valid long answer from a record.

    Args:
        record (Dict[str, Any]): A dictionary containing annotations with long answer data.

    Returns:
        Tuple[int, int]: The start and end byte positions. Returns (-1, -1) if no valid long answer is found.
    """
    start_byte = -1
    end_byte = -1
    long_answers = record["annotations"]["long_answer"]
    for j in range(len(long_answers)):
        if long_answers[j]["start_byte"] != -1:
            start_byte = long_answers[j]["start_byte"]
            end_byte = long_answers[j]["end_byte"]
            break
    return start_byte, end_byte


def get_timestamp() -> str:
    """
    Returns the current timestamp as a formatted string.

    This function generates the current date and time, formatted as a string in the
    format 'DD-MM-YYYY_HH-MM-SS'. It can be useful for creating unique filenames
    or logging events with a timestamp.

    Returns:
        str: The current timestamp in 'DD-MM-YYYY_HH-MM-SS' format.

    Example:
        >>> timestamp = get_timestamp()
        >>> print(timestamp)
        '18-08-2024_15-45-30'
    """
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-3.5-turbo-0125"
    datasets = [
        {"name": "msmarco", "loading_function": load_data_ms_marco, "data_size": 10},
        {"name": "naquestions", "loading_function": load_data_natural, "data_size": 10},
    ]

    for dataset in datasets:
        date = get_timestamp()
        dataset_name = dataset["name"]

        final_data = dataset["loading_function"](dataset["data_size"])
        path = Path(f"commands/jobs_{dataset_name}.jsonl")
        commands_filepath = Path(f"commands/results_{dataset_name}_{date}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)

        make_jobs(model=model, prompt=SYSTEM_PROMPT, filename=path, dataset=final_data)
        run_api_request_processor(
            requests_filepath=path,
            save_filepath=commands_filepath,
            request_url="https://api.openai.com/v1/chat/completions",
        )
        final_save_path = Path(f"datasets/{dataset_name}.parquet")
        save_in_file(commands_filepath, save_path=final_save_path)