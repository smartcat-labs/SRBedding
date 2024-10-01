import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from prompts import SYSTEM_PROMPT

sys.path.append("..")
from utils import make_cache_dir, make_path
from utils_openAI import batch_requests, save_jobs


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
            "custom_id": f'task-{sample["query_id"]}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(sample)},
                ],
            },
        }
        for sample in dataset
    ]
    save_jobs(filename, jobs)


def load_data_ms_marco(
    data_size: int,
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
    for i in range(0, data_size):
        final_data.append(
            {
                "query_id": str(ms_marco["query_id"][i]),
                "query": ms_marco["query"][i],
                "passage_text": ms_marco["passages"][i]["passage_text"],
            }
        )

    return final_data


def load_data_natural(
    data_size: int,
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
    while len(result) < data_size and i < len(validation_dataset):
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


if __name__ == "__main__":
    model = "gpt-3.5-turbo-0125"
    datasets = [
        {
            "name": "msmarco", 
            "loading_function": load_data_ms_marco, 
            "data_size": 10},
        {
            "name": "naquestions",
            "loading_function": load_data_natural,
            "data_size": 10,
        },
    ]

    for dataset in datasets:
        # date = get_timestamp()
        dataset_name = dataset["name"]
        final_data = dataset["loading_function"](dataset["data_size"])
        path = Path(f"commands/jobs_{dataset_name}.jsonl")
        make_path(path.parent)

        make_jobs(model=model, prompt=SYSTEM_PROMPT, filename=path, dataset=final_data)

        batch_requests(jobs_file=path, dataset_name=dataset_name)
