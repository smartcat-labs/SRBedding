import json
import os
from pathlib import Path
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

from utils import make_path


def environment_setup():
    """
    Sets up the environment by loading environment variables and configuring the OpenAI API key.

    This function loads environment variables from a `.env` file and sets the OpenAI API key
    using the `OPENAI_API_KEY` environment variable. It should be called before making API requests.

    Returns:
        None

    Example:
        >>> environment_setup()
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


def get_batch_id(file: Path) -> str:
    """
    Reads a file and returns its contents as a stripped string.

    Args:
        file (Path): The file path from which the contents will be read.

    Returns:
        str: The contents of the file with leading and trailing whitespace removed.
    """
    with open(file, "r") as file:
        text = file.read()
    return text.strip()


def save_failed_ids(failed: List[Dict[str, str]], dataset_name: str) -> None:
    """
    Saves a list of failed IDs to a JSON file.

    Args:
        failed (List[Dict[str, str]]): A list of dictionaries containing failed IDs.
        dataset_name (str): The name of the dataset to include in the filename.

    Returns:
        None
    """
    file_path = Path(f"failed/failed_{dataset_name}.json")
    make_path(file_path.parent)
    with open(file_path, "w") as f:
        json.dump(failed, f, default=str, indent=4)


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


def make_jobs(
    prompt: str,
    filename: Path,
    dataset: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo-0125",
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
            "custom_id": f'task-{sample["id"]}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(sample["sentence"])},
                ],
            },
        }
        for sample in dataset
    ]
    save_jobs(filename, jobs)


def batch_requests(jobs_file: Path, dataset_name: str):
    """
    Creates a batch job request to OpenAI's API and saves the batch job ID to a file.

    Args:
        jobs_file (Path): The path to the file containing job requests to be sent.
        dataset_name (str): The name of the dataset, used to name the output file.

    Returns:
        None
    """
    environment_setup()
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_file = client.files.create(file=open(jobs_file, "rb"), purpose="batch")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch_job.id)
    file_path = f"commands/number_{dataset_name}.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(batch_job.id)
