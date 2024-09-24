from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from openai import OpenAI
import pandas as pd
from prompts import SYSTEM_PROMPT


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
            "custom_id": f'task-{sample["id"]}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(sample['sentence'])},
                ],
            },
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


def load_sts_data(
) -> List[Dict[str, Any]]:
    datasets_dir = make_cache_dir()
    sts = load_dataset("mteb/stsbenchmark-sts",  cache_dir=datasets_dir)
    sts_test_df = pd.DataFrame(sts['test'])

    final_data = []
    for i, row in sts_test_df.iterrows():
        if i > 10:
            break
        final_data.append(
            {
                "id": row['sid'],
                "sentence": row['sentence2'],
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


def batch_requests(jobs_file: Path, dataset_name: str):
    batch_file = client.files.create(file=open(jobs_file, "rb"), purpose="batch")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch_job.id)
    file_path = f"commands/number_{dataset_name}.txt"

    # Open the file in write mode and write the number
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(batch_job.id)


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-3.5-turbo-0125"

    dataset_name = "STS"

    final_data = load_sts_data()
    path = Path(f"commands/jobs_{dataset_name}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    make_jobs(model=model, prompt=SYSTEM_PROMPT, filename=path, dataset=final_data)

    batch_requests(jobs_file=path, dataset_name=dataset_name)
