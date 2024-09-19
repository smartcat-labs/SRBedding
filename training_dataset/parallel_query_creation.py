import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import openai
import pandas as pd
from dotenv import load_dotenv

sys.path.append("..")
from prompt_4o_testing import PROMPT

from training_dataset.api_request_parallel_processor import run_api_request_processor


def save_jobs(
    sentences: List[str],
    filename: Path,
    prompt_template: str,
    model: str = "gpt-3.5-turbo-0125",
) -> None:
    """
    Saves a list of sentences as formatted jobs into a specified .jsonl file.

    This function creates a list of job dictionaries where each job contains:
    - The model to be used (default: "gpt-3.5-turbo-0125").
    - The response format, which is set to a JSON object.
    - A temperature setting for the model (default: 0).
    - Metadata containing the context, which is the sentence.
    - A message formatted using the provided prompt template.

    The jobs are saved line by line in a JSON format to the specified file.

    Args:
        sentences (List[str]): A list of sentences to be used as context for each job.
        filename (Path): The path to the .jsonl file where the jobs will be saved.
        prompt_template (str): A template string for generating the message content.
        model (str, optional): The model to be specified for each job. Defaults to "gpt-3.5-turbo-0125".

    Returns:
        None

    Example:
        >>> save_jobs(["sentence1", "sentence2"], Path("output.jsonl"), "Process this: {context}")
    """
    jobs = [
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "metadata": {"context": index},
            "messages": [
                {
                    "role": "system",
                    "content": prompt_template.format(context=sentence),
                }
            ],
        }
        for index, sentence in enumerate(sentences)
    ]
    with open(filename, "w", encoding="UTF-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")


def make_dataset(
    processed_commands_path: Path,
    contexts: List[str],
    save_path: Path,
    dataset_name: str,
) -> None:
    returned_dict = {
        "context": [],
        "short_query": [],
        "medium_query": [],
        "long_query": [],
        "keywords": [],
        "scores": [],
    }
    failed = []
    # Open and iterate through the .jsonl file
    with open(processed_commands_path, "r", encoding="utf8") as file:
        for line in file:
            try:
                data = json.loads(line)
                context_id = data[-1]["context"]
                context = contexts[context_id]
                returned_data = data[1]["choices"][0]["message"]["content"]
                returned_data = json.loads(returned_data)
                returned_dict["context"].append(context)
                returned_dict["short_query"].append(returned_data["short_query"])
                returned_dict["medium_query"].append(returned_data["medium_query"])
                returned_dict["long_query"].append(returned_data["long_query"])
                returned_dict["keywords"].append(returned_data["keywords"])
                returned_dict["scores"].append(returned_data["scores"])
            except Exception as e:
                failed.append({"context": context, "exception": e})
    if failed:
        save_failed_ids(failed, dataset_name=dataset_name)

    dataset = pd.DataFrame(returned_dict)
    dataset.to_parquet(save_path, engine="pyarrow")


def save_failed_ids(failed, dataset_name):
    file_path = Path(f"datasets/failed_{dataset_name}.json")

    # Write the IDs to a text file, one per line
    with open(file_path, "w") as f:
        # Convert exceptions to string because exceptions are not JSON serializable
        json.dump(failed, f, default=str, indent=4)


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


def generate_query(contexts: List[str], save_filepath: Path):
    """
    Generates a dataset of queries based on given contexts and saves it to a specified file.

    This function processes a list of contexts to create a dataset of queries using an API.
    It performs the following steps:
    1. Sets up the environment (e.g., loads API keys).
    2. Generates a unique timestamped filename for saving commands and processed commands.
    3. Saves the initial commands in a .jsonl file.
    4. Sends the commands to the OpenAI API to get processed commands.
    5. Converts the processed commands into a dataset and saves it as a Parquet file.

    Args:
        contexts (List[str]): A list of context strings to generate queries from.
        save_filepath (Path): The path where the resulting dataset will be saved in Parquet format.

    Returns:
        None

    Example:
        >>> generate_query(["Context 1", "Context 2"], Path("dataset.parquet"))
    """
    environment_setup()

    timestamp = get_timestamp()
    dataset_name = save_filepath.stem
    command_path = Path(f"commands/comands_{dataset_name}_{timestamp}.jsonl")
    processed_command_path = Path(
        f"commands/processed_commands_{dataset_name}_{timestamp}.jsonl"
    )

    command_path.parent.mkdir(parents=True, exist_ok=True)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    save_jobs(contexts, command_path, PROMPT)
    run_api_request_processor(
        requests_filepath=command_path,
        save_filepath=processed_command_path,
        request_url="https://api.openai.com/v1/chat/completions",
    )
    make_dataset(
        processed_commands_path=processed_command_path,
        save_path=save_filepath,
        contexts=contexts,
        dataset_name=dataset_name,
    )


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


if __name__ == "__main__":
    with open("chunking_example/chunking_test_example.json", "r") as file:
        contexts = json.load(file)
    sentences = contexts["contexts"]
    generate_query(contexts=sentences, save_filepath=Path("datasets/train.parquet"))
