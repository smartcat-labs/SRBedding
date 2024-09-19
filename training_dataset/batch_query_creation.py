import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import openai
from dotenv import load_dotenv
from prompt_4o_testing import PROMPT_A
import sys
sys.path.append("..")


def save_jobs(
    sentences: List[str],
    filename: Path,
    prompt_template: str,
    dataset_name: str,
    model: str = "gpt-4o-mini",
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
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_template.format(context=sentence),
                    }
                ],
            },
        }
        for index, sentence in enumerate(sentences)
    ]
    with open(filename, "w", encoding="UTF-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")

    save_sentences(sentences, dataset_name)


def save_sentences(sentences: List[str], dataset_name: str):
    filename = Path(f"datasets/contexts_{dataset_name}.json")
    sentences_dict = {index: sentence for index, sentence in enumerate(sentences)}

    with open(filename, "w", encoding="UTF-8") as f:
        json.dump(sentences_dict, f, ensure_ascii=False, indent=4)


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


def batch_requests(jobs_file: Path, dataset_name: str):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    batch_file = client.files.create(file=open(jobs_file, "rb"), purpose="batch")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch_job.id)
    save_batch_id(dataset_name, batch_job)


def save_batch_id(dataset_name, batch_job):
    file_path = f"commands/number_{dataset_name}.txt"

    # Open the file in write mode and write the number
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(batch_job.id)


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

    command_path.parent.mkdir(parents=True, exist_ok=True)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    save_jobs(contexts, command_path, PROMPT_A, dataset_name)

    batch_requests(jobs_file=command_path, dataset_name=dataset_name)


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
    dataset_name = ['wiki_mini', 'news_mini', 'literature_mini']
    model = "gpt-4o-mini"
    for dataset in dataset_name:
        with open(f"datasets/{dataset}.json", "r") as file:
            contexts = json.load(file)
        sentences = contexts["contexts"][:20]
        generate_query(contexts=sentences, save_filepath=Path(f"datasets/{dataset}_{model}.parquet"))