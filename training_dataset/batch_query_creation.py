import json
import sys
from pathlib import Path
from typing import Dict

from query_prompt import PROMPT_OLD

sys.path.append("..")
from utils import get_timestamp, make_path
from utils_openAI import batch_requests, environment_setup, make_jobs


def save_sentences(sentences_dict: Dict[int, str], dataset_name: str):
    filename = Path(f"datasets/contexts_{dataset_name}.json")

    with open(filename, "w", encoding="UTF-8") as f:
        json.dump(sentences_dict, f, ensure_ascii=False, indent=4)


def make_data_for_job(contexts):
    return [{"id": key, "sentence": value} for key, value in contexts.items()]


def generate_query(contexts: Dict[int, str], save_filepath: Path):
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

    make_path(command_path.parent)
    make_path(save_filepath.parent)

    save_sentences(contexts, dataset_name)

    job_structure = make_data_for_job(contexts)
    make_jobs(prompt=PROMPT_OLD, filename=command_path, dataset=job_structure)
    batch_requests(jobs_file=command_path, dataset_name=dataset_name)


if __name__ == "__main__":
    with open("chunking_example/chunking_test_example.json", "r") as file:
        contexts = json.load(file)
    sentences = contexts["contexts"]
    sentences_dict = {index: sentence for index, sentence in enumerate(sentences)}

    generate_query(
        contexts=sentences_dict, save_filepath=Path("datasets/train.parquet")
    )
