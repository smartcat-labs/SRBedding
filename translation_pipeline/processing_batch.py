import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI

sys.path.append("..")
from utils import make_path
from utils_openAI import environment_setup, get_batch_id, save_failed_ids


def make_dataset(
    processed_commands_path: Path, dataset_name: str
) -> Dict[str, str | List[str]]:
    """
    Creates a dataset by extracting queries and passage texts from a processed commands JSONL file.

    Args:
        processed_commands_path (Path): The file path to the processed commands in JSONL format.
        dataset_name (str): The name of the dataset, used for saving failed IDs.

    Returns:
        Dict[str, str | List[str]]: A dictionary containing lists of 'id', 'query', and 'passage_text'.
    """
    returned_dict = {"id": [], "query": [], "passage_text": []}
    failed = []
    # Open and iterate through the .jsonl file
    with open(processed_commands_path, "r") as file:
        for line in file:
            id_ = None
            try:
                data = json.loads(line)
                id_ = data["custom_id"].replace("task-", "")
                returned_data = data["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                tranlation = json.loads(returned_data)
                query = tranlation["query"]
                passages = tranlation["passage_text"]

                returned_dict["query"].append(query)
                returned_dict["passage_text"].append(passages)
                # returned_dict["explanation"].append(tranlation["explanation"])
                returned_dict["id"].append(id_)
            except Exception as e:
                failed.append({"id": id_, "exception": e})
    if failed:
        save_failed_ids(failed, dataset_name=dataset_name)

    # pprint(returned_dict)
    return returned_dict


def save_in_file(processed_commands_path: Path, save_path: Path) -> None:
    """
    Saves a dataset to a Parquet file by processing commands from a JSONL file.

    Args:
        processed_commands_path (Path): The file path to the processed commands in JSONL format.
        save_path (Path): The file path where the dataset will be saved as a Parquet file.

    Returns:
        None
    """
    data_for_df = make_dataset(processed_commands_path, save_path.stem)
    make_path(save_path.parent)
    dataset = pd.DataFrame(data_for_df)
    dataset["passage_text"] = dataset["passage_text"].apply(
        lambda x: np.array(x, dtype=str)
    )
    dataset.to_parquet(save_path, engine="pyarrow")


if __name__ == "__main__":
    environment_setup()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    datasets = ["msmarco", "naquestions"]
    for dataset_name in datasets:
        file_path = f"commands/number_{dataset_name}.txt"
        batch_id = get_batch_id(file=file_path)

        batch_job = client.batches.retrieve(batch_id)
        print(batch_job.status)
        if batch_job.status != "completed":
            continue
        print(batch_job)
        result_file_name = f"commands/results_{dataset_name}.jsonl"
        result_file_id = batch_job.output_file_id
        result = client.files.content(result_file_id).content
        with open(result_file_name, "wb") as file:
            file.write(result)

        final_save_path = Path(f"datasets/{dataset_name}.parquet")
        save_in_file(result_file_name, save_path=final_save_path)
