import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI
from pprint import pprint
from datasets import load_dataset

sys.path.append("..")


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
    returned_dict = {}
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
                sentence = tranlation["sentence"]

                returned_dict[id_] = sentence
            except Exception as e:
                failed.append({"id": id_, "exception": e})
    if failed:
        save_failed_ids(failed, dataset_name=dataset_name)

    # pprint(returned_dict)
    return returned_dict


def make_cache_dir() -> Path:
    """
    Creates and returns the cache directory path for storing datasets.

    Returns:
        Path: The expanded user path to the cache directory.
    """
    return Path("~/Datasets/SRBendding").expanduser()


def make_final(tranlated_sentences):
    datasets_dir = make_cache_dir()
    sts = load_dataset("mteb/stsbenchmark-sts", cache_dir=datasets_dir)
    sts_test_df = pd.DataFrame(sts["test"])
    for i, row in sts_test_df.iterrows():
        sid = row["sid"]
        srb = tranlated_sentences[sid]
        sts_test_df.at[i, "sentence2"] = srb  # Modify the 'sentence2' column directly
        print(srb)
    return sts_test_df

def save_in_file(processed_commands_path: Path, save_path: Path) -> None:
    tranlated_sentences = make_dataset(processed_commands_path, save_path.stem)
    dataset = make_final(tranlated_sentences)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(save_path, engine="pyarrow")


def get_batch_id(file: Path) -> str:
    with open(file, "r") as file:
        text = file.read()
    return text.strip()


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    dataset_name = "STS"
    file_path = f"commands/number_{dataset_name}.txt"
    batch_id = get_batch_id(file=file_path)
    print(batch_id)
    batch_job = client.batches.retrieve(batch_id)
    print(batch_job.status)
    print(batch_job)
    result_file_name = f"commands/results_{dataset_name}.jsonl"
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    with open(result_file_name, "wb") as file:
        file.write(result)

    final_save_path = Path(f"datasets/{dataset_name}.parquet")
    save_in_file(result_file_name, save_path=final_save_path)