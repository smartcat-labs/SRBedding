import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

sys.path.append("..")
from utils import make_cache_dir, make_path
from utils_openAI import get_batch_id, save_failed_ids


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

    return returned_dict


def make_final(tranlated_sentences):
    datasets_dir = make_cache_dir()
    sts = load_dataset("mteb/stsbenchmark-sts", cache_dir=datasets_dir)
    sts_test_df = pd.DataFrame(sts["test"])
    for i, row in sts_test_df.iterrows():
        if i > 10:
            break
        sid = row["sid"]
        srb = tranlated_sentences[sid]
        sts_test_df.at[i, "sentence2"] = srb  # Modify the 'sentence2' column directly
    return sts_test_df


def save_in_file(processed_commands_path: Path, save_path: Path) -> None:
    tranlated_sentences = make_dataset(processed_commands_path, save_path.stem)
    dataset = make_final(tranlated_sentences)
    make_path(save_path.parent)
    dataset.to_parquet(save_path, engine="pyarrow")


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    dataset_name = "STS"
    file_path = f"commands/number_{dataset_name}.txt"
    batch_id = get_batch_id(file=file_path)
    batch_job = client.batches.retrieve(batch_id)
    print(batch_job.status)
    if batch_job.status != "completed":
        exit()

    result_file_name = f"commands/results_{dataset_name}.jsonl"
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    with open(result_file_name, "wb") as file:
        file.write(result)

    final_save_path = Path(f"datasets/{dataset_name}.parquet")
    save_in_file(result_file_name, save_path=final_save_path)
