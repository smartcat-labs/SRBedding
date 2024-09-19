import json
import os
from pathlib import Path
from typing import Dict, List

import openai
import pandas as pd

from datasets_loading import get_datasets


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
                context_id = data["custom_id"].replace("task-", "")
                context = contexts[context_id]
                returned_data = data["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                returned_data = json.loads(returned_data)
                short_query = returned_data["short_query"]
                medium_query = returned_data["medium_query"]
                long_query = returned_data["long_query"]
                keywords = returned_data["keywords"]
                scores = returned_data["scores"]

                # Add the data to the returned_dict
                returned_dict["short_query"].append(short_query)
                returned_dict["medium_query"].append(medium_query)
                returned_dict["long_query"].append(long_query)
                returned_dict["keywords"].append(keywords)
                returned_dict["scores"].append(scores)
                returned_dict["context"].append(context)
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


def get_batch_id(file: Path) -> str:
    with open(file, "r") as file:
        text = file.read()
    return text.strip()


def load_contexts(dataset_name: str) -> Dict[int, str]:
    filename = Path(f"datasets/contexts_{dataset_name}_train.json")

    with open(filename, "r", encoding="UTF-8") as f:
        sentences_dict = json.load(f)

    return sentences_dict


if __name__ == "__main__":
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    datasets = get_datasets()
    for dataset_name in datasets.keys():
        file_path = f"commands/number_{dataset_name}_train.txt"
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

        contexts = load_contexts(dataset_name)
        make_dataset(
            dataset_name=dataset_name,
            contexts=contexts,
            processed_commands_path=result_file_name,
            save_path=final_save_path,
        )
