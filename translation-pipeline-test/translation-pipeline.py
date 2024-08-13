import json
import os
import sys
from pprint import pprint

import openai
import sentence_transformers
import transformers
from datasets import load_dataset
from openai import OpenAI

from prompts import SYSTEM_PROMPT  # Place it where it needs to go


def make_jobs(model, prompt, filename, dataset):  # type hint
    jobs = [
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "metadata": {"query_id": sample["query_id"]},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(sample)},
            ],
        }
        for sample in dataset
    ]
    with open(filename, "w") as f:  # Convert to method
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


def load_data(
    dataset_name="microsoft/ms_marco",
):  # Make it a config variable + type hint
    data = load_dataset(dataset_name, "v1.1")
    data_test_split = data["test"]
    ms_marco = data_test_split.select_columns(["passages", "query", "query_id"])

    final_data = []
    for i in range(3):
        final_data.append(
            {
                "query_id": ms_marco["query_id"][i],
                "query": ms_marco["query"][i],
                "passage_text": ms_marco["passages"][i]["passage_text"],
            }
        )

    return final_data


if __name__ == "__main__":
    # 1. load data
    # 2. Make jobs
    # 3. Write jobs to file, jsonl
    # 4. make requests to OpenAI API in parallel
    # Save to parquet in your format
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-3.5-turbo-0125"

    final_data = load_data()

    make_jobs(model, SYSTEM_PROMPT, "test", final_data)
