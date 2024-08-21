import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from prompts import SYSTEM_PROMPT

sys.path.append("..") 
from api_request_parallel_processor import run_api_request_processor



def make_jobs(model, prompt, filename, dataset):  # type hint
    jobs = [
        {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "metadata": {"id": sample["query_id"]},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(sample)},
            ],
        }
        for sample in dataset
    ]
    save_jobs(filename, jobs)

def save_jobs(filename, jobs):
    with open(filename, "w", encoding='utf-8') as f: 
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")

def load_data_ms_marco(dataset_name="microsoft/ms_marco"):
    datasets_dir = make_cache_dir()
    data = load_dataset(dataset_name, "v1.1", cache_dir=datasets_dir)
    data_test_split = data["test"]
    ms_marco = data_test_split.select_columns(["passages", "query", "query_id"])

    final_data = []
    for i in range(100):
        final_data.append(
            {
                "query_id": str(ms_marco["query_id"][i]),
                "query": ms_marco["query"][i],
                "passage_text": ms_marco["passages"][i]["passage_text"],
            }
        )

    return final_data

def make_cache_dir():
    return Path("~/Datasets/SRBendding").expanduser()

def save_failed_ids(failed, dataset_name):
    file_path = Path(f'translation_pipeline_test/failed_{dataset_name}.json')

    # Write the IDs to a text file, one per line
    with open(file_path, "w") as f:
        # Convert exceptions to string because exceptions are not JSON serializable
        json.dump(failed, f, default=str, indent=4)


def make_dataset(file_path: Path, dataset_name): #file_path is a path to chat gpt translation results
        returned_dict = {
             "id": [],
             "query": [],
             "passage_text": [],
        }
        failed = []
        # Open and iterate through the .jsonl file
        with open(file_path, 'r') as file:
            for line in file:
                id_ = None
                try:
                    data = json.loads(line)
                    id_ = data[-1]['id']
                    returned_data = data[1]['choices'][0]['message']['content']
                    tranlation = json.loads(returned_data) # gpt message i.e. translation in this case
                    returned_dict['id'].append(id_)
                    returned_dict['query'].append(tranlation['query'])
                    returned_dict['passage_text'].append(tranlation['passage_text'])
                except Exception as e:
                    failed.append({
                        'id': id_,
                        'exception': e
                    })
        if failed:
            save_failed_ids(failed, dataset_name=dataset_name)
        return returned_dict

def save_in_file(processed_commands_path, save_path: Path):
    data_for_df = make_dataset(processed_commands_path, save_path.stem)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.DataFrame(data_for_df)
    # dataset['id'] = dataset['id'].astype(str)
    # dataset['query'] = dataset['query'].astype(str)
    dataset['passage_text'] = dataset['passage_text'].apply(lambda x: np.array(x, dtype=str))
    dataset.to_parquet(save_path, engine='pyarrow')

def load_data_natural(dataset_name:str = "google-research-datasets/natural_questions"):
    dir = make_cache_dir()
    data = load_dataset(dataset_name, "dev", cache_dir=dir)  
    validation_dataset = data['validation']

    result = []
    i = 0
    while len(result) < 100 or i >= len(validation_dataset):
    # for i in range(len(validation_dataset) - 7720):
        record = validation_dataset[i]
        id = record['id']
        start_byte, end_byte = get_start_and_end_byte(record)
        context = make_context(record, start_byte, end_byte)
        question = record['question']['text']
        current = {
            "query_id": id,
            "query": question,
            "passage_text": [context],

        }
        i += 1
        if context != "" and len(context)< 48875:
            result.append(current)

    return result

def make_context(info, start_byte, end_byte):
    context = ""    
    token_element = info['document']['tokens']
    for j in range(len(token_element['token'])):
        if not token_element['is_html'][j] and start_byte <= token_element['start_byte'][j] <= end_byte:
            context += (token_element['token'][j]).strip() + " "
    context = context.strip()
    return context

def get_start_and_end_byte(info):
    start_byte = -1
    end_byte = -1
    long_answers = info['annotations']['long_answer']
    for j in range(len(long_answers)):
        if long_answers[j]["start_byte"] != -1:
            start_byte = long_answers[j]["start_byte"]
            end_byte = long_answers[j]["end_byte"]
            break
    return start_byte,end_byte

if __name__ == "__main__":

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-3.5-turbo-0125"
    datasets = [
        {'name': "msmarco",
         'loading_function': load_data_ms_marco},
        {'name': "naquestions",
         'loading_function': load_data_natural}
    ]

    for dataset in datasets:
        final_data = dataset['loading_function']()
        path = Path("translation_pipeline_test/test.jsonl")
        commands_filepath = Path(f"translation_pipeline_test/{dataset['name']}_test_results.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        make_jobs(model=model, prompt=SYSTEM_PROMPT, filename=path, dataset=final_data)
        run_api_request_processor(requests_filepath=path, save_filepath=commands_filepath, request_url="https://api.openai.com/v1/chat/completions")
        final_save_path = Path(f"datasets/{dataset['name']}.parquet")
        save_in_file(commands_filepath, save_path=final_save_path)

