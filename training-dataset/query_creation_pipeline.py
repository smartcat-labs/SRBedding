from typing import List
from datasets import load_dataset
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas
import openai
from openai import APIError
import os
import json
import re
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint
from pathlib import Path
import tiktoken
import sys

sys.path.append("..")
from api_request_parallel_processor import run_api_request_processor


PROMPT = """
### Goal ###
Tre primary objective is to produce multiple queries in the Serbian language and a list of keywords in the Serbian language from the provided context. The context repesents an answer to the query and the keywords best 
describe the context. The goal is to have query-context pairs that corelate with each other and a list of keywords that would spead-up the search in the future.

### Process Overview ###
1. Carefully read and analyze the given context text.
2. Identify all relevant keywords and what the context text is about.
3. Find the queries that best represents the given context text.

### Formatting Rules ###
- Keyword value MUST be a LIST of strings with max lenght of 5 or [null] if no relevant information is provided.
- Use double quotes for strings and escape internal quotes with a backslash (\).
- Keep the queries concise and general about the context text.
- Ensure the output is a valid JSON file, parsable by Python's json.loads().
- Strictly use only the information provided in the context text. Do not add, infer, or imagine any additional details beyond what is explicitly stated.
- Remember to answer in Serbian.

### Query description###
- All queries must be WH-questions.
- All queries must begin with a capital letter and end with a questoin mark (?).

### Output Format ###
{{
 "keywords": ["The keyword that best represent the given context with max lenght of 5"],
 "short_query": "A short query that best suits the given context. It should be only a few words long and general.",
 "medium_query": "A minium lenght query that best suits the given context. It should be a lenght of min 10 words and max 18.",
 "long_query": "A long query that best suits the given context. It should be longer than 19 words and very specific to the context."
}}

### Context ###
{context}
"""


def save_jobs(sentences, filename, prompt_template, model: str = "gpt-3.5-turbo-0125"):    
    jobs = [
                {
                "model": model,
                "response_format": {'type': 'json_object'},
                "temperature": 0,
                "metadata": {"context": sentence},
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_template.format(
                            context=sentence
                        ),
                    }
                ],
            }
                for sentence in sentences
            ]
    with open(filename, "w", encoding='UTF-8') as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")





def make_dataset_data(file_path: Path):
        returned_dict = {
             "context": [],
             "short_query": [],
             "medium_query": [],
             "long_query": [],
             "keywords": []
        }
        # Open and iterate through the .jsonl file
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                context = data[-1]['context']
                returned_data = data[1]['choices'][0]['message']['content']
                returned_data = json.loads(returned_data)
                returned_dict['context'].append(context)
                returned_dict['short_query'].append(returned_data['short_query'])
                returned_dict['medium_query'].append(returned_data['medium_query'])
                returned_dict['long_query'].append(returned_data['long_query'])
                returned_dict['keywords'].append(returned_data['keywords'])
        return returned_dict

def make_dataset(processed_commands: Path, save_filepath: Path): 
    data_for_df = make_dataset_data(Path(processed_commands))
    dataset = pd.DataFrame(data_for_df)
    # print(dataset.head())
    table = pa.Table.from_pandas(dataset)
    pq.write_table(table, save_filepath)

def run(contexts: List[str], save_filepath: Path):
    load_dotenv()
    command_path = Path("datasets/training_comands.ljson")
    processed_command_path = Path("datasets/processed_commands.ljson")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    save_jobs(contexts, command_path, PROMPT)
    run_api_request_processor(requests_filepath=command_path, save_filepath=processed_command_path, request_url="https://api.openai.com/v1/chat/completions")
    make_dataset(processed_command_path, save_filepath)


if __name__ == "__main__":
     contexts = ["Pajton je veoma popularan programski jezik opšte namene. Postao je poznat po svojoj jednostavnosti, lakoći učenja i brzini programiranja. Mnogi profesionalni programeri koriste Pajton bar kao pomoćni jezik, jer pomoću njega brzo i lako automatizuju razne poslove. ",
                 "Za izvršavanje programa koje pišemo na Pajtonu, potreban nam je program koji se zove Pajton interpreter. Ovaj program tumači (interpretira), a zatim i izvršava Pajton naredbe. Pajton interpreteri mogu da prihvate cele programe i da ih izvrše, a mogu da rade i u interaktivnom režimu, ",
                 "Još jedan način da pokrenete Pajton školjku je da otvorite komandni prozor (na Windows sistemima to se radi pokretanjem programa cmd), a zatim u komandnom prozoru otkucate Python (ovde podrazumevamo da je Pajton instaliran tako da je dostupan iz svakog foldera, u protivnom treba se prvo pozicionirati u folder u kome se nalazi Pajton interpreter)."]
     run(contexts=contexts, save_filepath=Path("datasets/train.parquet"))