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
from api_request_parallel_processor import run_api_request_processor

PROMPT = """
### Goal ###
You are a helpful question generation assistant. The primary objective is to produce multiple queries in the Serbian language and a list of keywords in the Serbian language from the provided context. 
The context repesents an answer to the query and the keywords best describe the context. 
The goal is to have query-context pairs that corelate with each other and a list of keywords that would spead-up the search in the future.

### Process Overview ###
1. Carefully read and analyze the given context text.
2. Identify all relevant keywords and what the context text is about.
3. Find the queries that best represents the given context text.

### Formatting Rules ###
- Keyword value MUST be a LIST of strings with 5 keywords for each context or [null] if no relevant information is provided.
- Use double quotes for strings and escape internal quotes with a backslash (\).
- Keep the queries concise and general about the context text.
- Ensure the output is a valid JSON file, parsable by Python's json.loads().
- Strictly use only the information provided in the context text. Do not add, infer, or imagine any additional details beyond what is explicitly stated.
- Remember to answer in Serbian.

### Query description###
All queries must be complete sentences that have a meaning and realte to specific information explicitly mentioned in the context.
One query has to ask for only one information from the context.
- A query is sometimes:
   - A question that starts with a capial letter and ends with a question mark (?).
   - A statement that starts with a capital letter and ends with a period (.).
### Score description ###
   - A score is a relatedness of the context and a query. You must output a score for each query. You must score each query objectively and without bias.
   - A score must be on a scale from 1 to 5, where:
        - 1 = The answer cannot be found in the context.
        - 2 = The answer is unclear from the context. The information in the context is scarse making the answer difficult to determine.
        - 3 = Inferring the answer from the context requires interpretation. The context provides some relevant information.
        - 4 = The answer is intelligible from the context. The context provides sufficient information to understand the answer.
        - 5 = The answer is directly and explicitly provided in the context.

### Output Format ###
{{
 "keywords": ["The keyword that best represent the given context with max lenght of 5"],
 "short_query": "A short query that best suits the given context. It must be a simple sentence of lenght of 4 words and general.",
 "medium_query": "A minium lenght query that best suits the given context. It should be a lenght of min 10 words and max 18.",
 "long_query": "A long query that best suits the given context. It should be longer than 19 words and very specific to the context."
 "scores": {{
    "short_query": A  score from 1 to 5 based on previos score description. It is the relatedness of short query and the given context,
    "medium_query": A  score from 1 to 5 based on previos score description. It is the relatedness of medium query and the given context,
    "long_query": A  score from 1 to 5 based on previos score description. It is the relatedness of long query and the given context.

 }}
}}

### Context ###
{context}
"""

        
        # - 1 = A human could not find the answer from the context.
        # - 2 = A human would have diffictuly deriving the answer from the context.
        # - 3 = A human could interpret the answer from the context.
        # - 4 = A human could easily derive the answer from the context.
        # - 5 = A human could immediatly give the answer from the context.

def save_jobs(sentences: List[str], filename: Path, prompt_template: str, model: str = "gpt-3.5-turbo-0125") -> None:  
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
                "response_format": {'type': 'json_object'},
                "temperature": 0,
                "metadata": {"context": index},
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_template.format(
                            context=sentence
                        ),
                    }
                ],
            }
                for index, sentence in enumerate(sentences)
            ]
    with open(filename, "w", encoding='UTF-8') as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")

def make_dataset(processed_commands_path: Path, contexts: List[str], save_path: Path) -> None:
        
        returned_dict = {
             "context": [],
             "short_query": [],
             "medium_query": [],
             "long_query": [],
             "keywords": [],
             "scores": []
        }
        # Open and iterate through the .jsonl file
        with open(processed_commands_path, 'r', encoding='utf8') as file:
            for line in file:
                data = json.loads(line)
                context_id = data[-1]['context']
                context = contexts[context_id]
                returned_data = data[1]['choices'][0]['message']['content']
                returned_data = json.loads(returned_data)
                returned_dict['context'].append(context)
                returned_dict['short_query'].append(returned_data['short_query'])
                returned_dict['medium_query'].append(returned_data['medium_query'])
                returned_dict['long_query'].append(returned_data['long_query'])
                returned_dict['keywords'].append(returned_data['keywords'])
                returned_dict['scores'].append(returned_data['scores'])
        
        dataset = pd.DataFrame(returned_dict)
        dataset.to_parquet(save_path, engine='pyarrow')

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
    processed_command_path = Path(f"commands/processed_commands_{dataset_name}_{timestamp}.jsonl")

    command_path.parent.mkdir(parents=True, exist_ok=True)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_jobs(contexts, command_path, PROMPT)
    run_api_request_processor(requests_filepath=command_path, save_filepath=processed_command_path, request_url="https://api.openai.com/v1/chat/completions")
    make_dataset(processed_commands_path=processed_command_path, save_path=save_filepath, contexts=contexts)

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
     with open('chunking_example/chunking_test_example.json', 'r') as file:
        contexts = json.load(file) 
     sentences = contexts['contexts']
     generate_query(contexts=sentences, save_filepath=Path("datasets/train.parquet"))
