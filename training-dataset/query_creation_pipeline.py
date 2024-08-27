from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import os
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
import openai
import os
import json
from pathlib import Path
import sys

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
   - A score is a similarity between the context and a query. You must output a score for each query.
   - A score must be on a scale from 1 to 5, where:
      - 1 = not similar at all
      - 2 = vaguely similar
      - 3 = moderately similar
      - 4 = almost completely similar
      - 5 = completely similar

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

def make_dataset_data(file_path: Path) -> Dict[str, List[Any]]:
        """
        Processes a .jsonl file to create a dataset dictionary.

        This function reads a .jsonl file where each line contains a JSON object. It extracts 
        specific fields from the JSON objects and organizes them into a dictionary with keys 
        such as 'context', 'short_query', 'medium_query', 'long_query', 'keywords', and 'scores'.

        The resulting dictionary contains lists of values corresponding to each key, which can 
        be used as a dataset for further processing or model training.

        Args:
            file_path (Path): The path to the .jsonl file containing the data.

        Returns:
            Dict[str, List]: A dictionary with the following keys:
                - 'context': List of context strings extracted from the file.
                - 'short_query': List of short query strings extracted from the file.
                - 'medium_query': List of medium query strings extracted from the file.
                - 'long_query': List of long query strings extracted from the file.
                - 'keywords': List of keyword lists extracted from the file.
                - 'scores': List of Dict scores extracted from the file.

        Example:
            >>> dataset = make_dataset_data(Path("data.jsonl"))
        """
        returned_dict = {
             "context": [],
             "short_query": [],
             "medium_query": [],
             "long_query": [],
             "keywords": [],
             "scores": []
        }
        # Open and iterate through the .jsonl file
        with open(file_path, 'r', encoding='utf8') as file:
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
                returned_dict['scores'].append(returned_data['scores'])
        
        return returned_dict

def make_dataset(processed_commands: Path, save_filepath: Path): 
    """
    Creates and saves a dataset from processed commands.

    This function reads processed command data from a .jsonl file, converts it into a 
    pandas DataFrame, and then saves it as a Parquet file. The DataFrame is created 
    using the `make_dataset_data` function, and the resulting dataset is saved to 
    the specified file path in Parquet format.

    Args:
        processed_commands (Path): The path to the .jsonl file containing the processed commands.
        save_filepath (Path): The path where the resulting dataset will be saved in Parquet format.

    Returns:
        None

    Example:
        >>> make_dataset(Path("commands.jsonl"), Path("dataset.parquet"))
    """
    data_for_df = make_dataset_data(Path(processed_commands))
    dataset = pd.DataFrame(data_for_df)
    table = pa.Table.from_pandas(dataset)
    pq.write_table(table, save_filepath)

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
    
    save_jobs(contexts, command_path, PROMPT)
    run_api_request_processor(requests_filepath=command_path, save_filepath=processed_command_path, request_url="https://api.openai.com/v1/chat/completions")
    make_dataset(processed_commands=processed_command_path, save_filepath=save_filepath)

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
     contexts = ["Pajton je veoma popularan programski jezik opšte namene. Postao je poznat po svojoj jednostavnosti, lakoći učenja i brzini programiranja. Mnogi profesionalni programeri koriste Pajton bar kao pomoćni jezik, jer pomoću njega brzo i lako automatizuju razne poslove. ",
                 "Za izvršavanje programa koje pišemo na Pajtonu, potreban nam je program koji se zove Pajton interpreter. Ovaj program tumači (interpretira), a zatim i izvršava Pajton naredbe. Pajton interpreteri mogu da prihvate cele programe i da ih izvrše, a mogu da rade i u interaktivnom režimu, ",
                 "Još jedan način da pokrenete Pajton školjku je da otvorite komandni prozor (na Windows sistemima to se radi pokretanjem programa cmd), a zatim u komandnom prozoru otkucate Python (ovde podrazumevamo da je Pajton instaliran tako da je dostupan iz svakog foldera, u protivnom treba se prvo pozicionirati u folder u kome se nalazi Pajton interpreter).",
                 "Novi Sad je Evropska prestonica kulture 2022. Novi Sad je, posle Beograda, drugi grad u Srbiji po broju stanovnika (bez podataka za područje Kosova i Metohije). Na poslednjem zvaničnom popisu iz 2011. godine, sam grad je imao 231.798[2] stanovnika. Na opštinskom području Novog Sada (uključujući i prigradska nasenja) broj stanovnika je 2011. godine iznosio 341.625.[",
                 "Najstariji arheološki ostaci (iz vremena kamenog doba) pronađeni su sa obe strane Dunava, na području današnjeg Petrovaradina (koji je u kontinuitetu nastanjen od praistorije do danas) i području današnje Klise. Istraživanjem ostataka naselja iz mlađeg bronzanog doba (3000. godina pne.) na području današnjeg Petrovaradina, arheolozi su pronašli i bedeme pojačane koljem i palisadama iz tog perioda, koji svedoče da je još u vreme vučedolske kulture ovde postojalo utvrđeno naselje.",
                 "Sredinom 6. veka, područje naseljavaju Sloveni. U 9. veku, tvrđava Petrikon (ili na slovenskim jezicima - Petrik) ulazi u sastav Bugarskog carstva, a u 11. veku njom vlada sremski vojvoda Sermon, čiji su zlatnici u 19. veku pronađeni u jednom petrovaradinskom vinogradu. Pošto je Bugarska poražena od Vizantije a Sermon ubijen, tvrđava ponovo postaje deo Vizantije, da bi, posle borbi Vizantinaca i Mađara, krajem 12. veka, ušla u sastav srednjovekovne Kraljevine Ugarske. Na području Bačke, ugarska vlast se ustaljuje nešto ranije, tokom 10. veka.",
                 "U vreme Osmanske uprave poznat pod imenom Varadin, Petrovaradin je bio sedište nahije u okviru Sremskog sandžaka. Podgrađe tvrđave imalo je oko 200 kuća, tu se nalazila Sulejman-hanova džamija, a postojale su i dve manje džamije, Hadži-Ibrahimova i Huseinova. Pored dve turske mahale, u sastavu grada nalazila se i hrišćanska četvrt sa 35 kuća, naseljenih isključivo Srbima.[8] Od praistorije pa sve do kraja 17. veka, centar urbanog života na području današnjeg grada nalazio se na sremskoj strani Dunava, na prostoru današnjeg Petrovaradina, koji je svojim značajem uvek zasenjivao naselja na bačkoj strani.",
                 "Tokom većeg dela 18. i 19. veka Novi Sad je bio središte kulturnog, političkog i društvenog života celokupnog srpskog naroda, koji tada nije imao sopstvenu državu, a prema podacima iz prve polovine 19. veka, ovo je bio i najveći grad nastanjen Srbima (Oko 1820. godine Novi Sad je imao oko 20.000 stanovnika, od kojih su oko dve trećine bili Srbi, a današnji najveći srpski grad, Beograd, nije dostigao približan broj stanovnika pre 1853. godine). Zbog svog kulturnog i političkog uticaja, Novi Sad je postao poznat kao Srpska Atina.",
                 "Posle sprovedenih izbora po svim vojvođanskim mestima (od 18. do 24. novembra), u Novom Sadu se 25. novembra 1918. godine sastala Velika narodna skupština Srba, Bunjevaca i ostalih Slovena Banata, Bačke i Baranje, koja zvanično proglašava otcepljenje ovih regiona od Ugarske i njihovo prisajedinjenje Srbiji, a na istoj skupštini formira se i pokrajinska vlada (Narodna uprava) Banata, Bačke i Baranje sa sedištem u Novom Sadu. 1. decembra 1918. godine, proglašeno je Kraljevstvo Srba, Hrvata i Slovenaca, a Novi Sad ulazi u tu novu državu kao sastavni deo Kraljevine Srbije.",
                 " Novom Sadu je, tokom celog rata, delovao pokret otpora i oslobodilački pokret pod vođstvom Komunističke Partije Jugoslavije. U gradu se nalazilo sedište okružnog komiteta Komunističke Partije Jugoslavije, koji je u okviru narodnooslobodilačkog pokreta bio deo Pokrajinskog komiteta KPJ za Vojvodinu. U narodnooslobodilačkoj borbi, u partizanskim odredima i vojvođanskim brigadama, neposredno je učestvovalo 2.365 Novosađana, pripadnika svih nacionalnosti, (Srba, Mađara, Slovaka i ostalih)"
                 ]
     generate_query(contexts=contexts, save_filepath=Path("datasets/train.parquet"))