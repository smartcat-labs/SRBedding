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


def create_sentences(texts):
    sentences = []
    for text in texts:
        matches = re.findall(r'<s>(.*?)</s>', text)
        sentences.extend(matches)
    
    return sentences


def load_datset_with_cashe(dataset_name:str):
    dir = Path("~/Datasets/SRBendding").expanduser()
    dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(dataset_name, cache_dir=dir)


def get_wiki_sentences():
    dataset = load_datset_with_cashe("jerteh/SrpWiki")
    return dataset['train']['text']


def get_news_sentences():
    dataset = load_datset_with_cashe("jerteh/SrpKorNews")
    text_news = dataset['train']['text']
    return create_sentences(text_news)


def get_sience_sentences():
    dataset = load_datset_with_cashe("procesaur/STARS")
    text_sci = dataset['train']['text']
    return create_sentences(text_sci[:450_000])

def get_literature_sentences():
    dataset = load_datset_with_cashe("jerteh/SrpELTeC")


def get_datasets():
    datasets = [{
    "loading_function": get_wiki_sentences,
    "final_lenght": 8000,
    "chunked_lenght": 100,
    "random_step":100,
    "random_step_start": 55
},
]