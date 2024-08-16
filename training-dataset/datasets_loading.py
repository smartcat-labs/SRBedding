from datasets import load_dataset
import re
from pathlib import Path


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
    text_lit = dataset['train']['text']
    return create_sentences(text_lit)


def get_datasets():
    return {
        "wiki": {
            "loading_function": get_wiki_sentences,
            "final_lenght": 8_000,
            "chunked_lenght": 30,
            "random_step":10_000,
            "random_step_start": 2_000
        },
        "news": {
            "loading_function": get_news_sentences,
            "final_lenght": 2_000,
            "chunked_lenght": 30,
            "random_step":100_000,
            "random_step_start": 2_000
        },
        "science": {
            "loading_function": get_sience_sentences,
            "final_lenght": 1_250,
            "chunked_lenght": 30,
            "random_step":10_000,
            "random_step_start": 2_000
        },
        "literature": {
            "loading_function": get_literature_sentences,
            "final_lenght": 1_250,
            "chunked_lenght": 30,
            "random_step":100,
            "random_step_start": 0
        },
    }
