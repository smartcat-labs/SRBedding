import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, load_dataset


def create_sentences(texts: List[str]) -> List[str]:
    """
    Extracts and returns sentences wrapped in <s> and </s> tags from a list of texts.

    This function processes a list of text strings, searching each text for sentences
    enclosed within `<s>` and `</s>` tags. It extracts these sentences and returns
    them as a list.

    Args:
        texts (List[str]): A list of text strings to search for sentences.

    Returns:
        List[str]: A list of sentences found between `<s>` and `</s>` tags in the input texts.

    Example:
        >>> texts = ["<s>Sentence 1</s> Some other text <s>Sentence 2</s>"]
        >>> sentences = create_sentences(texts)
        >>> print(sentences)
        ['Sentence 1', 'Sentence 2']
    """
    sentences = []
    for text in texts:
        matches = re.findall(r"<s>(.*?)</s>", text)
        for match in matches:
            if not match.endswith("."):
                match += "."
            sentences.append(match)

    return sentences


def load_datset_with_cashe(dataset_name: str) -> Dataset:
    """
    Loads a dataset using the Hugging Face `datasets` library with a specified cache directory.

    This function loads a dataset by name using the Hugging Face `datasets` library,
    storing the dataset in a specified cache directory. If the directory does not exist,
    it is created.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dataset: The loaded dataset from the Hugging Face `datasets` library.

    Example:
        >>> dataset = load_dataset_with_cache("dataset_name")
    """
    dir = Path("~/Datasets/SRBendding").expanduser()
    # dir = Path("D:/Datasets/SRBendding")

    dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(dataset_name, cache_dir=dir)


def add_period_to_sentences(sentences):
    updated_sentences = []
    for sentence in sentences:
        if not sentence.endswith("."):
            sentence += "."
        updated_sentences.append(sentence)
    return updated_sentences


def get_wiki_sentences() -> List[str]:
    """
    Loads and returns sentences from the Serbian Wikipedia dataset.

    This function loads the "jerteh/SrpWiki" dataset using the `load_dataset_with_cache` function
    and returns the text from the training split of the dataset.

    Returns:
        List[str]: A list of sentences from the Serbian Wikipedia dataset.

    Example:
        >>> wiki_sentences = get_wiki_sentences()
    """
    dataset = load_datset_with_cashe("jerteh/SrpWiki")
    wiki_sentences = dataset["train"]["text"]
    return add_period_to_sentences(wiki_sentences)


def get_news_sentences() -> List[str]:
    """
    Loads and returns sentences from the Serbian news dataset.

    This function loads the "jerteh/SrpKorNews" dataset using the `load_dataset_with_cache` function.
    It extracts sentences wrapped in `<s>` and `</s>` tags from the text in the training split.

    Returns:
        List[str]: A list of sentences extracted from the Serbian news dataset.

    Example:
        >>> news_sentences = get_news_sentences()
    """
    dataset = load_datset_with_cashe("jerteh/SrpKorNews")
    text_news = dataset["train"]["text"]
    return create_sentences(text_news)


def get_sience_sentences() -> List[str]:
    """
    Loads and returns sentences from the Serbian science dataset.

    This function loads the "procesaur/STARS" dataset using the `load_dataset_with_cache` function.
    It extracts the first 450,000 text entries from the training split and creates sentences
    wrapped in `<s>` and `</s>` tags.

    Returns:
        List[str]: A list of sentences extracted from the Serbian science dataset.

    Example:
        >>> science_sentences = get_sience_sentences()
    """
    dataset = load_datset_with_cashe("procesaur/STARS")
    text_sci = dataset["train"]["text"]
    return create_sentences(text_sci[:450_000])


def get_literature_sentences() -> List[str]:
    """
    Loads and returns sentences from the Serbian literature dataset.

    This function loads the "jerteh/SrpELTeC" dataset using the `load_dataset_with_cache` function.
    It extracts sentences wrapped in `<s>` and `</s>` tags from the text in the training split.

    Returns:
        List[str]: A list of sentences extracted from the Serbian literature dataset.

    Example:
        >>> literature_sentences = get_literature_sentences()
    """
    dataset = load_datset_with_cashe("jerteh/SrpELTeC")
    text_lit = dataset["train"]["text"]
    return create_sentences(text_lit)


def get_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of datasets with their corresponding loading functions and parameters.

    This function provides a dictionary containing different datasets, each associated with
    its specific loading function and parameters. These parameters include the final length
    of the subset to be generated, the length of each chunk, and the random steps used
    for selecting chunks of sentences.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where the keys are dataset names and the values
        are dictionaries containing:
            - "loading_function" (Callable): The function used to load sentences from the dataset.
            - "final_lenght" (int): The target number of sentences in the final subset.
            - "chunked_lenght" (int): The number of sentences in each chunk.
            - "random_step" (int): The maximum additional step between chunks, selected randomly.
            - "random_step_start" (int): The minimum additional step between chunks.

    Example:
        >>> datasets = get_datasets()
        >>> wiki_params = datasets["wiki"]
        >>> loading_function = wiki_params["loading_function"]
        >>> sentences = loading_function()
    """
    return {
        "wiki": {
            "loading_function": get_wiki_sentences,
            "final_lenght": 52_000,
            "chunked_lenght": 40,
            "jump": 500,
        },
        "news": {
            "loading_function": get_news_sentences,
            "final_lenght": 90_000,
            "chunked_lenght": 40,
            "jump": 600,
        },
        # "science": {
        #     "loading_function": get_sience_sentences,
        #     "final_lenght": 90_000,
        #     "chunked_lenght": 40,
        #     "jump": 200,
        # },
        "literature": {
            "loading_function": get_literature_sentences,
            "final_lenght": 36_000,
            "chunked_lenght": 30,
            "jump": 200,
        },
    }
