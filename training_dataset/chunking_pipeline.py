import json
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from statistics import median
from typing import Dict, List

import numpy as np
import openai
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("..")
from api_request_parallel_processor import run_api_request_processor


def save_jobs(
    sentences: List[Dict[str, str]],
    filename: Path,
    model: str = "text-embedding-3-small",
) -> None:
    jobs = [
        {
            "model": model,
            "metadata": {"id": sentence["id"]},
            "input": sentence["combined_sentence"],
        }
        for sentence in sentences
    ]
    with open(filename, "w", encoding="UTF-8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")


def return_dic(sentences: List[str]) -> List[Dict[str, str | int]]:
    """
    Generates a list of dictionaries from a list of sentences.

    Each dictionary contains a sentence and its corresponding index in the original list.

    Parameters:
    sentences (List[str]): A list of sentences.

    Returns:
    List[Dict[str, str|int]]: A list where each element is a dictionary with keys:
        - 'sentence' [str]: The original sentence.
        - 'id' [int]: The index of the sentence in the input list.
         Example:
        >>> texts = ["Sentence 1" "Sentence 2"]
        >>> sentences_dic = return_dic(texts)
        >>> print(sentences_dic)
    """
    sent_lst_dic = [{"sentence": x, "id": i} for i, x in enumerate(sentences)]
    return sent_lst_dic


def combine_sentences(
    sentences: List[Dict[str, str]], buffer_size: int
) -> List[Dict[str, str]]:
    """
    Combines each sentence in a list with its surrounding sentences based on a buffer size.

    For each sentence, the function concatenates the sentences within the specified buffer size before and after it,
    creating a new key 'combined_sentence' in each dictionary.

    Parameters:
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'sentence' key.
    buffer_size (int): The number of sentences before and after the current sentence to include in the combination.

    Returns:
    List[Dict[str, str]]: The updated list of dictionaries with each dictionary containing an additional 'combined_sentence' key.
     Example:
    >>> texts = ["Sentence 1", "Sentence 2"]
    >>> sentences_dic = return_dic(texts)
    >>> print(sentences_dic)
    """
    for i in range(len(sentences)):
        combined_sentence = ""

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]["sentence"] + " "
        combined_sentence += sentences[i]["sentence"]

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += " " + sentences[j]["sentence"]

        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def _make_embedding_dict(file_path):
    returned_dict = {}
    with open(file_path, "r", encoding="UTF-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                indx = data[-1]["id"]
                embedding = data[1]["data"][0]["embedding"]
                returned_dict[indx] = embedding
            except Exception as _:
                print("failed " + str(indx))
                returned_dict[indx] = [0.0] * 1536
    return returned_dict


def save_combined_embeddings_jobs(sentences: List[Dict[str, str]]) -> None:
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    sentence_path = Path(f"commands/cobined_sentences_{date}.jsonl")
    saved_path = Path(f"commands/combined_sentences_embeddings_{date}.jsonl")
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    save_jobs(sentences, sentence_path)
    run_api_request_processor(
        sentence_path, saved_path, "https://api.openai.com/v1/embeddings"
    )
    return _make_embedding_dict(saved_path)


def generate_embeddings(sentences: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Generates embeddings for combined sentences and adds them to each sentence dictionary.

    The function uses the `oaiembeds.embed_documents` method to generate embeddings for the 'combined_sentence'
    and appends the resulting embeddings as a new key 'combined_sentence_embedding' in each dictionary.

    Parameters:
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'combined_sentence' key.

    Returns:
    List[Dict[str, str]]: The updated list of dictionaries with an additional 'combined_sentence_embedding' key.

    Example:
    >>> sentences = [
    ...     {'sentence': 'This is the first sentence.', 'combined_sentence': 'This is the first sentence. This is the second sentence.'},
    ...     {'sentence': 'This is the second sentence.', 'combined_sentence': 'This is the first sentence. This is the second sentence. This is the third   sentence.'}
    ... ]
    >>> embeddings = generate_embeddings(sentences)
    >>> print(embeddings)
    """
    sent_dict = save_combined_embeddings_jobs(sentences)
    for sentence in sentences:
        sentence["combined_sentence_embedding"] = sent_dict[sentence["id"]]

    return sentences


def calculate_cosine_distances(
    sentences: List[Dict[str, str]],
) -> tuple[List[int], List[str]]:
    """
    Calculates cosine distances between the embeddings of consecutive sentences and adds the distance to each sentence dictionary.

    The function computes the cosine distance between the 'combined_sentence_embedding' of each sentence and the next one in the list.
    It appends the distance as a new key 'distance_to_next' in each dictionary and returns the list of distances along with the updated list of dictionaries.

    Parameters:
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'combined_sentence_embedding' key.

    Returns:
    Tuple[List[float], List[Dict[str, str]]]:
        - A list of cosine distances between consecutive sentence embeddings.
        - The updated list of dictionaries with an additional 'distance_to_next' key.

    Example:
    >>> sentences = [
    ...     {'sentence': 'This is the first sentence.', 'combined_sentence_embedding': [0.1, 0.2, 0.3]},
    ...     {'sentence': 'This is the second sentence.', 'combined_sentence_embedding': [0.4, 0.5, 0.6]}]
    >>> distances, updated_sentences = calculate_cosine_distances(sentences)
    >>> print(distances)
    >>> print(updated_sentences)
    """
    distances = []

    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)

        sentences[i]["distance_to_next"] = distance

    return distances, sentences


def get_breakpoint(
    distances: List[int], sentences: List[str], threshold: int
) -> List[str]:
    """
    Splits a list of sentences into chunks based on a distance threshold.
    If you want more chunks, lower the percentile cutoff.

    The function identifies breakpoints in the sentence list where the cosine distances between sentence embeddings
    exceed a specified percentile threshold. It then splits the sentences into chunks at those breakpoints.

    Parameters:
    distances (List[int]): A list of cosine distances between consecutive sentence embeddings.
    sentences (List[Dict[str, str]]): A list of dictionaries, each containing a 'sentence' key.
    threshold (int): The percentile threshold for determining breakpoints. Lower thresholds result in more chunks.

    Returns:
    List[str]: A list of combined text chunks, where each chunk is formed by joining sentences that fall between breakpoints.

    Example:
    >>> distances = [0.1, 0.4, 0.7, 0.2]
    >>> sentences = [
    ...     {'sentence': 'This is the first sentence.'},
    ...     {'sentence': 'This is the second sentence.'},
    ...     {'sentence': 'This is the third sentence.'}
    ... ]
    >>> chunks = get_breakpoint(distances, sentences, 50)
    >>> print(chunks)
    """
    breakpoint_distance_threshold = np.percentile(distances, threshold)
    indices_above_thresh = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]
    start_index = 0
    chunks = []

    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group])
        chunks.append(combined_text)

        start_index = index + 1

    if start_index < len(sentences):
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a string based on a specified encoding.

    The function uses the specified encoding to tokenize the input string and returns the number of tokens generated.

    Parameters:
    string (str): The input string to be tokenized.
    encoding_name (str): The name of the encoding to use for tokenization.

    Returns:
    int: The number of tokens in the input string based on the specified encoding.

    Example:
    >>> string = "This is a sample sentence."
    >>> encoding_name = "cl100k_base"
    >>> num_tokens = num_tokens_from_string(string, encoding_name)
    >>> print(num_tokens)
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_threshold(distances: List[int], sentences: List[str]) -> int:
    """
    Determines the optimal threshold value for splitting sentences into chunks based on token length.

    The function iterates over a range of threshold values, using each one to generate chunks of sentences.
    It then calculates the median token length of the chunks and returns the first threshold that results in
    a median chunk length greater than 90 tokens. If no such threshold is found, it returns 90.

    Parameters:
    distances (List[float]): A list of cosine distances between consecutive sentence embeddings.
    sentences (List[str]): A list of sentence strings to be chunked.

    Returns:
    int: The optimal threshold value that results in chunks with a median token length greater than 90.

    Example:
    >>> distances = [0.1, 0.4, 0.7, 0.2]
    >>> sentences = [
    ...     "This is the first sentence.",
    ...     "This is the second sentence.",
    ...     "This is the third sentence."
    ... ]
    >>> threshold = get_threshold(distances, sentences)
    >>> print(threshold)
    """
    for threshold in range(40, 100, 5):
        breaks = get_breakpoint(
            distances=distances, sentences=sentences, threshold=threshold
        )

        chunk_lengths = []
        for chunk in breaks:
            tokens = num_tokens_from_string(chunk, "cl100k_base")
            chunk_lengths.append(tokens)

        median_length = median(chunk_lengths)

        if median_length > 110:
            return threshold

    return 90


def split_chunk(big_chunk: str, smallest_size: int, largest_size: int) -> List[str]:
    """
    Splits a large chunk of text into smaller chunks based on token size.

    The function divides a large chunk of text into smaller chunks by splitting the text at sentence boundaries.
    It ensures that each resulting chunk has a token size between 50 and 450 tokens.
    If a chunk exceeds 450 tokens, the function discards that chunk and moves on.

    Parameters:
    big_chunk (str): The large chunk of text to be split.

    Returns:
    List[str]: A list of smaller text chunks that each contain between 50 and 450 tokens.

    Example:
    >>> big_chunk = "This is the first sentence. This is the second sentence. This is the third sentence."
    >>> small_chunks = split_chunk(big_chunk)
    >>> print(small_chunks)
    """
    splits = []
    sentences = big_chunk.split(".")
    current_sentence = ""
    for sentence in sentences:
        token_size = num_tokens_from_string(current_sentence, "cl100k_base")
        if smallest_size < token_size < largest_size:
            splits.append(current_sentence)
            current_sentence = ""
        elif token_size >= largest_size:
            current_sentence = ""
        else:
            current_sentence += "." + sentence
    return splits


def remove_start_dot(chunks):
    updated_chunks = []
    for chunk in chunks:
        if chunk.startswith("."):
            chunk = chunk[1:].strip()
        updated_chunks.append(chunk)
    return updated_chunks


def get_filtered_chunks(chunks: List[str]) -> List[str]:
    """
    Filters and refines chunks of text based on token size.

    The function processes a list of text chunks, filtering out those with fewer than 50 tokens and splitting
    those with more than 450 tokens into smaller chunks. It returns a list of chunks that each contain between
    50 and 450 tokens.

    Parameters:
    chunks (List[str]): A list of text chunks to be filtered and refined.

    Returns:
    List[str]: A list of text chunks, each containing between 50 and 450 tokens.

    Example:
    >>> chunks = [
    ...     "This is a short chunk.",
    ...     "This is a longer chunk that should be included because it has a moderate number of tokens.",
    ...     "This is a very long chunk that exceeds 450 tokens and needs to be split into smaller chunks."
    ... ]
    >>> filtered_chunks = get_filtered_chunks(chunks)
    >>> print(filtered_chunks)
    """
    smallest_size = 175
    largest_size = 500
    filtered = []
    for chunk in chunks:
        token_num = num_tokens_from_string(chunk, "cl100k_base")
        if token_num < smallest_size:
            continue
        if token_num < largest_size:
            filtered.append(chunk)
        else:
            filtered.extend(split_chunk(chunk, smallest_size, largest_size))
    return remove_start_dot(filtered)


def get_chunks(sentences: List[str], buffer_size: int) -> List[str]:
    """
    Generates optimized text chunks from a list of sentences using cosine distances and token thresholds.

    This function processes a list of sentences by performing several steps:
    1. Converts sentences into dictionaries with IDs.
    2. Combines sentences based on a specified buffer size.
    3. Generates embeddings for the combined sentences.
    4. Calculates cosine distances between the embeddings of consecutive sentences.
    5. Determines an optimal threshold for splitting sentences into chunks.
    6. Breaks the sentences into chunks based on the threshold.
    7. Filters and refines the chunks based on token size.

    Parameters:
    sentences (List[str]): A list of sentences to be processed into chunks.
    buffer_size (int): The number of neighboring sentences to consider when combining sentences.

    Returns:
    List[str]: A list of optimized text chunks.

    Example:
    >>> sentences = [
    ...     "This is the first sentence.",
    ...     "This is the second sentence.",
    ...     "This is the third sentence.",
    ...     "This is the fourth sentence.",
    ...     "This is the fifth sentence."
    ... ]
    >>> buffer_size = 2
    >>> chunks = get_chunks(sentences, buffer_size)
    >>> print(chunks)
    """
    sentences_dic = return_dic(sentences)
    sentences_comb = combine_sentences(sentences_dic, buffer_size)
    sentences_embed = generate_embeddings(sentences_comb)
    distances, sentences = calculate_cosine_distances(sentences_embed)
    threshold = get_threshold(distances, sentences)
    sentences_breaks = get_breakpoint(distances, sentences, threshold)
    chunks = get_filtered_chunks(sentences_breaks)
    return chunks


if __name__ == "__main__":
    # set environment
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(
        "chunking_example/chunking_test_example.json", "r", encoding="UTF-8"
    ) as file:
        contexts = json.load(file)
    sentences = contexts["contexts"]
    chunks = get_chunks(sentences=sentences, buffer_size=1)
    print(len(chunks))
    pprint(chunks)
