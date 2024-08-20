import random
from pathlib import Path
from typing import List

from chunking_pipeline import get_chunks
from datasets_loading import get_datasets
from query_creation_pipeline import generate_query


def make_subset_from_all_sentences(all_sentences: List[str], final_lenght: int, chunked_lenght:int, random_step: int, random_step_start: int = 2000) -> List[str]:
    """
    Creates a subset of sentences by selecting chunks from the input list with random steps.

    This function selects chunks of sentences from the input list `all_sentences`, adding 
    them to the final list until the desired number of sentences is reached or until the 
    input list is exhausted. The chunks are selected based on the `chunked_lenght`, and 
    a random step size determines the starting point of the next chunk.

    Args:
        all_sentences (List[str]): A list of all available sentences.
        final_lenght (int): The target number of sentences in the final subset.
        chunked_lenght (int): The number of sentences to include in each chunk.
        random_step (int): The maximum additional step between chunks, selected randomly.
        random_step_start (int, optional): The minimum additional step between chunks. Defaults to 2000.

    Returns:
        List[str]: A list of sentences forming the final subset.

    Example:
        >>> sentences = ["Sentence 1", "Sentence 2", "Sentence 3", ...]
        >>> subset = make_subset_from_all_sentences(sentences, final_lenght=100, chunked_lenght=10, random_step=500)
    """
    start = 0
    final_sentences = []
    while start+chunked_lenght < len(all_sentences) and len(final_sentences) < final_lenght:
        final_sentences.extend(all_sentences[start:start+chunked_lenght])
        start += chunked_lenght + random.randint(random_step_start, random_step)
    return final_sentences

if __name__== "__main__":
    datasets = get_datasets()

    for dataset_name, dataset_args in datasets.items():
        all_sentences =dataset_args['loading_function']()
        print(dataset_name)
        filthered_sentences = make_subset_from_all_sentences(all_sentences=all_sentences,
                                                        final_lenght = 30,
                                                        chunked_lenght = 10,
                                                        random_step = dataset_args['random_step'],
                                                        random_step_start = dataset_args['random_step_start'],
                                                        )
        finall_contexts = get_chunks(filthered_sentences, buffer_size=2)
        generate_query(contexts=finall_contexts, save_filepath=Path(f"datasets/{dataset_name}_train.parquet"))