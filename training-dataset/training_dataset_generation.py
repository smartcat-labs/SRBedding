from pathlib import Path
import random
from kaggle_datasets_loading import get_datasets
from query_creation_pipeline import run

def make_subset_from_all_sentences(all_sentences: list[str], final_lenght: int, chunked_lenght:int, random_step: int, random_step_start: int = 2000):
    start = 0
    final_sentences = []
    while start+chunked_lenght < len(all_sentences) and len(final_sentences) < final_lenght:
        final_sentences.extend(all_sentences[start:start+chunked_lenght])
        start += chunked_lenght + random.randint(random_step_start, random_step)
    return final_sentences

if __name__== "__name__":
    datasets = get_datasets()
    for dataset_name, dataset_args in datasets.items():
        all_sentences =dataset_args['loading_function']
        finall_sentences = make_subset_from_all_sentences(all_sentences=all_sentences,
                                                        final_lenght = dataset_args['final_lenght'],
                                                        chunked_lenght = dataset_args['chunked_lenght'],
                                                        random_step = dataset_args['random_step'],
                                                        random_step_start = dataset_args['random_step_start'],
                                                        )
        
        run(contexts=finall_sentences, save_filepath=Path(f"datasets/{dataset_name}.parquet"))