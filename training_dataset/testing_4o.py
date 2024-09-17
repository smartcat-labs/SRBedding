import random
from pathlib import Path
from typing import List

import pandas as pd

from chunking_pipeline import get_chunks
from datasets_loading import get_datasets
from parallel_query_creation import generate_query
# from batch_query_creation import generate_query

def get_contexts(dataset_path):
    data = pd.read_parquet(dataset_path)
    data = data["context"][:20].tolist()
    return data



if __name__== "__main__":
    datasets = get_datasets()

    for dataset_name in [Path("datasets/wiki.parquet"), ]:
        finall_contexts = get_contexts(dataset_name)
        name = dataset_name.stem
        generate_query(contexts=finall_contexts, save_filepath=Path(f"datasets/{name}_prompt_a.parquet"))