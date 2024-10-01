import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from prompts import SYSTEM_PROMPT

sys.path.append("..")
from utils import make_cache_dir
from utils_openAI import batch_requests, make_jobs


def load_sts_data() -> List[Dict[str, Any]]:
    datasets_dir = make_cache_dir()
    sts = load_dataset("mteb/stsbenchmark-sts", cache_dir=datasets_dir)
    sts_test_df = pd.DataFrame(sts["test"])

    final_data = []
    for i, row in sts_test_df.iterrows():
        if i > 10:
            break
        final_data.append(
            {
                "id": row["sid"],
                "sentence": row["sentence2"],
            }
        )

    return final_data


if __name__ == "__main__":
    model = "gpt-3.5-turbo-0125"

    dataset_name = "STS"

    final_data = load_sts_data()
    path = Path(f"commands/jobs_{dataset_name}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    make_jobs(prompt=SYSTEM_PROMPT, filename=path, dataset=final_data, model=model)

    batch_requests(jobs_file=path, dataset_name=dataset_name)
