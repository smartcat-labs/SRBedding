from pathlib import Path

import pandas
from datasets import Dataset
from sentence_transformers import SentenceTransformer

path = Path("datasets/MaCoCu-sr-en.parqet")
df = pandas.read_parquet(path)


data_dict = df[["sentence1", "sentence2"]].rename(
    columns={"sentence1": "english", "sentence2": "non_english"}
)
dataset = Dataset.from_pandas(data_dict)
dataset = dataset.remove_columns(["__index_level_0__"])


teacher_model = SentenceTransformer("intfloat/e5-base-v2")
inference_batch_size = 64


def encode_sentences(examples):
    # Encode all sentences in the "english" column in batches
    encoded = teacher_model.encode(
        examples["english"],
        batch_size=inference_batch_size,
        # show_progress_bar=True,  # Show a progress bar for feedback
    )
    return {"label": encoded}  # Add encoded representations to the dataset


small_dataset = dataset.select(range(70))

# Use the Hugging Face map function to process the dataset efficiently
train_dataset = small_dataset.map(
    encode_sentences, batched=True, batch_size=inference_batch_size
)
df = train_dataset.to_pandas()

path = Path("datasets/with_label.parquet")
df.to_parquet(path)
path = Path(path)
df_aa = pandas.read_parquet(path)
print(df_aa.head())
