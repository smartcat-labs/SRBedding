import logging
import math
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import pandas
import pyarrow.parquet as pq
import sentence_transformers.losses as losses
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from sentence_transformers.readers import InputExample

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)


class QueryType(Enum):
    SHORT = "short_query"
    MEDIUM = "medium_query"
    LONG = "long_query"


def load_df(file: Path) -> pandas.DataFrame:
    loaded_table = pq.read_table(file)
    return loaded_table.to_pandas()


def convert_dataset(
    dataframe: pandas.DataFrame, question_type: str
) -> List[InputExample]:
    dataset_samples = []
    for _, row in dataframe.iterrows():
        score = float(row["scores"][question_type]) / 5.0
        sample = InputExample(texts=[row["context"], row[question_type]], label=score)
        dataset_samples.append(sample)
    return dataset_samples


def convert_to_hf_dataset(input_examples: List[InputExample]) -> Dataset:
    # Convert each InputExample into a dictionary
    data_dict = {
        "sentence1": [ex.texts[0] for ex in input_examples],
        "sentence2": [ex.texts[1] for ex in input_examples],
        "score": [ex.label for ex in input_examples],
    }

    # Create a Hugging Face Dataset
    return Dataset.from_dict(data_dict)


def get_train_and_eval_datasets(
    dataset_name: Path,
) -> Tuple[Dataset, Dataset, Dataset, List]:
    # NOTE francuzi su 70:15:15 ovde je 80:10:10
    df = load_df(file=dataset_name)
    training_samples = convert_dataset(df, QueryType.LONG.value)

    random.shuffle(training_samples)

    # Manually split the dataset while retaining the original structure
    dataset_size = len(training_samples)
    train_size = int(0.8 * dataset_size)
    dev_size = int(0.1 * dataset_size)

    train_samples = training_samples[:train_size]
    dev_samples = training_samples[train_size : train_size + dev_size]
    eval_samples = training_samples[train_size + dev_size :]

    # Convert lists to Hugging Face Datasets
    train_dataset = convert_to_hf_dataset(train_samples)
    dev_dataset = convert_to_hf_dataset(dev_samples)
    eval_dataset = convert_to_hf_dataset(eval_samples)

    return train_dataset, dev_dataset, eval_dataset, eval_samples


def make_sentence_transformer(
    model_name: str, max_seq_length: int = 512
) -> SentenceTransformer:
    # word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # # Apply mean pooling to get one fixed sized sentence vector
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                             pooling_mode_cls_token=False,
    #                             pooling_mode_max_tokens=False,
    #                             pooling_mode_mean_tokens=True)
    # return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


def train_a_model(
    sentence_transformer: SentenceTransformer,
    args: SentenceTransformerTrainingArguments,
    train_dataset,
    eval_dataset,
):
    # sentence_transformer = make_sentence_transformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model=sentence_transformer)
    train_loss = losses.MatryoshkaLoss(
        sentence_transformer, train_loss, [768, 512, 256, 128, 64]
    )
    eval_path = Path(args.output_dir)
    # # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = make_evaluator(eval_dataset, sentence_transformer, eval_path.parent)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=sentence_transformer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # # (Optional) Evaluate the trained model on the test set
    make_evaluator(eval_dataset, sentence_transformer, eval_path.parent)

    # 8. Save the trained model
    # TODO da li ovako cuvati
    sentence_transformer.save_pretrained(f"{eval_path.parent}/final_model")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")


def make_evaluator(dataset, sentence_transformer, savePath: Path):
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=dataset["sentence1"],
        sentences2=dataset["sentence2"],
        scores=dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
        write_csv=True,
    )
    result_path = Path(f"output/{savePath.name}/eval/")
    result_path.mkdir(exist_ok=True)
    dev_evaluator(model=sentence_transformer, output_path=result_path)
    return dev_evaluator


def load_model(model_save_path: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model from a specified path.

    :param model_save_path: The directory where the model is saved.
    :return: The loaded SentenceTransformer model.
    """
    model = SentenceTransformer(model_save_path)
    return model


def main_pipeline(
    num_epochs: int, batch_size: int, model_name: str, dataset_name: Path
):
    train_dataset, dev_dataset, eval_dataset, _ = get_train_and_eval_datasets(
        dataset_name
    )
    model_save_path = Path(
        f'output/bi_encoder_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/model'
    )
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    train_bi_encoder(
        num_epochs, batch_size, model_name, train_dataset, eval_dataset, model_save_path
    )


def train_bi_encoder(
    num_epochs, batch_size, model_name, train_dataset, eval_dataset, model_save_path
):
    warmup_steps = math.ceil(len(train_dataset) * num_epochs * 0.1)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=model_save_path,
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="proba",  # Will be used in W&B if `wandb` is installed
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,  # Automatically load the best model at the end of training
        metric_for_best_model="eval_loss",  # Assuming you're using loss as the evaluation metric
        greater_is_better=False,

        # KeyError: "The `metric_for_best_model` training argument is set to 'eval_cosine_loss', which is not found in the evaluation metrics. The available evaluation metrics are: ['eval_loss', 'eval_sts-dev_pearson_cosine', 'eval_sts-dev_spearman_cosine', 'eval_sts-dev_pearson_manhattan', 'eval_sts-dev_spearman_manhattan', 'eval_sts-dev_pearson_euclidean', 'eval_sts-dev_spearman_euclidean', 'eval_sts-dev_pearson_dot', 'eval_sts-dev_spearman_dot', 'eval_sts-dev_pearson_max', 'eval_sts-dev_spearman_max', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']. Consider changing the `metric_for_best_model` via the TrainingArguments."
    )
    train_a_model(
        sentence_transformer=make_sentence_transformer(model_name),
        args=args,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
    )


if __name__ == "__main__":
    main_pipeline(
        2, 16, "mixedbread-ai/mxbai-embed-large-v1", Path("datasets/train.parquet")
    )
