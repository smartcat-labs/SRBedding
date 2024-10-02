import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple

import pandas
import pyarrow.parquet as pq
import sentence_transformers.losses as losses
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback, TrainerControl, TrainerState

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)


class QueryType(Enum):
    SHORT = "short_query"
    MEDIUM = "medium_query"
    LONG = "long_query"


def make_path(save_path: str):
    model_save_path = Path(save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)
    return model_save_path


def load_pandas_df(file: Path) -> pandas.DataFrame:
    loaded_table = pq.read_table(file)
    return loaded_table.to_pandas()


def convert_to_hf_dataset(dataframe: pandas.DataFrame, question_type: str) -> Dataset:
    data_dict = {
        "anchor": [],
        "positive": [],
    }
    for _, row in dataframe.iterrows():
        data_dict["anchor"].append(row[question_type])
        data_dict["positive"].append(row["context"])
    return Dataset.from_dict(data_dict)


def sanity_check(train_df, eval_df):
    dataset_counts_train = train_df["dataset"].value_counts()
    dataset_counts_eval = eval_df["dataset"].value_counts()
    dataset_proportions = dataset_counts_train / dataset_counts_train.sum()
    print(dataset_proportions)
    dataset_proportions = dataset_counts_eval / dataset_counts_eval.sum()
    print(dataset_proportions)


def get_train_and_eval_datasets(
    dataset_name: Path, question_type: str
) -> Tuple[Dataset, Dataset]:
    df = load_pandas_df(file=dataset_name)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = convert_to_hf_dataset(train_df, question_type)
    eval_dataset = convert_to_hf_dataset(eval_df, question_type)

    return train_dataset, eval_dataset


def make_sentence_transformer(
    model_name: str, max_seq_length: int = 512
) -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


class EvalLoggingCallback(TrainerCallback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    def write_in_log_file(self, logs, json_file):
        log_file = make_path(f"{self.save_path}/logs")
        log_file = log_file / json_file
        with open(log_file, "a") as f:
            print(logs)
            f.write(json.dumps(logs))
            f.write("\n")

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        _ = logs.pop("total_flos", None)

        if "loss" in logs:
            self.write_in_log_file(logs, "on_log_1.jsonl")
        if "train_loss" in logs:
            self.write_in_log_file(logs, "on_log_2.jsonl")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print("Evaluation started")
        eval_output = state.log_history[-1]  # Last logged evaluation metrics
        eval_output["epoch"] = state.epoch

        self.write_in_log_file(eval_output, "on_evaluate.jsonl")


def train_a_model(
    sentence_transformer: SentenceTransformer,
    args: SentenceTransformerTrainingArguments,
    dataset_name,
):
    train_dataset, eval_dataset = get_train_and_eval_datasets(
        dataset_name, QueryType.SHORT.value
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=sentence_transformer)
    bi_encoder_path = Path(args.output_dir).parent
    dev_evaluator = make_evaluator(eval_dataset, sentence_transformer, bi_encoder_path)

    trainer = SentenceTransformerTrainer(
        model=sentence_transformer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=[EvalLoggingCallback(save_path=bi_encoder_path)],
    )
    trainer.train()

    make_evaluator(eval_dataset, sentence_transformer, bi_encoder_path)

    sentence_transformer.save_pretrained(f"{bi_encoder_path}/final_model")


def getDictionariesForEval(dataset):
    query_dict = {}
    context_dict = {}
    query_to_contexts = {}

    context_id_map = {}
    context_counter = 0

    for idx, (query, context) in enumerate(
        zip(dataset["anchorn"], dataset["positive"])
    ):
        query_id = str(idx)
        query_dict[query_id] = query

        if context not in context_id_map:
            context_id_map[context] = str(context_counter)
            context_dict[str(context_counter)] = context
            context_counter += 1

        context_id = context_id_map[context]

        if query_id not in query_to_contexts:
            query_to_contexts[query_id] = []

        query_to_contexts[query_id].append(context_id)

    return query_dict, context_dict, query_to_contexts


def make_evaluator(dataset, sentence_transformer, savePath: Path):
    queries, corpus, relevan_docs = getDictionariesForEval(dataset)
    dev_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevan_docs,
        name="sts-dev",
        write_csv=True,
    )
    result_path = make_path(f"{savePath}/eval/")
    dev_evaluator(model=sentence_transformer, output_path=result_path)
    return dev_evaluator


def main_pipeline(
    num_epochs: int, batch_size: int, model_name: str, dataset_name: Path
):
    model_save_path = make_path(
        f'output/bi_encoder_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    )
    train_bi_encoder(num_epochs, batch_size, model_name, dataset_name, model_save_path)


def train_bi_encoder(num_epochs, batch_size, model_name, dataset_name, model_save_path):
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"{model_save_path}/model",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        lr_scheduler_type="constant_with_warmup",
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="proba",  # Will be used in W&B if `wandb` is installed
        load_best_model_at_end=True,  # Automatically load the best model at the end of training
        metric_for_best_model="eval_loss",  # Assuming you're using loss as the evaluation metric
        greater_is_better=False,
        disable_tqdm=False,
    )
    train_a_model(
        sentence_transformer=make_sentence_transformer(model_name),
        args=args,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main_pipeline(
        num_epochs=10,
        batch_size=16,
        model_name="BAAI/bge-base-en-v1.5",
        dataset_name=Path("datasets/TRAIN11k_fixed_v2.parquet"),
    )
