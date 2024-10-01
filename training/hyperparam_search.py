import json
import logging
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
    models,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.readers import InputExample
from transformers import TrainerCallback, TrainerControl, TrainerState

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
        # score = float(row["scores"][question_type]) / 5.0
        sample = InputExample(
            texts=[row[question_type], row["context"]]
        )  ## anchor and positive
        dataset_samples.append(sample)
    return dataset_samples


def convert_to_hf_dataset(input_examples: List[InputExample]) -> Dataset:
    # Convert each InputExample into a dictionary
    data_dict = {
        "anchor": [ex.texts[0] for ex in input_examples],
        "positive": [ex.texts[1] for ex in input_examples],
    }

    # Create a Hugging Face Dataset
    return Dataset.from_dict(data_dict)


def get_train_and_eval_datasets(
    dataset_name: Path,
) -> Tuple[Dataset, Dataset, Dataset, List]:
    # NOTE 90:10
    df = load_df(file=dataset_name)
    training_samples = convert_dataset(df, QueryType.LONG.value)

    random.shuffle(training_samples)

    # Manually split the dataset while retaining the original structure
    dataset_size = len(training_samples)
    train_size = int(0.9 * dataset_size)

    train_samples = training_samples[:train_size]
    dev_samples = training_samples[train_size:]

    # Convert lists to Hugging Face Datasets
    train_dataset = convert_to_hf_dataset(train_samples)
    dev_dataset = convert_to_hf_dataset(dev_samples)

    return train_dataset, dev_dataset


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
        log_file = Path(f"{self.save_path}/logs/{json_file}")
        log_file.parent.mkdir(exist_ok=True)
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


def hpo_search_space(trial):
    return {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 10),
        "per_device_train_batch_size": trial.suggest_int(
            "per_device_train_batch_size", 16, 64
        ),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
    }


def hpo_model_init(trial):
    word_embedding_model = models.Transformer(
        "mixedbread-ai/mxbai-embed-large-v1", max_seq_length=512
    )
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


# 5. Define the Loss Initialization
def hpo_loss_init(model):
    return losses.MultipleNegativesRankingLoss(model)


# 6. Define the Objective Function
def hpo_compute_objective(metrics):
    return metrics["eval_sts-dev_dot_ndcg@10"]


def train_a_model(
    args: SentenceTransformerTrainingArguments,
    train_dataset,
    eval_dataset,
):
    bi_encoder_path = Path(args.output_dir).parent
    dev_evaluator = make_evaluator(eval_dataset, bi_encoder_path)

    trainer = SentenceTransformerTrainer(
        model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=hpo_model_init,  # TODO check why is this set
        loss=hpo_loss_init,
        evaluator=dev_evaluator,
        callbacks=[EvalLoggingCallback(save_path=bi_encoder_path)],
    )
    best_trial = trainer.hyperparameter_search(
        hp_space=hpo_search_space,
        compute_objective=hpo_compute_objective,
        n_trials=20,
        direction="maximize",
        backend="optuna",
    )
    print(best_trial)


def getDictionariesForEval(dataset):
    relevant_docs = {}
    queries = {str(index): value for index, value in enumerate(dataset["anchor"])}
    corpus = {str(index): value for index, value in enumerate(dataset["positive"])}
    relevant_docs = {
        str(index): [str(index)] for index, _ in enumerate(dataset["positive"])
    }
    return queries, corpus, relevant_docs


def make_evaluator(dataset, savePath: Path):
    queries, corpus, relevan_docs = getDictionariesForEval(dataset)
    dev_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevan_docs,
        name="sts-dev",
        write_csv=True,
    )
    return dev_evaluator


def make_dirs(model_save_path):
    model_save_path.mkdir(exist_ok=True, parents=True)


def main_pipeline(dataset_name: Path):
    train_dataset, eval_dataset = get_train_and_eval_datasets(dataset_name)
    model_save_path = Path(
        f'output/bi_encoder_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    )
    make_dirs(model_save_path)
    train_bi_encoder(train_dataset, eval_dataset, model_save_path)


def train_bi_encoder(train_dataset, eval_dataset, model_save_path):
    # warmup_steps = math.ceil(len(train_dataset) * num_epochs * 0.1)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"{model_save_path}/model",
        # Optional training parameters:
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
        # load_best_model_at_end=True,  # Automatically load the best model at the end of training
        # metric_for_best_model="eval_loss",  # Assuming you're using loss as the evaluation metric
        # greater_is_better=False,
        disable_tqdm=False,
        # KeyError: "The `metric_for_best_model` training argument is set to 'eval_cosine_loss', which is not found in the evaluation metrics. The available evaluation metrics are: ['eval_loss', 'eval_sts-dev_pearson_cosine', 'eval_sts-dev_spearman_cosine', 'eval_sts-dev_pearson_manhattan', 'eval_sts-dev_spearman_manhattan', 'eval_sts-dev_pearson_euclidean', 'eval_sts-dev_spearman_euclidean', 'eval_sts-dev_pearson_dot', 'eval_sts-dev_spearman_dot', 'eval_sts-dev_pearson_max', 'eval_sts-dev_spearman_max', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']. Consider changing the `metric_for_best_model` via the TrainingArguments."
    )
    train_a_model(
        args=args,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
    )


if __name__ == "__main__":
    main_pipeline(Path("datasets/TRAIN11k_fixed.parquet"))
