import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
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
from training_utils import EvalLoggingCallback, make_path

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)


class QueryType(Enum):
    SHORT = "short_query"
    MEDIUM = "medium_query"
    LONG = "long_query"


def convert_to_hf_dataset(dataframe: pd.DataFrame, question_type: str) -> Dataset:
    """
    Converts a Pandas DataFrame into a Hugging Face Dataset format.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data to be converted.
        question_type (str): The column name in the DataFrame to be used as the 'anchor'.

    Returns:
        Dataset: A Hugging Face Dataset containing 'anchor' and 'positive' pairs.

    This function extracts the specified 'anchor' column and the 'context' column from the input DataFrame
    and organizes them into a dictionary format suitable for creating a Hugging Face Dataset.
    """
    data_dict = {
        "anchor": [],
        "positive": [],
    }
    for _, row in dataframe.iterrows():
        data_dict["anchor"].append(row[question_type])
        data_dict["positive"].append(row["context"])
    return Dataset.from_dict(data_dict)


def sanity_check(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """
    Performs a sanity check on the dataset distributions of training and evaluation DataFrames.

    Args:
        train_df (pd.DataFrame): The training DataFrame containing a 'dataset' column.
        eval_df (pd.DataFrame): The evaluation DataFrame containing a 'dataset' column.

    Returns:
        None

    This function calculates and prints the proportions of each dataset in the training and evaluation
    DataFrames to ensure that the distributions are as expected. It helps in identifying any imbalances
    between the two datasets.
    """
    dataset_counts_train = train_df["dataset"].value_counts()
    dataset_counts_eval = eval_df["dataset"].value_counts()
    dataset_proportions = dataset_counts_train / dataset_counts_train.sum()
    print(dataset_proportions)
    dataset_proportions = dataset_counts_eval / dataset_counts_eval.sum()
    print(dataset_proportions)


def get_train_and_eval_datasets(
    dataset_name: Path, question_type: str
) -> Tuple[Dataset, Dataset]:
    """
    Splits a dataset into training and evaluation sets and converts them to Hugging Face Datasets.

    Args:
        dataset_name (Path): The file path of the dataset to be loaded.
        question_type (str): The type of question to be used for creating the datasets (e.g., 'short_query').

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and evaluation datasets.

    This function loads a dataset from a specified path, splits it into training and evaluation sets
    with an 80:20 ratio, and converts these DataFrames into Hugging Face Dataset objects for further
    processing in machine learning tasks.
    """
    df = pd.read_parquet(dataset_name)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = convert_to_hf_dataset(train_df, question_type)
    eval_dataset = convert_to_hf_dataset(eval_df, question_type)

    return train_dataset, eval_dataset


def make_sentence_transformer(
    model_name: str, max_seq_length: int = 512
) -> SentenceTransformer:
    """
    Creates a SentenceTransformer model with a specified transformer model and mean pooling.

    Args:
        model_name (str): The name of the transformer model to be used (e.g., 'distilbert-base-nli-mean-tokens').
        max_seq_length (int, optional): The maximum sequence length for the input sentences. Default is 512.

    Returns:
        SentenceTransformer: An instance of the SentenceTransformer model configured with the specified transformer and pooling.
    """
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        pooling_mode_mean_tokens=True,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def train_a_model(
    sentence_transformer: SentenceTransformer,
    args: SentenceTransformerTrainingArguments,
    dataset_name: str,
) -> None:
    """
    Trains a SentenceTransformer model on a specified dataset.

    Args:
        sentence_transformer (SentenceTransformer): The SentenceTransformer model to be trained.
        args (SentenceTransformerTrainingArguments): The training arguments including parameters such as
            learning rate, batch size, and output directory.
        dataset_name: The name or path of the dataset to be used for training and evaluation.

    Returns:
        None

    This function initializes the training and evaluation datasets, sets up the loss function, and creates
    a trainer for the SentenceTransformer model. It trains the model using the specified datasets and
    parameters, evaluates the model after training, and saves the final model to the specified output directory.
    """
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


def getDictionariesForEval(
    dataset: Dataset,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Generates dictionaries for evaluation from a given dataset.

    Args:
        dataset (Dataset): The dataset containing queries and their corresponding positive contexts.

    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
            - query_dict: A dictionary mapping query IDs to their respective queries.
            - context_dict: A dictionary mapping context IDs to their respective contexts.
            - query_to_contexts: A dictionary mapping query IDs to lists of associated context IDs.

    This function iterates over the dataset to create mappings for queries and contexts, ensuring
    unique IDs for each context. It returns three dictionaries: one for queries, one for contexts,
    and one mapping each query to its associated contexts, which can be used for evaluation purposes.
    """
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


def make_evaluator(
    dataset, sentence_transformer, savePath: Path
) -> InformationRetrievalEvaluator:
    """
    Creates an evaluator for information retrieval based on the provided dataset and sentence transformer.

    Args:
        dataset (Dataset): The dataset containing queries, corpus, and relevant documents.
        sentence_transformer (SentenceTransformer): The sentence transformer model used for evaluation.
        savePath (Path): The path where evaluation results will be saved.

    Returns:
        InformationRetrievalEvaluator: An instance of the evaluator configured for the dataset.

    This function extracts queries, corpus, and relevant documents from the dataset, initializes
    an `InformationRetrievalEvaluator`, and runs the evaluation, saving the results to the specified
    path. The evaluator can be used to assess the performance of the sentence transformer in
    retrieving relevant documents for given queries.
    """
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


def train_bi_encoder(num_epochs, batch_size, model_name, dataset_name, model_save_path):
    """
    Trains a bi-encoder model using specified hyperparameters.

    Args:
        num_epochs (int): The number of epochs for training the model.
        batch_size (int): The batch size for training the model.
        model_name (str): The name of the pre-trained model to use for training.
        dataset_name (Path): The path to the dataset for training.
        model_save_path (str): The path where the trained model will be saved.

    This function sets up the training arguments for a bi-encoder model using the
    `SentenceTransformerTrainingArguments` class. It configures various training parameters,
    such as the learning rate, batch sizes, and logging steps. It then calls the
    `train_a_model` function to initiate the training process with the specified
    parameters and dataset. The model will be saved in the specified output directory
    after training is complete.
    """
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
    model_save_path = make_path(
        f'output/bi_encoder_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    )
    train_bi_encoder(
        num_epochs=20,
        batch_size=10,
        model_name="BAAI/bge-base-en-v1.5",
        dataset_name=Path("datasets/TRAIN11k_fixed_v2.parquet"),
        model_save_path=model_save_path,
    )
