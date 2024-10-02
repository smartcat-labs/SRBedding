import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    MSEEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
)
from sentence_transformers.losses import MSELoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from training_utils import EvalLoggingCallback


def load_datset_with_cashe(dataset_name: str) -> Dataset:
    """
    Loads a dataset with caching to a specified directory.

    Args:
        dataset_name (str): The name of the dataset to be loaded.

    Returns:
        Dataset: The loaded dataset from the specified dataset name.
    """
    dir = Path("~/Datasets/SRBendding").expanduser()

    dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(dataset_name, cache_dir=dir)


def get_train_and_eval_datasets(dataset_name: Path) -> Tuple[Dataset, Dataset]:
    """
    Loads a dataset from a specified parquet file and splits it into training and evaluation sets.

    Args:
        dataset_name (Path): The file path of the parquet dataset to be loaded.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing two datasets:
            - train_dataset: The training subset of the dataset.
            - eval_dataset: The evaluation subset of the dataset.

    This function utilizes the `load_dataset` function from the `datasets` library to read
    the dataset from the provided parquet file. It then splits the dataset into training
    and evaluation sets using an 98:2 ratio with a fixed seed for reproducibility.
    The resulting datasets can be used for model training and evaluation in machine
    learning tasks.
    """
    dataset = load_dataset(
        "parquet",
        data_files=dataset_name,
    )
    train_test = dataset["train"].train_test_split(test_size=0.02, seed=42)

    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    return train_dataset, eval_dataset


def divide_score_by_5(example: Dataset) -> Dataset:
    """
    Divides the 'score' field of the provided dataset example by 5 in order to normalize it.

    Args:
        example (Dataset): A dataset example that contains a 'score' field.

    Returns:
        Dataset: The modified dataset example with the 'score' field divided by 5.
    """
    example["score"] = example["score"] / 5.0
    return example


def get_sts_dataset(dataset_name: str) -> Dataset:
    """
    Loads the STS dataset, normalizes the 'score' field, and returns the training split.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dataset: The training portion of the loaded dataset with the 'score' field
                  normalized by dividing it by 5.
    Note:
        Ensure that the specified dataset contains a 'score' field for normalization
        to be applicable.
    """
    dataset = load_datset_with_cashe(dataset_name=dataset_name)
    dataset = dataset.map(divide_score_by_5)
    return dataset["train"]


def train(
    teacher_model_name: str,
    student_model_name: str,
    train_datset: str,
    sts_dataset: str,
    num_train_epochs: int = 5,
    num_evaluation_steps: int = 5000,
    batch_size: int = 16,
) -> None:
    """
    Trains a student model using a teacher model with Mean Squared Error (MSE) loss
    and evaluates its performance on multiple datasets.

    Args:
        teacher_model_name (str): The name or path of the teacher model to use for training.
        student_model_name (str): The name or path of the student model to train.
        train_datset (str): The dataset to use for training.
        sts_dataset (str): The STS dataset to use for evaluation.
        num_train_epochs (int, optional): The number of training epochs. Defaults to 5.
        num_evaluation_steps (int, optional): The number of steps between evaluations. Defaults to 5000.
        batch_size (int, optional): The batch size for training and evaluation. Defaults to 16.

    Returns:
        None

    This function initializes both the teacher and student models using the provided
    model names. It prepares the training and evaluation datasets, defines the loss
    function (Mean Squared Error), and sets up various evaluators, including MSE and
    translation accuracy evaluators.

    The training process is then executed using the `SentenceTransformerTrainer`,
    which incorporates logging and saves the final trained student model to the specified
    output directory. The training is designed for multilingual scenarios, specifically
    focusing on English-Serbian (en-sr) translations.
    """
    student_max_seq_length = (
        512  # Student model max. lengths for inputs (number of word pieces)
    )

    output_dir = (
        "output/make_multilingual_"
        + "en_sr"
        + "_"
        + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )

    teacher_model = SentenceTransformer(teacher_model_name)
    logging.info(f"Teacher model: {teacher_model}")

    student_model = SentenceTransformer(student_model_name)
    student_model.max_seq_length = student_max_seq_length
    logging.info(f"Student model: {student_model}")

    train_dataset, eval_dataset = get_train_and_eval_datasets(dataset_name=train_datset)

    logging.info("Prepared datasets for training:", train_dataset)

    # MSELoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss) needs one text columns and one
    # column with embeddings from the teacher model
    train_loss = MSELoss(model=student_model)

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = MSEEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name="en-sr",
        teacher_model=teacher_model,
        batch_size=batch_size,  # ovde je pre bio inference
    )

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of
    # source[i] is the closest to target[i] out of all available target sentences
    dev_trans_acc = TranslationEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name="en-sr",
        batch_size=batch_size,
    )
    evaluators = [dev_mse, dev_trans_acc]

    sts_dataset = get_sts_dataset(sts_dataset)

    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sts_dataset["sentence1"],
        sentences2=sts_dataset["sentence2_srb"],
        scores=sts_dataset["score"],
        batch_size=batch_size,
        name="sts17-test",
        show_progress_bar=False,
    )
    evaluators.append(test_evaluator)

    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sts_dataset["sentence1"],
        sentences2=sts_dataset["sentence2_eng"],
        scores=sts_dataset["score"],
        batch_size=batch_size,
        name="sts17-test",
        show_progress_bar=False,
    )
    evaluators.append(test_evaluator)

    evaluator = SequentialEvaluator(
        evaluators, main_score_function=lambda scores: np.mean(scores)
    )

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        learning_rate=2e-5,
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=num_evaluation_steps,
        save_strategy="steps",
        save_steps=num_evaluation_steps,
        save_total_limit=2,
        logging_steps=100,
        run_name="multilingual-en-sr",  # Will be used in W&B if `wandb` is installed
    )

    trainer = SentenceTransformerTrainer(
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
        callbacks=[EvalLoggingCallback(save_path=output_dir)],
    )
    trainer.train()

    final_output_dir = f"{output_dir}/final_model"
    student_model.save(final_output_dir)

    # model_name = student_model_name if "/" not in student_model_name else student_model_name.split("/")[-1]
    # try:
    #     student_model.push_to_hub(f"{model_name}-multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}")
    # except Exception:
    #     logging.error(
    #         f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
    #         f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
    #         f"and saving it using `model.push_to_hub('{model_name}-multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}')`."
    #     )


if __name__ == "__main__":
    train(
        teacher_model_name="intfloat/e5-base-v2",
        student_model_name="intfloat/e5-base-v2",
        train_datset="datasets/with_label.parquet",
        sts_dataset="smartcat/STS_parallel_en_sr",
        num_train_epochs=6,
        batch_size=16,
        num_evaluation_steps=400,
    )
