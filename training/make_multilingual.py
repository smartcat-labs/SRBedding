import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets, load_dataset
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
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback, TrainerControl, TrainerState


def make_path(save_path: str):
    model_save_path = Path(save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)
    return model_save_path


class EvalLoggingCallback(TrainerCallback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    # def on_step_begin(self, args, state, control, **kwargs):
    #     print("next step")
    #     print(kwargs['lr_scheduler'])
    #     # print(kwargs['model'].classifier.out_proj.weight.grad.norm())
    #     print("end step")

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
        # if state.is_local_process_zero:
        #     print("logs in if")
        #     print(logs)

        # Capture the last logged metrics
        if "loss" in logs:
            self.write_in_log_file(logs, "on_log_1.jsonl")
        if "train_loss" in logs:
            self.write_in_log_file(logs, "on_log_2.jsonl")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print("Evaluation started")
        eval_output = state.log_history[-1]  # Last logged evaluation metrics
        eval_output["epoch"] = state.epoch

        self.write_in_log_file(eval_output, "on_evaluate.jsonl")


def load_datset_with_cashe(dataset_name: str) -> Dataset:
    dir = Path("~/Datasets/SRBendding").expanduser()
    # dir = Path("D:/Datasets/SRBendding")

    dir.mkdir(parents=True, exist_ok=True)
    return load_dataset(dataset_name, cache_dir=dir)


# def convert_to_hf_dataset(dataframe: pandas.DataFrame, chunk_size=50000) -> Dataset:
#     # Convert each InputExample into a dictionary
#     # data_dict = dataframe[["sentence1", "sentence2", "label"]].rename(columns={"sentence1": "english", "sentence2": "non_english"})
#     print("usao")
#     chunks = []
#     for chunk in range(0, len(dataframe), chunk_size):
#         chunk_df = dataframe.iloc[chunk : chunk + chunk_size]  # Get chunk of rows
#         chunk_dataset = Dataset.from_pandas(chunk_df)
#         chunks.append(chunk_dataset)

#     # Use concatenate_datasets to merge all chunks
#     return concatenate_datasets(chunks)


# def load_pandas_df(file: Path) -> pandas.DataFrame:
#     loaded_table = pq.read_table(file)
#     return loaded_table.to_pandas().reset_index(drop=True)


def get_train_and_eval_datasets(dataset_name: Path) -> Tuple[Dataset, Dataset]:
    # df = load_pandas_df(file=dataset_name)
    # train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    # # sanity_check(train_df, eval_df)
    # # Convert lists to Hugging Face Datasets
    # eval_dataset = convert_to_hf_dataset(eval_df)
    # print("Izasao2")
    # train_dataset = convert_to_hf_dataset(train_df)
    # print("Izasao1")
    dataset = load_dataset(
        "parquet",
        data_files=dataset_name,
    )
    train_test = dataset["train"].train_test_split(test_size=0.02, seed=42)

    # Rename splits for clarity
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    return train_dataset, eval_dataset


def divide_score_by_5(example):
    example["score"] = example["score"] / 5.0
    return example


def get_sts_dataset(dataset_name):
    dataset = load_datset_with_cashe(dataset_name=dataset_name)
    dataset = dataset.map(divide_score_by_5)
    return dataset["train"]


# We want the student EN embeddings to be similar to the teacher EN embeddings and
# the student non-EN embeddings to be similar to the teacher EN embeddings


def train(
    teacher_model_name,
    student_model_name,
    train_datset,
    sts_dataset,
    inference_batch_size=128,  # NOTE ovaj batch je usao u eval slucajno
    num_train_epochs=5,
    num_evaluation_steps=5000,
    batch_size=16,
):
    def encode_sentences(examples):
        # Encode all sentences in the "english" column in batches
        encoded = teacher_model.encode(
            examples["english"],
            batch_size=inference_batch_size,
            # show_progress_bar=True,  # Show a progress bar for feedback
        )
        return {"label": encoded}

    # teacher_model_name = "paraphrase-distilroberta-base-v2"
    # student_model_name = "xlm-roberta-base"

    # NOTE kod njih je 128
    student_max_seq_length = (
        512  # Student model max. lengths for inputs (number of word pieces)
    )

    # NOTE ovo je isto, oni su pre ovoh imali set souce i target jezika koje prevode mi nemamo
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

    # NOTE mi train i eval ucitavamo iz parquet vraca se df sa redom english,
    # non_english i label, ovaj index je posledica prebacivanja iy pandasa df u huggingface Dataset
    # oni su ucitavali HF Datasets za svaki jeyik i uzimali max ne ynam koliko redova
    # pa su svaki dodali u DatasetDict
    train_dataset, eval_dataset = get_train_and_eval_datasets(dataset_name=train_datset)

    # NOTE oni su nad DatasetDict radili map ja sam samo nad jednim Dataset-om ali u poslednjoj
    # veryiji se ucitavaju labele iz fajla pa se ni ne korisni ova funkcija
    # print(train_dataset)
    # train_dataset = train_dataset.map(encode_sentences, batched=True, batch_size=inference_batch_size)
    # eval_dataset = eval_dataset.map(encode_sentences, batched=True, batch_size=inference_batch_size)
    logging.info("Prepared datasets for training:", train_dataset)

    # 3. Define our training loss
    # MSELoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss) needs one text columns and one
    # column with embeddings from the teacher model
    # NOTE isto
    train_loss = MSELoss(model=student_model)

    # 4. Define evaluators for use during training. This is useful to keep track of alongside the evaluation loss.

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    # NOTE isto
    dev_mse = MSEEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name="en-sr",
        teacher_model=teacher_model,
        batch_size=batch_size,  # ovde je pre bio inference
    )

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of
    # source[i] is the closest to target[i] out of all available target sentences
    # NOTE isto
    dev_trans_acc = TranslationEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name="en-sr",
        batch_size=batch_size,
    )
    evaluators = [dev_mse, dev_trans_acc]

    sts_dataset = get_sts_dataset(sts_dataset)

    print(sts_dataset)
    print(eval_dataset)
    print(train_dataset)

    # NOTE isto
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sts_dataset["sentence1"],
        sentences2=sts_dataset["sentence2_srb"],
        scores=sts_dataset["score"],  # NOTE nama se score delio pri ucitavanju
        batch_size=batch_size,
        name="sts17-test",
        show_progress_bar=False,
    )
    evaluators.append(test_evaluator)

    # NOTE dodato
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sts_dataset["sentence1"],
        sentences2=sts_dataset["sentence2_eng"],
        scores=sts_dataset["score"],
        batch_size=batch_size,
        name="sts17-test",
        show_progress_bar=False,
    )
    evaluators.append(test_evaluator)

    # NOTE isto
    evaluator = SequentialEvaluator(
        evaluators, main_score_function=lambda scores: np.mean(scores)
    )

    # NOTE isto
    # 5. Define the training arguments
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

    # NOTE dodat callback
    # 6. Create the trainer & start training
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

    # NOTE isto
    # 7. Save the trained & evaluated model locally
    final_output_dir = f"{output_dir}/final_model"
    student_model.save(final_output_dir)

    # # 8. (Optional) save the model to the Hugging Face Hub!
    # # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
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
        num_evaluation_steps=400
    )
