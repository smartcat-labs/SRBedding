import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas
import pyarrow.parquet as pq
from datasets import Dataset
from sentence_transformers import LoggingHandler, SentenceTransformer
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
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState


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


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


teacher_model_name = "paraphrase-distilroberta-base-v2"
student_model_name = "xlm-roberta-base"

student_max_seq_length = (
    512  # Student model max. lengths for inputs (number of word pieces)
)
inference_batch_size = 64  # Batch size at inference

num_train_epochs = 5  # Train for x epochs
num_evaluation_steps = 5000  # Evaluate performance after every xxxx steps


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

dataset_to_use = "datasets/wikimatrix.parquet"


def convert_to_hf_dataset(dataframe: pandas.DataFrame) -> Dataset:
    # Convert each InputExample into a dictionary
    data_dict = {"english": [], "non_english": []}
    for _, row in dataframe.iterrows():
        data_dict["english"].append(row["sentence1"])
        data_dict["non_english"].append(row["sentence2"])
    # Create a Hugging Face Dataset
    return Dataset.from_dict(data_dict)


def load_pandas_df(file: Path) -> pandas.DataFrame:
    loaded_table = pq.read_table(file)
    return loaded_table.to_pandas()


def get_train_and_eval_datasets(dataset_name: Path) -> Tuple[Dataset, Dataset]:
    df = load_pandas_df(file=dataset_name)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    # sanity_check(train_df, eval_df)
    # Convert lists to Hugging Face Datasets
    train_dataset = convert_to_hf_dataset(train_df)
    eval_dataset = convert_to_hf_dataset(eval_df)

    return train_dataset, eval_dataset

def get_sts_dataset():
    pass


# We want the student EN embeddings to be similar to the teacher EN embeddings and
# the student non-EN embeddings to be similar to the teacher EN embeddings
def prepare_dataset(batch):
    return {
        "english": batch["english"],
        "non_english": batch["non_english"],
        "label": teacher_model.encode(
            batch["english"], batch_size=inference_batch_size, show_progress_bar=False
        ),
    }


train_dataset, eval_dataset = get_train_and_eval_datasets(dataset_name=dataset_to_use)
column_names = train_dataset.column_names

train_dataset_dict = train_dataset.map(
    prepare_dataset, batched=True, batch_size=30000, remove_columns=column_names
)
eval_dataset = eval_dataset.map(
    prepare_dataset, batched=True, batch_size=30000, remove_columns=column_names
)
logging.info("Prepared datasets for training:", train_dataset_dict)

# 3. Define our training loss
# MSELoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss) needs one text columns and one
# column with embeddings from the teacher model
train_loss = MSELoss(model=student_model)

# 4. Define evaluators for use during training. This is useful to keep track of alongside the evaluation loss.

logger.info("Creating evaluators")

# Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
dev_mse = MSEEvaluator(
    source_sentences=eval_dataset["english"],
    target_sentences=eval_dataset["non_english"],
    name="en-sr",
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
)

# TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of
# source[i] is the closest to target[i] out of all available target sentences
dev_trans_acc = TranslationEvaluator(
    source_sentences=eval_dataset["english"],
    target_sentences=eval_dataset["non_english"],
    name="en-sr",
    batch_size=inference_batch_size,
)


evaluators = [dev_mse, dev_trans_acc]

test_dataset = get_sts_dataset()

# TODO
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=[
        score / 5.0 for score in test_dataset["score"]
    ],  # Convert 0-5 scores to 0-1 scores
    batch_size=inference_batch_size,
    name=f"sts17-{subset}-test",
    show_progress_bar=False,
)
evaluators.append(test_evaluator)

evaluator = SequentialEvaluator(
    evaluators, main_score_function=lambda scores: np.mean(scores)
)


# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
