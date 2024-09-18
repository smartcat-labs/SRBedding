import json
import logging
import math
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from pprint import pprint

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
            texts=[row[question_type].replace("?", ""), row["context"]]
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
    # NOTE francuzi su 70:15:15 ovde je 80:10:10
    df = load_df(file=dataset_name)
    training_samples = convert_dataset(df, QueryType.SHORT.value)
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

    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(eval_dataset))
    return train_dataset, dev_dataset, eval_dataset, eval_samples


def make_sentence_transformer(
    model_name: str, max_seq_length: int = 512
) -> SentenceTransformer:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.model_max_length = max_seq_length  # Set the max length for the model
    # tokenizer.padding_side = (
    #     "right"  # You can set "left" if you want to pad on the left side
    # )
    # # tokenizer.pad_token = tokenizer.eos_token  # Ensure the pad token is set
    # model = SentenceTransformer(model_name)
    # # Add the padding and truncation to the encode method
    # model.tokenizer = tokenizer
    # model.tokenizer_kwargs = {
    #     "padding": "max_length",
    #     "truncation": True,
    #     "max_length": max_seq_length,
    #     "return_tensors": "pt",  # Assuming you want PyTorch tensors as output
    # }
    # return model

    # word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # # Apply mean pooling to get one fixed sized sentence vector
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                             pooling_mode_cls_token=False,
    #                             pooling_mode_max_tokens=False,
    #                             pooling_mode_mean_tokens=True)
    # return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    m = SentenceTransformer(model_name)
    m.max_seq_length = max_seq_length
    return m


def get_scheduler_with_warmup_hold_decay(
    optimizer, num_warmup_steps, num_training_steps, hold_steps
):
    # Define custom LR schedule function
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(
                max(1, num_warmup_steps)
            )  # Linear warmup
        elif current_step < num_warmup_steps + hold_steps:
            return 1.0  # Hold learning rate constant
        else:
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(num_training_steps - num_warmup_steps - hold_steps),
            )  # Linear decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


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


def train_a_model(
    sentence_transformer: SentenceTransformer,
    args: SentenceTransformerTrainingArguments,
    train_dataset,
    eval_dataset,
):
    train_loss = losses.MultipleNegativesRankingLoss(model=sentence_transformer)
    # train_loss = losses.MatryoshkaLoss(
    #     sentence_transformer, train_loss, [768, 512, 256, 128, 64]
    # )
    bi_encoder_path = Path(args.output_dir).parent
    # # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = make_evaluator(eval_dataset, sentence_transformer, bi_encoder_path)

    optimizer = AdamW(
        sentence_transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Calculate total steps
    total_steps = len(train_dataset) * args.num_train_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    hold_steps = int(
        0.4 * total_steps
    )  # Hold LR for 40% of total steps (adjust as needed)

    # Use the custom LR scheduler with warmup, hold, and decay
    lr_scheduler = get_scheduler_with_warmup_hold_decay(
        optimizer, warmup_steps, total_steps, hold_steps
    )

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=sentence_transformer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=[EvalLoggingCallback(save_path=bi_encoder_path)],
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()

    # # (Optional) Evaluate the trained model on the test set
    make_evaluator(eval_dataset, sentence_transformer, bi_encoder_path)

    # 8. Save the trained model
    # TODO da li ovako cuvati
    sentence_transformer.save_pretrained(f"{bi_encoder_path}/final_model")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")


def getDictionariesForEval(dataset):
    relevant_docs = {}
    queries = {str(index): value for index, value in enumerate(dataset["anchor"])}
    corpus = {str(index): value for index, value in enumerate(dataset["positive"])}
    relevant_docs = {
        str(index): [str(index)] for index, _ in enumerate(dataset["positive"])
    }
    return queries, corpus, relevant_docs


def make_evaluator(dataset, sentence_transformer, savePath: Path):
    queries, corpus, relevan_docs = getDictionariesForEval(dataset)
    dev_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevan_docs,
        name="sts-dev",
        write_csv=True,
    )
    result_path = Path(f"{savePath}/eval/")
    result_path.mkdir(exist_ok=True)
    dev_evaluator(model=sentence_transformer, output_path=result_path)
    return dev_evaluator


def make_dirs(model_save_path):
    model_save_path.mkdir(exist_ok=True, parents=True)


def main_pipeline(
    num_epochs: int, batch_size: int, model_name: str, dataset_name: Path
):
    train_dataset, dev_dataset, eval_dataset, _ = get_train_and_eval_datasets(
        dataset_name
    )
    model_save_path = Path(
        f'output/bi_encoder_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    )
    make_dirs(model_save_path)
    train_bi_encoder(
        num_epochs, batch_size, model_name, train_dataset, eval_dataset, model_save_path
    )


def train_bi_encoder(
    num_epochs, batch_size, model_name, train_dataset, eval_dataset, model_save_path
):
    # warmup_steps = math.ceil(len(train_dataset) * num_epochs * 0.1)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"{model_save_path}/model",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        # lr_scheduler_type="constant_with_warmup",
        # lr_scheduler_kwargs={'last_epoch': 4},
        weight_decay=0.01,
        warmup_ratio=0.2,
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
        # warmup_steps=warmup_steps,
        load_best_model_at_end=True,  # Automatically load the best model at the end of training
        metric_for_best_model="eval_loss",  # Assuming you're using loss as the evaluation metric
        greater_is_better=False,
        disable_tqdm=False,
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
        10, 16, "BAAI/bge-base-en-v1.5", Path("datasets/TRAIN11k_fixed.parquet")
    )
