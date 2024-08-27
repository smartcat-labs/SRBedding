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
import torch
import tqdm
from datasets import Dataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments,
                                   models, util)
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import \
    CECorrelationEvaluator
from sentence_transformers.evaluation import (EmbeddingSimilarityEvaluator,
                                              SimilarityFunction)
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

class QueryType(Enum):
    SHORT = 'short_query'
    MEDIUM = 'medium_query'
    LONG = 'long_query'


def load_df(file: Path) -> pandas.DataFrame:
    loaded_table = pq.read_table(file)
    return loaded_table.to_pandas()


def convert_dataset(dataframe: pandas.DataFrame, question_type: str) -> List[InputExample]:
    dataset_samples = []
    for _, row in dataframe.iterrows():
        score = float(row['scores'][question_type])/5.0
        sample = InputExample(texts=[row['context'], row[question_type]],
                                 label=score)
        dataset_samples.append(sample)
    return dataset_samples

def convert_to_hf_dataset(input_examples: List[InputExample]) -> Dataset:
    # Convert each InputExample into a dictionary
    data_dict = {
        "sentence1": [ex.texts[0] for ex in input_examples],
        "sentence2": [ex.texts[1] for ex in input_examples],
        "score": [ex.label for ex in input_examples]
    }
    
    # Create a Hugging Face Dataset
    return Dataset.from_dict(data_dict)

def get_train_and_eval_datasets(dataset_name: Path) -> Tuple[Dataset, Dataset, Dataset, List]:
    # NOTE francuzi su 70:15:15 ovde je 80:10:10
    df = load_df(file=dataset_name)
    training_samples = convert_dataset(df, QueryType.LONG.value)

    random.shuffle(training_samples)

    # Manually split the dataset while retaining the original structure
    dataset_size = len(training_samples)
    train_size = int(0.8 * dataset_size)
    dev_size = int(0.1 * dataset_size)

    train_samples = training_samples[:train_size]
    dev_samples = training_samples[train_size:train_size + dev_size]
    eval_samples = training_samples[train_size + dev_size:]

    # Convert lists to Hugging Face Datasets
    train_dataset = convert_to_hf_dataset(train_samples)
    dev_dataset = convert_to_hf_dataset(dev_samples)
    eval_dataset = convert_to_hf_dataset(eval_samples)

    return train_dataset, dev_dataset, eval_dataset, eval_samples

def make_sentence_transformer(model_name: str, max_seq_length: int = 512) -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False,
                                pooling_mode_mean_tokens=True)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])

def train_a_model(sentence_transformer:SentenceTransformer, args: SentenceTransformerTrainingArguments, train_dataset, eval_dataset):
    # sentence_transformer = make_sentence_transformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model=sentence_transformer)
    train_loss = losses.MatryoshkaLoss(sentence_transformer, train_loss, [768, 512, 256, 128, 64])

    # # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = make_evaluator(eval_dataset, sentence_transformer)

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
    test_evaluator = make_evaluator(eval_dataset, sentence_transformer)

    # 8. Save the trained model
    # TODO da li ovako cuvati
    sentence_transformer.save_pretrained("output/mpnet-base-all-nli-triplet/final")

    # 9. (Optional) Push it to the Hugging Face Hub
    # model.push_to_hub("mpnet-base-all-nli-triplet")

def make_evaluator(dataset, sentence_transformer):
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=dataset['sentence1'],
        sentences2=dataset['sentence2'],
        scores=dataset['score'],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
        write_csv=True
    )
    result_path = Path('results/')
    result_path.mkdir(exist_ok=True)
    dev_evaluator(model=sentence_transformer, output_path='results/')
    return dev_evaluator


def make_gold_samle(df_train, batch_size):
    gold_samples = []
    batch_size = batch_size
    for df in df_train:
        score = float(df['score'])  # Already normalized scores to range 0 ... 1
        gold_samples.append(InputExample(texts=[df['sentence1'], df['sentence2']], label=score))
        gold_samples.append(InputExample(texts=[df['sentence2'], df['sentence1']], label=score))
    # We wrap gold_samples (which is a List[InputExample]) into a pytorch DataLoader
    return DataLoader(gold_samples, shuffle=True, batch_size=batch_size), gold_samples

def get_silver_datset(gold_samples):

    # Generation of the sentences
    sentences = set()

    for sample in gold_samples:
        sentences.update(sample.texts)

    sentences = list(sentences) # unique sentences
    sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)} # storing id and sentence in dictionary
    duplicates = set((sent2idx[data.texts[0]], sent2idx[data.texts[1]]) for data in gold_samples) # not to include gold pairs of sentences again
    return sentences, sent2idx, duplicates


def load_model(model_save_path: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model from a specified path.

    :param model_save_path: The directory where the model is saved.
    :return: The loaded SentenceTransformer model.
    """
    model = SentenceTransformer(model_save_path)
    return model


def main_pipeline(num_epochs: int, batch_size: int, model_name: str, dataset_name: Path):
    train_dataset, dev_dataset, eval_dataset, eval_with_text = get_train_and_eval_datasets(dataset_name)
    model_save_path = Path('output/trained_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    model_save_path.parent.mkdir(exist_ok=True, parents=True)
    train_bi_encoder(num_epochs, batch_size, model_name, train_dataset, eval_dataset, model_save_path)

    cross_encoder_path = 'output/cross-encoder/stsb_indomain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    gold_samples = train_cross_encoder(num_epochs, batch_size, model_name, train_dataset, eval_with_text, cross_encoder_path)
    
    semantic_search_model, silver_data = train_bi_encoder_on_silver(num_epochs, batch_size, model_save_path, gold_samples)

    cross_encoder = CrossEncoder(cross_encoder_path)
    silver_scores = cross_encoder.predict(silver_data)
    # All model predictions should be between [0,1]
    assert all(0.0 <= score <= 1.0 for score in silver_scores)

    train_dataloader = make_combined_data(batch_size, gold_samples, silver_data, silver_scores)

    # Configure the training.
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    final_model_save_path = model_save_path.with_name(model_save_path.name + "_final")

    args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=final_model_save_path,
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
            metric_for_best_model="cosine_loss",  # Assuming you're using loss as the evaluation metric
            greater_is_better=False,
        )
    
    train_a_model(sentence_transformer=semantic_search_model, args=args, eval_dataset=eval_dataset, train_dataset=train_dataset)

def make_combined_data(batch_size, gold_samples, silver_data, silver_scores):
    silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for \
        data, score in zip(silver_data, silver_scores))
    train_dataloader = DataLoader(gold_samples + silver_samples, shuffle=True, batch_size=batch_size)
    return train_dataloader

def train_bi_encoder_on_silver(num_epochs, batch_size, model_save_path, gold_samples):
    sentences, sent2idx, duplicates = get_silver_datset(gold_samples=gold_samples)
    semantic_search_model = load_model(model_save_path=f'{model_save_path}/checkpoint-{num_epochs}/')
    embeddings = semantic_search_model.encode(sentences, batch_size=batch_size, convert_to_tensor=True)
    top_k=2
    silver_data = []
    progress = tqdm.tqdm(unit="docs", total=len(sent2idx))
    for idx in range(len(sentences)):
        sentence_embedding = embeddings[idx]
        cos_scores = util.cos_sim(sentence_embedding, embeddings)[0]
        cos_scores = cos_scores.cpu()
        progress.update(1)

        # We use torch.topk to find the highest 5 scores
        top_results = torch.topk(cos_scores, k=top_k+1)
        
        for score, iid in zip(top_results[0], top_results[1]):
            if iid != idx and (iid, idx) not in duplicates:
                silver_data.append((sentences[idx], sentences[iid]))
                duplicates.add((idx,iid))            
    progress.reset()
    progress.close()
    return semantic_search_model,silver_data

def train_cross_encoder(num_epochs, batch_size, model_name, train_dataset, eval_with_text, cross_encoder_path):
    cross_encoder = CrossEncoder(model_name, num_labels=1)

    evaluator = CECorrelationEvaluator.from_input_examples(eval_with_text, name='sts-dev')
    train_dataloader, gold_samples = make_gold_samle(train_dataset, batch_size)
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    cross_encoder.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            optimizer_params={'lr': 1e-5, 
                                'eps': 1e-6,},
            warmup_steps=warmup_steps,
            output_path=cross_encoder_path)
            
    return gold_samples

def train_bi_encoder(num_epochs, batch_size, model_name, train_dataset, eval_dataset, model_save_path):
    warmup_steps = math.ceil(len(train_dataset) * num_epochs  * 0.1)

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
            metric_for_best_model="cosine_loss",  # Assuming you're using loss as the evaluation metric
            greater_is_better=False,
        )
    train_a_model(sentence_transformer=make_sentence_transformer(model_name), args=args, eval_dataset=eval_dataset, train_dataset=train_dataset)


if __name__ == "__main__":
    main_pipeline(2, 16, "google-bert/bert-base-multilingual-cased", Path("datasets/train.parquet"))


