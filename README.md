# SRBedding

## Project setup
Before runnning the project setup the environment  
`poetry shell`  
`poetry update`  

## Evaluation jupyter notebook
Inside or evaluation-pipetine add datasets folder and results.  
For loading SQuAD-sr you need to add the [squad-sr-lat.json](https://www.kaggle.com/datasets/aleksacvetanovic/squad-sr) into the datasets folder.  
First run the make-evaluation-datasets.ipynb. This will create all the files needed  
Then run  
`cd evaluation-pipetine/`  
`python evaluation-pipieline.py`  


## Training dataset creation  
Run the following commands for creating the training dataset:    
- `cd training_dataset`  
- `python .\main_training.py`  
- `python .\batch_loading.py`  
The .parquet files will be saved in the datasets folder.

## Translating dataset  
The folder translation_pipeline is used for translating [ms_marco](https://huggingface.co/datasets/microsoft/ms_marco) and  [natural_questions](https://huggingface.co/datasets/google-research-datasets/natural_questions) from English to Serbian. Translated queries and contexts from this datasets will be used for evaluation. 
Run the following commands:
- `cd  translation_pipeline`  
- `python .\sending_batch.py`  
- `python .\processing_batch.py` 

The folder translation_sts is used for translating one sentence pair from the [sts dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts) for the distiladion evaluator.
Run the following commands:
- `cd  translation_sts`  
- `python .\sending_batch.py`  
- `python .\processing_batch.py`
