# SRBedding

## Project setup
Before runnning the project setup the environment  
`poetry shell`  
`poetry update`  

## Evaluation jupyter notebook
Inside or evaluation-pipetine-test add datasets folder.  
For loading SQuAD-sr you need to add the [squad-sr-lat.json](https://www.kaggle.com/datasets/aleksacvetanovic/squad-sr) into the datasets folder.  
First run the make-evaluation-datasets.ipynb, then the final-evaluation.ipynb  
If running for the first try, you have to make all the jsonl files by running the script to generate the queries and context files then running the terminal line  
`python evaluation-pipetine-test/api_request_parallel_processor.py   --requests_filepath evaluation-pipetine-test/processed_datasets/queries   --save_filepath evaluation-pipetine-test/datasets/text-embedding-3-small-{datasetname}-queries.jsonl   --request_url https://api.openai.com/v1/embeddings   --max_requests_per_minute 1500   --max_tokens_per_minute 6250000   --token_encoding_name cl100k_base   --max_attempts 5   --logging_level 20`
