# SRBedding

## Project setup
Before runnning the project setup the environment  
`poetry shell`  
`poetry update`  

## Evaluation jupyter notebook
Inside or evaluation-pipetine-test add datasets folder.  
For loading SQuAD-sr you need to add the [squad-sr-lat.json](https://www.kaggle.com/datasets/aleksacvetanovic/squad-sr) into the datasets folder.  
First run the make-evaluation-datasets.ipynb. This will create all the files needed
Then run
`cd evaluation-pipetine-test/`  
`python evaluation-pipieline.py`