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
Inside od training-dataset folder add `datasets` and `commands` folder  
For creating the training dataset run  
`cd training-dataset`  
`python .\training_dataset_generation.py`  