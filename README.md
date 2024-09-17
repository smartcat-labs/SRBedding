# SRBedding

## Project setup
Before runnning the project setup the environment:
If there is an active .venv - deactivate, then:  
`poetry shell`  
`poetry update`

Setting up Poetry virtual environment in Jupyter notebook in VSC:
- 'install poetry'
- 'poetry init'
- create .toml file
- 'poetry shell'
When .venv file is created in the repository:
- 'poetry add ipkernel'
- ipython kernel install --user --name=[name of virtual environment]
- close Jupyter notebook
- ctrl + shift + p >Reaload Window
- ctrl + shift + p >Select Interpreter to start Jupyter notebook
- reopen Jupyter notebook
- select kernel

## Evaluation jupyter notebook
Inside or evaluation-pipetine add datasets folder and results.  
For loading SQuAD-sr you need to add the [squad-sr-lat.json](https://www.kaggle.com/datasets/aleksacvetanovic/squad-sr) into the datasets folder.  
First run the make-evaluation-datasets.ipynb. This will create all the files needed  
Then run  
`cd evaluation-pipetine/`  
`python evaluation-pipieline.py`  



## Training dataset creation  
Run the following commands for creating the training dataset:    
- `cd training-dataset`  
- `python .\training_dataset_generation.py`  
