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