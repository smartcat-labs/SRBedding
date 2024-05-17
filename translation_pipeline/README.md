# [WIP] English Text Translation Pipeline 
 
This pipeline is designed to translate English text into other languages. It consists of several components, each responsible for a specific task in the translation process.

## Pipeline Components

1. **Data Loader**: This component loads the English text from the source files.

2. **Translator**: This component translates the English text into the target language (e.g Serbian by default) using a an API/translation model (for now OpenAI and Google Translate are supported).

3. **Data Saver**: This component saves the translated text to the final destinations. (Pandas CSV in local file system, HuggingFace data hub, etc.)

## Invoking Factories

Factories are used to create instances of the pipeline components. To invoke a factory, you need to call its `create` method with the appropriate arguments. For example, to create a `MTEBDataLoader` with instance of `DataLoader`, you can do:

`from factories import DataLoaderFactory`

`data_loader = DataLoaderFactory.create('mteb')`

## Running tests with pytest

Tests can be ran with the following comamnd:
`poetry run python -m pytest`

## Running the Pipeline
To run the pipeline, you need to create instances of all the components and connect them together. Then, you can call the run method of the pipeline with the appropriate arguments.

Running with Poetry
You can run the pipeline with the poetry run command. Here's an example: 
`poetry run python ./pipelines/translation_pipeline.py `