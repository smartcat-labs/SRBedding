from datasets import Dataset, load_dataset

from loaders.base import DataLoader


class MTEBLoader(DataLoader):

    def __init__(self) -> None:
        """
        Initialize the MTEBLoader class.
        """

        self.iterator = None

        self.dataset = None
        self.corpus_dataset = None
        self.queries_dataset = None

        super().__init__()

    def load_data(
        self, dataset_name: str, split: str
    ) -> tuple[Dataset, Dataset, Dataset]:
        """
        Load the dataset from HuggingFace for the MTEB benchmark.

        Args:
        - dataset_name (str): Name of the dataset on HuggingFace. ('mteb/quora' for example)
        - split (str): Split of the dataset to load. Usually is one of the following 'train', 'dev', test'.
        """

        if split:
            dataset = load_dataset(dataset_name, split=split)
            dataset_corpus = self.filter_split(
                load_dataset(dataset_name, "corpus"),
                "_id",
                dataset,
                "corpus-id",
            )

            dataset_queries = self.filter_split(
                load_dataset(dataset_name, "queries"),
                "_id",
                dataset,
                "query-id",
            )

        self.dataset = load_dataset(dataset_name)
        self.corpus_dataset = load_dataset(dataset_name, "corpus")
        self.queries_dataset = load_dataset(dataset_name, "queries")

        return dataset, dataset_corpus, dataset_queries

    def filter_split(
        self,
        dataset: Dataset,
        dataset_id_col: str,
        index_dataset: Dataset,
        index_id_col: str,
    ) -> Dataset:
        """
        Filter the dataset to only include examples from the given split.
        """
        return dataset.filter(
            lambda example: example[dataset_id_col] in index_dataset[index_id_col]
        )

    def __iter__(self):
        return self
