from abc import ABC, abstractmethod

from datasets import Dataset


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, path: str) -> Dataset:
        pass


class DataLoaderFactory:
    @staticmethod
    def create_data_loader(data_loader_type: str) -> DataLoader:
        if data_loader_type == "mteb":
            from loaders.mteb_loader import MTEBLoader

            return MTEBLoader()
        else:
            raise ValueError("Invalid data loader type!")
