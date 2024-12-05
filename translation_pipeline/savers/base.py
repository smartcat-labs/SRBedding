from abc import ABC, abstractmethod

from datasets import Dataset


class DataSaver(ABC):
    @abstractmethod
    def save_data(self, data: Dataset, path: str):
        pass


class JSONSaver(DataSaver):
    def save_data(self, data: Dataset, path: str):
        data.to_json(path)


class DataSaverFactory:
    @staticmethod
    def create_data_saver(data_saver_type: str) -> DataSaver:
        if data_saver_type == "huggingface":
            from savers.huggingface_saver import HuggingFaceSaver

            return HuggingFaceSaver()
        elif data_saver_type == "json":
            return JSONSaver()
        else:
            raise ValueError("Invalid data saver type!")
