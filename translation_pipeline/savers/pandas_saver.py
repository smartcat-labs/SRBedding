import pandas as pd

from savers.base import DataSaver

# TODO: Implement the PandasSaver class


class PandasSaver(DataSaver):
    def save_data(self, data: pd.DataFrame, path: str):
        data.to_csv(path, index=False)
