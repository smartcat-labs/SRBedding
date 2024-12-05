from loaders.base import DataLoader
from savers.base import DataSaver
from translators.base import Translator

from pipelines.base import Pipeline


class TranslationPipeline(Pipeline):

    def __init__(
        self, data_loader: DataLoader, translator: Translator, data_saver: DataSaver
    ) -> None:
        """
        Initialize the TranslationPipeline class.

        Args:
        - data_loader (DataLoader): DataLoader object to load the data.
        - translator (Translator): Translator object to translate the data.
        - data_saver (DataSaver): DataSaver object to save the data.
        """

        self.data_loader = data_loader
        self.translator = translator
        self.data_saver = data_saver

        super().__init__()

    def run(self, input: str, save_to_path: str = "./local_data/") -> str:
        """Run the all steps in the pipeline sequentially."""

        data = self.data_loader.load_data(input)
        translated_data = self.translator.translate(data)
        self.data_saver.save_data(translated_data, input, save_to_path)

        return translated_data
