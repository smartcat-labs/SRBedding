import argparse
import asyncio

from loaders.base import DataLoaderFactory
from pipelines.translation_pipeline import TranslationPipeline
from savers.base import DataSaverFactory
from translators.base import TranslatorFactory
from utils.logger import configure_logging

# WIP:

logger = configure_logging()


def parse_input_args() -> argparse.Namespace:
    """Parse command line arguments."""

    logger.info("Parsing command line arguments.")

    parser = argparse.ArgumentParser(
        description="Translate dataset to a given language."
    )

    parser.add_argument("--dataset", type=str, required=True, help="Dataset path/name.")
    parser.add_argument(
        "--language",
        type=str,
        default="serbian",
        required=True,
        help="Language to translate to.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the translated dataset to HuggingFace.",
    )

    return parser.parse_args()


def main():
    args = parse_input_args()  # TODO: add args support

    loader = DataLoaderFactory.get_loader("mteb")
    translator = TranslatorFactory.create_translator("openai")
    data_saver = DataSaverFactory.get_saver("huggingface")

    pipeline = TranslationPipeline(loader, translator, data_saver)

    asyncio.run(pipeline.execute())


if __name__ == "__main__":
    main()
