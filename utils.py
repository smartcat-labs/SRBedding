from datetime import datetime
from pathlib import Path


def get_timestamp() -> str:
    """
    Returns the current timestamp as a formatted string.

    This function generates the current date and time, formatted as a string in the
    format 'DD-MM-YYYY_HH-MM-SS'. It can be useful for creating unique filenames
    or logging events with a timestamp.

    Returns:
        str: The current timestamp in 'DD-MM-YYYY_HH-MM-SS' format.

    Example:
        >>> timestamp = get_timestamp()
        >>> print(timestamp)
        '18-08-2024_15-45-30'
    """
    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")


def make_path(save_path: str):
    """
    Creates the specified directory path if it does not already exist.

    Args:
        save_path (str): The directory path to create.

    Returns:
        Path: The created or existing directory path as a Path object.
    """
    model_save_path = Path(save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)
    return model_save_path


def make_cache_dir() -> Path:
    """
    Creates and returns the cache directory path for storing datasets.

    Returns:
        Path: The expanded user path to the cache directory.
    """
    return Path("~/Datasets/SRBendding").expanduser()
