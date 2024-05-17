import pytest
from datasets.exceptions import DatasetNotFoundError
from loaders.mteb_loader import MTEBLoader


@pytest.fixture
def loader():
    """Return an instance of the MTEBLoader class."""

    return MTEBLoader()


def test_load_data_scifact(loader):
    """Test loading data for a valid SCIFACT dataset."""

    dataset_name = "mteb/scifact"
    split = "test"
    queries_data, corpus_data, test_data = loader.load_data(dataset_name, split)
    assert queries_data is not None
    assert corpus_data is not None
    assert test_data is not None


def test_load_data_invalid_dataset(loader):
    """Test loading data for an invalid dataset."""

    dataset_name = "invalid_dataset"
    split = "train"
    with pytest.raises(DatasetNotFoundError):
        loader.load_data(dataset_name, split)


def test_load_data_invalid_split(loader):
    """Test loading data for an invalid split."""

    dataset_name = "mteb/scifact"
    split = "invalid_split"
    with pytest.raises(ValueError):
        loader.load_data(dataset_name, split)
