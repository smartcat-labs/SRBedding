from savers.base import DataSaver


class HuggingFaceSaver(DataSaver):

    def __init__(self) -> None:
        """
        Initialize the HuggingFaceSaver class.
        """

        super().__init__()

    def save_data(self, data, path, push_to_hub=False):
        """Save the data to disk or push it to the HuggingFace hub."""

        if push_to_hub:
            data.push_to_hub(path, use_auth_token=True)
        else:
            data.save_to_disk(path)
