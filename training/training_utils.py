import json
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState


def make_path(save_path: str) -> Path:
    """
    Creates a directory for saving files if it does not already exist.

    Args:
        save_path (str): The path where the directory should be created.

    Returns:
        Path: The Path object representing the created directory.

    This function ensures that the directory structure is created, including any necessary parent directories.
    """
    model_save_path = Path(save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)
    return model_save_path


class EvalLoggingCallback(TrainerCallback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    def write_in_log_file(self, logs, json_file):
        """
        Appends log entries to a specified JSON log file.

        Args:
            self: The instance of the class containing the method.
            logs (Any): The log information to be written to the log file.
            json_file (str): The name of the JSON log file (including the .json extension) to which the logs will be appended.

        Returns:
            None
        """
        log_file = make_path(f"{self.save_path}/logs")
        log_file = log_file / json_file
        with open(log_file, "a") as f:
            print(logs)
            f.write(json.dumps(logs))
            f.write("\n")

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        """
        Handles logging during the training process.

        Args:
            self: The instance of the class containing the method.
            args: Additional arguments passed during logging.
            state (TrainerState): The current state of the trainer, containing information about the training process.
            control (TrainerControl): Controls the training process (e.g., stopping training).
            logs (dict, optional): A dictionary containing logging information (e.g., losses). Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        This method processes log entries during training. It removes the 'total_flos' key from the logs if present
        and checks for the presence of 'loss' or 'train_loss' keys to write their respective log entries to separate
        JSONL files.
        """
        _ = logs.pop("total_flos", None)

        if "loss" in logs:
            self.write_in_log_file(logs, "on_log_1.jsonl")
        if "train_loss" in logs:
            self.write_in_log_file(logs, "on_log_2.jsonl")

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Handles logging during the evaluation process.

        Args:
            self: The instance of the class containing the method.
            args: Additional arguments passed during the evaluation.
            state (TrainerState): The current state of the trainer, containing information about the training process.
            control (TrainerControl): Controls the training process (e.g., stopping training).
            **kwargs: Additional keyword arguments.

        Returns:
            None

        This method logs the evaluation metrics when the evaluation process starts. It retrieves the last logged
        evaluation metrics from the trainer's log history, adds the current epoch information, and writes this
        evaluation data to a JSONL file.
        """
        print("Evaluation started")
        eval_output = state.log_history[-1]  # Last logged evaluation metrics
        eval_output["epoch"] = state.epoch

        self.write_in_log_file(eval_output, "on_evaluate.jsonl")
