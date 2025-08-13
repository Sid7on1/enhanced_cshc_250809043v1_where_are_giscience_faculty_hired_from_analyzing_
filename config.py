import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from enhanced_cs.utils.data import Dataset
from enhanced_cs.utils.logging import Logger

logger = Logger(os.path.basename(__file__))

# Set default PyTorch device to CPU or GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    """Configuration class for the project.

    This class provides a centralized location for managing and accessing
    configuration settings, parameters, and constants. It also includes methods
    for loading and saving configurations to/from files.

    ...

    Attributes
    ----------
    dataset_path : str
        Path to the dataset directory.
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Dimension of hidden layers in the model.
    output_dim : int
        Dimension of the output layer in the model.
    learning_rate : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size for training and evaluation.
    num_epochs : int
        Number of epochs to train the model.
    checkpoint_dir : str
        Directory to save model checkpoints.
    log_dir : str
        Directory to save log files.
    seed : int
        Random seed for reproducibility.

    Methods
    -------
    load_from_file(file_path: str) -> None:
        Load configuration from a JSON or INI format file.
    save_to_file(file_path: str, file_format: str = 'json') -> None:
        Save configuration to a JSON or INI format file.
    set_params(params: Dict[str, Any]) -> None:
        Set multiple parameters at once from a dictionary.

    """

    def __init__(
        self,
        dataset_path: str,
        embedding_dim: int = 300,
        hidden_dim: int = 64,
        output_dim: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        seed: int = 42,
    ):
        """Initialize the configuration with default values.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset directory.
        embedding_dim : int, optional
            Dimension of word embeddings, by default 300.
        hidden_dim : int, optional
            Dimension of hidden layers in the model, by default 64.
        output_dim : int, optional
            Dimension of the output layer in the model, by default 2.
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.001.
        batch_size : int, optional
            Batch size for training and evaluation, by default 32.
        num_epochs : int, optional
            Number of epochs to train the model, by default 10.
        checkpoint_dir : str, optional
            Directory to save model checkpoints, by default "checkpoints".
        log_dir : str, optional
            Directory to save log files, by default "logs".
        seed : int, optional
            Random seed for reproducibility, by default 42.

        """
        self.dataset_path = dataset_path
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.seed = seed

    def load_from_file(self, file_path: str) -> None:
        """Load configuration from a JSON or INI format file.

        Parameters
        ----------
        file_path : str
            Path to the configuration file.

        Raises
        ------
        ValueError
            If the file format is neither JSON nor INI.

        """
        if file_path.endswith(".json"):
            self.__dict__.update(
                **self._load_json(file_path)
            )  # Update attributes with loaded config
        elif file_path.endswith(".ini"):
            self.load_from_ini(file_path)
        else:
            raise ValueError("Unsupported file format. Use JSON or INI format.")

    @staticmethod
    def _load_json(file_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON configuration file.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the loaded configuration.

        """
        import json

        with open(file_path, "r") as f:
            return json.load(f)

    def load_from_ini(self, file_path: str) -> None:
        """Load configuration from an INI format file.

        Parameters
        ----------
        file_path : str
            Path to the INI configuration file.

        """
        import configparser

        config = configparser.ConfigParser()
        config.read(file_path)

        self.set_params({section: dict(config[section]) for section in config.sections()})

    def save_to_file(self, file_path: str, file_format: str = "json") -> None:
        """Save configuration to a JSON or INI format file.

        Parameters
        ----------
        file_path : str
            Path to the output configuration file.
        file_format : str, optional
            Format of the output file, either 'json' or 'ini', by default 'json'.

        Raises
        ------
        ValueError
            If the specified file format is not supported.

        """
        if file_format == "json":
            self._save_json(file_path)
        elif file_format == "ini":
            self._save_ini(file_path)
        else:
            raise ValueError("Unsupported file format. Use 'json' or 'ini'.")

    def _save_json(self, file_path: str) -> None:
        """Save configuration to a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the output JSON configuration file.

        """
        import json

        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def _save_ini(self, file_path: str) -> None:
        """Save configuration to an INI format file.

        Parameters
        ----------
        file_path : str
            Path to the output INI configuration file.

        """
        import configparser

        config = configparser.ConfigParser()

        for section, params in self.__dict__.items():
            if not section.startswith("_"):  # Ignore private attributes
                config[section] = params

        with open(file_path, "w") as f:
            config.write(f)

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set multiple parameters at once from a dictionary.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary containing parameter names and values.

        """
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def device(self):
        """Get the device (CPU or GPU) for PyTorch operations."""
        return device

    @device.setter
    def device(self, device: str):
        """Set the device for PyTorch operations."""
        self._device = torch.device(device)

    @property
    def collate_fn(self):
        """Get the collate function for PyTorch DataLoader."""
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, collate_fn):
        """Set the collate function for PyTorch DataLoader."""
        self._collate_fn = collate_fn

    @staticmethod
    def get_dataset(dataset_path: str) -> Dataset:
        """Load and preprocess the dataset.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset directory.

        Returns
        -------
        Dataset
            Preprocessed dataset object.

        """
        # Example: Load and preprocess the dataset
        data = pd.read_csv(os.path.join(dataset_path, "data.csv"))
        # ... preprocessing steps ...

        # Create Dataset object and return
        return Dataset(data)

    @staticmethod
    def get_vocab_size(dataset: Dataset) -> int:
        """Get the vocabulary size from the dataset.

        Parameters
        ----------
        dataset : Dataset
            Preprocessed dataset object.

        Returns
        -------
        int
            Vocabulary size.

        """
        return len(dataset.vocab)


class ModelConfig:
    """Configuration class for the model.

    This class provides model-specific configuration settings and methods.

    ...

    Attributes
    ----------
    input_dim : int
        Dimension of the model input.
    output_dim : int
        Dimension of the model output.
    hidden_dims : List[int]
        List of hidden dimensions for each layer.
    dropout : float
        Dropout probability for regularization.
    bidirectional : bool
        Whether to use bidirectional RNNs.

    Methods
    -------
    init_weights(m: nn.Module) -> None:
        Initialize weights of the model.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        """Initialize the model configuration.

        Parameters
        ----------
        input_dim : int
            Dimension of the model input.
        output_dim : int
            Dimension of the model output.
        hidden_dims : List[int]
            List of hidden dimensions for each layer.
        dropout : float, optional
            Dropout probability for regularization, by default 0.5.
        bidirectional : bool, optional
            Whether to use bidirectional RNNs, by default True.

        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.bidirectional = bidirectional

    def init_weights(self, m: nn.Module) -> None:
        """Initialize weights of the model.

        Parameters
        ----------
        m : nn.Module
            Model or layer to initialize weights for.

        """
        if isinstance(m, nn.RNN) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)


class TrainingConfig:
    """Configuration class for training settings.

    This class provides training-specific configuration settings and methods.

    ...

    Attributes
    ----------
    optimizer : str
        Name of the optimizer to use.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) coefficient.
    scheduler : str
        Name of the learning rate scheduler to use.
    scheduler_params : Dict[str, Any]
        Parameters for the learning rate scheduler.

    Methods
    -------
    get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        Get the optimizer for the given model.
    get_scheduler(
        optimizer: torch.optim.Optimizer, num_iterations: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        Get the learning rate scheduler for the given optimizer.

    """

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        scheduler: str = "step_lr",
        scheduler_params: Dict[str, Any] = {},
    ):
        """Initialize the training configuration.

        Parameters
        ----------
        optimizer : str, optional
            Name of the optimizer to use, by default "adam".
        learning_rate : float, optional
            Learning rate for the optimizer, by default 0.001.
        weight_decay : float, optional
            Weight decay (L2 regularization) coefficient, by default 0.0.
        scheduler : str, optional
            Name of the learning rate scheduler to use, by default "step_lr".
        scheduler_params : Dict[str, Any], optional
            Parameters for the learning rate scheduler, by default {}.

        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Get the optimizer for the given model.

        Parameters
        ----------
        model : nn.Module
            Model to create an optimizer for.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer instance.

        Raises
        ------
        ValueError
            If the specified optimizer is not supported.

        """
        optimizer_map = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }

        if self.optimizer not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        optimizer_class = optimizer_map[self.optimizer]
        optimizer = optimizer_class(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        return optimizer

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, num_iterations: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get the learning rate scheduler for the given optimizer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to create a scheduler for.
        num_iterations : int
            Total number of training iterations.

        Returns
        -------
        Optional[torch.optim.lr_scheduler._LRScheduler]
            Learning rate scheduler instance, or None if no scheduler is specified.

        Raises
        ------
        ValueError
            If the specified scheduler is not supported.

        """
        scheduler = None
        scheduler_map = {
            "step_lr": torch.optim.lr_scheduler.StepLR,
            "exp_lr": torch.optim.lr_scheduler.ExponentialLR,
            "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        if self.scheduler in scheduler_map:
            scheduler_class = scheduler_map[self.scheduler]
            scheduler = scheduler_class(
                optimizer, step_size=self.scheduler_params.get("step_size", 10), gamma=0.1
            )

        return scheduler


# Example usage
if __name__ == "__main__":
    # Create configuration instances
    config = Config(dataset_path="path/to/dataset")
    model_config = ModelConfig(input_dim=config.embedding_dim, output_dim=config.output_dim)
    training_config = TrainingConfig()

    # Load configuration from a JSON file
    config_file = "path/to/config.json"
    config.load_from_file(config_file)

    # Save configuration to a JSON file
    output_config_file = "path/to/output_config.json"
    config.save_to_file(output_config_file)

    # Get dataset and vocabulary size
    dataset = config.get_dataset(config.dataset_path)
    vocab_size = config.get_vocab_size(dataset)
    print(f"Vocabulary size: {vocab_size}")

    # Example: Logging
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    # Example: Model and training configuration
    model = Model(model_config)
    optimizer = training_config.get_optimizer(model)
    scheduler = training_config.get_scheduler(optimizer, num_iterations=1000)