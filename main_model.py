import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GIScienceFacultyHiringModel(nn.Module):
    """
    Main NLP model implementation for analyzing GIScience faculty hiring.

    Attributes:
    - input_dim (int): Input dimension of the model.
    - hidden_dim (int): Hidden dimension of the model.
    - output_dim (int): Output dimension of the model.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the GIScienceFacultyHiringModel.

        Args:
        - input_dim (int): Input dimension of the model.
        - hidden_dim (int): Hidden dimension of the model.
        - output_dim (int): Output dimension of the model.
        """
        super(GIScienceFacultyHiringModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GIScienceFacultyHiringDataset(Dataset):
    """
    Custom dataset class for GIScience faculty hiring data.

    Attributes:
    - data (List[Dict]): List of dictionaries containing data.
    - labels (List[int]): List of labels corresponding to the data.
    """

    def __init__(self, data: List[Dict], labels: List[int]):
        """
        Initializes the GIScienceFacultyHiringDataset.

        Args:
        - data (List[Dict]): List of dictionaries containing data.
        - labels (List[int]): List of labels corresponding to the data.
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict, int]:
        """
        Returns a data point and its corresponding label.

        Args:
        - idx (int): Index of the data point.

        Returns:
        - Tuple[Dict, int]: Data point and its label.
        """
        return self.data[idx], self.labels[idx]

def create_dataloader(data: List[Dict], labels: List[int], batch_size: int) -> DataLoader:
    """
    Creates a DataLoader instance for the given data and labels.

    Args:
    - data (List[Dict]): List of dictionaries containing data.
    - labels (List[int]): List of labels corresponding to the data.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - DataLoader: DataLoader instance.
    """
    dataset = GIScienceFacultyHiringDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model: GIScienceFacultyHiringModel, dataloader: DataLoader, epochs: int, learning_rate: float) -> None:
    """
    Trains the GIScienceFacultyHiringModel using the given dataloader and hyperparameters.

    Args:
    - model (GIScienceFacultyHiringModel): Model instance to train.
    - dataloader (DataLoader): DataLoader instance for training data.
    - epochs (int): Number of epochs to train the model.
    - learning_rate (float): Learning rate for the optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for batch in dataloader:
            data, labels = batch
            data = torch.tensor(data)
            labels = torch.tensor(labels)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate_model(model: GIScienceFacultyHiringModel, dataloader: DataLoader) -> float:
    """
    Evaluates the GIScienceFacultyHiringModel using the given dataloader.

    Args:
    - model (GIScienceFacultyHiringModel): Model instance to evaluate.
    - dataloader (DataLoader): DataLoader instance for evaluation data.

    Returns:
    - float: Accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch
            data = torch.tensor(data)
            labels = torch.tensor(labels)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    logging.info(f'Accuracy: {accuracy:.4f}')
    return accuracy

def main() -> None:
    """
    Main function for the GIScience faculty hiring model.
    """
    # Load data
    data = pd.read_csv('data.csv')
    labels = data['label']
    data = data.drop('label', axis=1)

    # Create dataloader
    dataloader = create_dataloader(data, labels, batch_size=32)

    # Create model
    model = GIScienceFacultyHiringModel(input_dim=10, hidden_dim=20, output_dim=2)

    # Train model
    train_model(model, dataloader, epochs=10, learning_rate=0.001)

    # Evaluate model
    evaluate_model(model, dataloader)

if __name__ == '__main__':
    main()