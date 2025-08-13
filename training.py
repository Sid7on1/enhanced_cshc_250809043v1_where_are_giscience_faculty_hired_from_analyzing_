import logging
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration for the training pipeline."""
    def __init__(self, 
                 batch_size: int = 32, 
                 num_epochs: int = 10, 
                 learning_rate: float = 0.001, 
                 validation_split: float = 0.2):
        """
        Initialize the training configuration.

        Args:
        - batch_size (int): The batch size for training.
        - num_epochs (int): The number of epochs for training.
        - learning_rate (float): The learning rate for the optimizer.
        - validation_split (float): The proportion of data for validation.
        """
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.validation_split = validation_split

class FacultyHiringDataset(Dataset):
    """Dataset for faculty hiring data."""
    def __init__(self, data: pd.DataFrame, labels: pd.Series):
        """
        Initialize the dataset.

        Args:
        - data (pd.DataFrame): The feature data.
        - labels (pd.Series): The target labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Get a data point and its label.

        Args:
        - index (int): The index of the data point.

        Returns:
        - data (torch.Tensor): The data point.
        - label (torch.Tensor): The label.
        """
        data = torch.tensor(self.data.iloc[index].values, dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[index], dtype=torch.long)
        return data, label

class FacultyHiringModel(torch.nn.Module):
    """Model for faculty hiring prediction."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the model.

        Args:
        - input_dim (int): The input dimension.
        - hidden_dim (int): The hidden dimension.
        - output_dim (int): The output dimension.
        """
        super(FacultyHiringModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
        - x (torch.Tensor): The input.

        Returns:
        - output (torch.Tensor): The output.
        """
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output

class TrainingPipeline:
    """Training pipeline for faculty hiring prediction."""
    def __init__(self, config: TrainingConfig, model: FacultyHiringModel, device: torch.device):
        """
        Initialize the training pipeline.

        Args:
        - config (TrainingConfig): The training configuration.
        - model (FacultyHiringModel): The model.
        - device (torch.device): The device for training.
        """
        self.config = config
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self, train_loader: DataLoader):
        """
        Train the model.

        Args:
        - train_loader (DataLoader): The training data loader.
        """
        self.model.train()
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        logger.info(f'Training loss: {total_loss / len(train_loader)}')

    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate the model.

        Args:
        - val_loader (DataLoader): The validation data loader.

        Returns:
        - accuracy (float): The accuracy.
        - report (str): The classification report.
        - matrix (np.ndarray): The confusion matrix.
        """
        self.model.eval()
        total_correct = 0
        labels = []
        predictions = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == target).sum().item()
                labels.extend(target.cpu().numpy())
                predictions.extend(predicted.cpu().numpy())
        accuracy = total_correct / len(val_loader.dataset)
        report = classification_report(labels, predictions)
        matrix = confusion_matrix(labels, predictions)
        logger.info(f'Validation accuracy: {accuracy:.4f}')
        logger.info(f'Validation report:\n{report}')
        logger.info(f'Validation matrix:\n{matrix}')
        return accuracy, report, matrix

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the data.

    Args:
    - file_path (str): The file path.

    Returns:
    - data (pd.DataFrame): The feature data.
    - labels (pd.Series): The target labels.
    """
    data = pd.read_csv(file_path)
    labels = data['label']
    data = data.drop('label', axis=1)
    return data, labels

def main():
    # Load the data
    data, labels = load_data('data.csv')

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create the dataset and data loader
    train_dataset = FacultyHiringDataset(train_data, train_labels)
    val_dataset = FacultyHiringDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create the model and training pipeline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FacultyHiringModel(input_dim=10, hidden_dim=20, output_dim=2)
    config = TrainingConfig(batch_size=32, num_epochs=10, learning_rate=0.001, validation_split=0.2)
    pipeline = TrainingPipeline(config, model, device)

    # Train the model
    for epoch in range(config.num_epochs):
        logger.info(f'Epoch {epoch+1}')
        pipeline.train(train_loader)
        accuracy, report, matrix = pipeline.evaluate(val_loader)

if __name__ == '__main__':
    main()