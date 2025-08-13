import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPDataset(Dataset):
    """
    Custom dataset class for NLP tasks.

    Args:
        data (pd.DataFrame): DataFrame containing text data and labels.
        tokenizer (AutoTokenizer): Tokenizer instance for preprocessing text data.
        max_length (int): Maximum length of text sequences.
    """
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class NLPEvaluation:
    """
    Class for evaluating NLP models.

    Args:
        model (AutoModelForSequenceClassification): Pre-trained model instance.
        tokenizer (AutoTokenizer): Tokenizer instance for preprocessing text data.
        device (str): Device to use for computations (e.g., 'cpu', 'cuda').
    """
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            data (pd.DataFrame): DataFrame containing text data and labels.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
        """
        dataset = NLPDataset(data, self.tokenizer, max_length=512)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels_batch)
                logits = outputs.logits
                predictions_batch = torch.argmax(logits, dim=1)

                predictions.extend(predictions_batch.cpu().numpy())
                labels.extend(labels_batch.cpu().numpy())

        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)

        return {
            'accuracy': accuracy,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1-score': report['macro avg']['f1-score']
        }

    def train(self, data: pd.DataFrame, epochs: int = 5) -> None:
        """
        Train the model on the given data.

        Args:
            data (pd.DataFrame): DataFrame containing text data and labels.
            epochs (int): Number of training epochs.
        """
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

        train_dataset = NLPDataset(train_data, self.tokenizer, max_length=512)
        val_dataset = NLPDataset(val_data, self.tokenizer, max_length=512)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')

            self.model.eval()
            val_loss = 0
            correct = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)

                    val_loss += loss.item()
                    correct += (predictions == labels).sum().item()

            accuracy = correct / len(val_data)
            logger.info(f'Epoch {epoch+1}, Val Loss: {val_loss / len(val_dataloader)}, Val Acc: {accuracy:.4f}')

def main() -> None:
    # Load pre-trained model and tokenizer
    model_name = 'distilbert-base-uncased'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set device (cpu or cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load data
    data = pd.read_csv('data.csv')

    # Create NLPEvaluation instance
    evaluator = NLPEvaluation(model, tokenizer, device)

    # Train the model
    evaluator.train(data, epochs=5)

    # Evaluate the model
    metrics = evaluator.evaluate(data)
    logger.info(f'Evaluation Metrics: {metrics}')

if __name__ == '__main__':
    main()