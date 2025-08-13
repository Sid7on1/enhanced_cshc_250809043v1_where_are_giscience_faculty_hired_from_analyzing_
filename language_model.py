# language_model.py

import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(LanguageModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class LanguageModelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: AutoTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=512,
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

class LanguageModelTrainer:
    def __init__(self, model: LanguageModel, device: torch.device, batch_size: int):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def train(self, train_data: LanguageModelDataset, epochs: int):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    def evaluate(self, test_data: LanguageModelDataset):
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        predictions = []
        labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels.extend(batch['labels'].cpu().numpy())

                outputs = self.model(input_ids, attention_mask)
                logits = outputs.logits.cpu().numpy()
                predictions.extend(np.argmax(logits, axis=1))

        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        matrix = confusion_matrix(labels, predictions)

        logger.info(f'Accuracy: {accuracy:.4f}')
        logger.info(f'Classification Report:\n{report}')
        logger.info(f'Confusion Matrix:\n{matrix}')

def load_data(file_path: str) -> Tuple[List[str], List[str]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            text, label = line.strip().split('\t')
            data.append((text, label))

    texts, labels = zip(*data)
    return list(texts), list(labels)

def main():
    model_name = 'bert-base-uncased'
    num_classes = 10
    batch_size = 32
    epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    texts, labels = load_data('data.txt')
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data = LanguageModelDataset(train_texts, train_labels, tokenizer)
    test_data = LanguageModelDataset(test_texts, test_labels, tokenizer)

    model = LanguageModel(model_name, num_classes)
    model.to(device)

    trainer = LanguageModelTrainer(model, device, batch_size)
    trainer.train(train_data, epochs)
    trainer.evaluate(test_data)

if __name__ == '__main__':
    main()