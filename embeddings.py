import logging
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
from embeddings.config import Config
from embeddings.exceptions import EmbeddingsError
from embeddings.utils import load_model, load_tokenizer, get_device

logger = logging.getLogger(__name__)

class Embeddings:
    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()
        self.model = load_model(config.model_name)
        self.tokenizer = load_tokenizer(config.model_name)
        self.scaler = StandardScaler()

    def _preprocess_text(self, text: str) -> torch.Tensor:
        """Preprocess text by tokenizing and encoding it."""
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.max_length,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        return inputs["input_ids"].to(self.device)

    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for a given text."""
        inputs = self._preprocess_text(text)
        outputs = self.model(inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def get_word_embeddings(self, word: str) -> torch.Tensor:
        """Get embeddings for a single word."""
        try:
            embeddings = self._get_embeddings(word)
            return embeddings
        except Exception as e:
            raise EmbeddingsError(f"Failed to get embeddings for word '{word}': {str(e)}")

    def get_sentence_embeddings(self, sentence: str) -> torch.Tensor:
        """Get embeddings for a sentence."""
        try:
            embeddings = self._get_embeddings(sentence)
            return embeddings
        except Exception as e:
            raise EmbeddingsError(f"Failed to get embeddings for sentence '{sentence}': {str(e)}")

    def calculate_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> float:
        """Calculate similarity between two embeddings."""
        try:
            similarity = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())
            return similarity
        except Exception as e:
            raise EmbeddingsError(f"Failed to calculate similarity: {str(e)}")

    def get_word_similarity(self, word1: str, word2: str) -> float:
        """Get similarity between two words."""
        try:
            embeddings1 = self.get_word_embeddings(word1)
            embeddings2 = self.get_word_embeddings(word2)
            similarity = self.calculate_similarity(embeddings1, embeddings2)
            return similarity
        except Exception as e:
            raise EmbeddingsError(f"Failed to get similarity between words '{word1}' and '{word2}': {str(e)}")

    def get_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """Get similarity between two sentences."""
        try:
            embeddings1 = self.get_sentence_embeddings(sentence1)
            embeddings2 = self.get_sentence_embeddings(sentence2)
            similarity = self.calculate_similarity(embeddings1, embeddings2)
            return similarity
        except Exception as e:
            raise EmbeddingsError(f"Failed to get similarity between sentences '{sentence1}' and '{sentence2}': {str(e)}")

class Config:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.max_length = 512

class EmbeddingsError(Exception):
    pass

def load_model(model_name: str) -> nn.Module:
    model = AutoModel.from_pretrained(model_name)
    return model

def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    config = Config()
    embeddings = Embeddings(config)
    word1 = "hello"
    word2 = "world"
    sentence1 = "This is a test sentence."
    sentence2 = "This is another test sentence."
    print(embeddings.get_word_similarity(word1, word2))
    print(embeddings.get_sentence_similarity(sentence1, sentence2))