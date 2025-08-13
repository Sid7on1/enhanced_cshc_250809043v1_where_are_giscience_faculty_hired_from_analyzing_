import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define constants
VELOCITY_THRESHOLD = 0.5
FLOW_THEORY_THRESHOLD = 0.8

# Define custom exception classes
class InferenceError(Exception):
    """Base class for inference-related exceptions."""
    pass

class ModelNotTrainedError(InferenceError):
    """Raised when the model is not trained."""
    pass

class InvalidInputError(InferenceError):
    """Raised when the input is invalid."""
    pass

# Define data structures/models
class GIScienceFaculty:
    def __init__(self, name: str, department: str, university: str):
        """
        Initialize a GIScienceFaculty object.

        Args:
        - name (str): The name of the faculty member.
        - department (str): The department of the faculty member.
        - university (str): The university of the faculty member.
        """
        self.name = name
        self.department = department
        self.university = university

# Define helper classes and utilities
class FacultyDataset(Dataset):
    def __init__(self, data: List[Dict], labels: List[int]):
        """
        Initialize a FacultyDataset object.

        Args:
        - data (List[Dict]): A list of dictionaries containing faculty data.
        - labels (List[int]): A list of labels corresponding to the data.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

class FacultyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initialize a FacultyTransformer object.
        """
        self.imputer = SimpleImputer(strategy='mean')
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the transformer to the data.

        Args:
        - X (np.ndarray): The data to fit.
        - y (np.ndarray): The target values (optional).
        """
        self.imputer.fit(X)
        self.encoder.fit(X)
        return self

    def transform(self, X: np.ndarray):
        """
        Transform the data.

        Args:
        - X (np.ndarray): The data to transform.

        Returns:
        - np.ndarray: The transformed data.
        """
        X_imputed = self.imputer.transform(X)
        X_encoded = self.encoder.transform(X_imputed)
        return X_encoded

# Define the main class
class InferencePipeline:
    def __init__(self, model: str = 'logistic_regression'):
        """
        Initialize an InferencePipeline object.

        Args:
        - model (str): The model to use (default: 'logistic_regression').
        """
        self.model = model
        self.trained_model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.

        Args:
        - X (np.ndarray): The training data.
        - y (np.ndarray): The target values.

        Raises:
        - ModelNotTrainedError: If the model is not trained.
        """
        if self.trained_model is None:
            raise ModelNotTrainedError("Model is not trained.")

        # Define the pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        if self.model == 'logistic_regression':
            self.trained_model = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('classifier', LogisticRegression())])
        elif self.model == 'random_forest':
            self.trained_model = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('classifier', RandomForestClassifier())])
        elif self.model == 'svm':
            self.trained_model = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('classifier', SVC())])

        # Train the model
        self.trained_model.fit(X, y)

    def predict(self, X: np.ndarray):
        """
        Make predictions on the data.

        Args:
        - X (np.ndarray): The data to predict.

        Returns:
        - np.ndarray: The predicted values.

        Raises:
        - ModelNotTrainedError: If the model is not trained.
        """
        if self.trained_model is None:
            raise ModelNotTrainedError("Model is not trained.")

        # Make predictions
        predictions = self.trained_model.predict(X)
        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the model on the data.

        Args:
        - X (np.ndarray): The data to evaluate.
        - y (np.ndarray): The target values.

        Returns:
        - Dict: A dictionary containing the evaluation metrics.

        Raises:
        - ModelNotTrainedError: If the model is not trained.
        """
        if self.trained_model is None:
            raise ModelNotTrainedError("Model is not trained.")

        # Evaluate the model
        predictions = self.trained_model.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        matrix = confusion_matrix(y, predictions)

        # Calculate the velocity and flow theory metrics
        velocity = np.mean(np.abs(predictions - y))
        flow_theory = np.mean(np.abs(predictions - y) / (1 + np.abs(y)))

        # Create a dictionary containing the evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': matrix,
            'velocity': velocity,
            'flow_theory': flow_theory
        }

        return metrics

# Define the main function
def main():
    # Load the data
    data = pd.read_csv('data.csv')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

    # Create an instance of the InferencePipeline class
    pipeline = InferencePipeline(model='logistic_regression')

    # Train the model
    pipeline.train(X_train, y_train)

    # Make predictions on the test data
    predictions = pipeline.predict(X_test)

    # Evaluate the model on the test data
    metrics = pipeline.evaluate(X_test, y_test)

    # Print the evaluation metrics
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Classification Report:")
    print(metrics['classification_report'])
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"Velocity: {metrics['velocity']:.3f}")
    print(f"Flow Theory: {metrics['flow_theory']:.3f}")

if __name__ == '__main__':
    main()