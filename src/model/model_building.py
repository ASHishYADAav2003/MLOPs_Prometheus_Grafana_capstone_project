import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging
import mlflow
import mlflow.sklearn
import dagshub
import os

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ASHishYADAav2003"
repo_name = "MLOPs_Prometheus_Grafana_capstone_project"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/ASHishYADAav2003/MLOPs_Prometheus_Grafana_capstone_project.mlflow')
# dagshub.init(repo_owner='ASHishYADAav2003', repo_name='MLOPs_Prometheus_Grafana_capstone_project', mlflow=True)
#  -------------------------------------------------------------------------------------


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise




def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    # Load data
    train_data = pd.read_csv("./data/processed/train_bow.csv")

    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Create folder
    os.makedirs("models", exist_ok=True)

    # Save model (THIS FILE DVC NEEDS)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ model.pkl created successfully")

if __name__ == "__main__":
    main()