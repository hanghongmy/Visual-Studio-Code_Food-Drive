import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import joblib
import subprocess
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils.monitoring import TrainingMonitor
from prometheus_client import start_http_server, Counter, Gauge
from logging_handlers import RotatingFileHandler

# Configure logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add a rotating file handler for the train module
file_handler = RotatingFileHandler(
    f'{log_directory}/train.log',
    maxBytes=10485760,  # 10MB
    backupCount=5  # Keep up to 5 backup files
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('ml_app.train')
logger.addHandler(file_handler)

# Start Prometheus metrics server on port 8002
start_http_server(8003)

# Define Prometheus metrics
training_loss = Gauge('training_loss', 'Training loss')
validation_accuracy = Gauge('validation_accuracy', 'Validation accuracy')
epoch_count = Counter('epoch_count', 'Number of epochs completed')

logger.info("Training script started.")

# Set the MLflow tracking URI to the mlflow service in the Docker Compose network
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Verify the connection to the MLflow server
try:
    mlflow.set_experiment("CMPT2500")
except Exception as e:
    logger.error(f"Failed to connect to MLflow server: {e}")
    logger.info("Ensure the MLflow server is running and the MLFLOW_TRACKING_URI is set correctly.")
    exit(1)


class Trainer:
    def __init__(self, config_path: str):
        """
        Initializes the Trainer class with paths to processed data and model saving.

        :param config_path: Path to the configuration YAML file.
        """
        self.config_path = config_path

        # Load config file
        with open(config_path, "r") as path:
            self.config = yaml.safe_load(path)

        self.train_data_path = os.path.join("data","processed", self.config.get("train_data_path"))
        self.test_data_path = os.path.join("data","processed", self.config.get("test_data_path"))   

        self.models_dir = self.config["models_dir"]
        self.train_df = None
        self.test_df = None
        self.models = {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree": DecisionTreeRegressor(),
            "Random_Forest": RandomForestRegressor(
                n_estimators=self.config["random_forest"]["n_estimators"],
                max_depth=self.config["random_forest"]["max_depth"],
            )
        }
        self.param_grids = self.config["param_grids"]
        self.results_regular = []  # Store results for default models
        self.results_tuned = []  # Store results for tuned models

        # Start MLflow Experiment
        mlflow.set_experiment(self.config["mlflow_experiment_name"])

        # Enable MLflow autologging for Scikit-learn models
        mlflow.sklearn.autolog()

    def load_data(self):
        """Step 1: Load the training and testing datasets."""
        try:
            self.train_df = pd.read_csv(self.train_data_path)
            self.test_df = pd.read_csv(self.test_data_path)
            logger.info(f"Step 1: Training data loaded from {self.train_data_path} ({self.train_df.shape[0]} rows).")
            logger.info(f"Step 1: Testing data loaded from {self.test_data_path} ({self.test_df.shape[0]} rows).")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_value(self):
        """Step 2: Handle missing values in datasets."""
        for df in [self.train_df, self.test_df]:
            df.dropna(how='all', inplace=True)  # Drop fully empty rows
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill numeric NaNs
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna("Unknown")  # Fill categorical NaNs

        logger.info("Step 2: Missing values handled - Numeric with median, Categorical with 'Unknown'.")

    def prepare_data(self):
        """Step 3: Prepare features and target for model training."""
        feature_columns = ['time_spent', 'doors_in_route', 'assessed_value']
        target_column = 'donation_bags_collected'

        self.X_train = self.train_df[feature_columns]
        self.y_train = self.train_df[target_column]
        self.X_test = self.test_df[feature_columns]
        self.y_test = self.test_df[target_column]

        logger.info("Step 3: Data prepared for training.")

    def train_model(self, model_name, model, tuned=False):
        """Train a model and log results."""
        with mlflow.start_run(run_name=f"{model_name}_Tuned" if tuned else model_name):
            for epoch in range(1, 6):
                model.fit(self.X_train, self.y_train)

                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)

                # Log metrics to MLflow
                mlflow.log_param("tuned", tuned)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)

                # Update Prometheus metrics
                epoch_count.inc()
                training_loss.set(mse)
                validation_accuracy.set(r2)
            
                logger.info(f"Epoch {epoch} - {model_name} - MSE: {mse:.2f}, R² Score: {r2:.4f}")



            result = {
                "Model": model_name + ("_Tuned" if tuned else ""),
                "MSE": mse,
                "R² Score": r2
            }

            if tuned:
                self.results_tuned.append(result)
            else:
                self.results_regular.append(result)

            model_save_path = os.path.join(self.models_dir, f"{result['Model']}.pkl")
            joblib.dump(model, model_save_path)
            logger.info(f"Model '{result['Model']}' saved to {model_save_path}.")

    def hypertune_model(self, model_name, model):
        """Perform hyperparameter tuning using GridSearchCV."""
        if model_name not in self.param_grids:
            return model

        logger.info(f"Hyperparameter tuning for {model_name}...")
        param_grid = self.param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logger.info(f"Best hyperparameters for {model_name}: {best_params}")

        return grid_search.best_estimator_

    def train_models(self):
        """Train models with and without hyperparameter tuning."""
        os.makedirs(self.models_dir, exist_ok=True)

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} without tuning...")
            self.train_model(model_name, model, tuned=False)

            if model_name in self.param_grids:
                tuned_model = self.hypertune_model(model_name, model)
                self.train_model(model_name, tuned_model, tuned=True)

    def track_data_with_dvc(self):
        """Step 5: Track processed data with DVC."""
        try:
            subprocess.run(["dvc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            subprocess.run(["dvc", "add", self.train_data_path], check=True)
            subprocess.run(["dvc", "add", self.test_data_path], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Updated processed dataset"], check=True)
            subprocess.run(["dvc", "push"], check=True)
            logger.info("Step 5: Processed data tracked and pushed with DVC.")
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC tracking failed: {e}")
        except FileNotFoundError:
            logger.error("DVC is not installed. Please install DVC to track data. Using make init-dvc to install")

    def train_pipeline(self):
        """Run the full training pipeline."""
        logger.info("Starting training pipeline...")
        self.load_data()
        self.handle_missing_value()
        self.prepare_data()
        self.train_models()
        self.track_data_with_dvc()
        logger.info("Training pipeline complete.")

if __name__ == "__main__":
    logger.info("Starting training script...")
    # Start Promethrues metrics server
    #start_http_server(8003)
    logger.info("Training started...")
    logger.info("Training completed")
    trainer = Trainer(config_path="configs/train_config.yaml")
    trainer.train_pipeline()
    #subprocess.run(["./venv/bin/mlflow", "server", "--host", "127.0.0.1", "--port", "5000"])
