import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_client import start_http_server, Counter, Gauge
import time

# Configure logging
logging.basicConfig(
    filename='logs/train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_app.train')

# Prometheus metrics
training_loss = Gauge('training_loss', 'Training loss of the model')
epoch_count = Counter('training_epoch_total', 'Total number of epochs completed')
validation_accuracy = Gauge('validation_accuracy', 'Validation accuracy of the model')# Start Prometheus metrics server

start_http_server(8002)
# Set MLflow tracking URI
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Verify MLflow connection
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
        self._load_config()
        self._initialize_models()

    def _load_config(self):
        """Load configuration from the YAML file."""
        with open(self.config_path, "r") as path:
            self.config = yaml.safe_load(path)
        self.train_data_path = os.path.join("data", "processed", self.config.get("train_data_path"))
        self.test_data_path = os.path.join("data", "processed", self.config.get("test_data_path"))
        self.models_dir = self.config["models_dir"]

    def _initialize_models(self):
        """Initialize models and hyperparameter grids."""
        self.models = {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree": DecisionTreeRegressor(),
            "Random_Forest": RandomForestRegressor(
                n_estimators=self.config["random_forest"]["n_estimators"],
                max_depth=self.config["random_forest"]["max_depth"],
            )
        }
        self.param_grids = self.config.get("param_grids", {})
        self.results = []

    def load_data(self):
        """Load training and testing datasets."""
        try:
            self.train_df = pd.read_csv(self.train_data_path)
            self.test_df = pd.read_csv(self.test_data_path)
            logger.info(f"Training data loaded: {self.train_df.shape[0]} rows.")
            logger.info(f"Testing data loaded: {self.test_df.shape[0]} rows.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self):
        """Handle missing values in datasets."""
        for df in [self.train_df, self.test_df]:
            df.dropna(how='all', inplace=True)
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            categorical_cols = df.select_dtypes(include=['object']).columns
            df[categorical_cols] = df[categorical_cols].fillna("Unknown")
        logger.info("Missing values handled.")

    def prepare_data(self):
        """Prepare features and target for training."""
        feature_columns = ['time_spent', 'doors_in_route', 'assessed_value']
        target_column = 'donation_bags_collected'
        self.X_train = self.train_df[feature_columns]
        self.y_train = self.train_df[target_column]
        self.X_test = self.test_df[feature_columns]
        self.y_test = self.test_df[target_column]
        logger.info("Data prepared for training.")

    def train_model(self, model_name, model, tuned=False):
        """Train a model and log results."""
        with mlflow.start_run(run_name=f"{model_name}_Tuned" if tuned else model_name):
            for epoch in range(1, 6):  # Simulate 5 epochs
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)

                # Log metrics to MLflow
                mlflow.log_param("tuned", tuned)
                mlflow.log_metric("mse", mse, step=epoch)
                mlflow.log_metric("r2", r2, step=epoch)

                # Update Prometheus metrics
                epoch_count.inc()
                training_loss.set(mse)
                validation_accuracy.set(r2)

                logger.info(f"Epoch {epoch} - {model_name} - MSE: {mse:.2f}, R²: {r2:.4f}")

            # Save the model
            model_save_path = os.path.join(self.models_dir, f"{model_name}{'_Tuned' if tuned else ''}.pkl")
            joblib.dump(model, model_save_path)
            logger.info(f"Model saved to {model_save_path}.")

            self.results.append({"Model": model_name, "MSE": mse, "R²": r2})

    def hypertune_model(self, model_name, model):
        """Perform hyperparameter tuning using GridSearchCV."""
        if model_name not in self.param_grids:
            return model
        logger.info(f"Hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(model, self.param_grids[model_name], cv=3, scoring="r2", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        logger.info(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_models(self):
        """Train models with and without hyperparameter tuning."""
        os.makedirs(self.models_dir, exist_ok=True)
        for model_name, model in self.models.items():
            self.train_model(model_name, model, tuned=False)
            if model_name in self.param_grids:
                tuned_model = self.hypertune_model(model_name, model)
                self.train_model(model_name, tuned_model, tuned=True)

    def train_pipeline(self):
        """Run the full training pipeline."""
        logger.info("Starting training pipeline...")
        self.load_data()
        self.handle_missing_values()
        self.prepare_data()
        self.train_models()
        logger.info("Training pipeline complete.")

#if __name__ == "__main__":
#    logger.info("Starting training script...")
#    trainer = Trainer(config_path="configs/train_config.yaml")
#    trainer.train_pipeline()
#    logger.info("Training completed.")
#    while True:
#        time.sleep(1)  # Keep the application running to expose metrics

if __name__ == "__main__":
    logger.info("Starting training script...")

    # Set MLflow experiment
    experiment_name = "FoodDrive_Training"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"Using MLflow experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Failed to set MLflow experiment: {e}")
        exit(1)

    # Initialize the Trainer and run the training pipeline
    trainer = Trainer(config_path="configs/train_config.yaml")
    trainer.train_pipeline()

    # Log the configuration file as an artifact
    try:
        mlflow.log_artifact("configs/train_config.yaml")
        logger.info("Configuration file logged as an artifact.")
    except Exception as e:
        logger.error(f"Failed to log configuration file: {e}")

    logger.info("Training completed.")
    while True:
        time.sleep(1)  # Keep the application running to expose metrics