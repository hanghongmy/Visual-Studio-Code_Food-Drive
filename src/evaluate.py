import os
import joblib
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
import yaml
import subprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Evaluator:
    def __init__(self, config_path="configs/predict_config.yaml"):
        """
        Initializes the Evaluator with paths to the test dataset and trained models.
        
        :param test_data_path: Path to the processed test data (Food Drive 2024)
        :param models_dir: Directory where trained models are stored
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

            self.test_data_path = os.path.join("data", "processed", self.config["test_data_path"])
            self.models_dir = self.config["models_dir"]
            self.df = None
            self.models = {}
            self.metrics = {}
            
        # Set up mlflow experiment
        mlflow.set_experiment(self.config["mlflow_experiment_name"])

    def pull_data_with_dvc(self):
        logging.info("Pulling data from DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True)
            logging.info("Data pulled successfully.")
        except Exception as e:
            logging.error(f"Error pulling data from DVC: {e}")
            raise

    def load_data(self):
        """Step 1: Load the processed test dataset."""
        try:
            self.df = pd.read_csv(self.test_data_path)
            logging.info(f"Step 1: Test data loaded successfully from {self.test_data_path}.")
        except Exception as e:
            logging.error(f"Error loading test data: {e}")
            raise

    def load_models(self):
        """Step 2: Load all trained models from the models directory."""
        try:
            if not os.path.exists(self.models_dir):
                logging.error(f"Models directory '{self.models_dir}' does not exist. Ensure models are trained first.")
                os.makedirs(self.models_dir, exist_ok=True)
                return

            model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
            if not model_files:
                logging.error("No models found in the directory. Train models before evaluating.")
                return

            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                self.models[model_file] = joblib.load(model_path)

            logging.info(f"Step 2: Loaded {len(self.models)} models from {self.models_dir}. Available models: {model_files}")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

    def evaluate_models(self):
        """Step 3: Evaluate each model on the test dataset."""
        if self.df is None:
            logging.error("Test data not loaded. Run load_data() first.")
            return
        if not self.models:
            logging.error("No models loaded. Run load_models() first.")
            return

        # Define features and target
        feature_columns = ["time_spent", "doors_in_route", "assessed_value"]  # Adjust if necessary
        target_column = "donation_bags_collected"

        if not all(col in self.df.columns for col in feature_columns + [target_column]):
            logging.error("Required feature columns are missing in the dataset.")
            return
        
        X_test = self.df[feature_columns]
        y_test = self.df[target_column]

        # Evaluate each model
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.metrics[model_name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R² Score": r2}

                # Log metrics to MLflow
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R² Score", r2)
        
        logging.info("Step 3: Model evaluation complete.")

    def display_results(self):
        """Step 4: Print model performance metrics and identify the best model."""
        if not self.metrics:
            logging.error("No evaluation metrics found. Run evaluate_models() first.")
            return

        # Convert metrics to DataFrame for better visualization
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.sort_values(by="R² Score", ascending=False)  # Sort by best R² Score

        print("\n=== Model Performance Metrics ===")
        print(metrics_df.to_string())  # Print formatted table

        # Identify the best model (highest R² Score)
        best_model_name = metrics_df.index[0]
        best_r2 = metrics_df.loc[best_model_name, 'R² Score']
        best_rmse = metrics_df.loc[best_model_name, 'RMSE']

        print(f"\nBest Model: {best_model_name} with R² Score = {best_r2:.4f}, RMSE = {best_rmse:.2f}")

        # Save the best model
        best_model_path = os.path.join(self.models_dir, "best_model.pkl")
        joblib.dump(self.models[best_model_name], best_model_path)
        logging.info(f"Best model '{best_model_name}' saved as 'best_model.pkl' in {self.models_dir}.")

    def evaluate_pipeline(self):
        """Runs the full evaluation pipeline step by step."""
        logging.info("Starting evaluation pipeline...")
        self.load_data()
        self.load_models()
        self.evaluate_models()
        self.display_results()
        logging.info("Evaluation pipeline complete.")

# Example Usage
if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_pipeline()
    # Start the MLflow server
    subprocess.run(["mlflow", "server", "--host", "127.0.0.1", "--port", "6002"])
