import os
import joblib
import pandas as pd
import logging
import yaml
import mlflow
import mlflow.sklearn
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Predictor:
    def __init__(self, config_path="configs/predict_config.yaml"):
        """
        Initializes the Predictor class.

        :param model_dir: Directory where trained models are stored.
        :param test_data_path: Path to the processed test dataset.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.model_dir = self.config["models_dir"]
        self.test_data_path = os.path.join("data", "processed", self.config["test_data_path"])
        self.output_path = self.config["predictions_output"]       
        self.mlflow_experiment_name = self.config["mlflow_experiment_name"]
        self.model = None
        self.df = None          
        # Set up mlflow experiment
        mlflow.set_experiment(self.mlflow_experiment_name)

    def load_best_model(self):
        """Step 1: Load the best trained model from the models directory."""
        try:
            best_model_path = os.path.join(self.model_dir, "best_model.pkl")
            if not os.path.exists(best_model_path):
                logging.error(f"Best model not found in {self.model_dir}. Run evaluation first.")
                return

            self.model = joblib.load(best_model_path)
            logging.info(f"Best model loaded from {best_model_path}.")
        except Exception as e:
            logging.error(f"Error loading best model: {e}")
            raise

    def load_test_data(self):
        """Step 2: Load the test dataset for prediction."""
        try:
            self.df = pd.read_csv(self.test_data_path)
            logging.info(f"Step 2: Test data loaded from {self.test_data_path}. ({self.df.shape[0]} rows)")
        except Exception as e:
            logging.error(f"Error loading test data: {e}")
            raise

    def make_predictions(self):
        """Step 3: Use the best model to make predictions on new data as integers."""
        if self.model is None:
            logging.error("No model loaded. Run load_best_model() first.")
            return
        if self.df is None:
            logging.error("No test data loaded. Run load_test_data() first.")
            return

        # Define feature columns (same as used in training)
        feature_columns = ["time_spent", "doors_in_route", "assessed_value"]
        if not all(col in self.df.columns for col in feature_columns):
            logging.error(f"Required feature columns are missing: {feature_columns}")
            return

        # Predict donation bags collected and round to the nearest integer
        X_test = self.df[feature_columns]
        self.df["predicted_donation_bags"] = self.model.predict(X_test).round().astype(int)

        logging.info("Step 3: Predictions made successfully (rounded to integers).")

    def save_predictions(self, output_path="data/predictions/Food_Drive_2024_Predictions.csv"):
        """Step 4: Save predictions to a CSV file."""
        if self.df is None:
            logging.error("No predictions available to save. Run make_predictions() first.")
            return

        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False, encoding='utf-8')
            logging.info(f"Step 4: Predictions saved to {output_path}.")
            
            # Log predictions to mlflow
            with mlflow.start_run():
                mlflow.log_artifact(self.output_path)
                logging.info("Predictions logged to mlflow.")
        except Exception as e:
            logging.error(f"Error saving predictions: {e}")
            
    def track_predictions_with_dvc(self):
        """Track predictions file with DVC."""
        try:
            subprocess.run(["dvc", "add", self.output_path], check=True)
            subprocess.run(["git", "add", self.output_path + ".dvc"], check=True)
            subprocess.run(["git", "commit", "-m", "Tracked predictions with DVC"], check=True)
            subprocess.run(["dvc", "push"], check=True)
            logging.info("Predictions successfully tracked and pushed to DVC.")
        except Exception as e:
            logging.error(f"DVC tracking failed: {e}")
            
    def predict_pipeline(self):
        """Runs the full prediction pipeline step by step."""
        logging.info("Starting prediction pipeline...")
        self.load_best_model()
        self.load_test_data()
        self.make_predictions()
        self.save_predictions()
        logging.info("Prediction pipeline complete.")


# Example Usage
if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict_pipeline()
    subprocess.run(["./venv/bin/mlflow", "server", "--host", "127.0.0.1", "--port", "5000"])
