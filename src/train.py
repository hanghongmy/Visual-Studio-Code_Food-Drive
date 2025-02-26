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

class Trainer:
    def __init__(self, config_path:str):
        """
        Initializes the Trainer class with paths to processed data and model saving.

        :param train_data_path: Path to the training dataset (2023).
        :param test_data_path: Path to the testing dataset (2024).
        :param models_dir: Directory where all trained models will be saved.
        """
        self.config_path = config_path
        
        #Load config file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.train_data_path = self.config["train_data_path"]
        self.test_data_path = self.config["test_data_path"]
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

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Start MLflow run
        mlflow.set_experiment(self.config["mlflow_experiment_name"])
    
    def load_data(self):
        """Step 1: Load the training and testing datasets."""
        try:
            self.train_df = pd.read_csv(self.train_data_path)
            self.test_df = pd.read_csv(self.test_data_path)
            logging.info(f" Step 1: Training data loaded from {self.train_data_path} ({self.train_df.shape[0]} rows).")
            logging.info(f" Step 1: Testing data loaded from {self.test_data_path} ({self.test_df.shape[0]} rows).")
        except Exception as e:
            logging.error(f" Error loading data: {e}")
            raise
    
    def handle_missing_value(self):
        with mlflow.start_run(run_name="Handle Missing Values"):
            for df in [self.train_df, self.test_df]:
                df.dropna(how='all', inplace=True)  # Drop fully empty rows
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill numeric NaNs
                categorical_cols = df.select_dtypes(include=['object']).columns
                df[categorical_cols] = df[categorical_cols].fillna("Unknown")  # Fill categorical NaNs

            logging.info("Step 2: Missing values handled - Numeric with median, Categorical with 'Unknown'.")

    
    def prepare_data(self):
        
        # Define features and target
        feature_columns = ['time_spent', 'doors_in_route', 'assessed_value']
        target_column = 'donation_bags_collected'

        # Train/Test split
        self.X_train = self.train_df[feature_columns]
        self.y_train = self.train_df[target_column]
        self.X_test = self.test_df[feature_columns]
        self.y_test = self.test_df[target_column]
        
        logging.info("Step 2: Data prepared for training.")

    def train_model(self, model_name, model, tuned=False):
        """Trains a model and evaluates it."""
        with mlflow.start_run(run_name=f"{model_name}_Tuned" if tuned else model_name):
            # Fit model
            model.fit(self.X_train, self.y_train)

            # Evaluate model
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            mlflow.log_param("tuned", tuned)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            result = {
                "Model": model_name + ("_Tuned" if tuned else ""),
                "MSE": mse,
                "R² Score": r2
            }
            
            # Save result based on whether it's tuned or regular
            if tuned:
                self.results_tuned.append(result)
            else:
                self.results_regular.append(result)

            logging.info(f" {model_name} - MSE: {mse:.2f}, R² Score: {r2:.4f}")

            # Save trained model
            model_save_path = os.path.join(self.models_dir, f"{result['Model']}.pkl")
            joblib.dump(model, model_save_path)
            logging.info(f"Model '{result['Model']}' saved to {model_save_path}.")

    def hypertune_model(self, model_name, model):
        """Performs hyperparameter tuning using GridSearchCV if parameters are defined."""
        if model_name not in self.param_grids:
            return model  # No hyperparameter tuning needed

        logging.info(f" Hyperparameter tuning for {model_name}...")
        param_grid = self.param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logging.info(f" Best hyperparameters for {model_name}: {best_params}")

        return grid_search.best_estimator_  # Return the best model with optimal hyperparameters

    def train_models(self):
        """Step 3: Train multiple models with and without hyperparameter tuning and save them."""
        os.makedirs(self.models_dir, exist_ok=True)  # Ensure models directory exists

        for model_name, model in self.models.items():
            logging.info(f" Training {model_name} without tuning...")
            self.train_model(model_name, model, tuned=False)  # Train regular model

            # Perform hyperparameter tuning
            if model_name in self.param_grids:
                logging.info(f"Hyperparameter tuning for {model_name}...")
                tuned_model = self.hypertune_model(model_name, model)
                self.train_model(model_name, tuned_model, tuned=True)  # Train tuned model

        logging.info(f"Step 3: All models trained and saved.")
    
    def track_data_with_dvc(self):
        """Step 5: Track processed data with DVC."""
        with mlflow.start_run(run_name="DVC Tracking"):
            try:
                subprocess.run(["dvc", "add", self.train_data_path], check=True)
                subprocess.run(["dvc", "add", self.test_data_path], check=True)
                subprocess.run(["git", "add", "."], check=True)
                subprocess.run(["git", "commit", "-m", "Updated processed dataset"], check=True)
                subprocess.run(["dvc", "push"], check=True)
                logging.info("Step 5: Processed data tracked and pushed with DVC.")
            except Exception as e:
                logging.error(f"DVC tracking failed: {e}")

    def print_results(self):
        """Step 4: Print model performance results before and after hyperparameter tuning."""
        regular_df = pd.DataFrame(self.results_regular).sort_values(by="R² Score", ascending=False)
        tuned_df = pd.DataFrame(self.results_tuned).sort_values(by="R² Score", ascending=False)

        print("\n=== Model Performance Summary (Before Tuning) ===")
        print(regular_df.to_string(index=False))

        print("\n=== Model Performance Summary (After Hyperparameter Tuning) ===")
        print(tuned_df.to_string(index=False))

    def train_pipeline(self):
        """Runs the full training pipeline step by step."""
        logging.info("Starting training pipeline...")
        self.load_data()
        self.handle_missing_value()
        self.prepare_data()
        self.train_models()
        self.track_data_with_dvc()
        self.print_results()
        logging.info("Training pipeline complete.")

# Example Usage:
if __name__ == "__main__":
    trainer = Trainer(config_path="configs/train_config.yaml")
    trainer.train_pipeline()  import os
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

class Trainer:
    def __init__(self, config_path:str):
        """
        Initializes the Trainer class with paths to processed data and model saving.

        :param train_data_path: Path to the training dataset (2023).
        :param test_data_path: Path to the testing dataset (2024).
        :param models_dir: Directory where all trained models will be saved.
        """
        self.config_path = config_path
        
        #Load config file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.train_data_path = self.config["train_data_path"]
        self.test_data_path = self.config["test_data_path"]
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

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Start MLflow run
        mlflow.set_experiment(self.config["mlflow_experiment_name"])
    
    def load_data(self):
        """Step 1: Load the training and testing datasets."""
        try:
            self.train_df = pd.read_csv(self.train_data_path)
            self.test_df = pd.read_csv(self.test_data_path)
            logging.info(f" Step 1: Training data loaded from {self.train_data_path} ({self.train_df.shape[0]} rows).")
            logging.info(f" Step 1: Testing data loaded from {self.test_data_path} ({self.test_df.shape[0]} rows).")
        except Exception as e:
            logging.error(f" Error loading data: {e}")
            raise
    
    def handle_missing_value(self):
        with mlflow.start_run(run_name="Handle Missing Values"):
            for df in [self.train_df, self.test_df]:
                df.dropna(how='all', inplace=True)  # Drop fully empty rows
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill numeric NaNs
                categorical_cols = df.select_dtypes(include=['object']).columns
                df[categorical_cols] = df[categorical_cols].fillna("Unknown")  # Fill categorical NaNs

            logging.info("Step 2: Missing values handled - Numeric with median, Categorical with 'Unknown'.")

    
    def prepare_data(self):
        
        # Define features and target
        feature_columns = ['time_spent', 'doors_in_route', 'assessed_value']
        target_column = 'donation_bags_collected'

        # Train/Test split
        self.X_train = self.train_df[feature_columns]
        self.y_train = self.train_df[target_column]
        self.X_test = self.test_df[feature_columns]
        self.y_test = self.test_df[target_column]
        
        logging.info("Step 2: Data prepared for training.")

    def train_model(self, model_name, model, tuned=False):
        """Trains a model and evaluates it."""
        with mlflow.start_run(run_name=f"{model_name}_Tuned" if tuned else model_name):
            # Fit model
            model.fit(self.X_train, self.y_train)

            # Evaluate model
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            mlflow.log_param("tuned", tuned)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            result = {
                "Model": model_name + ("_Tuned" if tuned else ""),
                "MSE": mse,
                "R² Score": r2
            }
            
            # Save result based on whether it's tuned or regular
            if tuned:
                self.results_tuned.append(result)
            else:
                self.results_regular.append(result)

            logging.info(f" {model_name} - MSE: {mse:.2f}, R² Score: {r2:.4f}")

            # Save trained model
            model_save_path = os.path.join(self.models_dir, f"{result['Model']}.pkl")
            joblib.dump(model, model_save_path)
            logging.info(f"Model '{result['Model']}' saved to {model_save_path}.")

    def hypertune_model(self, model_name, model):
        """Performs hyperparameter tuning using GridSearchCV if parameters are defined."""
        if model_name not in self.param_grids:
            return model  # No hyperparameter tuning needed

        logging.info(f" Hyperparameter tuning for {model_name}...")
        param_grid = self.param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logging.info(f" Best hyperparameters for {model_name}: {best_params}")

        return grid_search.best_estimator_  # Return the best model with optimal hyperparameters

    def train_models(self):
        """Step 3: Train multiple models with and without hyperparameter tuning and save them."""
        os.makedirs(self.models_dir, exist_ok=True)  # Ensure models directory exists

        for model_name, model in self.models.items():
            logging.info(f" Training {model_name} without tuning...")
            self.train_model(model_name, model, tuned=False)  # Train regular model

            # Perform hyperparameter tuning
            if model_name in self.param_grids:
                logging.info(f"Hyperparameter tuning for {model_name}...")
                tuned_model = self.hypertune_model(model_name, model)
                self.train_model(model_name, tuned_model, tuned=True)  # Train tuned model

        logging.info(f"Step 3: All models trained and saved.")
    
    def track_data_with_dvc(self):
        """Step 5: Track processed data with DVC."""
        with mlflow.start_run(run_name="DVC Tracking"):
            try:
                subprocess.run(["dvc", "add", self.train_data_path], check=True)
                subprocess.run(["dvc", "add", self.test_data_path], check=True)
                subprocess.run(["git", "add", "."], check=True)
                subprocess.run(["git", "commit", "-m", "Updated processed dataset"], check=True)
                subprocess.run(["dvc", "push"], check=True)
                logging.info("Step 5: Processed data tracked and pushed with DVC.")
            except Exception as e:
                logging.error(f"DVC tracking failed: {e}")

    def print_results(self):
        """Step 4: Print model performance results before and after hyperparameter tuning."""
        regular_df = pd.DataFrame(self.results_regular).sort_values(by="R² Score", ascending=False)
        tuned_df = pd.DataFrame(self.results_tuned).sort_values(by="R² Score", ascending=False)

        print("\n=== Model Performance Summary (Before Tuning) ===")
        print(regular_df.to_string(index=False))

        print("\n=== Model Performance Summary (After Hyperparameter Tuning) ===")
        print(tuned_df.to_string(index=False))

    def train_pipeline(self):
        """Runs the full training pipeline step by step."""
        logging.info("Starting training pipeline...")
        self.load_data()
        self.handle_missing_value()
        self.prepare_data()
        self.train_models()
        self.track_data_with_dvc()
        self.print_results()
        logging.info("Training pipeline complete.")

# Example Usage:
if __name__ == "__main__":
    trainer = Trainer(config_path="configs/train_config.yaml")
    trainer.train_pipeline()