import os
import pandas as pd
import logging
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

class Trainer:
    def __init__(self, train_data_path: str, test_data_path: str, models_dir: str = "models/"):
        """
        Initializes the Trainer class with paths to processed data and model saving.

        :param train_data_path: Path to the training dataset (2023).
        :param test_data_path: Path to the testing dataset (2024).
        :param models_dir: Directory where all trained models will be saved.
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.models_dir = models_dir
        self.train_df = None
        self.test_df = None
        self.models = {
            "Linear_Regression": LinearRegression(),
            "Decision_Tree": DecisionTreeRegressor(),
            "Random_Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor()
        }
        self.param_grids = {
            "Decision_Tree": {"max_depth": [5, 10, 20, None]},
            "Random_Forest": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
            "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
        }
        self.results_regular = []  # Store results for default models
        self.results_tuned = []  # Store results for tuned models

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
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
    
    def prepare_data(self):
        """Step 2: Prepare features and target variable for model training."""
        if self.train_df is None or self.test_df is None:
            logging.error(" Data not loaded. Run load_data() first.")
            return
        
        # Define features and target
        feature_columns = ['time_spent', 'doors_in_route', 'assessed_value']
        target_column = 'donation_bags_collected'

        # Ensure all required columns exist
        for df in [self.train_df, self.test_df]:
            missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
            if missing_cols:
                logging.error(f" Missing columns in dataset: {missing_cols}")
                return

        # Train/Test split
        self.X_train = self.train_df[feature_columns]
        self.y_train = self.train_df[target_column]
        self.X_test = self.test_df[feature_columns]
        self.y_test = self.test_df[target_column]
        
        logging.info("Step 2: Data prepared for training.")

    def train_model(self, model_name, model, tuned=False):
        """Trains a model and evaluates it."""
        # Fit model
        model.fit(self.X_train, self.y_train)

        # Evaluate model
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

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
        self.prepare_data()
        self.train_models()
        self.print_results()
        logging.info("Training pipeline complete.")

# Example Usage:
if __name__ == "__main__":
    trainer = Trainer(
        train_data_path="data/processed/Food_Drive_2023_Processed.csv",  # Training Data (Food Drive 2023)
        test_data_path="data/processed/Food_Drive_2024_Processed.csv",   # Testing Data (Food Drive 2024)
        models_dir="models/"                                            # Save models to this directory
    )
    trainer.train_pipeline()
