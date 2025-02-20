import os
import joblib
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureAnalysis:
    def __init__(self, data_path="data/processed/Food_Drive_2023_Processed.csv", model_path="models/XGBoost.pkl", reports_dir="reports/"):
        """
        Initializes the Feature Analysis class with data path, model path, and output directory.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.reports_dir = reports_dir
        self.df = None
        self.model = None

        # Create reports directory if not exists
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_data(self):
        """Load dataset and model."""
        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded from {self.data_path} with {self.df.shape[0]} rows.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
        
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def feature_importance(self):
        """Generate and save feature importance plot."""
        if not hasattr(self.model, "feature_importances_"):
            logging.error("Model does not support feature importance analysis.")
            return
        
        feature_names = ["time_spent", "doors_in_route", "assessed_value"]
        importance_values = self.model.feature_importances_

        feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance_values})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="Blues_r")
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        
        # Save plot
        save_path = os.path.join(self.reports_dir, "feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Feature importance plot saved at {save_path}")

        # Print DataFrame
        print("\n=== Feature Importance ===")
        print(feature_importance_df)

    def correlation_heatmap(self):
        """Generate and save correlation heatmap."""
        plt.figure(figsize=(10, 7))
        correlation_matrix = self.df[["time_spent", "doors_in_route", "assessed_value", "donation_bags_collected"]].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")

        # Save plot
        save_path = os.path.join(self.reports_dir, "correlation_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Correlation heatmap saved at {save_path}")

    def run_analysis(self):
        """Runs the full feature analysis pipeline."""
        logging.info("Starting feature analysis...")
        self.load_data()
        self.feature_importance()
        self.correlation_heatmap()
        logging.info("Feature analysis completed.")

# Example Usage
if __name__ == "__main__":
    analyzer = FeatureAnalysis()
    analyzer.run_analysis()
