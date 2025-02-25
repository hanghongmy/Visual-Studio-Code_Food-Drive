import pandas as pd
import os
import logging
import subprocess
import mlflow
import argparse
import yaml

class Preprocessor:
    def __init__(self, raw_data_path, external_data_path, processed_data_path):
        """
        Initializes the Preprocessor with paths to raw and processed data.

        :param raw_data_path: Path to the raw data file.
        :param processed_data_path: Path to save the processed data.
        """
        self.raw_data_path = raw_data_path
        self.external_data_path = external_data_path
        self.processed_data_path = processed_data_path
        self.df = None
        self.external_df = None

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load_data(self):
        """Step 1: Load raw data and external data"""
        with mlflow.start_run(run_name="Load Data"):
            try:
                if not os.path.exists(self.raw_data_path):
                    logging.error(f"Cannot find raw data file at {self.raw_data_path}")
                    return
                self.df = pd.read_csv(self.raw_data_path)
                self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
                logging.info(f"Loaded raw data ({self.df.shape[0]} rows).")
                mlflow.log_param("Raw Data Path", self.df.shape[0])
                
                if os.path.exists(self.external_data_path):
                    self.external_df = pd.read_csv(self.external_data_path)
                    self.external_df.columns = self.external_df.columns.str.strip().str.lower().str.replace(' ', '_')
                    logging.info(f"Loaded external data from {self.external_df.shape[0]} rows.")
                    mlflow.log_param("External Data Path", self.external_df.shape[0])
                else:
                    logging.error(f"Cannot find external data file at {self.external_data_path}")
                    self.external_df = None
            except Exception as e:
                logging.error(f"Error in loading data: {e}")
                raise
            

    def clean_data(self):
        """Step 2: Clean the raw data."""
        with mlflow.start_run(run_name="Clean Data"):
            if self.df is None:
                logging.error("Data not loaded. Cannot clean data.")
        # Remove duplicate rows
        self.df.drop_duplicates(inplace=True)
        logging.info("Step 2.1: Duplicates removed.")

        # Rename important columns for consistency
        rename_columns = {
            '#_of_adult_volunteers_who_participated_in_this_route': 'number_of_adult',
            '#_of_youth_volunteers_who_participated_in_this_route': 'number_of_youth',
            'time_spent_collecting_donations': 'time_spent',
            '#_of_doors_in_route': 'doors_in_route',
            '#_of_donation_bags_collected': 'donation_bags_collected'
        }
        self.df.rename(columns=rename_columns, inplace=True)
        
        # Drop unnecessary columns
        columns_to_remove = [
            'id', 'start_time', 'completion_time', 'email', 'name', 'how_did_you_receive_the_form?', 'email_addresses',
             'other_drop-off_locations','how_many_routes_did_you_complete?', 'additional_routes_completed_(2_routes)', 
             'route_number/name','additional_routes_completed_(3_routes)', 'additional_routes_completed_(3_routes)2', 
             'additional_routes_completed_(more_than_3_routes)', 'additional_routes_completed_(more_than_3_routes)2', 
             'additional_routes_completed_(more_than_3_routes)3', 'comments_or_feedback'
        ]
        self.df.drop(columns=[col for col in columns_to_remove if col in self.df.columns], inplace=True)
        logging.info("Step: 2.2: Uncessary columns removed.")
        

        # Convert categorical time spent to numeric
        time_mapping = {
            '0 - 30 Minutes': 15,
            '30 - 60 Minutes': 45,
            '1 Hour - 1.5 Hours': 90,
            '2+ Hours': 150
        }
        
        if 'time_spent' in self.df.columns:
            self.df['time_spent'] = self.df['time_spent'].replace(time_mapping)
            pd.set_option('future.no_silent_downcasting', True)  # Opt-in to future behavior
            self.df['time_spent'] = pd.to_numeric(self.df['time_spent'], errors='coerce')
            self.df.fillna({'time_spent': self.df['time_spent'].median()}, inplace=True)
        logging.info("Step 2.3: 'Time Spent' column converted and missing values handled.")
    
        # Handle other missing values
        self.df['doors_in_route'] = pd.to_numeric(self.df['doors_in_route'], errors='coerce').fillna(self.df['doors_in_route'].median())
        self.df['donation_bags_collected'] = pd.to_numeric(self.df['donation_bags_collected'], errors='coerce').fillna(0)
        logging.info("Step 2.4: Other missing values handled.")
        
        # Extract neighbourhood from Ward/Stake and standardize
        if 'ward/stake' in self.df.columns:
            self.df['neighbourhood'] = self.df['ward/stake'].apply(lambda x: x.split(' Ward')[0] if pd.notna(x) else x)
            self.df['neighbourhood'] = self.df['neighbourhood'].str.upper()
            logging.info("Step 3.3: Neighbourhood extracted and standardized.")

        
    def merge_data(self):
        """Step 3: Merge the raw data with external data."""
        with mlflow.start_run(run_name="Merge Data"):
            if self.df is None:
                logging.error("Data not loaded. Cannot merge data.")
                return
            if self.external_df is not None:
                self.df = self.df.merge(self.external_df, on='neighbourhood', how='left')
                logging.info("Step 3: External data merged.")
            else:
                logging.warning("Skipping merge step as external data is missing.")
     
    def save_processed_data(self):
        """Step 4: Save processed data and track with DVC."""
        with mlflow.start_run(run_name="Save Processed Data"):
            if self.df is None:
                logging.error("No processed data available. Run preprocessing steps first.")
                return

            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            self.df.to_csv(self.processed_data_path, index=False)
            logging.info(f"Processed data saved at {self.processed_data_path}.")
            mlflow.log_param("processed_data_rows", self.df.shape[0])

            # Track processed data with DVC
            try:
                subprocess.run(["dvc", "add", self.processed_data_path], check=True)
                subprocess.run(["git", "add", "."], check=True)
                subprocess.run(["git", "commit", "-m", "Updated processed dataset"], check=True)
                subprocess.run(["dvc", "push"], check=True)
                logging.info("Processed data tracked and pushed with DVC.")
            except Exception as e:
                logging.error(f"DVC tracking failed: {e}")

    def preprocess(self):
        """Runs the full preprocessing pipeline step by step."""
        logging.info("Starting preprocessing pipeline...")
        self.load_data()
        self.clean_data()
        self.merge_data()
        self.save_processed_data()
        logging.info("Preprocessing pipeline complete.")

class Preprocessor_2023:
    def __init__(self,raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.df = None

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    def load_data(self):
        try:
            self.df = pd.read_csv(self.raw_data_path,encoding='latin1')
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            logging.info(f"Data loaded from {self.raw_data_path}.")
        except Exception as e:
            logging.error(f"Error in loading data: {e}")
            raise
    def rename_columns(self):
        rename_map = {
            'date': 'collection_date',
            'location': 'drop_off_location',
            'stake': 'stake',
            'ward/branch': 'ward/stake',
            '#_of_adult_volunteers': 'number_of_adult',
            '#_of_youth_volunteers': 'number_of_youth',
            'donation_bags_collected': 'donation_bags_collected',
            'time_to_complete_(min)': 'time_spent',
            'routes_completed': 'routes_completed',
            'doors_in_route': 'doors_in_route',
            'neighbourhood': 'neighbourhood',
            'Ward": "ward",'
            'assessed_value': 'assessed_value'
        }
        self.df.rename(columns=rename_map, inplace=True)
        logging.info("Columns renamed for 2023 dataset.")
    def save_processed_data(self):
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        self.df.to_csv(self.processed_data_path, index=False)
        logging.info(f"Processed data saved at {self.processed_data_path}.")
    
    def drop_columns(self):
        columns_to_remove = ['ward']
        self.df.drop(columns=[col for col in columns_to_remove if col in self.df.columns], inplace=True)
        logging.info("Uncessary columns removed for 2023 dataset.") 
        
    def preprocess(self):
        self.load_data()
        self.rename_columns()
        self.save_processed_data()
        self.drop_columns()
        logging.info("Preprocessing pipeline for 2023 dataset complete.")
       
                         

if __name__ == "__main__":
    logging.info("Starting preprocessing for both 2023 and 2024 datasets...")

    # Process 2023 dataset (cleaning only)
    preprocessor_2023 = Preprocessor_2023(
        raw_data_path="data/external/Food_Drive_2023.csv",
        processed_data_path="data/processed/Food_Drive_2023_Processed.csv"
    )
    preprocessor_2023.preprocess()

    # Process 2024 dataset (cleaning + merging with external data)
    preprocessor_2024 = Preprocessor(
        raw_data_path="data/raw/Food_Drive_Data_Collection_2024_original.csv",
        external_data_path="data/external/Property_Assessment_Data__Current_Calendar_Year__20240925.csv",
        processed_data_path="data/processed/Food_Drive_2024_Processed.csv"
    )
    preprocessor_2024.preprocess()

    logging.info("Preprocessing for both 2023 and 2024 datasets completed successfully.")