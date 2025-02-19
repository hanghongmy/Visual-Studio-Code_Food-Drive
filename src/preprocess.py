import pandas as pd
import os
import logging

class Preprocessor:
    def __init__(self, raw_data_path: str = 'ml_project/data/raw/Food Drive Data Collection 2024_original.csv',
                 external_data_path: str = 'ml_project/data/external/Property_Assessment_Data__Current_Calendar_Year__20240925.csv',
                 processed_data_path: str = 'ml_project/data/processed/Food_Drive_Processed.csv'):
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
        """Step 1: Load raw data from a CSV file and clean column names."""
        try:
            logging.info(f"Looking for file at: {os.path.abspath(self.raw_data_path)}")
            self.df = pd.read_csv(self.raw_data_path, encoding='utf-8')
            logging.info(f"Step 1: Data loaded successfully from {self.raw_data_path}.")
            
            # Standardize column names
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            logging.info(f"Standardized column names: {self.df.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error in loading data: {e}")
            raise

    def load_external_data(self):
        """Step 2: Load external data with assessed value, latitude, longitude, and neighbourhood."""
        try:
            logging.info(f"Looking for external file at: {os.path.abspath(self.external_data_path)}")
            
            # Load with correct column names
            self.external_df = pd.read_csv(self.external_data_path, usecols=['Assessed Value', 'Latitude', 'Longitude', 'Neighbourhood'], encoding='utf-8')
            
            # Standardize column names
            self.external_df.columns = self.external_df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Aggregate external data by neighbourhood to avoid duplication
            self.external_df = self.external_df.groupby('neighbourhood', as_index=False).agg({
                'assessed_value': 'median',  # Use median to avoid extreme values
                'latitude': 'mean',  # Average latitude for merging
                'longitude': 'mean'  # Average longitude for merging
            })
            
            logging.info(f"Step 2: External data loaded and aggregated successfully from {self.external_data_path}.")
            logging.info(f"Columns in external dataset after aggregation: {self.external_df.columns.tolist()}")
        except Exception as e:
            logging.error(f"Error in loading external data: {e}")
            raise

    def clean_data(self):
        """Step 3: Clean the dataset by handling missing values, removing duplicates, renaming columns, and dropping unnecessary columns."""
        if self.df is None:
            logging.error("Data not loaded. Run load_data() first.")
            return

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
            self.df['time_spent'] = pd.to_numeric(self.df['time_spent'], errors='coerce')
            self.df['time_spent'].fillna(self.df['time_spent'].median(), inplace=True)
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
        if self.df is None or self.external_df is None:
            logging.error("Data not loaded. Run load_data() and load_external_data() first.")
            return
        
        # Merge datasets based on neighbourhood
        self.df = pd.merge(self.df, self.external_df, on='neighbourhood', how='left')
        
        # Handle missing assessed values and coordinates
        missing_count_before = self.df['assessed_value'].isna().sum()
        self.df[['assessed_value', 'latitude', 'longitude']] = self.df.groupby('stake')[['assessed_value', 'latitude', 'longitude']].transform(lambda x: x.fillna(x.mean()))
        self.df[['assessed_value', 'latitude', 'longitude']] = self.df.groupby('drop_off_location')[['assessed_value', 'latitude', 'longitude']].transform(lambda x: x.fillna(x.mean()))
        missing_count_after = self.df['assessed_value'].isna().sum()
        logging.info(f"Step 4: External data merged. Missing assessed values before: {missing_count_before}, after: {missing_count_after}")
        
    def save_processed_data(self):
        """Step 5: Save the cleaned and processed dataset."""
        if self.df is None:
            logging.error("No data to save. Run preprocessing steps first.")
            return
        
        try:
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            self.df.to_csv(self.processed_data_path, index=False, encoding='utf-8')
            logging.info(f"Step 5: Processed data saved successfully at {self.processed_data_path}.")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")

    def preprocess(self):
        """Runs the full preprocessing pipeline step by step."""
        logging.info("Starting preprocessing pipeline...")
        self.load_data()
        self.load_external_data()
        self.clean_data()
        self.merge_data()
        self.save_processed_data()
        logging.info("Preprocessing pipeline complete.")
    
class Preprocessor_2023:
    def __init__(self, raw_data_path, processed_data_path):
        """
        Preprocessor for Food Drive 2023 dataset.
        This class **only cleans the dataset** (no merging).
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.df = None

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load_data(self):
        """Step 1: Load raw Food Drive 2023 data."""
        try:
            self.df = pd.read_csv(self.raw_data_path, encoding='utf-8')
            self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
            logging.info(f"Step 1: Data loaded from {self.raw_data_path}.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def rename_columns(self):
        """Step 2: Rename columns for consistency."""
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
            'assessed_value': 'assessed_value'
        }
        self.df.rename(columns=rename_map, inplace=True)
        logging.info("Step 2: Columns renamed for 2023 dataset.")

    def save_processed_data(self):
        """Step 3: Save processed 2023 dataset."""
        self.df.to_csv(self.processed_data_path, index=False, encoding='utf-8')
        logging.info(f"Step 3: Processed data saved at {self.processed_data_path}.")

    def preprocess(self):
        """Run preprocessing pipeline."""
        self.load_data()
        self.rename_columns()
        self.save_processed_data()
        logging.info("Preprocessing for 2023 complete.")


# **Run Both Preprocessors**
if __name__ == "__main__":
    preprocessor_2023 = Preprocessor_2023(
        raw_data_path="ml_project/data/external/Food_Drive_2023.csv",
        processed_data_path="ml_project/data/processed/Food_Drive_2023_Processed.csv"
    )
    preprocessor_2023.preprocess()

    preprocessor_2024 = Preprocessor(
        raw_data_path="ml_project/data/raw/Food Drive Data Collection 2024_original.csv",
        external_data_path="ml_project/data/external/Property_Assessment_Data__Current_Calendar_Year__20240925.csv",
        processed_data_path="ml_project/data/processed/Food_Drive_2024_Processed.csv"
    )
    preprocessor_2024.preprocess()