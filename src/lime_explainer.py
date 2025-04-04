import pandas as pd
import lime
import lime.lime_tabular
import logging

logger = logging.getLogger("lime_explainer")

# Load training data for LIME
def load_training_data(train_data_path, feature_columns):
    """Load training data for LIME."""
    try:
        training_data = pd.read_csv(train_data_path)
        X_train = training_data[feature_columns].values  # Extract feature values
        return X_train
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

# Initialize LIME explainer
def initialize_lime_explainer(X_train, feature_columns, class_names):
    """Initialize the LIME explainer."""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_columns,
            class_names=class_names,
            mode='regression'
        )
        return explainer
    except Exception as e:
        logger.error(f"Failed to initialize LIME explainer: {e}")
        raise