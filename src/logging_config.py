import logging
import os
from logging.handlers import RotatingFileHandler



def configure_logging(log_directory='logs'):
    # Create logs directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create module-specific loggers
    modules = ['train', 'predict', 'api']
    loggers = {}
    
    for module in modules:
        logger = logging.getLogger(f'ml_app.{module}')
        
        # Add a rotating file handler for each module
        file_handler = RotatingFileHandler(
            f'{log_directory}/{module}.log',
            maxBytes=10485760,  # 10MB
            backupCount=5  # Keep up to 5 backup files
        )
        
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        loggers[module] = logger
    
    return loggers