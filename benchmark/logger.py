import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "deepchem_benchmark", log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a global logger with console and file handlers.
    
    Parameters
    ----------
    name : str
        The name of the logger.
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"benchmark_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Create a default logger instance
logger = setup_logger()
