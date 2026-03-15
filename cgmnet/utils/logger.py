# cgmnet/utils/logger.py
"""
Configures a logger for the project.

This module provides a setup function to create a standardized logger that writes
to both the console and a file.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = 'cgmnet', log_dir: str = 'logs/') -> logging.Logger:
    """
    Sets up a logger that outputs to both a file and the console.

    The log file will be named with a timestamp in the specified directory. This
    function is idempotent; calling it multiple times with the same name will
    not add duplicate handlers.

    Example:
        >>> logger = setup_logger(name='my_experiment', log_dir='exp_logs/')
        >>> logger.info("This is an info message.")

    Args:
        name (str): The name of the logger.
        log_dir (str): The directory where the log file will be saved.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get the logger instance
    logger = logging.getLogger(name)

    # Return the existing logger if it's already been configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # --- Formatter ---
    # Defines the format for all log messages.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Console Handler ---
    # Writes logs to the standard output (your terminal).
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # --- File Handler ---
    # Ensures the log directory exists.
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Creates a unique log file with a timestamp.
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = Path(log_dir) / f'{name}_{timestamp}.log'
    
    # Writes logs to the specified file.
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
