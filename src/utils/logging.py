"""
Logging utilities for the GameBus-HealthBehaviorMining project.
"""
import logging
import os
import sys
from datetime import datetime

from config.settings import LOG_LEVEL, LOG_FORMAT
from config.paths import PROJECT_ROOT

def setup_logging(log_to_file: bool = True, log_level: str = None) -> logging.Logger:
    """
    Set up logging for the project.
    
    Args:
        log_to_file: Whether to log to a file
        log_level: Log level (defaults to settings.LOG_LEVEL)
        
    Returns:
        Logger instance
    """
    if log_level is None:
        log_level = LOG_LEVEL
        
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger("gamebus_health_mining")
    logger.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"gamebus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 