"""
Centralized logging configuration for AutoBool project.

This module provides a consistent logging setup across all components of the AutoBool project,
including training, dataset preparation, and API services.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to the levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(
    name: str,
    level: Union[str, int] = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both console and file output.
    
    Args:
        name: Logger name (typically __name__ or module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (defaults to logs/)
        console_output: Whether to output to console
        file_output: Whether to output to file
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for console
        colored_formatter = ColoredFormatter(
            f"%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        # Create logs directory
        if log_dir is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        # Use plain formatter for file (no colors)
        file_formatter = logging.Formatter(
            format_string,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get or create a logger with default AutoBool configuration.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments passed to setup_logger
    
    Returns:
        Configured logger instance
    """
    # Set default log level from environment
    default_level = os.getenv("AUTOBOOL_LOG_LEVEL", "INFO")
    kwargs.setdefault("level", default_level)
    
    return setup_logger(name, **kwargs)


def setup_training_logger(
    experiment_name: str,
    output_dir: Union[str, Path],
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logger specifically for training experiments.
    
    Args:
        experiment_name: Name of the training experiment
        output_dir: Training output directory
        level: Logging level
    
    Returns:
        Configured training logger
    """
    # Create logs subdirectory in output directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger_name = f"autobool.training.{experiment_name}"
    
    return setup_logger(
        name=logger_name,
        level=level,
        log_dir=log_dir,
        console_output=True,
        file_output=True
    )


def setup_api_logger(service_name: str = "entrez_api", level: str = "INFO") -> logging.Logger:
    """
    Set up logger for API services.
    
    Args:
        service_name: Name of the API service
        level: Logging level
    
    Returns:
        Configured API logger
    """
    logger_name = f"autobool.api.{service_name}"
    
    return setup_logger(
        name=logger_name,
        level=level,
        console_output=True,
        file_output=True
    )


def setup_dataset_logger(step_name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up logger for dataset preparation steps.
    
    Args:
        step_name: Name of the dataset preparation step
        level: Logging level
    
    Returns:
        Configured dataset logger
    """
    logger_name = f"autobool.dataset.{step_name}"
    
    return setup_logger(
        name=logger_name,
        level=level,
        console_output=True,
        file_output=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    logger = get_logger("test_logger")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print(f"Log file created in: logs/")