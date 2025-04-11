# src/utils/config.py

"""
Configuration utilities for the Fake News Detection project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Path object representing the project root directory
    """
    return Path(__file__).resolve().parents[2]

def load_config(config_name: str = "config.yaml") -> Dict[str, Any]:
    """
    Load a configuration file from the config directory.
    
    Args:
        config_name (str): Name of the configuration file to load
        
    Returns:
        Dict[str, Any]: Configuration as a dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the configuration file is not valid YAML
    """
    config_path = get_project_root() / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise

def get_path(path_name: str, config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Get a path from the configuration.
    
    Args:
        path_name (str): Name of the path to get (e.g., "data.raw", "models.saved")
        config (Optional[Dict[str, Any]]): Configuration dictionary
            If None, the default configuration is loaded
            
    Returns:
        Path: Resolved path
        
    Raises:
        KeyError: If the path is not found in the configuration
    """
    if config is None:
        config = load_config()

    # split the path_name by dots
    parts = path_name.split(".")

    # navigate through the config dictionary
    current = config
    for part in parts[:-1]:
        if part not in current:
            raise KeyError(f"Path component '{part}' not found in configuration")
        current = current[part]

    if parts[-1] not in current:
        raise KeyError(f"Path component '{parts[-1]}' not found in configuration")
    
    # get the path string and resolve it relative to the project root
    path_str = current[parts[-1]]
    path = get_project_root() / path_str

    return path

def get_data_path(subpath: Optional[str] = None) -> Path:
    """
    Get a path in the data directory.
    
    Args:
        subpath (Optional[str]): Subpath within the data directory
            
    Returns:
        Path: Resolved path
    """
    data_path = get_path("paths.data.raw")

    if subpath:
        return data_path / subpath
    return data_path

def get_models_path(subpath: Optional[str] = None) -> Path:
    """
    Get a path in the models directory.
    
    Args:
        subpath (Optional[str]): Subpath within the models directory
            
    Returns:
        Path: Resolved path
    """
    models_path = get_path("paths.models.saved")

    if subpath:
        return models_path / subpath
    return models_path

def get_logs_path(subpath: Optional[str] = None) -> Path:
    """
    Get a path in the logs directory.
    
    Args:
        subpath (Optional[str]): Subpath within the logs directory
            
    Returns:
        Path: Resolved path
    """
    logs_path = get_path("paths.logs")

    if subpath:
        return logs_path / subpath
    return logs_path