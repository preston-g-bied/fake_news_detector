# src/__init__.py

"""
Fake News Detection System
==========================

A multimodal machine learning system for detecting fake news by analyzing text,
images, and metadata.

This package contains all the source code for the project.
"""

import logging
import os
from pathlib import Path

# set up logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# package version
__version__ = "0.1.0"

# project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent