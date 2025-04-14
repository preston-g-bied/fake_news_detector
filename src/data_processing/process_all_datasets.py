# src/data/process_all_datasets.py

"""
Main script to process all datasets for the Fake News Detection project.
This script orchestrates the processing of text, image, and metadata datasets.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config import load_config, get_project_root
from src.data_processing.text_processor import LiarProcessor, PhemeProcessor, FakeNewsNetProcessor
from src.data_processing.image_processor import MediaevalProcessor, FakeNewsNetImageProcessor
from src.data_processing.metadata_processor import process_metadata, create_integrated_dataset

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{project_root}/logs/data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_text_datasets(config):
    """
    Process all text datasets.
    
    Args:
        config (dict): Configuration dictionary
    """
    logger.info("Processing text datasets...")

    # get paths from config
    raw_data_dir = project_root / config['paths']['data']['raw']
    processed_data_dir = project_root / config['paths']['data']['processed']

    # process LIAR dataset
    logger.info("Processing LIAR dataset...")
    liar_processor = LiarProcessor(config)
    liar_input_dir = raw_data_dir / "text/liar"
    liar_output_dir = processed_data_dir / "text/liar"
    liar_processor.process(liar_input_dir, liar_output_dir)

    # process PHEME dataset
    logger.info("Processing PHEME dataset...")
    pheme_processor = PhemeProcessor(config)
    pheme_input_dir = raw_data_dir / "text/pheme"
    pheme_output_dir = processed_data_dir / "text/pheme"
    pheme_processor.process(pheme_input_dir, pheme_output_dir)

    # process FakeNewsNet dataset
    logger.info("Processing FakeNewsNet dataset...")
    fakenewsnet_processor = FakeNewsNetProcessor(config)
    fakenewsnet_input_dir = raw_data_dir / "text/fakenewsnet"
    fakenewsnet_output_dir = processed_data_dir / "text/fakenewsnet"
    fakenewsnet_processor.process(fakenewsnet_input_dir, fakenewsnet_output_dir)

    logger.info("All text datasets processed successfully!")

def process_image_datasets(config):
    """
    Process all image datasets.
    
    Args:
        config (dict): Configuration dictionary
    """
    logger.info("Processing image datasets...")

    # get paths from config
    raw_data_dir = project_root / config['paths']['data']['raw']
    processed_data_dir = project_root / config['paths']['data']['processed']

    # process MediaEval dataset
    logger.info("Processing MediaEval dataset...")
    mediaeval_processor = MediaevalProcessor(config)
    mediaeval_input_dir = raw_data_dir / "images/mediaeval"
    mediaeval_output_dir = processed_data_dir / "images/mediaeval"
    mediaeval_processor.process(mediaeval_input_dir, mediaeval_output_dir)

    # process FakeNewsNet images
    logger.info("Processing FakeNewsNet images...")
    fakenewsnet_img_processor = FakeNewsNetImageProcessor(config)
    fakenewsnet_img_input_dir = raw_data_dir / "images/fakenewsnet"
    fakenewsnet_img_output_dir = processed_data_dir / "images/fakenewsnet"
    fakenewsnet_img_processor.process(fakenewsnet_img_input_dir, fakenewsnet_img_output_dir)

    logger.info("All image datasets processed successfully!")

def process_all_metadata(config):
    """
    Process and integrate metadata from all datasets.
    
    Args:
        config (dict): Configuration dictionary
    """
    logger.info("Processing metadata...")

    # get paths from config
    processed_data_dir = project_root / config['paths']['data']['processed']
    metadata_dir = processed_data_dir / "metadata"

    # ensure metadata directory exists
    os.makedirs(metadata_dir, exist_ok=True)

    # process metadata
    process_metadata(processed_data_dir, metadata_dir)

    # create integrated dataset
    create_integrated_dataset(processed_data_dir, metadata_dir)

    logger.info("Metadata processing completed!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process all datasets for Fake News Detection project')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--datasets', nargs='+',
                        choices=['text', 'images', 'metadata', 'all'],
                        default=['all'],
                        help='Specify which types of datasets to process')
    return parser.parse_args()

def main():
    """Main function to process all datasets."""
    args = parse_args()

    # load configuration
    if args.config.endswith('.yaml'):
        config = load_config(args.config)
    else:
        config = load_config()

    logger.info("Starting data processing pipeline...")

    # create necessary directories
    processed_data_dir = project_root / config['paths']['data']['processed']
    for data_type in ['text', 'images', 'metadata']:
        os.makedirs(processed_data_dir / data_type, exist_ok=True)

    # process datasets based on arguments
    datasets_to_process = args.datasets
    process_all = 'all' in datasets_to_process

    if process_all or 'text' in datasets_to_process:
        process_text_datasets(config)

    if process_all or 'images' in datasets_to_process:
        process_image_datasets(config)

    if process_all or 'metadata' in datasets_to_process:
        process_all_metadata(config)

    logger.info("Data processing pipeline completed successfully!")

if __name__ == "__main__":
    main()