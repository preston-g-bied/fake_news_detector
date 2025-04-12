# scripts/data_collection/download_mediaeval_images.py

"""
Script to download images from the MediaEval Verifying Multimedia Use dataset.
This script extracts image URLs from set_images.txt and downloads them,
organizing them by veracity (fake/real).
"""

import os
import sys
import logging
import argparse
import concurrent.futures
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config import get_project_root, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{project_root}/logs/mediaeval_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_image_urls(file_path):
    """
    Load image URLs and metadata from set_images.txt
    
    Args:
        file_path (Path): Path to set_images.txt
        
    Returns:
        pd.DataFrame: DataFrame with image_id, image_url, annotation, and event
    """
    try:
        # read file as text first to handle inconsistent lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # parse lines manually
        data = []
        for i, line in enumerate(lines, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 4: # ensure we have at least 4 parts
                # use the first 4 parts only
                image_id = parts[0]
                image_url = parts[1]
                annotation = parts[2]
                event = parts[3]
                data.append([image_id, image_url, annotation, event])
            else:
                logger.warning(f"Line {i} has fewer than 4 fields: {line.strip()}")

        # create DataFrame
        df = pd.DataFrame(data, columns=['image_id', 'image_url', 'annotation', 'event'])
        logger.info(f"Loaded {len(df)} image URLs from {file_path}")

        return df
    except Exception as e:
        logger.error(f"Error loading image URLs from {file_path}: {e}")
        sys.exit(1)

def download_image(row, output_dir):
    """
    Download an image from URL and save it to the appropriate directory
    
    Args:
        row (pd.Series): Row containing image_id, image_url, and annotation
        output_dir (Path): Base directory to save images
        
    Returns:
        tuple: (image_id, success_flag, error_message)
    """
    image_id = row['image_id']
    image_url = row['image_url']
    # convert annotation to lowercase and standardize
    annotation = row['annotation'].lower()
    if annotation == 'fake':
        subdir = 'fake'
    elif annotation == 'real':
        subdir = 'real'
    else:
        subdir = 'unknown'

    # create output directory if it doesn't exist
    save_dir = output_dir / subdir
    os.makedirs(save_dir, exist_ok=True)

    # determine file extension from URL
    file_ext = os.path.splitext(image_url)[1]
    if not file_ext:
        file_ext = '.jpg'   # default to jpg if no extension

    # full path to save the image
    save_path = save_dir / f"{image_id}{file_ext}"

    # skip if the file already exists
    if save_path.exists():
        return image_id, True, "Already exists"
    
    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()

        # save the image
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return image_id, True, "Success"
    except requests.exceptions.RequestException as e:
        error_msg = f"Error downloading {image_url}: {e}"
        return image_id, False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error for {image_url}: {e}"
        return image_id, False, error_msg
    
def download_images_parallel(df, output_dir, max_workers=10):
    """
    Download images in parallel using ThreadPoolExecutor
    
    Args:
        df (pd.DataFrame): DataFrame with image_id, image_url, and annotation
        output_dir (Path): Base directory to save images
        max_workers (int): Maximum number of parallel downloads
        
    Returns:
        dict: Summary statistics of download results
    """
    results = {'success': 0, 'failed': 0, 'already_exists': 0}
    failed_ids = []

    logger.info(f"Downloading {len(df)} images to {output_dir} using {max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit download tasks
        future_to_id = {
            executor.submit(download_image, row, output_dir): row['image_id']
            for _, row in df.iterrows()
        }

        # process results as they comltete
        for future in tqdm(concurrent.futures.as_completed(future_to_id),
                           total=len(future_to_id),
                           desc="Downloading images"):
            image_id = future_to_id[future]
            try:
                _, success, message = future.result()
                if success:
                    if "Already exists" in message:
                        results['already_exists'] += 1
                    else:
                        results['success'] += 1
                else:
                    results['failed'] += 1
                    failed_ids.append(image_id)
                    logger.warning(f"Failed to download image {image_id}: {message}")
            except Exception as e:
                results['failed'] += 1
                failed_ids.append(image_id)
                logger.error(f"Error processing result for image {image_id}: {e}")

    # log failed image IDs
    if failed_ids:
        failed_log_path = output_dir / "failed_downloads.txt"
        with open(failed_log_path, 'w') as f:
            for image_id in failed_ids:
                f.write(f"{image_id}\n")
        logger.info(f"List of failed downloads saved to {failed_log_path}")

    return results

def create_metadata_file(df, output_dir):
    """
    Create a cleaned metadata CSV file
    
    Args:
        df (pd.DataFrame): DataFrame with image data
        output_dir (Path): Directory to save metadata
    """
    try:
        # create metadata directory if it doesn't exist
        metadata_dir = output_dir.parent / 'metadata'
        os.makedirs(metadata_dir, exist_ok=True)

        # save cleaned metadata
        metadata_path = metadata_dir / 'mediaeval_images.csv'
        df.to_csv(metadata_path, index=False)

        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Download MediaEval images')
    parser.add_argument('--input', type=str,
                        help='Path to set_images.txt file')
    parser.add_argument('--output', type=str,
                        help='Directory to save downloaded images')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Maximum number of parallel downloads')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of images to download (0 for all)')
    return parser.parse_args()

def main():
    args = parse_args()

    # load config
    config = load_config()

    # set input/output paths
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = (get_project_root() / config['paths']['data']['raw'] /
                      'images/mediaeval/raw/set_images.txt')
        
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = (get_project_root() / config['paths']['data']['raw'] /
                      'images/mediaeval/images')
        
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # load image URLs
    df = load_image_urls(input_path)

    # apply limit if specified
    if args.limit > 0:
        logger.info(f"Limiting downloads to {args.limit} images")
        df = df.head(args.limit)

    # download images
    results = download_images_parallel(df, output_dir, args.max_workers)

    # create metadata file
    create_metadata_file(df, output_dir)

    # print summary
    logger.info("\n===== Download Summary =====")
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Successfully downloaded: {results['success']}")
    logger.info(f"Already existed: {results['already_exists']}")
    logger.info(f"Failed: {results['failed']}")
    
    if results['failed'] == 0:
        logger.info("\nAll images processed successfully!")
    else:
        logger.warning(f"\n{results['failed']} images failed to download.")
        logger.info("Check the log file for details and the failed_downloads.txt file.")

if __name__ == "__main__":
    main()