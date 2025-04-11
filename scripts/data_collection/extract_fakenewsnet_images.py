# scripts/data_collection/extract_fakenewsnet_images.py

"""
Script to extract and download images from the FakeNewsNet dataset.
This script processes the downloaded news content.json files from FakeNewsNet
and extracts image URLs for downloading.
"""

import os
import sys
import json
import logging
import argparse
import concurrent.futures
from pathlib import Path
import requests
from tqdm import tqdm
import time

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config import get_project_root, get_data_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{project_root}/logs/fakenewsnet_images.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_news_content_files(base_dir):
    """
    Find all news content.json files in the FakeNewsNet directory structure.
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        List of tuples (file_path, source, label, news_id)
    """
    content_files = []
    
    # Expected structure: base_dir/source/label/news_id/news content.json
    for source in ['politifact', 'gossipcop']:
        source_dir = base_dir / source
        if not source_dir.exists():
            continue
            
        for label in ['fake', 'real']:
            label_dir = source_dir / label
            if not label_dir.exists():
                continue
                
            for news_dir in label_dir.iterdir():
                if not news_dir.is_dir():
                    continue
                    
                content_file = news_dir / "news content.json"
                if content_file.exists():
                    content_files.append((content_file, source, label, news_dir.name))
    
    logger.info(f"Found {len(content_files)} news content files")
    return content_files

def extract_image_urls(content_file, source, label, news_id):
    """
    Extract image URLs from a news content.json file.
    
    Args:
        content_file: Path to the news content.json file
        source: Source of the news (politifact/gossipcop)
        label: Label of the news (fake/real)
        news_id: ID of the news article
        
    Returns:
        List of dictionaries with image information
    """
    try:
        with open(content_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        if 'images' not in content or not content['images']:
            return []
        
        images = []
        for i, image_url in enumerate(content['images']):
            if not image_url or not isinstance(image_url, str):
                continue
                
            images.append({
                'image_url': image_url,
                'source': source,
                'label': label,
                'news_id': news_id,
                'image_id': f"{news_id}_{i}"
            })
        
        return images
    except Exception as e:
        logger.error(f"Error processing {content_file}: {str(e)}")
        return []

def download_image(image_info, output_dir, retry_count=3):
    """
    Download an image from a URL.
    
    Args:
        image_info: Dictionary containing image information
        output_dir: Directory to save the image
        retry_count: Number of retries if download fails
    
    Returns:
        Tuple of (success, image_id, source, label)
    """
    image_url = image_info['image_url']
    source = image_info['source']
    label = image_info['label']
    news_id = image_info['news_id']
    image_id = image_info['image_id']
    
    # Create directory structure
    image_dir = output_dir / source / label
    os.makedirs(image_dir, exist_ok=True)
    
    # Also save by news ID to maintain the association
    news_dir = output_dir / 'by_news' / source / label / news_id
    os.makedirs(news_dir, exist_ok=True)
    
    image_path = image_dir / f"{image_id}.jpg"
    news_image_path = news_dir / f"{image_id.split('_')[-1]}.jpg"
    
    # Skip if image already exists
    if image_path.exists() and news_image_path.exists():
        return True, image_id, source, label
    
    # Extract file extension from URL or default to jpg
    url_extension = os.path.splitext(image_url.split('?')[0])[1].lower()
    if url_extension and len(url_extension) < 5:  # Reasonable extension length check
        extension = url_extension
    else:
        extension = ".jpg"
    
    image_path = image_dir / f"{image_id}{extension}"
    news_image_path = news_dir / f"{image_id.split('_')[-1]}{extension}"
    
    # Attempt to download with retries
    for attempt in range(retry_count):
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Save in the main source/label directory
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Also save in the news-specific directory
            with open(news_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True, image_id, source, label
            
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retry_count} failed for {image_id}: {str(e)}")
            time.sleep(1)  # Wait before retrying
    
    logger.error(f"Failed to download image {image_id} after {retry_count} attempts")
    return False, image_id, source, label

def main():
    """Main function to extract and download FakeNewsNet images."""
    parser = argparse.ArgumentParser(description='Extract and download images from FakeNewsNet')
    parser.add_argument('--input', type=str, default=None,
                      help='Path to FakeNewsNet directory (default: data/raw/text/fakenewsnet)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for downloaded images (default: data/raw/images/fakenewsnet)')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of parallel download workers (default: 4)')
    parser.add_argument('--max-images', type=int, default=None,
                      help='Maximum number of images to download (default: all)')
    args = parser.parse_args()
    
    # Set up input directory
    if args.input is None:
        input_dir = get_data_path('text/fakenewsnet')
    else:
        input_dir = Path(args.input)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Set up output directory
    if args.output is None:
        output_dir = get_data_path('images/fakenewsnet')
    else:
        output_dir = Path(args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find news content files
    content_files = find_news_content_files(input_dir)
    if not content_files:
        logger.error("No news content files found")
        sys.exit(1)
    
    # Extract image URLs
    logger.info("Extracting image URLs from news content files...")
    all_images = []
    for content_file, source, label, news_id in tqdm(content_files, desc="Extracting image URLs"):
        images = extract_image_urls(content_file, source, label, news_id)
        all_images.extend(images)
    
    logger.info(f"Found {len(all_images)} images in {len(content_files)} news articles")
    
    # Limit the number of images if specified
    if args.max_images is not None and len(all_images) > args.max_images:
        all_images = all_images[:args.max_images]
        logger.info(f"Limited to {args.max_images} images")
    
    # Save image metadata
    metadata_dir = output_dir / 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_file = metadata_dir / 'image_metadata.json'
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_images, f, indent=2)
    
    logger.info(f"Saved image metadata to {metadata_file}")
    
    # Download images in parallel
    logger.info("Downloading images...")
    success_count = 0
    failure_count = 0
    source_label_counts = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_image = {
            executor.submit(download_image, image, output_dir): image
            for image in all_images
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_image), 
                          total=len(all_images), 
                          desc="Downloading images"):
            success, image_id, source, label = future.result()
            key = f"{source}_{label}"
            if key not in source_label_counts:
                source_label_counts[key] = 0
                
            if success:
                success_count += 1
                source_label_counts[key] += 1
            else:
                failure_count += 1
    
    # Print summary
    logger.info(f"\n===== Download Summary =====")
    logger.info(f"Total images: {len(all_images)}")
    logger.info(f"Successfully downloaded: {success_count}")
    logger.info(f"Failed to download: {failure_count}")
    
    for key, count in source_label_counts.items():
        source, label = key.split('_')
        logger.info(f"{source} {label}: {count} images")
    
    if failure_count > 0:
        logger.warning(f"Some images could not be downloaded. Check the log for details.")
    else:
        logger.info(f"All images downloaded successfully!")

if __name__ == "__main__":
    main()