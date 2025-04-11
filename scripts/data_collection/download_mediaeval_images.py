# scripts/data_collection_download_mediaeval_images.py

"""
Improved script to download images from the MediaEval dataset with
better error handling, rate limiting, and alternative URL strategies.
"""

import os
import sys
import csv
import json
import logging
import argparse
import concurrent.futures
from pathlib import Path
import requests
from tqdm import tqdm
import time
import random
from urllib.parse import urlparse, unquote

# Add the project root to the path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{project_root}/logs/mediaeval_download_improved.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple rate limiter to avoid hitting rate limits."""
    
    def __init__(self, domains):
        """
        Initialize with domain-specific delays.
        
        Args:
            domains: Dictionary of domain to delay in seconds
        """
        self.domains = domains
        self.last_request = {}
        
        # Initialize last request time for all domains
        for domain in domains:
            self.last_request[domain] = 0
    
    def wait_if_needed(self, url):
        """
        Wait if needed to respect rate limits.
        
        Args:
            url: URL to check
        """
        domain = urlparse(url).netloc
        
        # Apply default delay if domain not in list
        if domain not in self.domains:
            domain = 'default'
        
        # Check if we need to wait
        now = time.time()
        elapsed = now - self.last_request.get(domain, 0)
        delay = self.domains[domain]
        
        if elapsed < delay:
            wait_time = delay - elapsed
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request[domain] = time.time()

def read_image_data(input_file):
    """
    Read image data from the input file.
    
    Args:
        input_file: Path to the input file
    
    Returns:
        List of dictionaries with image data
    """
    images = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                images.append(row)
        
        logger.info(f"Read {len(images)} images from {input_file}")
        return images
    except Exception as e:
        logger.error(f"Error reading {input_file}: {str(e)}")
        return []

def try_alternative_url(image_url):
    """
    Try to generate alternative URLs for common issues.
    
    Args:
        image_url: Original image URL
        
    Returns:
        List of alternative URLs to try
    """
    alternatives = []
    parsed = urlparse(image_url)
    
    # For Twitter proxy URLs, try to extract the original URL
    if 'twimg.com' in parsed.netloc and 'proxy.jpg' in parsed.path:
        # Extract original URL from proxy parameters
        query = parsed.query
        for param in query.split('&'):
            if 't=' in param:
                # Extract and decode the original URL
                try:
                    encoded_url = param.split('=')[1].split(',')[0]
                    # Handle different encoding schemes
                    if 'http' in encoded_url:
                        alternatives.append(encoded_url)
                    else:
                        # Try some common decoding patterns for Twitter
                        possible_urls = [
                            unquote(encoded_url),
                            # Add other decoding methods if needed
                        ]
                        for url in possible_urls:
                            if url.startswith('http'):
                                alternatives.append(url)
                except:
                    pass
    
    # For imgur URLs, try different extensions and formats
    if 'imgur.com' in parsed.netloc:
        # Extract the image ID
        path = parsed.path
        if path.endswith(('.jpg', '.png', '.gif')):
            img_id = os.path.splitext(os.path.basename(path))[0]
            # Remove size suffix like 'h' at the end (common in imgur URLs)
            if img_id[-1] in ['h', 'l', 'm', 's']:
                img_id = img_id[:-1]
            
            # Try different extensions and formats
            extensions = ['.jpg', '.png', '.gif']
            for ext in extensions:
                alternatives.append(f"https://i.imgur.com/{img_id}{ext}")
    
    return alternatives

def download_image(row, output_dir, rate_limiter, retry_count=3, allow_alternatives=True):
    """
    Download an image from a URL with rate limiting and alternative URLs.
    
    Args:
        row: Dictionary containing image information
        output_dir: Directory to save the image
        rate_limiter: RateLimiter instance
        retry_count: Number of retries if download fails
        allow_alternatives: Whether to try alternative URLs
        
    Returns:
        Tuple of (success, image_id, annotation, url_used)
    """
    image_id = row['image_id']
    image_url = row['image_url']
    annotation = row['annotation']
    event = row['event']
    
    # Normalize annotation to lowercase
    annotation = annotation.lower()
    
    # Create directory structure
    image_dir = output_dir / annotation
    os.makedirs(image_dir, exist_ok=True)
    
    # Create event-specific directory
    event_dir = output_dir / 'by_event' / event.lower()
    os.makedirs(event_dir, exist_ok=True)
    
    image_path = image_dir / f"{image_id}.jpg"
    event_image_path = event_dir / f"{image_id}.jpg"
    
    # Skip if image already exists
    if image_path.exists() and event_image_path.exists():
        return True, image_id, annotation, image_url
    
    # URLs to try - start with the original
    urls_to_try = [image_url]
    
    # Add alternative URLs if enabled
    if allow_alternatives:
        alt_urls = try_alternative_url(image_url)
        urls_to_try.extend(alt_urls)
    
    # Try each URL
    for url in urls_to_try:
        # Apply rate limiting
        rate_limiter.wait_if_needed(url)
        
        # Attempt to download with retries
        for attempt in range(retry_count):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                }
                
                response = requests.get(url, stream=True, timeout=15, headers=headers)
                response.raise_for_status()
                
                # Check if the response content is actually an image
                content_type = response.headers.get('Content-Type', '')
                if 'image' not in content_type:
                    logger.warning(f"URL {url} returned content type {content_type} - not an image")
                    raise ValueError("Not an image")
                
                # Save in the main annotation directory
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Also save in the event directory
                with open(event_image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return True, image_id, annotation, url
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retry_count} failed for {image_id} with URL {url}: {str(e)}")
                
                # Add a longer delay for rate limiting errors
                if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                    time.sleep(5 + random.random() * 5)  # 5-10 seconds
                else:
                    time.sleep(1 + random.random())  # 1-2 seconds
    
    logger.error(f"Failed to download image {image_id} after trying all URLs")
    return False, image_id, annotation, None

def main():
    """Main function to download MediaEval images with improved handling."""
    parser = argparse.ArgumentParser(description='Download images from MediaEval dataset (improved version)')
    parser.add_argument('--input', type=str, default=None,
                      help='Path to set_images.txt file (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for downloaded images (default: data/raw/images/mediaeval)')
    parser.add_argument('--workers', type=int, default=3,
                      help='Number of parallel download workers (default: 3)')
    parser.add_argument('--max-images', type=int, default=None,
                      help='Maximum number of images to download (default: all)')
    args = parser.parse_args()
    
    # Auto-detect input file if not specified
    if args.input is None:
        input_paths = [
            project_root / "data" / "raw" / "images" / "mediaeval" / "set_images.txt",
            project_root / "data" / "raw" / "images" / "mediaeval" / "raw" / "set_images.txt"
        ]
        
        input_file = None
        for path in input_paths:
            if path.exists():
                input_file = path
                break
                
        if input_file is None:
            logger.error("Could not find set_images.txt file")
            sys.exit(1)
    else:
        input_file = Path(args.input)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Set up output directory
    if args.output is None:
        output_dir = project_root / "data" / "raw" / "images" / "mediaeval" / "images"
    else:
        output_dir = Path(args.output)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image data
    images = read_image_data(input_file)
    if not images:
        logger.error("No images found in the input file")
        sys.exit(1)
    
    # Limit number of images if specified
    if args.max_images is not None and len(images) > args.max_images:
        logger.info(f"Limiting to {args.max_images} images")
        images = images[:args.max_images]
    
    # Set up rate limiter
    domain_delays = {
        'i.imgur.com': 2.0,       # 2 seconds between imgur requests
        'imgur.com': 2.0,
        'pbs.twimg.com': 1.5,     # 1.5 seconds between Twitter requests
        'twimg.com': 1.5,
        'o.twimg.com': 1.5,
        'default': 1.0            # 1 second for other domains
    }
    rate_limiter = RateLimiter(domain_delays)
    
    # Download images concurrently
    success_count = 0
    failure_count = 0
    fake_count = 0
    real_count = 0
    
    # Prepare metadata file to track successes/failures
    metadata = {
        "images": [],
        "stats": {
            "total": len(images),
            "success": 0,
            "failure": 0,
            "fake": 0,
            "real": 0
        }
    }
    
    # Use a smaller number of workers to be gentler on the servers
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, args.workers)) as executor:
        futures = []
        
        # Submit all tasks
        for image in images:
            future = executor.submit(download_image, image, output_dir, rate_limiter)
            futures.append((future, image))
        
        # Process results as they complete
        for future, image in tqdm(futures, desc="Downloading images"):
            success, image_id, annotation, url_used = future.result()
            
            # Update metadata
            result = {
                "image_id": image_id,
                "annotation": annotation,
                "original_url": image["image_url"],
                "url_used": url_used,
                "success": success,
                "event": image["event"]
            }
            metadata["images"].append(result)
            
            if success:
                success_count += 1
                metadata["stats"]["success"] += 1
                
                if annotation.lower() == 'fake':
                    fake_count += 1
                    metadata["stats"]["fake"] += 1
                elif annotation.lower() == 'real':
                    real_count += 1
                    metadata["stats"]["real"] += 1
            else:
                failure_count += 1
                metadata["stats"]["failure"] += 1
    
    # Save metadata
    metadata_dir = output_dir.parent / 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    
    with open(metadata_dir / 'download_results.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    logger.info(f"\n===== Download Summary =====")
    logger.info(f"Total images: {len(images)}")
    logger.info(f"Successfully downloaded: {success_count}")
    logger.info(f"Failed to download: {failure_count}")
    logger.info(f"Fake images: {fake_count}")
    logger.info(f"Real images: {real_count}")
    
    if failure_count > 0:
        logger.warning(f"Some images could not be downloaded. Check the metadata for details.")
    else:
        logger.info(f"All images downloaded successfully!")

if __name__ == "__main__":
    main()