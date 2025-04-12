# scripts/data_collection/extract_fakenewsnet_images.py

"""
Script to extract and download images from the FakeNewsNet dataset.
This script parses news content.json files to find image URLs and downloads them.
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
from urllib.parse import urlparse, unquote

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.config import get_project_root, load_config

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{project_root}/logs/fakenewsnet_images.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_image_urls(content_file):
    """
    Extract image URLs from a news content.json file
    
    Args:
        content_file (Path): Path to news content.json file
        
    Returns:
        list: List of image URLs found in the content
    """
    try:
        with open(content_file, 'r', encoding='utf-8') as f:
            content = json.load(f)

        image_urls = []

        # check if there are images in the top-level "images" field
        if 'images' in content and isinstance(content['images'], list):
            image_urls.extend(content['images'])

        # check if there are image URLs in the "text" field (HTML content)
        if 'text' in content and content['text']:
            # this is a simple extraction and might miss some images
            # a more robust approach would use a HTML parser
            text = content['text']
            img_tags = text.split('<img')
            for img_tag in img_tags[1:]:    # skip the first split which is before any img tag
                src_pos = img_tag.find('src="')
                if src_pos != -1:
                    src_pos += 5    # length of 'src="'
                    end_pos = img_tag.find('"', src_pos)
                    if end_pos != -1:
                        img_url = img_tag[src_pos:end_pos].strip()
                        if img_url and not img_url.startswith('data:'): # skip data urls
                            image_urls.append(img_url)

        # check top-level image field (sometimes it's singular "image" instead of "images")
        if 'image' in content and content['image'] and isinstance(content['image'], str):
            image_urls.append(content['image'])

        # deduplicate URLs
        image_urls = list(set(image_urls))

        return image_urls
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON file: {content_file}")
        return []
    except Exception as e:
        logger.error(f"Error extracting image URLs from {content_file}: {e}")
        return []
    
def find_all_content_files(base_dir):
    """
    Find all news content.json files in the dataset
    
    Args:
        base_dir (Path): Base directory of the FakeNewsNet dataset
        
    Returns:
        list: List of tuples (source, label, article_id, content_file_path)
    """
    content_files = []

    # sources in FakeNewsNet
    sources = ['gossipcop', 'politifact']

    # labels
    labels = ['fake', 'real']

    for source in sources:
        source_dir = base_dir / source
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            continue

        for label in labels:
            label_dir = source_dir / label
            if not label_dir.exists():
                logger.warning(f"Label directory not found: {label_dir}")
                continue

            # each subdirectory is an article ID
            for article_dir in label_dir.iterdir():
                if article_dir.is_dir():
                    article_id = article_dir.name
                    content_file = article_dir / "news content.json"

                    if content_file.exists():
                        content_files.append((source, label, article_id, content_file))

    logger.info(f"Found {len(content_files)} news content files")
    return content_files

def get_filename_from_url(url):
    """
    Extract a filename from the URL
    
    Args:
        url (str): Image URL
        
    Returns:
        str: Filename for the image
    """
    try:
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)

        # get the base filename
        filename = os.path.basename(path)

        # if no filename or it ends with / (directory), generate a random name
        if not filename or filename.endswith('/'):
            return f"image_{hash(url) % 10000:04d}.jpg"
        
        # check if the filename has an extension
        name, ext = os.path.splitext(filename)
        if not ext:
            # no extension, assume jpg
            filename = f"{name}.jpg"

        # remove query parameters if any
        filename = filename.split('?')[0]

        # make sure the filename is valid
        filename = ''.join(c for c in filename if c.isalnum() or c in '._-')

        # limit filename length
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = f"{name[:95]}{ext}"

        return filename
    except Exception:
        # fallback to a hash-based filename
        return f"image_{hash(url) % 10000:04d}.jpg"
    
def download_image(url, output_path):
    """
    Download an image from URL and save it
    
    Args:
        url (str): Image URL
        output_path (Path): Path to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # check if the content type is an image
        content_type = response.headers.get('Content-Type', '')
        if not (content_type.startswith('image/') or 'octet-stream' in content_type):
            logger.warning(f"Not an image content type: {content_type} for URL: {url}")
            return False
        
        # save the image
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error for {url}: {e}")
        return False
    
def process_article(source, label, article_id, content_file, output_base_dir):
    """
    Process a single article: extract image URLs and download images
    
    Args:
        source (str): News source (gossipcop or politifact)
        label (str): Label (fake or real)
        article_id (str): Article ID
        content_file (Path): Path to news content.json file
        output_base_dir (Path): Base directory to save images
        
    Returns:
        dict: Results summary for this article
    """
    results = {'article_id': article_id, 'total_urls': 0, 'downloaded': 0, 'failed': 0}

    # find image URLs in the article
    image_urls = find_image_urls(content_file)
    results['total_urls'] = len(image_urls)

    if not image_urls:
        return results
    
    # create output directory for this article's images
    output_dir = output_base_dir / source / label / article_id
    os.makedirs(output_dir, exist_ok=True)

    # download each image
    for url in image_urls:
        try:
            # skip invalid URLs
            if not url or not url.startswith(('http://', 'https://')):
                results['failed'] += 1
                continue

            # get filename from URL
            filename = get_filename_from_url(url)
            output_path = output_dir / filename

            # skip if file already exists
            if output_path.exists():
                results['downloaded'] += 1
                continue

            # download the image
            success = download_image(url, output_path)
            if success:
                results['downloaded'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            logger.error(f"Error processing URL {url} for article {article_id}: {e}")
            results['failed'] += 1

    return results

def process_articles_parallel(content_files, output_base_dir, max_workers=5):
    """
    Process articles in parallel
    
    Args:
        content_files (list): List of (source, label, article_id, content_file) tuples
        output_base_dir (Path): Base directory to save images
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        dict: Summary of results
    """
    results = {'total_articles': len(content_files), 'total_urls': 0,
               'downloaded': 0, 'failed': 0, 'articles_with_images': 0}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # submit tasks
        future_to_article = {
            executor.submit(
                process_article, source, label, article_id, content_file, output_base_dir
            ): (source, label, article_id)
            for source, label, article_id, content_file in content_files
        }

        # process results
        for future in tqdm(concurrent.futures.as_completed(future_to_article),
                           total=len(future_to_article),
                           desc="Processing articles"):
            source, label, article_id = future_to_article[future]
            try:
                article_result = future.result()

                # update overall results
                results['total_urls'] += article_result['total_urls']
                results['downloaded'] += article_result['downloaded']
                results['failed'] += article_result['failed']

                if article_result['total_urls'] > 0:
                    results['articles_with_images'] += 1

            except Exception as e:
                logger.error(f"Error processing article {source}/{label}/{article_id}: {e}")

    return results

def create_metadata_file(content_files, output_base_dir):
    """
    Create a metadata file mapping articles to their images
    
    Args:
        content_files (list): List of (source, label, article_id, content_file) tuples
        output_base_dir (Path): Base directory where images are saved
    """
    try:
        # create metadata directory
        metadata_dir = output_base_dir.parent / 'metadata'
        os.makedirs(metadata_dir, exist_ok=True)

        # create metadata file
        metadata_path = metadata_dir / 'fakenewsnet_images.json'

        metadata = []

        for source, label, article_id, content_file in tqdm(content_files,
                                                            desc="Creating metadata"):
            # check if this article has images
            article_img_dir = output_base_dir / source / label / article_id
            if not article_img_dir.exists():
                continue

            # find all images for this article
            images = [f.name for f in article_img_dir.iterdir() if f.is_file()]
            if not images:
                continue

            # add to metadata
            metadata.append({
                'source': source,
                'label': label,
                'article_id': article_id,
                'image_count': len(images),
                'images': images,
                'content_file': str(content_file)
            })

        # save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Extract images from FakeNewsNet dataset')
    parser.add_argument('--input', type=str,
                        help='Path to FakeNewsNet dataset directory')
    parser.add_argument('--output', type=str,
                        help='Directoru to save extracted images')
    parser.add_argument('--max_workers', type=int, default=0,
                        help='Maximum number of parallel workers')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of articles to process (0 for all)')
    return parser.parse_args()

def main():
    args = parse_args()

    # load config
    config = load_config()

    # set input/output paths
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = (get_project_root() / config['paths']['data']['raw'] /
                     'text/fakenewsnet')
        
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = (get_project_root() / config['paths']['data']['raw'] /
                      'images/fakenewsnet/images')
    
    # find all content files
    content_files = find_all_content_files(input_dir)

    # apply limit if specified
    if args.limit > 0:
        logger.info(f"Limiting processing to {args.limit} articles")
        content_files = content_files[:args.limit]

    # process articles
    results = process_articles_parallel(content_files, output_dir, args.max_workers)

    # create metadata file
    create_metadata_file(content_files, output_dir)

    # print summary
    logger.info("\n===== Extraction Summary =====")
    logger.info(f"Total articles processed: {results['total_articles']}")
    logger.info(f"Articles with images: {results['articles_with_images']}")
    logger.info(f"Total image URLs found: {results['total_urls']}")
    logger.info(f"Successfully downloaded: {results['downloaded']}")
    logger.info(f"Failed: {results['failed']}")

    if results['failed'] == 0:
        logger.info("\nAll images processed successfully!")
    else:
        logger.warning(f"\n{results['failed']} images failed to download.")
        logger.info("Check the log file for details.")

if __name__ == "__main__":
    main()