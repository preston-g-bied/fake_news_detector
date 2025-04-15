# src/data_processing/image_processor.py

"""
Image data processing module for the Fake News Detection project.
This module handles the processing of image data from different datasets.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import csv
import shutil
from PIL import Image, ImageFile, UnidentifiedImageError
import exifread
import warnings
from tqdm import tqdm

# allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Base class for image processing tasks"""

    def __init__(self, config: Dict[str, Any], dataset_name: str):
        """
        Initialize the image processor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            dataset_name (str): Name of the dataset being processed
        """
        self.config = config
        self.dataset_name = dataset_name
        self._setup_image_processing()

    def _setup_image_processing(self):
        """Setup image processing parameters from configuration"""
        image_config = self.config['preprocessing']['image']

        # set resize dimensions
        self.resize = image_config.get('resize', [224, 224])

        # set normalization parameters
        self.normalization = image_config.get('normalization', {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.225, 0.225]
        })

        # set augmentation parameters
        self.augmentation = image_config.get('augmentation', {
            'enabled': True,
            'horizontal_flip': True,
            'vertical_flip': False,
            'random_crop': True,
            'color_jitter': True
        })

    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Validate that an image file can be opened and is not corrupt.
        
        Args:
            image_path (Union[str, Path]): Path to image file
            
        Returns:
            bool: True if the image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()    # verify it's an image

            # try to open it again to make sure it's readable
            with Image.open(image_path) as img:
                img.load()

            return True
        except (IOError, SyntaxError, UnidentifiedImageError, ValueError, OSError) as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False
        
    def process_image(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Process a single image file.
        
        Args:
            image_path (Union[str, Path]): Path to image file
            output_path (Union[str, Path]): Path to save processed image
            
        Returns:
            Optional[Dict[str, Any]]: Image metadata if successful, None otherwise
        """
        try:
            # validate the image
            if not self.validate_image(image_path):
                return None
            
            # open the image
            with Image.open(image_path) as img:
                # extract basic properties
                width, height = img.size
                aspect_ratio = width / height

                # convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # resize if configured
                if self.resize:
                    img = img.resize(self.resize, Image.LANCZOS)

                # save the processed image
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)

            # extract EXIF data if available
            exif_data = self.extract_exif(image_path)

            # create metadata
            metadata = {
                'original_path': str(image_path),
                'processed_path': str(output_path),
                'original_size': (width, height),
                'aspect_ratio': aspect_ratio,
                'format': os.path.splitext(image_path)[1].lower()[1:],  # extract extension without dot
                'exif': exif_data
            }

            return metadata
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
        
    def extract_exif(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract EXIF metadata from an image.
        
        Args:
            image_path (Union[str, Path]): Path to image file
            
        Returns:
            Dict[str, Any]: EXIF metadata
        """
        exif_data = {}
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            # convert tags to a more usable format
            for tag, value in tags.items():
                # skip some binary tags
                if tag.startswith('JPEGThumbnail'):
                    continue

                # convert value to string
                exif_data[tag] = str(value)

            # extract some common EXIF tags specifically
            common_tags = {
                'DateTime': 'EXIF DateTimeOriginal',
                'Make': 'Image Make',
                'Model': 'Image Model',
                'Software': 'Image Software',
                'GPSInfo': 'GPS GPSInfo'
            }

            # create a simplified subset of common tags
            exif_data['common'] = {}
            for key, tag in common_tags.items():
                if tag in tags:
                    exif_data['common'][key] = str(tags[tag])

        except Exception as e:
            logger.debug(f"Could not extract EXIF from {image_path}: {e}")

        return exif_data
    
    def process(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process the dataset from input directory and save to output directory.
        To be implemented by subclasses for dataset-specific processing.
        
        Args:
            input_dir (Union[str, Path]): Directory with raw images
            output_dir (Union[str, Path]): Directory to save processed images
        """
        raise NotImplementedError("Must be implemented by subclasses")


class MediaevalProcessor(ImageProcessor):
    """Processor for the MediaEval dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MediaEval")

    def load_image_metadata(self, metadata_file: Union[str, Path]) -> pd.DataFrame:
        """
        Load image metadata from the set_images.txt file.
        
        Args:
            metadata_file (Union[str, Path]): Path to set_images.txt
            
        Returns:
            pd.DataFrame: DataFrame with image metadata
        """
        try:
            # read file as text first to handle inconsistent lines
            with open(metadata_file, 'r', encoding='utf-8') as f:
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
            logger.info(f"Loaded {len(df)} image metadata entries from {metadata_file}")

            return df
        except Exception as e:
            logger.error(f"Error loading image metadata from {metadata_file}: {e}")
            return pd.DataFrame()
        
    def load_tweet_image_mapping(self, mapping_file: Union[str, Path]) -> pd.DataFrame:
        """
        Load tweet-image mappings from the tweets_images.txt file.
        
        Args:
            mapping_file (Union[str, Path]): Path to tweets_images.txt
            
        Returns:
            pd.DataFrame: DataFrame with tweet-image mappings
        """
        try:
            # read file as text first to handle inconsistent lines
            with open(mapping_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # parse lines manually
            data = []
            for i, line in enumerate(lines, 1):
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # ensure we have at least 4 parts
                    # use the first 4 parts only
                    tweet_id = parts[0]
                    image_id = parts[1]
                    annotation = parts[2]
                    event = parts[3]
                    data.append([tweet_id, image_id, annotation, event])
                else:
                    logger.warning(f"Line {i} in tweet mapping has fewer than 4 fields: {line.strip()}")

            # create DataFrame
            df = pd.DataFrame(data, columns=['tweet_id', 'image_id', 'annotation', 'event'])
            logger.info(f"Loaded {len(df)} tweet-image mappings from {mapping_file}")

            return df
        except Exception as e:
            logger.error(f"Error loading tweet-image mappings from {mapping_file}: {e}")
            return pd.DataFrame()
        
    def process(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process the MediaEval dataset.
        
        Args:
            input_dir (Union[str, Path]): Directory with MediaEval data
            output_dir (Union[str, Path]): Directory to save processed data
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # look for raw directory with metadata files
        raw_dir = input_dir / "raw"
        if not raw_dir.exists():
            raw_dir = input_dir # try the input directory itself

        # look for images directory
        images_dir = input_dir / "images"
        if not images_dir.exists():
            logger.error(f"Could not find images directory in {input_dir}")
            return
        
        # find metadata files
        set_images_file = raw_dir / "set_images.txt"
        tweets_images_file = raw_dir / "tweets_images.txt"

        if not set_images_file.exists():
            logger.error(f"Could not find set_images.txt in {raw_dir}")
            return
        
        # load image metadata
        image_metadata_df = self.load_image_metadata(set_images_file)
        if image_metadata_df.empty:
            return
        
        # load tweet-image mappings if available
        tweet_image_df = None
        if tweets_images_file.exists():
            tweet_image_df = self.load_tweet_image_mapping(tweets_images_file)

        # create output directories for each class
        fake_dir = output_dir / "fake"
        real_dir = output_dir / "real"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        # create a list to hold processed metadata
        processed_metadata = []

        # process images by class
        for annotation in ['fake', 'real']:
            # filter images by annotation
            class_df = image_metadata_df[image_metadata_df['annotation'].str.lower() == annotation]

            class_dir = fake_dir if annotation == 'fake' else real_dir
            logger.info(f"Processing {len(class_df)} {annotation} images")

            # find all image files for this class
            class_images_dir = images_dir / annotation
            if not class_images_dir.exists():
                logger.warning(f"Directory not found: {class_images_dir}")
                continue

            # collect all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.gif']:
                image_files.extend(list(class_images_dir.glob(f"*{ext}")))

            # map image_id to file path
            image_id_to_file = {}
            for image_file in image_files:
                image_id = image_file.stem
                image_id_to_file[image_id] = image_file

            # process each image in the DataFrame
            for _, row in tqdm(class_df.iterrows(), total=len(class_df),
                               desc=f"Processing {annotation} images"):
                image_id = row['image_id']

                # find the image file
                if image_id in image_id_to_file:
                    image_file = image_id_to_file[image_id]
                else:
                    # try to find a file with this ID but different extension
                    found = False
                    for img_file in image_files:
                        if img_file.stem == image_id:
                            found = True
                            image_file = img_file
                            break

                    if not found:
                        logger.warning(f"Could not find image file for ID {image_id}")
                        continue

                # determine output path
                output_path = class_dir / f"{image_id}{image_file.suffix}"

                # process the image
                image_metadata = self.process_image(image_file, output_path)

                if image_metadata:
                    # add information from the DataFrame
                    image_metadata.update({
                        'image_id': image_id,
                        'annotation': annotation,
                        'event': row['event'],
                        'image_url': row['image_url']
                    })

                    # add tweet information if available
                    if tweet_image_df is not None and not tweet_image_df.empty:
                        # Check if the DataFrame has the expected columns
                        if 'image_id' in tweet_image_df.columns and 'tweet_id' in tweet_image_df.columns:
                            # find all tweets that use this image
                            related_tweets = tweet_image_df[tweet_image_df['image_id'] == image_id]
                            if not related_tweets.empty:
                                image_metadata['tweet_ids'] = related_tweets['tweet_id'].tolist()
                                image_metadata['tweet_count'] = len(related_tweets)
                        else:
                            # Log a warning if the columns don't match expectations
                            logger.warning(f"Tweet mapping DataFrame is missing expected columns. Available columns: {tweet_image_df.columns.tolist()}")

                    processed_metadata.append(image_metadata)
        
        # save processed metadata
        metadata_dir = output_dir.parent / "metadata"
        os.makedirs(metadata_dir, exist_ok=True)

        metadata_file = metadata_dir / "mediaeval_processed.json"
        with open(metadata_file, 'w') as f:
            json.dump(processed_metadata, f, indent=2)

        logger.info(f"Processed {len(processed_metadata)} images and saved metadata to {metadata_file}")

        # create a summary CSV file
        summary_file = metadata_dir / "mediaeval_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            fieldnames = ['image_id', 'annotation', 'event', 'processed_path',
                          'original_size', 'aspect_ratio', 'has_exif']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in processed_metadata:
                writer.writerow({
                    'image_id': item['image_id'],
                    'annotation': item['annotation'],
                    'event': item['event'],
                    'processed_path': item['processed_path'],
                    'original_size': f"{item['original_size'][0]}x{item['original_size'][1]}",
                    'aspect_ratio': f"{item['aspect_ratio']:.2f}",
                    'has_exif': 'common' in item['exif'] and len(item['exif']['common']) > 0
                })

        logger.info(f"Saved summary to {summary_file}")


class FakeNewsNetImageProcessor(ImageProcessor):
    """Processor for the FakeNewsNet images dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "FakeNewsNet-Images")

    def load_image_metadata(self, metadata_file: Union[str, Path]) -> pd.DataFrame:
        """
        Load image metadata from the FakeNewsNet metadata file.
        
        Args:
            metadata_file (Union[str, Path]): Path to metadata file
            
        Returns:
            pd.DataFrame: DataFrame with image metadata
        """
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # convert to DataFrame for easier processing
            df_data = []
            for item in metadata:
                source = item['source']
                label = item['label']
                article_id = item['article_id']
                for image_file in item.get('images', []):
                    df_data.append({
                        'source': source,
                        'label': label,
                        'article_id': article_id,
                        'image_file': image_file,
                        'content_file': item.get('content_file', '')
                    })

            df = pd.DataFrame(df_data)
            logger.info(f"Loaded {len(df)} image metadata entries from {metadata_file}")
            return df
        except Exception as e:
            logger.error(f"Error loading image metadata from {metadata_file}: {e}")
            return pd.DataFrame()

    def process(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process the FakeNewsNet images dataset.
        
        Args:
            input_dir (Union[str, Path]): Directory with FakeNewsNet images
            output_dir (Union[str, Path]): Directory to save processed images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # find images directory
        images_dir = input_dir / "images"
        if not images_dir.exists():
            images_dir = input_dir  # try the input directory itself

        # look for metadata file
        metadata_dir = input_dir / "metadata"
        metadata_file = metadata_dir / "fakenewsnet_images.json"

        if not metadata_file.exists():
            logger.error(f"Could not find metadata file {metadata_file}")
            return
        
        # load image metadata
        image_metadata_df = self.load_image_metadata(metadata_file)
        if image_metadata_df.empty:
            return
        
        # create output directories for each source and label
        for source in ['gossipcop', 'politifact']:
            for label in ['fake', 'real']:
                os.makedirs(output_dir / source / label, exist_ok=True)

        # create a list to hold processed metadata
        processed_metadata = []

        # process each image
        for idx, row in tqdm(image_metadata_df.iterrows(), total=len(image_metadata_df),
                             desc="Processing FakeNewsNet images"):
            source = row['source']
            label = row['label']
            article_id = row['article_id']
            image_file = row['image_file']

            # construct the full image path
            image_path = images_dir / source / label / article_id / image_file

            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                continue

            # determine output path
            output_path = output_dir / source / label / f"{article_id}_{image_file}"

            # process the image
            image_metadata = self.process_image(image_path, output_path)

            if image_metadata:
                # add information from the DataFrame
                image_metadata.update({
                    'source': source,
                    'label': label,
                    'article_id': article_id,
                    'original_filename': image_file,
                    'content_file': row['content_file']
                })

                processed_metadata.append(image_metadata)

        # save processed metadata
        metadata_out_dir = output_dir.parent / "metadata"
        os.makedirs(metadata_out_dir, exist_ok=True)

        metadata_out_file = metadata_out_dir / "fakenewsnet_processed.json"
        with open(metadata_out_file, 'w') as f:
            json.dump(processed_metadata, f, indent=2)

        logger.info(f"Processed {len(processed_metadata)} images and saved metadata to {metadata_out_file}")

        # create a summary CSV file
        summary_file = metadata_out_dir / "fakenewsnet_summary.csv"
        with open(summary_file, 'w', newline='') as f:
            fieldnames = ['source', 'label', 'article_id', 'image_file', 'processed_path',
                          'original_size', 'aspect_ratio', 'has_exif']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in processed_metadata:
                writer.writerow({
                    'source': item['source'],
                    'label': item['label'],
                    'article_id': item['article_id'],
                    'image_file': item['original_filename'],
                    'processed_path': item['processed_path'],
                    'original_size': f"{item['original_size'][0]}x{item['original_size'][1]}",
                    'aspect_ratio': f"{item['aspect_ratio']:.2f}",
                    'has_exif': 'common' in item['exif'] and len(item['exif']['common']) > 0
                })
        
        logger.info(f"Saved summary to {summary_file}")