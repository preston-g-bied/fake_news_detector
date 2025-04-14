# src/data_processing/metadata_processor.py

"""
Metadata processing module for the Fake News Detection project.
This module handles extraction and processing of metadata from different datasets.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)

def load_processed_data(data_dir: Union[str, Path], dataset: str) -> Dict[str, Any]:
    """
    Load processed data from a dataset.
    
    Args:
        data_dir (Union[str, Path]): Base data directory
        dataset (str): Dataset name (e.g., 'liar', 'pheme', 'fakenewsnet')
        
    Returns:
        Dict[str, Any]: Dictionary with loaded data
    """
    data_dir = Path(data_dir)

    try:
        # look for combined JSON file
        combined_file = data_dir / f"{dataset}/{dataset}_combined.json"
        if combined_file.exists():
            with open(combined_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded combined data for {dataset} ({len(data)} items)")
            return {'data': data, 'source': dataset}
        
        # if no combined file, look for individual files
        all_data = []
        for json_file in data_dir.glob(f"{dataset}/*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                all_data.extend(file_data)

        if all_data:
            logger.info(f"Loaded data from multiple files for {dataset} ({len(all_data)} items)")
            return {'data': all_data, 'source': dataset}
        
        logger.warning(f"No data found for {dataset}")
        return {'data': [], 'source': dataset}
    
    except Exception as e:
        logger.error(f"Error loading data for {dataset}: {e}")
        return {'data': [], 'source': dataset}
    
def extract_common_metadata(data_items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract common metadata fields from data items and convert to DataFrame.
    
    Args:
        data_items (List[Dict[str, Any]]): List of data items
        
    Returns:
        pd.DataFrame: DataFrame with common metadata
    """
    metadata_records = []

    for item in data_items:
        # basic fields that should be common across datasets
        metadata = {
            'id': item.get('id', ''),
            'source': item.get('source', ''),
            'label': item.get('label', ''),
            'original_label': item.get('original_label', item.get('label', '')),
            'has_text': bool(item.get('raw_text', '') or item.get('processed_text', '')),
            'has_images': False     # to be updated later when linking images
        }

        # add dataset-specific fields
        if 'metadata' in item:
            # copy all metadata fields
            item_metadata = item['metadata']
            for key, value in item_metadata.items():
                # don't overwrite existing keys and skip complex objects
                if key not in metadata and not isinstance(value, (dict, list)):
                    metadata[key] = value

        metadata_records.append(metadata)

    # convert to DataFrame
    df = pd.DataFrame(metadata_records)

    return df

def link_images_to_article(metadata_df: pd.DataFrame, image_metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Link images to their corresponding articles in the metadata.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame with article metadata
        image_metadata (List[Dict[str, Any]]): List of image metadata items
        
    Returns:
        pd.DataFrame: Updated DataFrame with image information
    """
    # create a mapping of article IDs to image coutns and paths
    article_to_images = {}

    for img in image_metadata:
        article_id = img.get('article_id', '')
        if not article_id:
            continue

        if article_id not in article_to_images:
            article_to_images[article_id] = {
                'count': 0,
                'images': []
            }

        article_to_images[article_id]['count'] += 1
        article_to_images[article_id]['images'].append({
            'path': img.get('processed_path', ''),
            'original_path': img.get('original_path', ''),
            'size': img.get('original_size', [0, 0]),
            'aspect_ratio': img.get('aspect_ratio', 0)
        })

    # update metadata DataFrame with image information
    metadata_df['image_count'] = metadata_df['id'].map(
        lambda x: article_to_images.get(x, {}).get('count', 0)
    )

    metadata_df['has_images'] = metadata_df['image_count'] > 0

    # add image paths as a JSON string (to keep in DataFrame)
    metadata_df['image_paths'] = metadata_df['id'].map(
        lambda x: json.dumps(article_to_images.get(x, {}).get('images', []))
    )

    return metadata_df

def create_dataset_statistics(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create statistics for the dataset.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame with metadata
        
    Returns:
        Dict[str, Any]: Dictionary with statistics
    """
    stats = {
        'total_items': len(metadata_df),
        'class_distribution': metadata_df['label'].value_counts().to_dict(),
        'sources': metadata_df['source'].value_counts().to_dict(),
        'items_with_text': int(metadata_df['has_text'].sum()),
        'items_with_images': int(metadata_df['has_images'].sum() if 'has_images' in metadata_df.columns else 0),
        'items_with_both': int((metadata_df['has_text'] & metadata_df['has_images']).sum()
                               if 'has_images' in metadata_df.columns else 0)
    }

    return stats

def process_metadata(processed_data_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Process metadata from all datasets and create an integrated metadata file.
    
    Args:
        processed_data_dir (Union[str, Path]): Directory with processed data
        output_dir (Union[str, Path]): Directory to save metadata
    """
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Processing metadata from all datasets...")

    # load processed text data
    text_data_dir = processed_data_dir / "text"

    # dictionary to hold metadata from each dataset
    all_metadata = {}

    # process each text dataset
    for dataset in ['liar', 'pheme', 'fakenewsnet']:
        dataset_data = load_processed_data(text_data_dir, dataset)
        if dataset_data['data']:
            # extract common metadata
            metadata_df = extract_common_metadata(dataset_data['data'])
            all_metadata[dataset] = metadata_df

            # save individual dataset metadata
            metadata_file = output_dir / f"{dataset}_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            logger.info(f"Saved metadata for {dataset} to {metadata_file} ({len(metadata_df)} items)")

    # load image metadata
    image_metadata_dir = processed_data_dir / "images/metadata"
    image_metadata = []

    # load MediaEval image metadata
    mediaeval_metadata_file = image_metadata_dir / "mediaeval_processed.json"
    if mediaeval_metadata_file.exists():
        try:
            with open(mediaeval_metadata_file, 'r', encoding='utf-8') as f:
                mediaeval_metadata = json.load(f)
                image_metadata.extend(mediaeval_metadata)
            logger.info(f"Loaded {len(mediaeval_metadata)} MediaEval image metadata items")
        except Exception as e:
            logger.error(f"Error loading MediaEval image metadata: {e}")

    # load FakeNewsNet image metadata
    fakenewsnet_metadata_file = image_metadata_dir / "fakenewsnet_processed.json"
    if fakenewsnet_metadata_file.exists():
        try:
            with open(fakenewsnet_metadata_file, 'r', encoding='utf-8') as f:
                fakenewsnet_metadata = json.load(f)
                image_metadata.extend(fakenewsnet_metadata)
            logger.info(f"Loaded {len(fakenewsnet_metadata)} FakeNewsNet image metadata items")
        except Exception as e:
            logger.error(f"Error loading FakeNewsNet image metadata: {e}")

    # update metadata with image information
    for dataset, metadata_df in all_metadata.items():
        # only link images for FakeNewsNet
        if dataset == 'fakenewsnet':
            # extract only FakeNewsNet image metadata
            fakenewsnet_images = [img for img in image_metadata if 'source' in img and
                                  img['source'] in ['gossipcop', 'politifact']]
            
            metadata_df = link_images_to_article(metadata_df, fakenewsnet_images)
            all_metadata[dataset] = metadata_df

            # save updated metadata
            metadata_file = output_dir / f"{dataset}_metadata_with_images.csv"
            metadata_df.to_csv(metadata_file, index=False)
            logger.info(f"Updated metadata for {dataset} with image information ({len(metadata_df)} items)")

    # combine all metadata
    combined_metadata = pd.concat(all_metadata.values(), ignore_index=True)

    # save combined metadata
    combined_metadata_file = output_dir / "combined_metadata.csv"
    combined_metadata.to_csv(combined_metadata_file, index=False)
    logger.info(f"Saved combined metadata to {combined_metadata_file} ({len(combined_metadata)} items)")

    # create dataset statistics
    stats = {}
    for dataset, metadata_df in all_metadata.items():
        stats[dataset] = create_dataset_statistics(metadata_df)

    stats['combined'] = create_dataset_statistics(combined_metadata)

    # save statistics
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved dataset statistics to {stats_file}")

def create_integrated_dataset(processed_data_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Create an integrated dataset from all processed data.
    
    Args:
        processed_data_dir (Union[str, Path]): Directory with processed data
        output_dir (Union[str, Path]): Directory to save integrated dataset
    """
    processed_data_dir = Path(processed_data_dir)
    output_dir = Path(output_dir)

    logger.info("Creating integrated dataset...")

    # load combined metadata
    metadata_file = output_dir / "combined_metadata.csv"
    if not metadata_file.exists():
        logger.error(f"Combined metadata file not found: {metadata_file}")
        return
    
    try:
        metadata = pd.read_csv(metadata_file)
        logger.info(f"Loaded combined metadata with {len(metadata)} items")

        # create a mapping from ID to dataset source
        id_to_source = dict(zip(metadata['id'], metadata['source']))

        # create integrated dataset structure
        integrated_data = {
            'metadata': metadata.to_dict(orient='records'),
            'text_data': {},
            'image_data': {}
        }

        # load text data for each source
        text_data_dir = processed_data_dir / "text"
        for dataset in ['liar', 'pheme', 'fakenewsnet']:
            dataset_data = load_processed_data(text_data_dir, dataset)
            if dataset_data['data']:
                # create mapping from ID to text data
                id_to_text = {}
                for item in dataset_data['data']:
                    item_id = item.get('id', '')
                    if item_id:
                        id_to_text[item_id] = {
                            'raw_text': item.get('raw_text', ''),
                            'processed_text': item.get('processed_text', '')
                        }

                integrated_data['text_data'][dataset] = id_to_text

        # load image data
        image_metadata_dir = processed_data_dir / "images/metadata"

        # load MediaEval image metadata
        mediaeval_metadata_file = image_metadata_dir / "mediaeval_processed.json"
        if mediaeval_metadata_file.exists():
            try:
                with open(mediaeval_metadata_file, 'r', encoding='utf-8') as f:
                    mediaeval_metadata = json.load(f)

                # group by image_id
                mediaeval_images = {}
                for item in mediaeval_metadata:
                    image_id = item.get('image_id', '')
                    if image_id:
                        mediaeval_images[image_id] = {
                            'path': item.get('processed_path', ''),
                            'annotation': item.get('annotation', ''),
                            'event': item.get('event', ''),
                            'original_size': item.get('original_size', [0, 0]),
                            'aspect_ratio': item.get('aspect_ratio', 0),
                            'format': item.get('format', '')
                        }
                integrated_data['image_data']['mediaeval'] = mediaeval_images
                logger.info(f"Added {len(mediaeval_images)} MediaEval images to integrated dataset")
            except Exception as e:
                logger.error(f"Error loading MediaEval image metadata: {e}")

        # load FakeNewsNet image metadata
        fakenewsnet_metadata_file = image_metadata_dir / "fakenewsnet_processed.json"
        if fakenewsnet_metadata_file.exists():
            try:
                with open(fakenewsnet_metadata_file, 'r', encoding='utf-8') as f:
                    fakenewsnet_metadata = json.load(f)
                
                # group by article_id
                fakenewsnet_images = {}
                for item in fakenewsnet_metadata:
                    article_id = item.get('article_id', '')
                    if article_id:
                        if article_id not in fakenewsnet_images:
                            fakenewsnet_images[article_id] = []
                        
                        fakenewsnet_images[article_id].append({
                            'path': item.get('processed_path', ''),
                            'original_filename': item.get('original_filename', ''),
                            'source': item.get('source', ''),
                            'label': item.get('label', ''),
                            'original_size': item.get('original_size', [0, 0]),
                            'aspect_ratio': item.get('aspect_ratio', 0),
                            'format': item.get('format', '')
                        })
                
                integrated_data['image_data']['fakenewsnet'] = fakenewsnet_images
                logger.info(f"Added {len(fakenewsnet_images)} FakeNewsNet article image sets to integrated dataset")
            except Exception as e:
                logger.error(f"Error loading FakeNewsNet image metadata: {e}")

        # save integrated dataset
        integrated_file = output_dir / "integrated_dataset.json"
        with open(integrated_file, 'w') as f:
            json.dump(integrated_data, f, indent=2)

        logger.info(f"Saved integrated dataset to {integrated_file}")

        # also create a simplified version that links items directly to their images
        simplified_data = []

        for item in metadata.to_dict(orient='records'):
            item_id = item.get('id', '')
            source = item.get('source', '')
            
            # get the corresponding text data
            text_data = {}
            if source in integrated_data['text_data'] and item_id in integrated_data['text_data'][source]:
                text_data = integrated_data['text_data'][source][item_id]
            
            # get any associated images
            images = []
            if source == 'fakenewsnet' and 'fakenewsnet' in integrated_data['image_data']:
                if item_id in integrated_data['image_data']['fakenewsnet']:
                    images = integrated_data['image_data']['fakenewsnet'][item_id]
            elif source == 'mediaeval' and 'mediaeval' in integrated_data['image_data']:
                if item_id in integrated_data['image_data']['mediaeval']:
                    images = [integrated_data['image_data']['mediaeval'][item_id]]
            
            # create the simplified item
            simplified_item = {
                'id': item_id,
                'source': source,
                'label': item.get('label', ''),
                'text': text_data,
                'images': images,
                'metadata': {k: v for k, v in item.items() if k not in ['id', 'source', 'label']}
            }
            
            simplified_data.append(simplified_item)
        
        # save simplified dataset
        simplified_file = output_dir / "simplified_dataset.json"
        with open(simplified_file, 'w') as f:
            json.dump(simplified_data, f, indent=2)
        
        logger.info(f"Saved simplified integrated dataset to {simplified_file}")

    except Exception as e:
        logger.error(f"Error creating integrated dataset: {e}")