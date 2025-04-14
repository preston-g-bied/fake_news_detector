# src/data_processing/dataset_splitter.py

"""
Dataset splitting module for the Fake News Detection project.
This module handles the creation of stratified train/validation/test splits.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetSplitter:
    """Class for creating dataset splits."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset splitter.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.random_state = config.get('training', {}).get('seed', -7)

        # set random seeds for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def load_integrated_data(self, metadata_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load the integrated dataset.
        
        Args:
            metadata_dir (Union[str, Path]): Directory with integrated dataset
            
        Returns:
            Optional[Dict[str, Any]]: Integrated dataset or None if not found
        """
        metadata_dir = Path(metadata_dir)
        integrated_file = metadata_dir / "integrated_dataset.json"

        if not integrated_file.exists():
            logger.error(f"Integrated dataset file not found: {integrated_file}")
            return None
        
        try:
            with open(integrated_file, 'r', encoding='utf-8') as f:
                integraded_data = json.load(f)

            logger.info(f"Loaded integrated dataset with {len(integraded_data['metadata'])} items")
            return integraded_data
        except Exception as e:
            logger.error(f"Error loading integrated dataset: {e}")
            return None
        
    def create_stratified_split(self,
                               metadata: List[Dict[str, Any]],
                               val_size: float = 0.15,
                               test_size: float = 0.15) -> Tuple[List[str], List[str], List[str]]:
        """
        Create stratified train/val/test splits based on metadata.
        
        Args:
            metadata (List[Dict[str, Any]]): List of metadata items
            val_size (float): Proportion of data to use for validation
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Lists of IDs for train, val, test splits
        """
        # convert metadata to DataFrame for easier handling
        df = pd.DataFrame(metadata)

        # filter out items with unknown labels
        df = df[df['label'].isin(['true', 'false', 'mixed'])]

        # calculate actual test size relative to filtered dataset
        effective_test_size = test_size / (1 - val_size)

        # first_split: training + validation vs test
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df['label']
        )

        # second split: training vs validation
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=self.random_state,
            stratify=df.loc[train_val_idx, 'label']
        )

        # get the IDs for each split
        train_ids = df.loc[train_idx, 'id'].tolist()
        val_ids = df.loc[val_idx, 'id'].tolist()
        test_ids = df.loc[test_idx, 'id'].tolist()

        logger.info(f"Created splits - Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")

        # print class distributions in each split
        for split_name, split_ids in [('Train', train_idx), ('Validation', val_idx), ('Test', test_idx)]:
            class_dist = df.loc[split_ids, 'label'].value_counts(normalize=True)
            logger.info(f"{split_name} class distribution: {dict(class_dist)}")

        return train_ids, val_ids, test_ids
    
    def create_source_aware_split(self,
                                  metadata: List[Dict[str, Any]],
                                  val_size: float = 0.15,
                                  test_size: float = 0.15) -> Tuple[List[str], List[str], List[str]]:
        """
        Create splits that maintain source distribution and class balance.
        
        Args:
            metadata (List[Dict[str, Any]]): List of metadata items
            val_size (float): Proportion of data to use for validation
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Lists of IDs for train, val, test splits
        """
        df = pd.DataFrame(metadata)

        # filter out items with unknown labels
        df = df[df['label'].isin(['true', 'false', 'mixed'])]

        # group by source and label to ensure balanced distribution
        grouped = df.groupby(['source', 'label'])

        train_ids, val_ids, test_ids = [], [], []

        # split each group individually to maintain source and label distribution
        for (source, label), group in tqdm(grouped, desc="Creating source-aware splits"):
            group_idx = group.index.tolist()

            if len(group_idx) < 3:  # too small to split
                train_ids.extend(group['id'].tolist())
                continue

            # split into train+val and test
            train_val_idx, test_idx = train_test_split(
                group_idx,
                test_size=test_size,
                random_state=self.random_state
            )

            # split train+val into train and val
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size / (1 - test_size),
                random_state=self.random_state
            )

            # add to respective splits
            train_ids.extend(df.loc[train_idx, 'id'].tolist())
            val_ids.extend(df.loc[val_idx, 'id'].tolist())
            test_ids.extend(df.loc[test_idx, 'id'].tolist())

        logger.info(f"Creat4ed source-aware splits - Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")

        # print distribution stats
        for split_name, split_ids in [('Train', train_ids), ('Validation', val_ids), ('Test', test_ids)]:
            split_df = df[df['id'].isin(split_ids)]
            source_dist = split_df['source'].value_counts(normalize=True)
            label_dist = split_df['label'].value_counts(normalize=True)
            logger.info(f"{split_name} source distribution: {dict(source_dist)}")
            logger.info(f"{split_name} label distribution: {dict(label_dist)}")

        return train_ids, val_ids, test_ids
    
    def create_cross_validation_folds(self,
                                      metadata: List[Dict[str, Any]],
                                      n_splits: int = 5) -> Dict[int, Dict[str, List[str]]]:
        """
        Create cross-validation folds for the dataset.
        
        Args:
            metadata (List[Dict[str, Any]]): List of metadata items
            n_splits (int): Number of folds
            
        Returns:
            Dict[int, Dict[str, List[str]]]: Dictionary with fold indices and train/val splits
        """
        df = pd.DataFrame(metadata)

        # filter out items with unknown labels
        df = df[df['label'].isin(['true', 'false', 'mixed'])]

        # initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # create folds
        folds = {}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            train_ids = df.iloc[train_idx]['id'].tolist()
            val_ids = df.iloc[val_idx]['id'].tolist()

            folds[fold_idx] = {
                'train': train_ids,
                'validation': val_ids
            }

            # log fold statistics
            train_label_dist = df.iloc[train_idx]['label'].value_counts(normalize=True)
            val_label_dist = df.iloc[val_idx]['label'].value_counts(normalize=True)

            logger.info(f"Fold {fold_idx} - Train: {len(train_ids)}, Validation: {len(val_ids)}")
            logger.info(f"Fold {fold_idx} train label distribution: {dict(train_label_dist)}")
            logger.info(f"Fold {fold_idx} validation label distribution: {dict(val_label_dist)}")

        return folds
    
    def process(self, processed_data_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Create and save dataset splits.
        
        Args:
            processed_data_dir (Union[str, Path]): Directory with processed data
            output_dir (Union[str, Path]): Directory to save splits
        """
        processed_data_dir = Path(processed_data_dir)
        output_dir = Path(output_dir)

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # load integrated dataset
        metadata_dir = processed_data_dir / "metadata"
        integrated_data = self.load_integrated_data(metadata_dir)

        if not integrated_data:
            return
        
        metadata = integrated_data['metadata']

        # create regular train/val/test split
        train_ids, val_ids, test_ids = self.create_stratified_split(
            metadata,
            val_size=0.15,
            test_size=0.15
        )

        # save splits
        splits = {
            'train': train_ids,
            'validation': val_ids,
            'test': test_ids
        }

        splits_file = output_dir / "dataset_splits.json"
        with open(splits_file, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2)

        logger.info(f"Saved dataset splits to {splits_file}")

        # create source-aware splits
        src_train_ids, src_val_ids, src_test_ids = self.create_source_aware_split(
            metadata,
            val_size=0.15,
            test_size=0.15
        )

        # save source-aware splits
        source_splits = {
            'train': src_train_ids,
            'validation': src_val_ids,
            'test': src_test_ids
        }

        source_splits_file = output_dir / "source_aware_splits.json"
        with open(source_splits_file, 'w', encoding='utf-8') as f:
            json.dump(source_splits, f, indent=2)

        logger.info(f"Saved source-aware splits to {source_splits_file}")

        # check if cross-validation is enabled in config
        if self.config.get('evaluation', {}).get('cross_validation', {}).get('enabled', False):
            n_splits = self.config.get('evaluation', {}).get('cross_validation', {}).get('n_splits', 5)

            # create CV folds
            cv_folds = self.create_cross_validation_folds(metadata, n_splits=n_splits)

            # save CV folds
            cv_folds_file = output_dir / "cv_folds.json"
            with open(cv_folds_file, 'w', encoding='utf-8') as f:
                json.dump(cv_folds, f, indent=2)

            logger.info(f"Saved {n_splits} cross-validation folds to {cv_folds_file}")