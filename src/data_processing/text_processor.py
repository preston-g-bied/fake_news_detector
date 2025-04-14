# src/data_processing/text_processor.py

"""
Text data processing module for the Fake News Detection project.
This module handles the processing of text data from different datasets.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from bs4 import BeautifulSoup
import warnings

logger = logging.getLogger(__name__)

class TextProcessor:
    """Base class for text processing tasks"""

    def __init__(self, config: Dict[str, Any], dataset_name: str):
        """
        Initialize the text processor.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            dataset_name (str): Name of the dataset being processed
        """
        self.config = config
        self.dataset_name = dataset_name
        self._setup_nlp_resources()

    def _setup_nlp_resources(self):
        """Setup NLP resources based on configuration"""
        # download required NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        # initialize tools based on configuration
        text_config = self.config['preprocessing']['text']

        # set language for processing
        self.language = text_config.get('language', 'english')

        # set up stopwords if configured
        self.remove_stopwords = text_config.get('remove_stopwords', True)
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words(self.language))

        # set up stemming if configured
        self.stemming = text_config.get('stemming', False)
        if self.stemming:
            self.stemmer = PorterStemmer()

        # set up lemmatization if configured
        self.lemmatization = text_config.get('lemmatization', True)
        if self.lemmatization:
            self.lemmatizer = WordNetLemmatizer()
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            try:
                nltk.data.find('taggers/universal_tagset')
            except LookupError:
                nltk.download('universal_tagset')

        # maximum text length for truncation
        self.max_length = text_config.get('max_length', 512)

        # load spaCy model for more advanced processing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Using basic text processing instead."
            )
            self.nlp = None

    def clean_text(self, text: str) -> str:
        """
        Clean text content by removing HTML, extra whitespace, etc.
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Cleaned text
        """
        if text is None:
            return ""
        
        # convert to string if not already
        text = str(text)

        # remove HTML tags
        if "<" in text and ">" in text: # only parse as HTML if tags likely exist
            try:
                text = BeautifulSoup(text, "lxml").get_text(separator=" ")
            except Exception as e:
                logger.warning(f"Error removing HTML: {e}")

        # replace URLs with [URL]
        text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

        # replace email addresses with [EMAIL]
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)

        # replace numbers with [NUMBER]
        text = re.sub(r'\b\d+\b', '[NUMBER]', text)

        # replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # remove leading and trailing whitespace
        text = text.strip()

        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words or tokens.
        
        Args:
            text (str): Cleaned text
            
        Returns:
            List[str]: List of tokens
        """
        if self.nlp is not None:
            # use spaCy for tokenization
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
        else:
            # fall back to NLTK for tokenization
            tokens = word_tokenize(text)

        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text (str): Raw text content
            
        Returns:
            str: Fully preprocessed text
        """
        # clean the text
        cleaned_text = self.clean_text(text)

        # tokenize
        tokens = self.tokenize(cleaned_text)

        # apply stopword removal if configured
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]

        # apply stemming or lemmatization (not both)
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatization:
            if self.nlp is not None:
                # use spaCy for lemmatization
                doc = self.nlp(" ".join(tokens))
                tokens = [token.lemma_ for token in doc]
            else:
                # fall back to NLTK
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # join tokens back into text
        processed_text = " ".join(tokens)

        # truncate if necessary
        if len(processed_text) > self.max_length * 4:   # rough character estimate
            processed_text = processed_text[:self.max_length * 4]

        return processed_text
    
    def extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from text data.
        To be implemented by subclasses for dataset-specific metadata extraction.
        
        Args:
            data (Dict[str, Any]): Raw data dictionary
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        raise NotImplementedError("Must be implemented by subclasses")
    
    def process(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """
        Process the dataset from input path and save to output path.
        To be implemented by subclasses for dataset-specific processing.
        
        Args:
            input_path (Union[str, Path]): Path to raw data
            output_path (Union[str, Path]): Path to save processed data
        """
        raise NotImplementedError("Must be implemented by subclasses")
    

class LiarProcessor(TextProcessor):
    """Processor for the LIAR dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "LIAR")

    def read_liar_tsv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read LIAR dataset TSV file.
        
        Args:
            file_path (Union[str, Path]): Path to TSV file
            
        Returns:
            pd.DataFrame: DataFrame with the LIAR data
        """
        # define column names from LIAR documentation
        column_names = [
            'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
            'state_info', 'party', 'barely_true_counts', 'false_counts',
            'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts',
            'context'
        ]

        try:
            # read the TSV file
            df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
            logger.info(f"Successfully read LIAR data from {file_path} with {len(df)} entries")
            return df
        except Exception as e:
            logger.error(f"Error reading LIAR data from {file_path}: {e}")
            return pd.DataFrame()
        
    def extract_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract metadata from LIAR row.
        
        Args:
            row (pd.Series): Row of LIAR DataFrame
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            'id': row['id'],
            'subject': row['subject'],
            'speaker': row['speaker'],
            'speaker_job': row['job_title'],
            'state': row['state_info'],
            'party': row['party'],
            'context': row['context'],
            'speaker_history': {
                'barely_true': int(row['barely_true_counts']),
                'false': int(row['false_counts']),
                'half_true': int(row['half_true_counts']),
                'mostly_true': int(row['mostly_true_counts']),
                'pants_on_fire': int(row['pants_on_fire_counts'])
            },
            'total_statements': sum([
                int(row['barely_true_counts']),
                int(row['false_counts']),
                int(row['half_true_counts']),
                int(row['mostly_true_counts']),
                int(row['pants_on_fire_counts'])
            ])
        }

        # calculate speaker reliability score based on history
        mostly_true_weight = 0.75
        half_true_weight = 0.5
        barely_true_weight = 0.25
        false_weight = 0.0
        pants_fire_weight = -0.5

        if metadata['total_statements'] > 0:
            metadata['reliability_score'] = (
                mostly_true_weight * int(row['mostly_true_counts']) +
                half_true_weight * int(row['half_true_counts']) +
                barely_true_weight * int(row['barely_true_counts']) +
                false_weight * int(row['false_counts']) +
                pants_fire_weight * int(row['pants_on_fire_counts'])
            ) / metadata['total_statements']
        else:
            metadata['reliability_score'] = np.nan

        return metadata
    
    def standardize_label(self, label: str) -> str:
        """
        Standardize the label from LIAR dataset.
        
        Args:
            label (str): Original label
            
        Returns:
            str: Standardized label
        """
        if label in ['pants-fire', 'false', 'barely-true']:
            return 'false'
        elif label in ['true', 'mostly-true']:
            return 'true'
        elif label in ['half-true']:
            return 'mixed'
        else:
            logger.warning(f"Unknown label: {label}, defaulting to 'unknown'")
            return 'unknown'
        
    def process(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process the LIAR dataset.
        
        Args:
            input_dir (Union[str, Path]): Directory with LIAR data
            output_dir (Union[str, Path]): Directory to save processed data
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # find all TSV files
        tsv_files = list(input_dir.glob('*.tsv'))

        if not tsv_files:
            logger.warning(f"No TSV files found in {input_dir}")
            return
        
        # create a list to hold all processed data
        all_processed_data = []

        # process each TSV file
        for tsv_file in tsv_files:
            file_name = tsv_file.stem   # train, test, or valid

            # read the data
            df = self.read_liar_tsv(tsv_file)
            if df.empty:
                continue

            # process each row
            processed_data = []
            for _, row in df.iterrows():
                # process text
                raw_text = row['statement']
                processed_text = self.preprocess_text(raw_text)

                # extract metadata
                metadata = self.extract_metadata(row)

                # standardize label
                original_label = row['label']
                standardized_label = self.standardize_label(original_label)

                # create processed entry
                processed_entry = {
                    'id': row['id'],
                    'split': file_name, # train, test, or valid
                    'raw_text': raw_text,
                    'processed_text': processed_text,
                    'original_label': original_label,
                    'label': standardized_label,
                    'metadata': metadata
                }

                processed_data.append(processed_entry)
                all_processed_data.append(processed_entry)

            # save processed data for this file
            output_file = output_dir / f"{file_name}.json"
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)

            logger.info(f"Processed {len(processed_data)} entries from {tsv_file.name} and saved to {output_file}")

        # save combined processed data
        combined_output_file = output_dir / "liar_combined.json"
        with open(combined_output_file, 'w') as f:
            json.dump(all_processed_data, f, indent=2)

        logger.info(f"Saved combined data with {len(all_processed_data)} entries to {combined_output_file}")


class PhemeProcessor(TextProcessor):
    """Processor for the PHEME dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "PHEME")

    def read_tweet(self, tweet_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Read a tweet JSON file.
        
        Args:
            tweet_file (Union[str, Path]): Path to tweet JSON file
            
        Returns:
            Dict[str, Any]: Tweet data
        """
        try:
            with open(tweet_file, 'r', encoding='utf-8') as f:
                tweet_data = json.load(f)
            return tweet_data
        except Exception as e:
            logger.error(f"Error reading tweet file {tweet_file}: {e}")
            return {}
        
    def read_annotation(self, annotation_file: Union[str, Path]) -> str:
        """
        Read and process a rumor annotation file.
        
        Args:
            annotation_file (Union[str, Path]): Path to annotation file
            
        Returns:
            str: Standardized label
        """
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            # import the conversion function
            from data.raw.text.pheme.convert_veracity_annotations import convert_annotations

            # convert annotations to standard format
            label = convert_annotations(annotation)

            # standardize the label format
            if label == "true":
                return "true"
            elif label == "false":
                return "false"
            elif label == "unverified":
                return "unverified"
            else:
                return "unknown"
        except Exception as e:
            logger.error(f"Error processing annotation file {annotation_file}: {e}")
            return "unknown"
        
    def process_rumor(self, rumor_dir: Path, event_name: str) -> Dict[str, Any]:
        """
        Process a single rumor directory.
        
        Args:
            rumor_dir (Path): Path to rumor directory
            event_name (str): Name of the event this rumor belongs to
            
        Returns:
            Dict[str, Any]: Processed rumor data
        """
        rumor_id = rumor_dir.name

        # check for source tweets
        source_tweets_dir = rumor_dir / "source-tweets"
        if not source_tweets_dir.exists():
            logger.warning(f"No source-tweets directory found for rumor {rumor_id}")
            return None
        
        # find the source tweet file (there should be only one)
        source_tweet_files = list(source_tweets_dir.glob("*.json"))
        if not source_tweet_files:
            logger.warning(f"No source tweet file found for rumor {rumor_id}")
            return None
        
        source_tweet_file = source_tweet_files[0]
        source_tweet = self.read_tweet(source_tweet_file)

        if not source_tweet:
            logger.warning(f"Empty source tweet for rumor {rumor_id}")
            return None
        
        # get the text content
        if 'text' in source_tweet:
            raw_text = source_tweet['text']
        else:
            logger.warning(f"No text field in source tweet for rumor {rumor_id}")
            return None
        
        # process the text
        processed_text = self.preprocess_text(raw_text)

        # read the annotation file if it exists
        annotation_file = rumor_dir / "annotation.json"
        if annotation_file.exists():
            label = self.read_annotation(annotation_file)
        else:
            label = "unknown"

        # count reactions
        reactions_dir = rumor_dir / "reactions"
        reaction_count = 0
        reaction_texts = []
        if reactions_dir.exists():
            reaction_files = list(reactions_dir.glob("**/*.json"))
            reaction_count = len(reaction_files)

            # process a sample of reactions (max 10)
            for reaction_file in reaction_files[:10]:
                reaction = self.read_tweet(reaction_file)
                if reaction and 'text' in reaction:
                    reaction_text = self.preprocess_text(reaction['text'])
                    reaction_texts.append(reaction_text)

        # extract metadata
        metadata = self.extract_metadata(source_tweet)
        metadata.update({
            'event': event_name,
            'rumor_id': rumor_id,
            'reaction_count': reaction_count,
            'is_rumor': True
        })

        # create processed rumor entry
        processed_rumor = {
            'id': rumor_id,
            'event': event_name,
            'raw_text': raw_text,
            'processed_text': processed_text,
            'label': label,
            'reactions': reaction_texts,
            'reaction_count': reaction_count,
            'is_rumor': True,
            'metadata': metadata
        }

        return processed_rumor
    
    def process_non_rumor(self, non_rumor_dir: Path, event_name: str) -> Dict[str, Any]:
        """
        Process a single non-rumor directory.
        
        Args:
            non_rumor_dir (Path): Path to non-rumor directory
            event_name (str): Name of the event this non-rumor belongs to
            
        Returns:
            Dict[str, Any]: Processed non-rumor data
        """
        non_rumor_id = non_rumor_dir.name
        
        # check for source tweets
        source_tweets_dir = non_rumor_dir / "source-tweets"
        if not source_tweets_dir.exists():
            logger.warning(f"No source-tweets directory found for non-rumor {non_rumor_id}")
            return None
        
        # find the source tweet file (there should be only one)
        source_tweet_files = list(source_tweets_dir.glob("*.json"))
        if not source_tweet_files:
            logger.warning(f"No source tweet file found for non-rumor {non_rumor_id}")
            return None
        
        source_tweet_file = source_tweet_files[0]
        source_tweet = self.read_tweet(source_tweet_file)
        
        if not source_tweet:
            logger.warning(f"Empty source tweet for non-rumor {non_rumor_id}")
            return None
        
        # get the text content
        if 'text' in source_tweet:
            raw_text = source_tweet['text']
        else:
            logger.warning(f"No text field in source tweet for non-rumor {non_rumor_id}")
            return None
        
        # process the text
        processed_text = self.preprocess_text(raw_text)
        
        # for non-rumors, we assign a label of "true" since they are factual content
        label = "true"
        
        # count reactions
        reactions_dir = non_rumor_dir / "reactions"
        reaction_count = 0
        reaction_texts = []
        if reactions_dir.exists():
            reaction_files = list(reactions_dir.glob("**/*.json"))
            reaction_count = len(reaction_files)
            
            # Process a sample of reactions (max 10)
            for reaction_file in reaction_files[:10]:
                reaction = self.read_tweet(reaction_file)
                if reaction and 'text' in reaction:
                    reaction_text = self.preprocess_text(reaction['text'])
                    reaction_texts.append(reaction_text)
        
        # extract metadata
        metadata = self.extract_metadata(source_tweet)
        metadata.update({
            'event': event_name,
            'non_rumor_id': non_rumor_id,
            'reaction_count': reaction_count,
            'is_rumor': False  # flag to indicate this is not a rumor
        })
        
        # create processed non-rumor entry
        processed_non_rumor = {
            'id': non_rumor_id,
            'event': event_name,
            'raw_text': raw_text,
            'processed_text': processed_text,
            'label': label,
            'reactions': reaction_texts,
            'reaction_count': reaction_count,
            'is_rumor': False,
            'metadata': metadata
        }
        
        return processed_non_rumor
    
    def extract_metadata(self, tweet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from a tweet.
        
        Args:
            tweet (Dict[str, Any]): Tweet data
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            'user': {},
            'tweet': {}
        }

        # extract user information
        if 'user' in tweet:
            user = tweet['user']
            metadata['user'] = {
                'id': user.get('id_str', ''),
                'screen_name': user.get('screen_name', ''),
                'verified': user.get('verified', False),
                'followers_count': user.get('followers_count', 0),
                'friends_count': user.get('friends_count', 0),
                'statuses_count': user.get('statuses_count', 0),
                'created_at': user.get('created_at', 0)
            }

        # extract tweet information
        metadata['tweet'] = {
            'id': tweet.get('id_str', ''),
            'created_at': tweet.get('created_at', ''),
            'retweet_count': tweet.get('retweet_count', 0),
            'favorite_count': tweet.get('favorite_count', 0),
            'lang': tweet.get('lang', ''),
            'has_media': 'media' in tweet.get('entities', {})
        }

        # extract URLs if available
        urls = []
        if 'entities' in tweet and 'urls' in tweet['entities']:
            for url_obj in tweet['entities']['urls']:
                if 'expanded_url' in url_obj:
                    urls.append(url_obj['expanded_url'])
        metadata['urls'] = urls

        # extract hashtags
        hashtags = []
        if 'entities' in tweet and 'hashtags' in tweet['entities']:
            for hashtag in tweet['entities']['hashtags']:
                if 'text' in hashtag:
                    hashtags.append(hashtag['text'])
        metadata['hashtags'] = hashtags

        return metadata
    
    def process(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process the PHEME dataset.
        
        Args:
            input_dir (Union[str, Path]): Directory with PHEME data
            output_dir (Union[str, Path]): Directory to save processed data
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # find the all-rnr-annotated-threads directory
        annotated_threads_dir = input_dir / "all-rnr-annotated-threads"
        if not annotated_threads_dir.exists():
            logger.error(f"Could not find all-rnr-annotated-threads directory in {input_dir}")
            return
        
        # final all event directories
        event_dirs = [d for d in annotated_threads_dir.iterdir() if d.is_dir()]

        if not event_dirs:
            logger.warning(f"No event directories found in {annotated_threads_dir}")
            return
        
        # process each event
        all_processed_items = []
        for event_dir in event_dirs:
            event_name = event_dir.name
            logger.info(f"Processing event: {event_name}")

            event_items = []

            # process rumors
            rumors_dir = event_dir / "rumors"
            if rumors_dir.exists():
                # find all rumor directories
                rumor_dirs = [d for d in rumors_dir.iterdir() if d.is_dir()]

                # process each rumor
                for rumor_dir in rumor_dirs:
                    processed_rumor = self.process_rumor(rumor_dir, event_name)
                    if processed_rumor:
                        event_items.append(processed_rumor)
                        all_processed_items.append(processed_rumor)

                logger.info(f"Processed {len(rumor_dirs)} rumors from event {event_name}")

            # process non-rumors
            non_rumors_dir = event_dir / "non-rumors"
            if non_rumors_dir.exists():
                # find all non-rumor directories
                non_rumor_dirs = [d for d in non_rumors_dir.iterdir() if d.is_dir()]

                # process each non-rumor
                for non_rumor_dir in non_rumor_dirs:
                    processed_non_rumor = self.process_non_rumor(non_rumor_dir, event_name)
                    if processed_non_rumor:
                        event_items.append(processed_non_rumor)
                        all_processed_items.append(processed_non_rumor)

                logger.info(f"Processed {len(non_rumor_dirs)} non-rumors from event {event_name}")

            # save processed items for this event
            output_file = output_dir / f"{event_name}.json"
            with open(output_file, 'w') as f:
                json.dump(event_items, f, indent=2)

            logger.info(f"Saved {len(event_items)} items from event {event_name} to {output_file}")

        # save combined processed items
        combined_output_file = output_dir / "pheme_combined.json"
        with open(combined_output_file, 'w') as f:
            json.dump(all_processed_items, f, indent=2)

        # log statistics
        rumors_count = sum(1 for item in all_processed_items if item.get('is_rumor', False))
        non_rumors_count = sum(1 for item in all_processed_items if not item.get('is_rumor', True))
        true_count = sum(1 for item in all_processed_items if item.get('label') == 'true')
        false_count = sum(1 for item in all_processed_items if item.get('label') == 'false')
        unverified_count = sum(1 for item in all_processed_items if item.get('label') == 'unverified')

        logger.info(f"Saved combined data with {len(all_processed_items)} to {combined_output_file}")
        logger.info(f"Statistics: {rumors_count} rumors, {non_rumors_count} non-rumors")
        logger.info(f"Label distribution: {true_count} true, {false_count} false, {unverified_count} unverified")

class FakeNewsNetProcessor(TextProcessor):
    """Processor for the FakeNewsNet dataset"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "FakeNewsNet")

    def read_news_content(self, content_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Read a news content.json file.
        
        Args:
            content_file (Union[str, Path]): Path to content.json file
            
        Returns:
            Dict[str, Any]: News content data
        """
        try:
            with open(content_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            return content
        except Exception as e:
            logger.error(f"Error reading news content file {content_file}: {e}")
            return {}
        
    def extract_metadata(self, content: Dict[str, Any], source: str, label: str, article_id: str) -> Dict[str, Any]:
        """
        Extract metadata from news content.
        
        Args:
            content (Dict[str, Any]): News content data
            source (str): News source (gossipcop or politifact)
            label (str): Veracity label (fake or real)
            article_id (str): Article ID
            
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        metadata = {
            'source': source,
            'label': label,
            'article_id': article_id,
            'title': content.get('title', ''),
            'url': content.get('url', ''),
            'date_published': content.get('publish_date', '')
        }

        # extract authors
        authors = content.get('authors', [])
        if isinstance(authors, list):
            metadata['authors'] = authors
        else:
            metadata['authors'] = [str(authors)]

        # eactract image URLs
        image_urls = []
        if 'images' in content and isinstance(content['images'], list):
            image_urls.extend(content['images'])
        if 'image' in content and content['image']:
            image_urls.append(content['image'])
        metadata['image_urls'] = image_urls
        metadata['image_count'] = len(image_urls)

        # additional metadata
        metadata['keywords'] = content.get('keywords', [])
        metadata['description'] = content.get('meta_description', '')

        return metadata
    
    def process_article(self, article_dir: Path, source: str, label: str) -> Dict[str, Any]:
        """
        Process a single article directory.
        
        Args:
            article_dir (Path): Path to article directory
            source (str): News source
            label (str): Veracity label
            
        Returns:
            Dict[str, Any]: Processed article data
        """
        article_id = article_dir.name
        
        # check for content file
        content_file = article_dir / "news content.json"
        if not content_file.exists():
            logger.warning(f"No content file found fo article {article_id}")
            return None
        
        # read the content file
        content = self.read_news_content(content_file)
        if not content:
            logger.warning(f"Empty content for article {article_id}")
            return None
        
        # extract text content
        raw_text = ""
        if 'text' in content and content['text']:
            raw_text = content['text']

        # if no text, try to use title + description
        if not raw_text and 'title' in content:
            raw_text = content['title']
            if 'meta_description' in content:
                raw_text += " " + content['meta_description']

        if not raw_text:
            logger.warning(f"No text content found for article {article_id}")
            return None
        
        # process the text
        processed_text = self.preprocess_text(raw_text)

        # extract metadata
        metadata = self.extract_metadata(content, source, label, article_id)

        # count tweets
        tweets_dir = article_dir / "tweets"
        tweet_count = 0
        if tweets_dir.exists():
            tweet_files = len(tweets_dir.glob("*.json"))
            tweet_count = len(tweet_files)

        # add tweet count to metadata
        metadata['tweet_count'] = tweet_count

        # standardize the label
        standardized_label = 'false' if label == 'fake' else 'true'

        # create processed article entry
        processed_article = {
            'id': f"{source}_{article_id}",
            'source': source,
            'raw_text': raw_text,
            'processed_text': processed_text,
            'original_label': label,
            'label': standardized_label,
            'metadata': metadata
        }

        return processed_article
    
    def process(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process the FakeNewsNet dataset.
        
        Args:
            input_dir (Union[str, Path]): Directory with FakeNewsNet data
            output_dir (Union[str, Path]): Directory to save processed data
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # sources in FakeNewsNet
        sources = ['gossipcop', 'politifact']

        # labels
        labels = ['fake', 'real']

        # process each source
        all_processed_articles = []
        for source in sources:
            source_dir = input_dir / source
            if not source_dir.exists():
                logger.warning(f"Source directory not found: {source_dir}")
                continue

            source_articles = []

            # process each label
            for label in labels:
                label_dir = source_dir / label
                if not label_dir.exists():
                    logger.warning(f"Label directory not found: {label_dir}")
                    continue

                # process articles for this source and label
                label_articles = []

                # find all article directories
                article_dirs = [d for d in label_dir.iterdir() if d.is_dir()]

                for article_dir in article_dirs:
                    processed_article = self.process_article(article_dir, source, label)
                    if processed_article:
                        label_articles.append(processed_article)
                        source_articles.append(processed_article)
                        all_processed_articles.append(processed_article)

                # save processed articles for this label
                output_file = output_dir / f"{source}_{label}.json"
                with open(output_file, 'w') as f:
                    json.dump(label_articles, f, indent=2)

                logger.info(f"Processed {len(label_articles)} articles from {source}/{label} and saved to {output_file}")

            # save all processed files for this source
            source_output_file = output_dir / f"{source}_combined.json"
            with open(source_output_file, 'w') as f:
                json.dump(source_articles, f, indent=2)

            logger.info(f"Saved combined data with {len(source_articles)} articles from {source} to {source_output_file}")

        # save all processed articles
        combined_output_file = output_dir / "fakenewsnet_combined.json"
        with open(combined_output_file, 'w') as f:
            json.dump(all_processed_articles, f, indent=2)

        logger.info(f"Saved combined data with {len(all_processed_articles)} articles to {combined_output_file}")