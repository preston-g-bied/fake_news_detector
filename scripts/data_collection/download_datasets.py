# scripts/data_collection/download_datasets.py

"""
Dataset downloader for the Fake News Detection project.
This script downloads and organizes the datasets specified in the config file.
"""

import os
import sys
import yaml
import logging
import argparse
import zipfile
import tarfile
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{project_root}/logs/data_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load the configuration file"""
    config_path = project_root / "config" / "config.yaml"
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)

def create_directories(config):
    """Create necessary directories for data."""
    try:
        # create data directories
        for path_type in ["raw", "processed", "external"]:
            path = project_root / config["paths"]["data"][path_type]
            for subdir in ["text", "images", "metadata"]:
                os.makedirs(path / subdir, exist_ok=True)
                # create .gitkeep file to include empty directories in git
                with open(path / subdir / ".gitkeep", "w") as f:
                    pass

        # create logs directory
        os.makedirs(project_root / config["paths"]["logs"], exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        sys.exit(1)

def download_file(url, destination):
    """
    Download a file from URL to the specified destination with progress bar.
    
    Args:
        url (str): URL to download from
        destination (Path): Destination path to save the file
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024   # 1 KB

        with open(destination, "wb") as file, tqdm(
            desc=f"Downloading {destination.name}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)

        if total_size != 0 and bar.n != total_size:
            logger.warning("Downloaded size does not match expected size")

        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while downloading {url}: {e}")
        return False
    
def extract_archive(archive_path, extract_dir):
    """
    Extract a zip or tar archive to the specified directory.
    
    Args:
        archive_path (Path): Path to the archive file
        extract_dir (Path): Directory to extract to
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        logger.info(f"Extracting {archive_path.name}")

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc=f"Extracting {archive_path.name}"):
                    zip_ref.extract(member, extract_dir)

        elif archive_path.suffix in [".tar", ".gz", ".bz2", ".xz"]:
            with tarfile.open(archive_path, "r:*") as tar_ref:
                members = tar_ref.getmembers()
                for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                    tar_ref.extract(member, extract_dir)

        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return False
    
def download_liar_dataset(config):
    """Download and extract the LIAR dataset."""
    try:
        dataset_config = next(d for d in config["data"]["text_datasets"] if d["name"] == "LIAR")
        url = dataset_config["url"]
        destination_dir = project_root / config["paths"]["data"]["raw"] / "text" / "liar"
        os.makedirs(destination_dir, exist_ok=True)

        # download zip file
        zip_path = destination_dir / "liar_dataset.zip"
        if not download_file(url, zip_path):
            return False
        
        # extract the archive
        if not extract_archive(zip_path, destination_dir):
            return False
        
        logger.info(f"LIAR dataset downloaded and extracted to {destination_dir}")
        return True
    except Exception as e:
        logger.error(f"Error processing LIAR dataset: {e}")
        return False
    
def download_pheme_dataset(config):
    """Create instructions for downloading PHEME dataset."""
    try:
        dataset_config = next(d for d in config["data"]["text_datasets"] if d["name"] == "PHEME")
        url = dataset_config["url"]
        destination_dir = project_root / config["paths"]["data"]["raw"] / "text" / "pheme"
        os.makedirs(destination_dir, exist_ok=True)

        # create a README that explains how to download this dataset manually
        # (since it requires a registration on figshare)
        readme_path = destination_dir / "download_instructions.md"
        with open(readme_path, "w") as f:
            f.write(f"""# PHEME Dataset

This dataset requires manual download from: {url}

## Steps to download:
1. Visit the URL above
2. Register or log in to figshare
3. Download the dataset
4. Extract the downloaded file to this directory

## Citation
{dataset_config["citation"]}
""")
        logger.info(f"PHEME dataset README created at {readme_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing PHEME dataset: {e}")
        return False
    
def download_fakenewsnet_dataset(config):
    """Create instructions for downloading FakeNewsNet dataset."""
    try:
        dataset_config = next(d for d in config["data"]["text_datasets"] if d["name"] == "FakeNewsNet")
        url = dataset_config["url"]
        destination_dir = project_root / config["paths"]["data"]["raw"] / "text" / "fakenewsnet"
        os.makedirs(destination_dir, exist_ok=True)

        # create a README with instructions
        # (since this dataset requires specific tools to download)
        readme_path = destination_dir / "download_instructions.md"
        with open(readme_path, "w") as f:
            f.write(f"""# FakeNewsNet Dataset

This dataset requires using the FakeNewsNet tools for downloading.
                    
## Steps to download:
1. Clone the repository: `git clone {url}`
2. Follow the instructions in the repository README to download the dataset
3. Copy or move the downloaded data to this directory

## Citation
{dataset_config["citation"]}

## Note
This dataset includes both text and images for news articles.
After downloading, you may want to organize the images into the images directory.
""")
            
        logger.info(f"FakeNewsNet dataset README created at {readme_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing FakeNewsNet dataset: {e}")
        return False
    
def download_mediaeval_dataset(config):
    """Create instructions for download MediaEval dataset."""
    try:
        dataset_config = next(d for d in config["data"]["image_datasets"] if d["name"] == "MediaEval")
        url = dataset_config["url"]
        destination_dir = project_root / config["paths"]["data"]["raw"] / "images" / "mediaeval"
        os.makedirs(destination_dir, exist_ok=True)

        # create a README with instructions
        readme_path = destination_dir / "download_instructions.md"
        with open(readme_path, "w") as f:
            f.write(f"""# MediaEval Verifying Multimedia Use Dataset

This dataset requires manual downloading and processing.
                    
## Steps to download:
1. Visit the repository: {url}
2. Follow the instructions to download the dataset
3. Extract the downloaded data to this directory

## Citation
{dataset_config["citation"]}
""")
        logger.info(f"MediaEval dataset README created at {readme_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing MediaEval dataset: {e}")
        return False
    
def create_dataset_overview(config):
    """Create a README in the data directory explaining all datasets."""
    try:
        data_dir = project_root / config["paths"]["data"]["raw"]
        readme_path = data_dir / "README.md"

        with open(readme_path, "w") as f:
            f.write("""# Dataset Overview for Fake News Detection Project
                    
This directory contains the raw datasets used in the project. Below is an overview of each dataset and its purpose.
                    
## Text Datasets

""")
            for dataset in config["data"]["text_datasets"]:
                f.write(f"""### {dataset["name"]}
- **Description**: {dataset["description"]}
- **URL**: {dataset["url"]}
- **Citation**: {dataset["citation"]}

""")
                
            f.write("""## Image Datasets
                    
""")
            for dataset in config["data"]["image_datasets"]:
                f.write(f"""### {dataset["name"]}
- **Description**: {dataset["description"]}
- **URL**: {dataset["url"]}
- **Citation**: {dataset["citation"]}

""")
            
            f.write("""## Directory Structure
                    
```
data/
├── raw/            # Original, immutable data
│   ├── text/       # Text datasets
│   ├── images/     # Image datasets
│   └── metadata/   # Metadata information
├── processed/      # Cleaned and processed data
└── external/       # External data sources
```

## Data Processing

See the Jupyter notebooks in the `notebooks/exploratory` directory for data exploration and analysis.                    
""")
        logger.info(f"Dataset overview README created at {readme_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating dataset overview: {e}")
        return False
    
def main():
    """Main function to download and organize all datasets."""
    parser = argparse.ArgumentParser(description='Download datasets for Fake News Detection project')
    parser.add_argument('--datasets', nargs='+', choices=['all', 'liar', 'pheme', 'fakenewsnet', 'mediaeval'],
                        default=['all'], help='Specify which datasets to download')
    args = parser.parse_args()

    logger.info("Starting dataset download process")

    # load configuration
    config = load_config()

    # create necessary directories
    create_directories(config)

    datasets_to_download = args.datasets
    all_datasets = datasets_to_download == ['all']

    # download datasets
    results = []

    if all_datasets or 'liar' in datasets_to_download:
        results.append(('LIAR', download_liar_dataset(config)))

    if all_datasets or 'pheme' in datasets_to_download:
        results.append(('PHEME', download_pheme_dataset(config)))

    if all_datasets or 'fakenewsnet' in datasets_to_download:
        results.append(('FakeNewsNet', download_fakenewsnet_dataset(config)))

    if all_datasets or 'mediaeval' in datasets_to_download:
        results.append(('MediaEval', download_mediaeval_dataset(config)))

    # create dataset overview
    create_dataset_overview(config)

    # print summary
    logger.info("\n===== Download Summary =====")
    for name, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{name}: {status}")

    if all(success for _, success in results):
        logger.info("\nAll datasets processed successfully!")
    else:
        logger.warning("\nSome datasets could not be processed. Check the log for details.")

    logger.info("\nNext steps: Explore the data and start preprocessing.")

if __name__ == "__main__":
    main()