# Dataset Overview for Fake News Detection Project
                    
This directory contains the raw datasets used in the project. Below is an overview of each dataset and its purpose.
                    
## Text Datasets

### LIAR
- **Description**: 12.8K human-labeled short statements from PolitiFact
- **URL**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **Citation**: Wang, W. Y. (2017). 'Liar, Liar Pants on Fire': A New Benchmark Dataset for Fake News Detection.

### FakeNewsNet
- **Description**: News content from GossipCop and PolitiFact
- **URL**: https://github.com/KaiDMML/FakeNewsNet
- **Citation**: Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media.

### PHEME
- **Description**: Twitter rumors and non-rumors
- **URL**: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
- **Citation**: Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., & Tolmie, P. (2016). Analysing how people orient to and spread rumours in social media by looking at conversational threads.

## Image Datasets
                    
### MediaEval
- **Description**: Images from tweets with verification labels
- **URL**: https://github.com/MKLab-ITI/image-verification-corpus
- **Citation**: Boididou, C., Papadopoulos, S., Zampoglou, M., Apostolidis, L., Papadopoulou, O., & Kompatsiaris, Y. (2018). Detection and visualization of misleading content on Twitter.

### FakeNewsNet-Images
- **Description**: Images associated with news articles
- **URL**: https://github.com/KaiDMML/FakeNewsNet
- **Citation**: Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media.

## Directory Structure
                    
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
