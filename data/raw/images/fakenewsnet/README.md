# FakeNewsNet Image Dataset

## Overview
This directory contains images extracted from news articles in the FakeNewsNet dataset. The images were extracted from the URLs listed in the `news content.json` files of each article.

## Extraction Process
- Images were extracted using the `extract_fakenewsnet_images.py` script
- Images maintain their original filenames from source URLs where possible
- Images that failed to download are logged in `extract_images.log`

## Directory Structure
```
fakenewsnet/
├── gossipcop/
│   ├── fake/
│   │   └── [article_id]/
│   │       └── [image files]
│   └── real/
│       └── [article_id]/
│           └── [image files]
└── politifact/
    ├── fake/
    │   └── [article_id]/
    │       └── [image files]
    └── real/
        └── [article_id]/
            └── [image files]
```

## Image Characteristics
- Format: Mostly JPG and PNG
- Resolution: Varies by source
- Association: Images are grouped by the article they appeared in
- Content: News-related images, including photos, graphs, and occasionally logos or advertisements

## Usage Notes
- Some articles have multiple images, others have none
- Image quality and relevance varies considerably
- Consider image preprocessing (resizing, normalization) before model training
- Be aware that some images may be decorative rather than substantive to the news content

## Citation
```
Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). 
FakeNewsNet: A Data Repository with News Content, Social Context and 
Dynamic Information for Studying Fake News on Social Media.
```