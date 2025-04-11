# MediaEval Verifying Multimedia Use Dataset

## Dataset Overview
This dataset contains images from tweets with verification labels for multimedia verification.

## Files
- `set_images.txt`: Contains fake and real images verified by online sources. Fields include:
  - `image_id`: Reference ID for each image
  - `image_url`: Online URL of the image
  - `annotation`: Veracity of the image (fake/real)
  - `event`: The event the image is associated with

- `tweets_images.txt`: Contains tweets and their associated images. Fields include:
  - `tweet_id`: ID of each tweet
  - `image_id`: Reference ID of the associated image
  - `annotation`: Veracity of each tweet
  - `event`: The event the tweet is associated with

- `tweets_images_update.txt`: Contains only pure fake tweets, with funny content or self-declared fake content removed.

- `tweets_event.txt`: Contains tweets with fake content that are no longer available online.

## Processing Steps

To prepare this dataset for our project:

1. Create a script to download the actual images from the URLs in `set_images.txt`:
   ```
   python scripts/data_collection/download_mediaeval_images.py
   ```

2. Create mappings between tweets and images for our analysis:
   ```
   python scripts/preprocessing/prepare_mediaeval_data.py
   ```

3. Organize the downloaded images into fake/real directories for easier processing.

## Recommended Structure After Processing

```
mediaeval/
├── raw/
│   ├── set_images.txt
│   ├── tweets_images.txt
│   ├── tweets_images_update.txt
│   └── tweets_event.txt
├── images/
│   ├── fake/
│   │   └── [image_ids].jpg
│   └── real/
│       └── [image_ids].jpg
└── metadata/
    └── tweet_image_mappings.csv
```

## Citation
Boididou, C., Papadopoulos, S., Zampoglou, M., Apostolidis, L., Papadopoulou, O., & Kompatsiaris, Y. (2018). Detection and visualization of misleading content on Twitter.