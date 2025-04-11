# PHEME Dataset

## Dataset Overview
The PHEME dataset contains Twitter rumors and non-rumors for detecting misinformation. The dataset focuses on rumor detection and verification, featuring conversations around rumors that have been annotated for veracity.

## Files
- `PHEME_veracity.tar.bz2`: The compressed dataset archive
- `convert_veracity_annotations.py`: Script to process the veracity annotations

## Processing Steps
1. Extract the archive: `tar -xjf PHEME_veracity.tar.bz2`
2. Run the conversion script: `python convert_veracity_annotations.py`

## Dataset Structure (after extraction)
The dataset is organized by events, with each event containing rumorous and non-rumorous content:

```
pheme/
├── all-rnr-annotated-threads/
├──── event1/
│   ├──── rumours/
│   │   └──── [rumor_ids]/
│   │       ├──── source-tweets/
│   │       ├──── reactions/
│   │       └──── annotation.json
│   └──── non-rumours/
│       └──── [non_rumor_ids]/
├──── event2/
...
```

## Citation
Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., & Tolmie, P. (2016). Analysing how people orient to and spread rumours in social media by looking at conversational threads.