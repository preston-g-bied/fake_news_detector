# PHEME Dataset

## Dataset Overview
The PHEME dataset contains Twitter rumors and non-rumors for detecting misinformation. The dataset focuses on rumor detection and verification, featuring conversations around rumors that have been annotated for veracity.

## Files
- `PHEME_veracity.tar.bz2`: The compressed dataset archive (after downloading from Figshare)
- `convert_veracity_annotations.py`: Script to process the veracity annotations

## Processing Steps
1. Extract the archive: `tar -xjf PHEME_veracity.tar.bz2`
2. This will create the `all-rnr-annotated-threads` directory containing the dataset
3. Process the veracity annotations using the conversion script

## Using the Conversion Script
The `convert_veracity_annotations.py` file contains a function to convert the annotations in the dataset into standardized labels:

```python
def convert_annotations(annotation, string = True):
    """
    Converts the raw annotation dictionary into standardized labels:
    - "true": The rumor is confirmed to be true
    - "false": The rumor is confirmed to be false (misinformation)
    - "unverified": The rumor couldn't be verified conclusively
    
    Parameters:
    annotation (dict): Dictionary with 'misinformation' and 'true' keys
    string (bool): If True, return string labels; if False, return numeric labels
                  (0 = false, 1 = true, 2 = unverified)
    
    Returns:
    str or int: The standardized label
    """
    # Function implementation...
```

To use this function, you would typically:

1. Load the annotation JSON from a rumor's annotation.json file
2. Pass the annotation to the function
3. Get back a standardized label ("true", "false", or "unverified")

Example usage:
```python
import json
from convert_veracity_annotations import convert_annotations

# Load an annotation file
with open("all-rnr-annotated-threads/charliehebdo/rumours/552784600502915073/annotation.json", "r") as f:
    annotation = json.load(f)

# Convert to standard label
label = convert_annotations(annotation)
print(f"This rumor is: {label}")
```

## Dataset Structure
The dataset is organized by events, with each event containing rumorous and non-rumorous content:

```
all-rnr-annotated-threads/
├── charliehebdo/
│   ├── rumours/
│   │   └── [rumor_ids]/
│   │       ├── source-tweets/
│   │       ├── reactions/
│   │       └── annotation.json
│   └── non-rumours/
│       └── [non_rumor_ids]/
├── ferguson/
...and other events
```

## Creating Standardized Labels
To process the entire dataset and create standardized labels for all rumors:

```python
import os
import json
from pathlib import Path
from convert_veracity_annotations import convert_annotations

# Path to the dataset
dataset_path = Path("all-rnr-annotated-threads")

# Events in the dataset
events = [d for d in dataset_path.iterdir() if d.is_dir()]

# Process each event
for event in events:
    rumors_path = event / "rumours"
    if not rumors_path.exists():
        continue
        
    # Process each rumor
    rumors = [d for d in rumors_path.iterdir() if d.is_dir()]
    for rumor in rumors:
        annotation_file = rumor / "annotation.json"
        if annotation_file.exists():
            with open(annotation_file, "r") as f:
                try:
                    annotation = json.load(f)
                    label = convert_annotations(annotation)
                    print(f"{event.name}/{rumor.name}: {label}")
                except json.JSONDecodeError:
                    print(f"Error parsing {annotation_file}")
```

## Citation
Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., & Tolmie, P. (2016). Analysing how people orient to and spread rumours in social media by looking at conversational threads.