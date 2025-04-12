# LIAR Dataset

## Dataset Overview
The LIAR dataset is a benchmark collection for fake news detection, consisting of 12,800 human-labeled short statements from PolitiFact. Each statement is evaluated for its truthfulness and assigned one of six labels ranging from "pants-fire" (completely false) to "true".

## Files
- `train.tsv`: Training dataset (10,269 statements)
- `valid.tsv`: Validation dataset (1,284 statements)
- `test.tsv`: Testing dataset (1,283 statements)
- `README`: Original dataset documentation (kept for reference)

## Dataset Structure
Each file is in TSV (Tab-Separated Values) format with the following columns:

1. **ID**: The ID of the statement (`[ID].json`)
2. **Label**: Truthfulness rating, one of:
   - `pants-fire` (completely false)
   - `false`
   - `barely-true`
   - `half-true`
   - `mostly-true`
   - `true`
3. **Statement**: The text of the claim being verified
4. **Subject**: The subject(s) of the statement
5. **Speaker**: The person who made the statement
6. **Speaker's Job**: The speaker's job title
7. **State Info**: State information
8. **Party**: The speaker's party affiliation
9. **Barely True Counts**: Historical count of barely true statements by this speaker
10. **False Counts**: Historical count of false statements by this speaker
11. **Half True Counts**: Historical count of half true statements by this speaker
12. **Mostly True Counts**: Historical count of mostly true statements by this speaker
13. **Pants on Fire Counts**: Historical count of pants on fire statements by this speaker
14. **Context**: Where or when the statement was made

## Usage Notes

### Preprocessing Considerations
- For binary classification, consider grouping labels:
  - True labels: "true", "mostly-true"
  - False labels: "false", "pants-fire"
  - Ambiguous (may exclude or use as third class): "barely-true", "half-true"
- The dataset includes metadata that may be useful for feature engineering:
  - Speaker's historical reliability (columns 9-13)
  - Context and subject information
  - Party affiliation

### Accessing Full Verdict Reports
The full-text verdict reports are not included in this dataset but can be accessed via the PolitiFact API:
```
wget http://www.politifact.com/api/v/2/statement/[ID]/?format=json
```

## Processing Steps
1. Extract the downloaded ZIP file: `unzip liar_dataset.zip`
2. The extracted files (`train.tsv`, `valid.tsv`, `test.tsv`) are ready to use

## Legal Notice
The original sources retain the copyright of the data. This dataset is provided "as is" with no guarantees. You are allowed to use this dataset for research purposes only.

## Citation
William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.