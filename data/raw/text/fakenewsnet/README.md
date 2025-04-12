# FakeNewsNet Dataset

## Setup Instructions

The FakeNewsNet dataset requires running their collection scripts to download the complete dataset, as the repository only provides the minimal version due to Twitter policies and copyright restrictions.

## Steps to Download the Complete Dataset

1. Clone the repository:
   ```
   git clone https://github.com/KaiDMML/FakeNewsNet.git
   cd FakeNewsNet/code
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure Twitter API keys:
   - Get Twitter API keys from [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
   - Update the `tweet_keys_file.json` in the `code/resources` folder with your API keys

4. Modify `config.json` to specify what components to download:
   - News articles
   - Tweets
   - Retweets
   - User profiles
   - User timeline tweets
   - User followers
   - User following

5. Start the key server:
   ```
   nohup python -m resource_server.app &> keys_server.out&
   ```

6. Start data collection:
   ```
   nohup python main.py &> data_collection.out&
   ```

7. After collection, copy or move the downloaded data to our project structure:
   ```
   cp -r FakeNewsNet/dataset/* /path/to/our/project/data/raw/text/fakenewsnet/
   ```

## Dataset Structure After Collection

```
fakenewsnet/
├── gossipcop/
│   ├── fake/
│   │   └── [article_ids]/
│   │       ├── news content.json
│   │       ├── tweets/
│   │       └── retweets/
│   └── real/
│       └── [article_ids]/
├── politifact/
│   ├── fake/
│   │   └── [article_ids]/
│   └── real/
│       └── [article_ids]/
└── user_data/
    └── user_profiles/
      └── [user_ids].json
```

## Note About Images
The FakeNewsNet dataset includes images associated with news articles. After collection, these will be stored within each article's directory in the content.json file as image URLs.

To prepare these for our image analysis:
1. Extract all image URLs from the collected news content
2. Download these images to our `data/raw/images/fakenewsnet` directory

## Citation
Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2018). FakeNewsNet: A Data Repository with News Content, Social Context and Dynamic Information for Studying Fake News on Social Media.