import os
import pandas as pd
import json
import numpy as np
from DataPaths import (
    path_train_key, path_dev_key, path_test_key,
    twitter_trainingDev_data_path, twitter_test_data_path
)

class TwitterDataProcessor:
    def __init__(self, dataset_type, key_path, data_path):
        self.dataset_type = dataset_type
        self.key_path = key_path
        self.data_path = data_path
        self.key_df = pd.read_json(key_path)
        self.src_dirs_sorted = []
        self.src_posts_df = None
        self.replies_df = None

    def process_key_data(self):
        key_taska_df = self.key_df['subtaskaenglish'].dropna().reset_index()
        key_taska_df.columns = ['id', 'label']

        # Selecting Twitter data based on dataset type
        if self.dataset_type == 'train':
            return key_taska_df.iloc[:4519]
        elif self.dataset_type == 'dev':
            return key_taska_df.iloc[:1049]
        elif self.dataset_type == 'test':
            return key_taska_df.iloc[:1066]

    def process_source_posts(self):
        self.src_dirs_sorted = sorted(next(os.walk(self.data_path))[1])
        src_posts = []

        for directory in self.src_dirs_sorted:
            for subdir in next(os.walk(f"{self.data_path}/{directory}"))[1]:
                tweet_src_path = f"{self.data_path}/{directory}/{subdir}/source-tweet"
                paths = f"{tweet_src_path}/{subdir}.json"

                with open(paths) as f:
                    data = json.load(f)
                    src_posts.append({
                        'text': data['text'],
                        'id': data['id'],
                        'inre': data['in_reply_to_status_id']
                    })

        self.src_posts_df = pd.DataFrame(src_posts)

    def process_reply_posts(self):
        replies = []

        for directory in self.src_dirs_sorted:
            for subdir in next(os.walk(f"{self.data_path}/{directory}"))[1]:
                tweet_src_path = f"{self.data_path}/{directory}/{subdir}/replies"
                files = next(os.walk(tweet_src_path))[2]

                for file in files:
                    paths = f"{tweet_src_path}/{file}"

                    with open(paths) as f:
                        data = json.load(f)
                        if 'text' in data:
                            replies.append({
                                'text': data['text'],
                                'id': data['id'],
                                'inre': str(data['in_reply_to_status_id']),
                                'source': directory
                            })

        self.replies_df = pd.DataFrame(replies)

    def clean_data(self):
        twitter_data = pd.concat([self.src_posts_df, self.replies_df])
        twitter_data['id'] = twitter_data['id'].astype(str).str.strip()
        twitter_data['inre'] = twitter_data['inre'].astype(str).str.strip()
        return twitter_data

    def merge_with_keys(self, clean_df, key_df):
        return pd.merge(clean_df, key_df, on="id")

    def create_final_dataset(self, merged_df):
        twitter_df = merged_df[['id', 'text']].rename(columns={'id': 'inre', 'text': 'inreText'})
        twitter_df1 = merged_df[['id', 'text']].rename(columns={'id': 'source', 'text': 'sourceText'})

        dataset = pd.merge(merged_df, twitter_df, how='left', on="inre")
        dataset = pd.merge(dataset, twitter_df1, how='left', on="source")
        return dataset

    def remove_redundant_data(self, df):
        df.loc[df['inre_x'] == df['source_x'], 'sourceText'] = np.nan
        return df

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False)

    def process(self):
        # Step-by-step processing
        key_df = self.process_key_data()
        self.process_source_posts()
        self.process_reply_posts()
        clean_df = self.clean_data()
        merged_df = self.merge_with_keys(clean_df, key_df)
        final_df = self.create_final_dataset(merged_df)
        final_df = self.remove_redundant_data(final_df)
        
        # Select columns to retain
        final_columns = ['text_x', 'id', 'inre_x', 'source_x', 'label_x', 'inreText', 'sourceText']
        final_df = final_df[final_columns]
        
        # Save final dataset to CSV
        self.save_to_csv(final_df, f'Twitter{self.dataset_type.capitalize()}DataSrc.csv')

# Processing train, dev, and test datasets
train_processor = TwitterDataProcessor('train', path_train_key, twitter_trainingDev_data_path)
dev_processor = TwitterDataProcessor('dev', path_dev_key, twitter_trainingDev_data_path)
test_processor = TwitterDataProcessor('test', path_test_key, twitter_test_data_path)

train_processor.process()
dev_processor.process()
test_processor.process()
