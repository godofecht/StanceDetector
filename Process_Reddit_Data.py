import os
import pandas as pd
import json
import numpy as np
from DataPaths import (
    path_train_key, path_dev_key, path_test_key,
    reddit_train_data_path, reddit_dev_data_path, reddit_test_data_path
)

class RedditDataProcessor:
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

        # Selecting Reddit data based on dataset type
        if self.dataset_type == 'train':
            return key_taska_df.iloc[4519:]
        elif self.dataset_type == 'dev':
            return key_taska_df.iloc[1049:]
        elif self.dataset_type == 'test':
            return key_taska_df.iloc[1066:]

    def process_source_posts(self):
        self.src_dirs_sorted = sorted(next(os.walk(self.data_path))[1])
        src_posts = []

        for directory in self.src_dirs_sorted:
            path = f"{self.data_path}/{directory}/source-tweet"
            files = sorted(next(os.walk(path))[2])

            for file in files:
                with open(f"{path}/{file}") as f:
                    data = json.load(f)['data']['children'][0]['data']
                    src_posts.append({
                        'text': data['title'],
                        'id': data['id'],
                        'inre': 'None'
                    })

        self.src_posts_df = pd.DataFrame(src_posts)

    def process_reply_posts(self):
        replies = []

        for directory in self.src_dirs_sorted:
            path = f"{self.data_path}/{directory}/replies"
            files = next(os.walk(path))[2]

            for file in files:
                with open(f"{path}/{file}") as f:
                    data = json.load(f)['data']
                    if 'body' in data:
                        replies.append({
                            'text': data['body'],
                            'id': data['id'],
                            'inre': data['parent_id'].split('_')[1],
                            'source': directory
                        })

        self.replies_df = pd.DataFrame(replies)

    def clean_data(self):
        reddit_data = pd.concat([self.src_posts_df, self.replies_df])
        reddit_data['id'] = reddit_data['id'].str.strip()
        reddit_data['inre'] = reddit_data['inre'].str.strip()
        return reddit_data

    def merge_with_keys(self, clean_df, key_df):
        return pd.merge(clean_df, key_df, on="id")

    def create_final_dataset(self, merged_df):
        reddit_df = merged_df[['id', 'text']].rename(columns={'id': 'inre', 'text': 'inreText'})
        reddit_df1 = merged_df[['id', 'text']].rename(columns={'id': 'source', 'text': 'sourceText'})

        dataset = pd.merge(merged_df, reddit_df, how='left', on="inre")
        dataset = pd.merge(dataset, reddit_df1, how='left', on="source")
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
        self.save_to_csv(final_df, f'Reddit{self.dataset_type.capitalize()}DataSrc.csv')

# Processing train, dev, and test datasets
train_processor = RedditDataProcessor('train', path_train_key, reddit_train_data_path)
dev_processor = RedditDataProcessor('dev', path_dev_key, reddit_dev_data_path)
test_processor = RedditDataProcessor('test', path_test_key, reddit_test_data_path)

train_processor.process()
dev_processor.process()
test_processor.process()
