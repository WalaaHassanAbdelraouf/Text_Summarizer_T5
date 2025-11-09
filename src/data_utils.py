"""
Data loading and preprocessing utilities
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
import config


def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', "", text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text


def load_and_prepare_data(data_path=config.DATA_PATH):
    print("=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Original dataset shape: {data.shape}")
    
    # Check for missing values
    print(f"\nMissing values:\n{data.isnull().sum()}")
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"\nDuplicates found: {duplicates}")
    
    # Drop duplicates
    data.drop_duplicates(inplace=True)
    print(f"After removing duplicates: {data.shape}")
    
    # Sample data
    df = data.sample(n=config.SAMPLE_SIZE, random_state=config.RANDOM_STATE)
    df = df.reset_index(drop=True)
    print(f"Sampled dataset: {df.shape}")
    
    # Display length statistics
    headline_length = df['headlines'].str.len()
    text_length = df['text'].str.len()
    print(f"\nMax Headline Length: {max(headline_length)}")
    print(f"Max Text Length: {max(text_length)}")
    
    # Clean text
    print("\nCleaning text...")
    df['text'] = df['text'].map(clean_text)
    df['headlines'] = df['headlines'].map(clean_text)
    
    # Remove empty or too short samples
    df = df[(df['text'].str.len() > 50) & (df['headlines'].str.len() > 10)]
    print(f"After cleaning: {df.shape}")
    
    return df


def split_data(df):
    print("\n" + "=" * 80)
    print("Splitting Data")
    print("=" * 80)
    
    # Split into train and temp
    train_df, temp_df = train_test_split(
        df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Split temp into validation and test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=config.VAL_SPLIT, 
        random_state=config.RANDOM_STATE
    )
    
    print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
    
    return train_df, val_df, test_df


def create_datasets(train_df, val_df, test_df):
    print("\nConverting to Hugging Face datasets...")
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)
    
    print("✓ Datasets created successfully")
    
    return train_ds, val_ds, test_ds