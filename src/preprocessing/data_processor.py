"""
Data preprocessing module for breast cancer diagnosis project.
"""

import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import urllib.request
import sys
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Add project root to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import (WISCONSIN_DATASET_URL, WISCONSIN_DATASET_PATH, 
                        WISCONSIN_COLUMNS, PROCESSED_DATA_DIR, 
                        RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
                        RESULTS_DIR)

def download_wisconsin_dataset():
    """
    Download the Wisconsin Breast Cancer dataset if not already present.
    """
    if not os.path.exists(WISCONSIN_DATASET_PATH):
        print(f"Downloading Wisconsin Breast Cancer dataset to {WISCONSIN_DATASET_PATH}")
        os.makedirs(os.path.dirname(WISCONSIN_DATASET_PATH), exist_ok=True)
        urllib.request.urlretrieve(WISCONSIN_DATASET_URL, WISCONSIN_DATASET_PATH)
        print("Download complete.")
    else:
        print(f"Wisconsin Breast Cancer dataset already exists at {WISCONSIN_DATASET_PATH}")

def load_wisconsin_dataset():
    """
    Load the Wisconsin Breast Cancer dataset.
    
    Returns:
        DataFrame: The loaded dataset
    """
    # Download dataset if not exists
    download_wisconsin_dataset()
    
    # Load dataset
    df = pd.read_csv(WISCONSIN_DATASET_PATH, header=None, names=WISCONSIN_COLUMNS)
    
    # Convert diagnosis to binary (M = 1 (malignant), B = 0 (benign))
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    print(f"Loaded Wisconsin dataset with {df.shape[0]} samples and {df.shape[1]} features.")
    return df

def preprocess_wisconsin_dataset(df):
    """
    Preprocess the Wisconsin Breast Cancer dataset.
    
    Args:
        df (DataFrame): The dataset to preprocess
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # Separate features and target
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    
    # Split data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Further split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data split: Training={X_train.shape[0]}, Validation={X_val.shape[0]}, Test={X_test.shape[0]}")
    
    # Save preprocessed data
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train.values)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val.values)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test.values)
    
    # Save feature names for later interpretability
    pd.Series(X.columns).to_csv(os.path.join(PROCESSED_DATA_DIR, 'feature_names.csv'), index=False)

    # Also save the scaler for future use
    import pickle
    with open(os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def load_processed_data():
    """
    Load preprocessed data.
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_feature_names():
    """
    Get feature names for interpretability.
    
    Returns:
        list: Feature names
    """
    return pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'feature_names.csv'))['0'].tolist()

def explore_data(df):
    """
    Explore the dataset and create visualizations.
    
    Args:
        df (DataFrame): The dataset to explore
    """
    # Create directory for EDA visualizations
    eda_dir = os.path.join(RESULTS_DIR, 'eda')
    os.makedirs(eda_dir, exist_ok=True)
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='diagnosis', data=df)
    plt.title('Class Distribution (0=Benign, 1=Malignant)')
    plt.savefig(os.path.join(eda_dir, 'class_distribution.png'))
    
    # Feature distributions
    feature_df = df.drop(['id', 'diagnosis'], axis=1)
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(feature_df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(eda_dir, 'feature_correlation.png'))
    
    # Distribution of a few important features by diagnosis
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_df.columns[:9]):  # Use first 9 features for visualization
        plt.subplot(3, 3, i+1)
        sns.boxplot(x='diagnosis', y=feature, data=df)
        plt.title(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'feature_distributions_by_class.png'))
    
    # Pair plot of select features
    plt.figure(figsize=(15, 15))
    selected_features = feature_df.columns[:5]  # Use first 5 features
    pair_df = df[list(selected_features) + ['diagnosis']]
    sns.pairplot(pair_df, hue='diagnosis')
    plt.savefig(os.path.join(eda_dir, 'feature_pairplot.png'))
    
    print(f"Saved exploratory data analysis plots to {eda_dir}")

if __name__ == "__main__":
    # Process the Wisconsin Breast Cancer dataset
    df = load_wisconsin_dataset()
    
    # Explore data
    explore_data(df)
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_wisconsin_dataset(df)
    print("\nPreprocessing complete. Files saved to:", PROCESSED_DATA_DIR)




