import os
import sys
import argparse
import torch
import random
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.CryptoFeatureEngineer import CryptoFeatureEngineer
from src.data.StablecoinFilter import StablecoinFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_LSTM_model')

class CryptoDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=30):
        self.X, self.y = [], []
        for coin_id in df['coin_id'].unique():
            coin_df = df[df['coin_id'] == coin_id]
            if len(coin_df) < seq_len:
                continue
            for i in range(seq_len, len(coin_df)):
                self.X.append(coin_df[feature_cols].iloc[i - seq_len:i].values)
                self.y.append(coin_df['target'].iloc[i])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return all_preds, all_labels

def train(args):
    # Filter stablecoins
    coin_files = [f for f in os.listdir(args.data_dir) if f.startswith('coin_') and f.endswith('.csv')]
    stablecoin_filter = StablecoinFilter(logger=logger)
    _, non_stablecoin_paths = stablecoin_filter.filter_stablecoins(args.data_dir)
    non_stablecoin_files = [os.path.basename(p) for p in non_stablecoin_paths]
    coin_files = [f for f in coin_files if f in non_stablecoin_files]
    logger.info(f"Processing {len(coin_files)} non-stablecoins")

    # Load data and filter coins with insufficient length
    dfs = []
    for f in coin_files:
        file_path = os.path.join(args.data_dir, f)
        coin_df = pd.read_csv(file_path)
        coin_id = f.replace('coin_', '').replace('.csv', '')
        coin_df['coin_id'] = coin_id
        # Pre-filter: Ensure enough data for seq_len + days_ahead
        if len(coin_df) >= args.seq_len + args.days_ahead:
            dfs.append(coin_df)
    if not dfs:
        raise ValueError("No coins with sufficient data after filtering")
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['coin_id', 'date'])

    # Create target
    df['future_price'] = df.groupby('coin_id')['price'].shift(-args.days_ahead)
    df['price_change_pct'] = (df['future_price'] - df['price']) / df['price']
    df['target'] = (df['price_change_pct'] > args.threshold).astype(int)
    df = df.dropna(subset=['target'])  # Remove rows where target couldn't be calculated

    # Secondary filter: Remove coins that lost all data after target creation
    valid_coins = df['coin_id'].unique()
    if len(valid_coins) == 0:
        raise ValueError("No valid coins remaining after target creation")
    logger.info(f"Valid coins after processing: {len(valid_coins)}")

    # Split into train/test coins
    random.seed(42)
    train_coin_ids = random.sample(list(valid_coins), int(len(valid_coins) * (1 - args.test_size)))
    test_coin_ids = [cid for cid in valid_coins if cid not in train_coin_ids]
    
    train_df = df[df['coin_id'].isin(train_coin_ids)]
    test_df = df[df['coin_id'].isin(test_coin_ids)]
    
    # Validate splits
    if len(train_df) == 0:
        raise ValueError("Empty training DataFrame after split")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # After feature engineering
    engineer = CryptoFeatureEngineer()
    train_df = engineer.add_all_features(train_df, drop_na=False)  # Don't drop NaNs yet
    test_df = engineer.add_all_features(test_df, drop_na=False)

    feature_cols = engineer.get_selected_features(train_df, args.feature_set)

    # Critical validation checks
    # Check if features exist
    missing_features = [col for col in feature_cols if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}. Check feature engineering.")

    # Check for all-NA features
    na_counts = train_df[feature_cols].isna().sum()
    all_na_features = na_counts[na_counts == len(train_df)].index.tolist()
    if all_na_features:
        raise ValueError(f"Features completely NaN: {all_na_features}")

    # Fill remaining NaNs and infinity values
    train_df[feature_cols] = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Final empty check
    if len(train_df[feature_cols]) == 0:
        raise ValueError("Final feature matrix is empty after NaN handling")

    # Now proceed with scaling
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_save_dir', type=str, default='models/lstm')
    parser.add_argument('--feature_set', type=str, default='standard')
    parser.add_argument('--days_ahead', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)