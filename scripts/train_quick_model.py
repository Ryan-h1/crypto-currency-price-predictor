import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import random
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import logging

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import StablecoinFilter
from src import StablecoinFilter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a quick cryptocurrency price prediction model')

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing cryptocurrency data CSV files')

    parser.add_argument('--model_save_dir', type=str, default='models/quick',
                        help='Directory to save trained models')

    parser.add_argument('--days_ahead', type=int, default=30,
                        help='Number of days to look ahead for prediction')

    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Price increase threshold (e.g., 0.15 = 15%)')

    parser.add_argument('--model_type', type=str, default='logistic',
                        choices=['random_forest', 'logistic'],
                        help='Type of model to train')

    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of coins to use for testing (default: 0.1)')

    parser.add_argument('--max_coins', type=int, default=100,
                        help='Maximum number of coins to use (for faster training)')

    parser.add_argument('--selection', type=str, default='train-leader-test-follower',
                        choices=['random', 'train-leader-test-follower'],
                        help='Method to select train/test coins: random or train-leader-test-follower')

    parser.add_argument('--filter_stablecoins', type=bool, default=True,
                        help='Whether to filter out stablecoins from the dataset')

    return parser.parse_args()


def load_and_process_coin(file_path, days_ahead, threshold):
    """Load and process a single coin's data."""
    try:
        # Load data
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if len(df) < 60:  # Skip coins with insufficient data
            return None

        # Add target variable
        df['future_price'] = df['price'].shift(-days_ahead)
        df['price_change_pct'] = (df['future_price'] - df['price']) / df['price']
        df['target'] = (df['price_change_pct'] > threshold).astype(int)

        # Add minimal features (only the most reliable ones)

        # Price moving averages (only two)
        df['ma7'] = df['price'].rolling(window=7).mean()
        df['ma30'] = df['price'].rolling(window=30).mean()

        # Price to moving average ratios - with safety checks
        df['price_to_ma7'] = np.where(df['ma7'] > 0, df['price'] / df['ma7'], 1.0)
        df['price_to_ma30'] = np.where(df['ma30'] > 0, df['price'] / df['ma30'], 1.0)

        # Simple returns
        df['weekly_return'] = df['price'].pct_change(7)

        # Simple volume features - with safety checks
        volume_ma7 = df['volume'].rolling(window=7).mean()
        df['volume_to_ma7'] = np.where(volume_ma7 > 0, df['volume'] / volume_ma7, 1.0)

        # Clean up data - replace inf/NaN
        for col in df.columns:
            if col not in ['date', 'price', 'market_cap', 'volume', 'coin_id', 'future_price', 'target']:
                # First replace infinities
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

                # Then handle NaNs
                non_nan_values = df[col].dropna()
                if len(non_nan_values) > 0:  # Only calculate median if we have non-NaN values
                    median_val = non_nan_values.median()
                    df[col] = df[col].fillna(median_val)
                else:
                    # If all values are NaN, fill with 0
                    df[col] = df[col].fillna(0)

        # Remove rows with NaN in critical columns
        df = df.dropna(
            subset=['target', 'ma7', 'ma30', 'price_to_ma7', 'price_to_ma30', 'weekly_return', 'volume_to_ma7'])

        if len(df) < 30:  # Skip if too many rows were removed
            return None

        # Add coin identifier from filename
        coin_id = os.path.basename(file_path).replace('coin_', '').replace('.csv', '')
        df['coin_id'] = coin_id

        # Calculate coin statistics for sorting (with NaN protection)
        # Use .mean(skipna=True) to handle any remaining NaNs
        stats = {
            'coin_id': coin_id,
            'avg_market_cap': df['market_cap'].replace([np.inf, -np.inf], np.nan).mean(skipna=True),
            'avg_volume': df['volume'].replace([np.inf, -np.inf], np.nan).mean(skipna=True),
            'avg_price': df['price'].replace([np.inf, -np.inf], np.nan).mean(skipna=True),
            'positive_rate': df['target'].mean(skipna=True),
            'rows': len(df)
        }

        # Make sure we don't have NaN in stats
        for key in stats:
            if key != 'coin_id' and (pd.isna(stats[key]) or np.isinf(stats[key])):
                stats[key] = 0.0

        return df, stats

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def create_leader_follower_split(coin_stats, test_size):
    """Create a split where test coins are randomly selected only from the bottom half (followers) 
    by market cap, and all remaining coins are used for training."""
    # Sort coins by market cap
    sorted_coins = sorted(coin_stats, key=lambda x: x['avg_market_cap'], reverse=True)

    # Divide into leaders (top half) and followers (bottom half)
    midpoint = len(sorted_coins) // 2
    leaders = sorted_coins[:midpoint]
    followers = sorted_coins[midpoint:]

    # Calculate how many followers we need for testing
    # Adjust test_size to account for testing only from bottom half
    total_coins = len(sorted_coins)
    num_test_coins = min(int(total_coins * test_size), len(followers))

    # Randomly select test coins from followers
    random.seed(42)
    selected_followers = random.sample(followers, num_test_coins)
    test_coins = [coin['coin_id'] for coin in selected_followers]

    # All coins not selected for testing are used for training
    test_coin_ids = set(test_coins)
    train_coins = [coin['coin_id'] for coin in sorted_coins if coin['coin_id'] not in test_coin_ids]

    # Calculate average statistics
    test_stats = [s for s in coin_stats if s['coin_id'] in test_coins]
    train_stats = [s for s in coin_stats if s['coin_id'] in train_coins]

    avg_test_mcap = sum(s['avg_market_cap'] for s in test_stats) / len(test_stats) if test_stats else 0
    avg_train_mcap = sum(s['avg_market_cap'] for s in train_stats) / len(train_stats) if train_stats else 0

    print(f"Train-leader-test-follower split by market cap:")
    print(f"  Train: {len(train_coins)} coins, Avg Market Cap: ${avg_train_mcap:.2f}")
    print(f"  Test (from bottom half): {len(test_coins)} coins, Avg Market Cap: ${avg_test_mcap:.2f}")

    return train_coins, test_coins


def train_quick_model(args):
    """Train a quick model with minimal features."""
    start_time = time.time()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_quick_model')

    # Get all coin files
    coin_files = [f for f in os.listdir(args.data_dir) if f.startswith('coin_') and f != 'coin_list.csv']

    # Filter out stablecoins if requested
    if args.filter_stablecoins:
        logger.info("Filtering out stablecoins...")
        stablecoin_filter = StablecoinFilter(logger=logger)
        _, non_stablecoin_paths = stablecoin_filter.filter_stablecoins(args.data_dir)
        # Extract just the filenames from the paths
        non_stablecoin_files = [os.path.basename(path) for path in non_stablecoin_paths]

        # Filter the original coin_files list to only include non-stablecoins
        coin_files = [f for f in coin_files if f in non_stablecoin_files]
        logger.info(f"After stablecoin filtering: {len(coin_files)} coins remaining")

    if len(coin_files) > args.max_coins:
        logger.info(f"Limiting to {args.max_coins} coins for faster training (out of {len(coin_files)} available)")
        random.seed(42)
        coin_files = random.sample(coin_files, args.max_coins)

    logger.info(f"Processing {len(coin_files)} coins...")

    # Process all coins
    all_coin_data = []
    valid_coin_ids = []
    coin_stats = []

    for coin_file in coin_files:
        file_path = os.path.join(args.data_dir, coin_file)
        result = load_and_process_coin(file_path, args.days_ahead, args.threshold)

        if result is not None:
            df, stats = result
            all_coin_data.append(df)
            valid_coin_ids.append(stats['coin_id'])
            coin_stats.append(stats)
            logger.info(
                f"Processed {stats['coin_id']}: {stats['rows']} rows, {stats['positive_rate'] * 100:.1f}% positive")

    logger.info(f"Successfully processed {len(valid_coin_ids)} coins")

    # Split coins for training and testing
    if args.selection == 'train-leader-test-follower':
        train_coins, test_coins = create_leader_follower_split(coin_stats, args.test_size)
    else:  # random
        random.seed(42)
        random.shuffle(valid_coin_ids)
        split_idx = int(len(valid_coin_ids) * (1 - args.test_size))
        train_coins = valid_coin_ids[:split_idx]
        test_coins = valid_coin_ids[split_idx:]

    logger.info(f"Training coins ({len(train_coins)}): {', '.join(train_coins[:5])}...")
    logger.info(f"Testing coins ({len(test_coins)}): {', '.join(test_coins)}")

    # Combine all data
    combined_df = pd.concat(all_coin_data, ignore_index=True)

    # Split into train and test sets
    train_df = combined_df[combined_df['coin_id'].isin(train_coins)]
    test_df = combined_df[combined_df['coin_id'].isin(test_coins)]

    logger.info(f"Training data: {len(train_df)} rows")
    logger.info(f"Testing data: {len(test_df)} rows")

    # Define features
    feature_cols = [
        'ma7', 'ma30',
        'price_to_ma7', 'price_to_ma30',
        'weekly_return',
        'volume_to_ma7'
    ]

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Check class balance
    train_balance = y_train.mean()
    test_balance = y_test.mean()
    logger.info(f"Class balance - Train: {train_balance:.2f} positive, Test: {test_balance:.2f} positive")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    if args.model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=8,  # Limit depth to prevent overfitting
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
    else:  # logistic
        model = LogisticRegression(
            C=0.1,  # More regularization
            class_weight='balanced',
            max_iter=500,
            random_state=42
        )

    logger.info(f"Training {args.model_type} model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate on training set
    y_train_pred = model.predict(X_train_scaled)
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }

    logger.info("Training Results:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")

    # Evaluate on test set
    y_test_pred = model.predict(X_test_scaled)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }

    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")

    # Evaluate per test coin
    coin_results = {}
    for coin in test_coins:
        coin_data = test_df[test_df['coin_id'] == coin]
        if len(coin_data) > 0:
            X_coin = coin_data[feature_cols]
            y_coin = coin_data['target']

            X_coin_scaled = scaler.transform(X_coin)
            y_coin_pred = model.predict(X_coin_scaled)

            # Get market cap for this coin
            coin_market_cap = next((s['avg_market_cap'] for s in coin_stats if s['coin_id'] == coin), 0)

            coin_metrics = {
                'accuracy': accuracy_score(y_coin, y_coin_pred),
                'precision': precision_score(y_coin, y_coin_pred, zero_division=0),
                'recall': recall_score(y_coin, y_coin_pred, zero_division=0),
                'f1_score': f1_score(y_coin, y_coin_pred, zero_division=0),
                'positives': int(y_coin.sum()),
                'total': len(y_coin),
                'market_cap': coin_market_cap
            }

            coin_results[coin] = coin_metrics

    # Show best and worst performing test coins
    coin_f1_scores = [(coin, data['f1_score'], data['market_cap']) for coin, data in coin_results.items()]
    coin_f1_scores.sort(key=lambda x: x[1], reverse=True)

    logger.info("\nBest performing test coins:")
    for coin, f1, mcap in coin_f1_scores[:3]:
        results = coin_results[coin]
        logger.info(
            f"  {coin}: F1={f1:.4f}, Acc={results['accuracy']:.4f}, +ve={results['positives']}/{results['total']}, MCap=${mcap:.2f}")

    logger.info("\nWorst performing test coins:")
    for coin, f1, mcap in coin_f1_scores[-3:]:
        results = coin_results[coin]
        logger.info(
            f"  {coin}: F1={f1:.4f}, Acc={results['accuracy']:.4f}, +ve={results['positives']}/{results['total']}, MCap=${mcap:.2f}")

    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save results
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Save model and scaler
    joblib.dump(model, os.path.join(args.model_save_dir, 'quick_model.joblib'))
    joblib.dump(scaler, os.path.join(args.model_save_dir, 'quick_scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(args.model_save_dir, 'quick_features.joblib'))

    # Save confusion matrix plot
    plt.savefig(os.path.join(args.model_save_dir, 'confusion_matrix.png'))
    plt.close()

    # Save feature importance for random forest
    if args.model_type == 'random_forest':
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_save_dir, 'feature_importance.png'))
        plt.close()

    # Create a plot showing performance by market cap
    if len(coin_results) > 5:  # Only if we have enough test coins
        plt.figure(figsize=(10, 6))
        mcaps = [data['market_cap'] for coin, data in coin_results.items()]
        f1s = [data['f1_score'] for coin, data in coin_results.items()]
        plt.scatter(mcaps, f1s, alpha=0.7)
        plt.xscale('log')  # Market caps vary by orders of magnitude
        plt.xlabel('Market Cap (log scale)')
        plt.ylabel('F1 Score')
        plt.title('Model Performance vs. Market Cap')
        plt.grid(True, alpha=0.3)

        # Add coin labels to a few points
        sorted_by_mcap = sorted(coin_results.items(), key=lambda x: x[1]['market_cap'], reverse=True)
        # Label top 3 and bottom 3 by market cap
        for coin, data in sorted_by_mcap[:3] + sorted_by_mcap[-3:]:
            plt.annotate(coin, (data['market_cap'], data['f1_score']))

        plt.tight_layout()
        plt.savefig(os.path.join(args.model_save_dir, 'performance_by_mcap.png'))
        plt.close()

    # Prepare results dictionary
    results = {
        'args': vars(args),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_seconds': time.time() - start_time,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'features': feature_cols,
        'train_coins': train_coins,
        'test_coins': test_coins,
        'coin_results': coin_results,
        'selection_method': args.selection
    }

    # Save detailed results
    with open(os.path.join(args.model_save_dir, 'quick_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Results saved to {args.model_save_dir}")

    return results


def main():
    args = parse_args()
    train_quick_model(args)


if __name__ == "__main__":
    main()
