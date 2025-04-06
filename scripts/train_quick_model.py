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
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.data import StablecoinFilter
from src.features import CryptoFeatureEngineer


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
                        choices=['random_forest', 'logistic', 'xgboost', 'lstm'],
                        help='Type of model to train')

    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of coins to use for testing (default: 0.1)')

    parser.add_argument('--max_coins', type=int, default=2000,
                        help='Maximum number of coins to use (for faster training)')

    parser.add_argument('--selection', type=str, default='train-leader-test-follower',
                        choices=['random', 'train-leader-test-follower'],
                        help='Method to select train/test coins: random or train-leader-test-follower')

    parser.add_argument('--filter_stablecoins', type=bool, default=True,
                        help='Whether to filter out stablecoins from the dataset')

    parser.add_argument('--feature_set', type=str, default='standard',
                        choices=['minimal', 'standard', 'full'],
                        help='Feature set to use for training (minimal: few basic features, '
                             'standard: balanced set of features, full: all available features)')

    parser.add_argument('--validation', type=str, default='coin_split',
                        choices=['coin_split', 'walk_forward'],
                        help='Validation strategy: coin_split (train on some coins, test on others) or '
                             'walk_forward (train on earlier data, test on later data)')

    parser.add_argument('--lstm_units', type=int, default=64,
                        help='Number of LSTM units (only used when model_type is lstm)')

    parser.add_argument('--lstm_dropout', type=float, default=0.2,
                        help='Dropout rate for LSTM layers (only used when model_type is lstm)')

    parser.add_argument('--lstm_epochs', type=int, default=50,
                        help='Number of epochs for LSTM training (only used when model_type is lstm)')

    parser.add_argument('--lstm_sequence_length', type=int, default=30,
                        help='Sequence length for LSTM input (only used when model_type is lstm)')

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


def prepare_lstm_data(X, y, sequence_length):
    """
    Prepare data for LSTM model by creating sequences.
    
    Args:
        X: Feature data
        y: Target data
        sequence_length: Length of sequences to create
        
    Returns:
        X_sequences, y_sequences
    """
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)


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

    # Add features using the feature engineering module
    logger.info(f"Adding features using feature set: {args.feature_set}")
    feature_engineer = CryptoFeatureEngineer()

    # Add features to the combined dataframe
    combined_df = feature_engineer.add_all_features(combined_df, drop_na=True)

    # Get selected features based on the chosen feature set
    feature_cols = feature_engineer.get_selected_features(combined_df, args.feature_set)

    logger.info(f"Selected {len(feature_cols)} features: {', '.join(feature_cols[:10])}...")

    # Split into train and test sets
    train_df = combined_df[combined_df['coin_id'].isin(train_coins)]
    test_df = combined_df[combined_df['coin_id'].isin(test_coins)]

    logger.info(f"Training data: {len(train_df)} rows")
    logger.info(f"Testing data: {len(test_df)} rows")

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
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
    elif args.model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective='binary:logistic',
            scale_pos_weight=len(y_train) / y_train.sum() - 1,  # Adjust for class imbalance
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
    elif args.model_type == 'lstm':
        # Prepare sequence data for LSTM
        logger.info(f"Preparing sequence data for LSTM with sequence length {args.lstm_sequence_length}...")
        
        # Make sure we have enough data for the sequence length
        if len(X_train) <= args.lstm_sequence_length:
            logger.warning(f"Training data length ({len(X_train)}) is too short for sequence length ({args.lstm_sequence_length}). Reducing sequence length.")
            args.lstm_sequence_length = max(5, len(X_train) // 3)
            
        X_train_seq, y_train_seq = prepare_lstm_data(X_train_scaled, y_train.values, args.lstm_sequence_length)
        
        # Check if we have enough data after sequence preparation
        if len(X_train_seq) < 10:
            logger.error("Not enough training data for LSTM after sequence preparation. Consider using a shorter sequence length or more data.")
            raise ValueError("Insufficient data for LSTM training")
            
        # Build LSTM model
        logger.info(f"Building LSTM model with {args.lstm_units} units and {args.lstm_dropout} dropout...")
        
        # Get input shape
        input_shape = (args.lstm_sequence_length, X_train.shape[1])
        
        # Create model
        model = Sequential([
            LSTM(args.lstm_units, input_shape=input_shape, return_sequences=True),
            Dropout(args.lstm_dropout),
            LSTM(args.lstm_units // 2),
            Dropout(args.lstm_dropout),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Class weights to handle imbalance
        class_weight = {
            0: 1.0,
            1: len(y_train_seq) / max(sum(y_train_seq), 1) - 1
        }
        
        # Fit model
        logger.info(f"Training LSTM model for up to {args.lstm_epochs} epochs...")
        model.fit(
            X_train_seq, y_train_seq,
            epochs=args.lstm_epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            class_weight=class_weight,
            verbose=1
        )
        
        # Prepare test sequences
        X_test_seq, y_test_seq = prepare_lstm_data(X_test_scaled, y_test.values, args.lstm_sequence_length)
        
        # If we don't have enough test data for sequences, we need to handle it
        if len(X_test_seq) == 0:
            logger.warning("Not enough test data for LSTM evaluation with sequences. Using a different approach.")
            # One approach: predict on the last {sequence_length} points of each test coin
            y_test_pred = []
            y_test_actual = []
            
            # Get predictions for each test coin separately
            for coin in test_coins:
                coin_data = test_df[test_df['coin_id'] == coin]
                if len(coin_data) > args.lstm_sequence_length:
                    X_coin = scaler.transform(coin_data[feature_cols])
                    X_coin_seq = np.array([X_coin[-args.lstm_sequence_length:]])
                    y_coin_pred = (model.predict(X_coin_seq) > 0.5).astype(int)
                    y_test_pred.append(y_coin_pred[0][0])
                    y_test_actual.append(coin_data['target'].iloc[-1])
            
            # If we still don't have predictions, we can't evaluate properly
            if not y_test_pred:
                logger.error("Cannot evaluate LSTM on test data - insufficient data.")
                y_test_pred = np.zeros(len(y_test))
                y_train_pred = model.predict(X_train_seq) > 0.5
            else:
                y_test_pred = np.array(y_test_pred)
                y_test = np.array(y_test_actual)
                
                # For training predictions, use the sequences we already have
                y_train_pred = model.predict(X_train_seq) > 0.5
        else:
            # Normal case - we have enough test data
            y_test_pred = model.predict(X_test_seq) > 0.5
            y_train_pred = model.predict(X_train_seq) > 0.5
            y_test = y_test_seq
    else:  # logistic
        model = LogisticRegression(
            C=0.1,  # More regularization
            class_weight='balanced',
            max_iter=500,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

    # Evaluate on training set
    train_metrics = {
        'accuracy': accuracy_score(y_train if args.model_type != 'lstm' else y_train_seq, y_train_pred),
        'precision': precision_score(y_train if args.model_type != 'lstm' else y_train_seq, y_train_pred, zero_division=0),
        'recall': recall_score(y_train if args.model_type != 'lstm' else y_train_seq, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train if args.model_type != 'lstm' else y_train_seq, y_train_pred, zero_division=0)
    }

    logger.info("Training Results:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")

    # Evaluate on test set
    test_metrics = {
        'accuracy': accuracy_score(y_test if args.model_type != 'lstm' else y_test_seq, y_test_pred),
        'precision': precision_score(y_test if args.model_type != 'lstm' else y_test_seq, y_test_pred, zero_division=0),
        'recall': recall_score(y_test if args.model_type != 'lstm' else y_test_seq, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test if args.model_type != 'lstm' else y_test_seq, y_test_pred, zero_division=0)
    }

    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")

    # Evaluate per test coin
    coin_results = {}
    if args.model_type != 'lstm':
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
    else:
        # For LSTM, we need a different approach for per-coin evaluation
        for coin in test_coins:
            coin_data = test_df[test_df['coin_id'] == coin]
            if len(coin_data) > args.lstm_sequence_length:
                X_coin = scaler.transform(coin_data[feature_cols])
                y_coin = coin_data['target'].values
                
                # Prepare sequences
                X_coin_seq, y_coin_seq = prepare_lstm_data(X_coin, y_coin, args.lstm_sequence_length)
                
                if len(X_coin_seq) > 0:
                    y_coin_pred = model.predict(X_coin_seq) > 0.5
                    
                    # Get market cap for this coin
                    coin_market_cap = next((s['avg_market_cap'] for s in coin_stats if s['coin_id'] == coin), 0)
                    
                    coin_metrics = {
                        'accuracy': accuracy_score(y_coin_seq, y_coin_pred),
                        'precision': precision_score(y_coin_seq, y_coin_pred, zero_division=0),
                        'recall': recall_score(y_coin_seq, y_coin_pred, zero_division=0),
                        'f1_score': f1_score(y_coin_seq, y_coin_pred, zero_division=0),
                        'positives': int(y_coin_seq.sum()),
                        'total': len(y_coin_seq),
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
    cm = confusion_matrix(y_test if args.model_type != 'lstm' else y_test_seq, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save results
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Save model and scaler
    if args.model_type == 'lstm':
        # Save Keras model differently
        model_path = os.path.join(args.model_save_dir, 'quick_model')
        model.save(model_path)
    else:
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
        'selection_method': args.selection,
        'feature_set': args.feature_set
    }

    # Save detailed results
    with open(os.path.join(args.model_save_dir, 'quick_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save additional info for LSTM
    if args.model_type == 'lstm':
        lstm_params = {
            'sequence_length': args.lstm_sequence_length,
            'units': args.lstm_units,
            'dropout': args.lstm_dropout
        }
        with open(os.path.join(args.model_save_dir, 'lstm_params.json'), 'w') as f:
            json.dump(lstm_params, f, indent=2)

    logger.info(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Results saved to {args.model_save_dir}")

    return results


def main():
    args = parse_args()
    train_quick_model(args)


if __name__ == "__main__":
    main()
