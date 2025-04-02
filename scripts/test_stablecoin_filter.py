#!/usr/bin/env python
"""
Script to test the StablecoinFilter functionality.
This will scan the data directory and identify stablecoins based on price volatility.
"""

import os
import sys
import pandas as pd
import logging
import argparse
from typing import List

# Add project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the StablecoinFilter
from src import StablecoinFilter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the StablecoinFilter module')

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing cryptocurrency data CSV files')

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed information about each coin')

    parser.add_argument('--threshold', type=float, default=0.03,
                        help='Price volatility threshold (e.g., 0.03 means ±3% from median price)')

    parser.add_argument('--min_days', type=int, default=30,
                        help='Minimum days of data required for analysis')

    parser.add_argument('--stable_ratio', type=float, default=0.9,
                        help='Required ratio of prices within threshold to be considered stable')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('stablecoin_filter_test')

    # Initialize the filter with user-provided parameters
    stablecoin_filter = StablecoinFilter(
        price_volatility_threshold=args.threshold,
        min_days_for_volatility_check=args.min_days,
        stable_ratio_threshold=args.stable_ratio,
        logger=logger
    )

    # Get a list of all coin files
    coin_files = [f for f in os.listdir(args.data_dir)
                  if f.startswith('coin_') and f.endswith('.csv') and f != 'coin_list.csv']

    logger.info(f"Found {len(coin_files)} coin files in {args.data_dir}")

    # Lists to store results
    detected_stablecoins = []
    non_stablecoins = []

    # Process each coin
    for coin_file in coin_files:
        coin_id = coin_file.replace('coin_', '').replace('.csv', '')
        file_path = os.path.join(args.data_dir, coin_file)

        try:
            # Check price stability for each coin
            df = pd.read_csv(file_path)
            if 'price' in df.columns and len(df) >= args.min_days:
                price_data = df['price']

                # Check if this is a stablecoin based on price stability
                is_stablecoin = stablecoin_filter.is_stablecoin(price_data, coin_id)

                # Store the result
                if is_stablecoin:
                    # Calculate price statistics for reporting
                    price_mean = price_data.median()
                    price_deviation = (price_data - price_mean).abs() / price_mean
                    stability_ratio = (price_deviation <= args.threshold).mean()

                    detected_stablecoins.append((coin_id, stability_ratio))

                    if args.verbose:
                        logger.info(
                            f"{coin_id}: STABLECOIN - {stability_ratio:.2%} of prices within ±{args.threshold:.1%} of median")
                else:
                    non_stablecoins.append(coin_id)

                    if args.verbose:
                        price_mean = price_data.median()
                        price_deviation = (price_data - price_mean).abs() / price_mean
                        stability_ratio = (price_deviation <= args.threshold).mean()
                        logger.info(
                            f"{coin_id}: NOT STABLECOIN - {stability_ratio:.2%} of prices within ±{args.threshold:.1%} of median")
            else:
                # Not enough data
                non_stablecoins.append(coin_id)
                if args.verbose:
                    logger.info(f"{coin_id}: NOT ANALYZED - insufficient data")
        except Exception as e:
            logger.error(f"Error processing {coin_file}: {e}")
            non_stablecoins.append(coin_id)

    # Sort stablecoins by stability ratio (most stable first)
    detected_stablecoins.sort(key=lambda x: x[1], reverse=True)

    # Print summary
    logger.info("\n===== RESULTS =====")
    logger.info(f"Total coins analyzed: {len(coin_files)}")
    logger.info(
        f"Detected stablecoins: {len(detected_stablecoins)} ({len(detected_stablecoins) / len(coin_files) * 100:.1f}%)")

    # Print all detected stablecoins with their stability ratio
    logger.info("\nStablecoins detected (ordered by stability):")
    for coin_id, stability_ratio in detected_stablecoins:
        logger.info(f"  - {coin_id}: {stability_ratio:.2%} price stability")

    # Also save the results to a file
    with open('stablecoin_detection_results.txt', 'w') as f:
        f.write(f"Total coins analyzed: {len(coin_files)}\n")
        f.write(f"Detected stablecoins: {len(detected_stablecoins)}\n\n")
        f.write("=== Configuration ===\n")
        f.write(f"Price volatility threshold: ±{args.threshold:.1%}\n")
        f.write(f"Minimum days of data: {args.min_days}\n")
        f.write(f"Stability ratio threshold: {args.stable_ratio:.1%}\n\n")
        f.write("=== Stablecoins (ordered by stability) ===\n")
        for coin_id, stability_ratio in detected_stablecoins:
            f.write(f"  - {coin_id}: {stability_ratio:.2%} price stability\n")

    logger.info("\nResults also saved to stablecoin_detection_results.txt")


if __name__ == "__main__":
    main()
