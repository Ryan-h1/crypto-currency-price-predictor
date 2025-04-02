import os
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional


class StablecoinFilter:
    """
    A class to detect and filter out stablecoins from cryptocurrency datasets.
    
    This filter uses price stability analysis to identify stablecoins by checking
    if a coin's price stays within a narrow range over time.
    """

    def __init__(self,
                 price_volatility_threshold: float = 0.03,
                 min_days_for_volatility_check: int = 30,
                 stable_ratio_threshold: float = 0.9,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the StablecoinFilter with configuration parameters.
        
        Args:
            price_volatility_threshold: Maximum allowed price volatility for stablecoins 
                                        (e.g., 0.03 means ±3% from mean price)
            min_days_for_volatility_check: Minimum number of days of data required to perform volatility check
            stable_ratio_threshold: Minimum ratio of prices that must be within threshold to be considered stable
            logger: Optional logger instance for logging
        """
        self.price_volatility_threshold = price_volatility_threshold
        self.min_days_for_volatility_check = min_days_for_volatility_check
        self.stable_ratio_threshold = stable_ratio_threshold
        self.logger = logger or logging.getLogger(__name__)

    def is_stablecoin(self, price_data: pd.Series, coin_id: str = '') -> bool:
        """
        Check if a coin is likely a stablecoin based on its price stability.
        
        Args:
            price_data: Series containing price data for the coin
            coin_id: Optional coin identifier for logging purposes
            
        Returns:
            bool: True if the coin shows stablecoin-like price stability, False otherwise
        """
        # Skip check if not enough data
        if len(price_data) < self.min_days_for_volatility_check:
            self.logger.debug(f"Skipping volatility check for {coin_id}: insufficient data points")
            return False

        # Calculate mean and standard deviation, handling potential outliers
        price_mean = price_data.median()  # Use median to reduce impact of outliers

        if price_mean == 0:
            return False  # Avoid division by zero

        # Calculate price deviation as percent of mean
        price_deviation = (price_data - price_mean).abs() / price_mean

        # Check if sufficient values stay within the volatility threshold
        stability_ratio = (price_deviation <= self.price_volatility_threshold).mean()

        is_stable = stability_ratio >= self.stable_ratio_threshold

        if is_stable:
            self.logger.debug(f"Detected {coin_id} as stablecoin via price stability analysis: "
                              f"{stability_ratio:.2%} of prices within ±{self.price_volatility_threshold:.1%} of mean")

        return is_stable

    def filter_stablecoins(self, data_dir: str) -> Tuple[List[str], List[str]]:
        """
        Scan a directory of coin CSV files and identify stablecoins based on price stability.
        
        Args:
            data_dir: Directory containing coin CSV files
            
        Returns:
            Tuple containing:
                - List of stablecoin file paths
                - List of non-stablecoin file paths
        """
        stablecoin_files = []
        non_stablecoin_files = []

        coin_files = [f for f in os.listdir(data_dir) if f.startswith('coin_') and f.endswith('.csv')]
        total_files = len(coin_files)

        self.logger.info(f"Scanning {total_files} coin files to identify stablecoins...")

        for i, file_name in enumerate(coin_files, 1):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{total_files} files processed")

            file_path = os.path.join(data_dir, file_name)
            coin_id = file_name.replace('coin_', '').replace('.csv', '')

            # Check price stability
            try:
                df = pd.read_csv(file_path)
                if 'price' in df.columns and len(df) >= self.min_days_for_volatility_check:
                    price_data = df['price']

                    if self.is_stablecoin(price_data, coin_id):
                        stablecoin_files.append(file_path)
                        self.logger.info(f"Detected stablecoin: {coin_id}")
                    else:
                        non_stablecoin_files.append(file_path)
                else:
                    # Not enough data for analysis, consider it non-stablecoin
                    self.logger.debug(f"Insufficient data for {coin_id}, treating as non-stablecoin")
                    non_stablecoin_files.append(file_path)
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {e}")
                # If we can't analyze it, default to non-stablecoin
                non_stablecoin_files.append(file_path)

        self.logger.info(f"Found {len(stablecoin_files)} stablecoins and {len(non_stablecoin_files)} non-stablecoins")
        return stablecoin_files, non_stablecoin_files

    def get_filtered_coin_files(self, data_dir: str) -> List[str]:
        """
        Get a list of coin files excluding stablecoins.
        
        Args:
            data_dir: Directory containing coin CSV files
            
        Returns:
            List of file paths for non-stablecoin coins
        """
        _, non_stablecoin_files = self.filter_stablecoins(data_dir)
        return non_stablecoin_files
