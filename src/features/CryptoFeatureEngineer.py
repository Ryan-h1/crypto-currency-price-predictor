import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class CryptoFeatureEngineer:
    """
    Feature engineering class for cryptocurrency price data.
    
    This class creates various features from cryptocurrency time series data, including:
    - Price moving averages and ratios
    - Volume indicators
    - Momentum indicators
    - Volatility measures
    - Market cycle features
    """

    def __init__(self,
                 price_windows: List[int] = None,
                 volume_windows: List[int] = None,
                 volatility_windows: List[int] = None,
                 rsi_windows: List[int] = None,
                 macd_params: Dict = None):
        """
        Initialize the feature engineer with desired parameters.
        
        Args:
            price_windows: List of window sizes for price-based features
            volume_windows: List of window sizes for volume-based features
            volatility_windows: List of window sizes for volatility calculation
            rsi_windows: List of periods for RSI calculation
            macd_params: Dictionary with MACD parameters (fast, slow, signal periods)
        """
        # Default window sizes if not provided
        self.price_windows = price_windows or [7, 14, 30, 60, 90]
        self.volume_windows = volume_windows or [7, 14, 30]
        self.volatility_windows = volatility_windows or [7, 14, 30]
        self.rsi_windows = rsi_windows or [7, 14, 30]
        self.macd_params = macd_params or {'fast': 12, 'slow': 26, 'signal': 9}

    def add_all_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Add all available features to the dataframe.
        
        Args:
            df: DataFrame with at least 'price' and 'volume' columns
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with all features added
        """
        if not all(col in df.columns for col in ['price', 'volume']):
            raise ValueError("DataFrame must contain 'price' and 'volume' columns")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Add all feature groups
        result_df = self.add_price_features(result_df, drop_na=False)
        result_df = self.add_volume_features(result_df, drop_na=False)
        result_df = self.add_volatility_features(result_df, drop_na=False)
        result_df = self.add_momentum_features(result_df, drop_na=False)
        result_df = self.add_technical_indicators(result_df, drop_na=False)

        # Drop NaN values at the end if requested
        if drop_na:
            # Get all feature columns (exclude price, volume, date, etc.)
            feature_cols = [col for col in result_df.columns
                            if col not in ['date', 'price', 'volume', 'market_cap',
                                           'coin_id', 'future_price', 'target']]
            result_df = result_df.dropna(subset=feature_cols)

        return result_df

    def add_price_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Add price-based features to the dataframe.
        
        Args:
            df: DataFrame with at least a 'price' column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with price features added
        """
        result_df = df.copy()

        # Simple moving averages
        for window in self.price_windows:
            result_df[f'ma{window}'] = result_df['price'].rolling(window=window).mean()

            # Price to moving average ratio (with safety check)
            ma_col = f'ma{window}'
            result_df[f'price_to_{ma_col}'] = np.where(
                result_df[ma_col] > 0,
                result_df['price'] / result_df[ma_col],
                1.0
            )

            # Exponential moving averages
            result_df[f'ema{window}'] = result_df['price'].ewm(span=window, adjust=False).mean()

            # Price percent change
            result_df[f'price_pct_change_{window}d'] = result_df['price'].pct_change(window)

        # Moving average crossovers (shorter MA / longer MA)
        for i, short_window in enumerate(self.price_windows[:-1]):
            for long_window in self.price_windows[i + 1:]:
                result_df[f'ma_crossover_{short_window}_{long_window}'] = np.where(
                    result_df[f'ma{long_window}'] > 0,
                    result_df[f'ma{short_window}'] / result_df[f'ma{long_window}'],
                    1.0
                )

        # Distance from high/low
        for window in self.price_windows:
            # Rolling high and low
            result_df[f'high_{window}d'] = result_df['price'].rolling(window).max()
            result_df[f'low_{window}d'] = result_df['price'].rolling(window).min()

            # Distance from high/low as percentage
            result_df[f'pct_from_high_{window}d'] = np.where(
                result_df[f'high_{window}d'] > 0,
                (result_df['price'] - result_df[f'high_{window}d']) / result_df[f'high_{window}d'],
                0
            )

            result_df[f'pct_from_low_{window}d'] = np.where(
                result_df[f'low_{window}d'] > 0,
                (result_df['price'] - result_df[f'low_{window}d']) / result_df[f'low_{window}d'],
                0
            )

        # Clean up infinity and NaN values in computed columns
        for col in result_df.columns:
            if col not in df.columns:  # Only process new columns
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                result_df[col] = result_df[col].ffill().fillna(0)

        # Drop NaN values if requested
        if drop_na:
            price_feature_cols = [col for col in result_df.columns if col not in df.columns]
            result_df = result_df.dropna(subset=price_feature_cols)

        return result_df

    def add_volume_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Add volume-based features to the dataframe.
        
        Args:
            df: DataFrame with 'volume' column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with volume features added
        """
        result_df = df.copy()

        # Simple volume features
        for window in self.volume_windows:
            # Volume moving average
            result_df[f'volume_ma{window}'] = result_df['volume'].rolling(window=window).mean()

            # Volume to moving average ratio
            vol_ma = result_df[f'volume_ma{window}']
            result_df[f'volume_to_ma{window}'] = np.where(vol_ma > 0, result_df['volume'] / vol_ma, 1.0)

            # Volume change
            result_df[f'volume_change_{window}d'] = result_df['volume'].pct_change(window)

            # Relative volume (compared to average)
            vol_ma = result_df['volume'].rolling(window=window).mean()
            result_df[f'relative_volume_{window}d'] = np.where(vol_ma > 0, result_df['volume'] / vol_ma, 1.0)

        # Price-volume relationship - safer implementation
        result_df['price_volume_correlation_7d'] = 0.0

        # Use vectorized operations where possible
        # Calculate rolling correlation in a safer way
        try:
            # Use pandas built-in rolling correlation with min_periods
            rolling_corr = result_df['price'].rolling(window=7, min_periods=5).corr(result_df['volume'])
            # Replace NaN with 0 and clip to valid range
            result_df['price_volume_correlation_7d'] = rolling_corr.fillna(0).clip(-1, 1)
        except Exception:
            # If the above fails, fall back to manual calculation with better error handling
            for i in range(7, len(result_df)):
                try:
                    if i >= 7:
                        price_slice = result_df['price'].iloc[i - 7:i + 1]
                        volume_slice = result_df['volume'].iloc[i - 7:i + 1]

                        # Check for valid data
                        if (len(price_slice) >= 5 and  # Need at least 5 valid points
                                len(volume_slice) >= 5 and
                                price_slice.std() > 0 and
                                volume_slice.std() > 0):

                            # Calculate correlation
                            correlation = price_slice.corr(volume_slice)

                            # Ensure it's a valid value
                            if pd.notna(correlation) and not np.isinf(correlation):
                                result_df.loc[result_df.index[i], 'price_volume_correlation_7d'] = correlation
                except Exception:
                    # If correlation fails for this window, keep as 0
                    pass

        # Clean up infinity and NaN values
        for col in result_df.columns:
            if col not in df.columns:  # Only process new columns
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                result_df[col] = result_df[col].ffill().fillna(0)

        if drop_na:
            volume_feature_cols = [col for col in result_df.columns if col not in df.columns]
            result_df = result_df.dropna(subset=volume_feature_cols)

        return result_df

    def add_volatility_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Add volatility-based features to the dataframe.
        
        Args:
            df: DataFrame with at least a 'price' column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with volatility features added
        """
        result_df = df.copy()

        # Calculate returns first
        result_df['daily_return'] = result_df['price'].pct_change()

        for window in self.volatility_windows:
            # Standard deviation of returns (volatility)
            result_df[f'volatility_{window}d'] = result_df['daily_return'].rolling(window=window).std()

            # Average true range-like measure
            result_df[f'price_range_{window}d'] = (
                                                          result_df['price'].rolling(window=window).max() -
                                                          result_df['price'].rolling(window=window).min()
                                                  ) / result_df['price'].rolling(window=window).mean()

            # Exponential volatility
            result_df[f'exp_volatility_{window}d'] = result_df['daily_return'].ewm(span=window).std()

        # Clean up infinity and NaN values
        for col in result_df.columns:
            if col not in df.columns:  # Only process new columns
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                result_df[col] = result_df[col].ffill().fillna(0)

        if drop_na:
            volatility_feature_cols = [col for col in result_df.columns if col not in df.columns]
            result_df = result_df.dropna(subset=volatility_feature_cols)

        return result_df

    def add_momentum_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Add momentum-based features to the dataframe.
        
        Args:
            df: DataFrame with at least a 'price' column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with momentum features added
        """
        result_df = df.copy()

        # Calculate returns if not already present
        if 'daily_return' not in result_df.columns:
            result_df['daily_return'] = result_df['price'].pct_change()

        # RSI (Relative Strength Index) - more robust implementation
        for window in self.rsi_windows:
            try:
                # Calculate up and down movements
                delta = result_df['price'].diff()

                # Handle up and down movements with vectorized operations
                up_moves = delta.clip(lower=0)
                down_moves = -delta.clip(upper=0)

                # Calculate averages with proper handling of zero values
                avg_up = up_moves.rolling(window=window).mean()
                avg_down = down_moves.rolling(window=window).mean()

                # Calculate RS with NaN/zero protection
                rs = np.where(avg_down > 0.00001, avg_up / avg_down, 100.0)  # If no down moves, RSI should be high

                # Calculate RSI with bounds
                result_df[f'rsi_{window}d'] = 100.0 - (100.0 / (1.0 + rs))

                # Ensure RSI is within 0-100 range
                result_df[f'rsi_{window}d'] = result_df[f'rsi_{window}d'].clip(0, 100)

            except Exception:
                # Fallback to the traditional calculation if the above fails
                delta = result_df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

                # Calculate RSI with safety checks
                rs = np.where(loss > 0.00001, gain / loss, 100.0)
                result_df[f'rsi_{window}d'] = 100.0 - (100.0 / (1.0 + rs))
                result_df[f'rsi_{window}d'] = result_df[f'rsi_{window}d'].clip(0, 100)

        # Momentum (current price / price n days ago) - with safety checks
        for window in [7, 14, 30, 60]:
            # Get past prices
            past_price = result_df['price'].shift(window)

            # Calculate momentum with division by zero protection
            result_df[f'momentum_{window}d'] = np.where(
                past_price > 0.0000001,  # Avoid division by very small numbers
                result_df['price'] / past_price,
                1.0  # Default to neutral value if no valid past price
            )

            # Cap extremely large values 
            result_df[f'momentum_{window}d'] = result_df[f'momentum_{window}d'].clip(upper=10.0)

        # Rate of change - with safety
        for window in [7, 14, 30, 60]:
            past_price = result_df['price'].shift(window)

            # Calculate rate of change with safety checks
            result_df[f'roc_{window}d'] = np.where(
                past_price > 0.0000001,
                (result_df['price'] - past_price) / past_price * 100.0,
                0.0
            )

            # Clip extreme values
            result_df[f'roc_{window}d'] = result_df[f'roc_{window}d'].clip(-100.0, 1000.0)

        # Clean up infinity and NaN values
        for col in result_df.columns:
            if col not in df.columns:  # Only process new columns
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                result_df[col] = result_df[col].ffill().fillna(0)

        if drop_na:
            momentum_feature_cols = [col for col in result_df.columns if col not in df.columns]
            result_df = result_df.dropna(subset=momentum_feature_cols)

        return result_df

    def add_technical_indicators(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Add advanced technical indicators to the dataframe.
        
        Args:
            df: DataFrame with at least a 'price' column
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            DataFrame with technical indicators added
        """
        result_df = df.copy()

        # MACD (Moving Average Convergence Divergence)
        fast = self.macd_params['fast']
        slow = self.macd_params['slow']
        signal = self.macd_params['signal']

        # Calculate MACD components - with error handling
        try:
            # Calculate EMAs
            result_df['ema_fast'] = result_df['price'].ewm(span=fast, adjust=False).mean()
            result_df['ema_slow'] = result_df['price'].ewm(span=slow, adjust=False).mean()

            # Calculate MACD line
            result_df['macd_line'] = result_df['ema_fast'] - result_df['ema_slow']

            # Calculate signal line 
            result_df['macd_signal'] = result_df['macd_line'].ewm(span=signal, adjust=False).mean()

            # Calculate histogram
            result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']

            # Scale MACD values relative to price for better comparison across coins
            avg_price = result_df['price'].mean()
            if avg_price > 0:
                scale_factor = 100.0 / avg_price  # Normalize to percentage of average price
                result_df['macd_line'] = result_df['macd_line'] * scale_factor
                result_df['macd_signal'] = result_df['macd_signal'] * scale_factor
                result_df['macd_histogram'] = result_df['macd_histogram'] * scale_factor

        except Exception:
            # If MACD calculation fails, create empty columns
            result_df['macd_line'] = 0.0
            result_df['macd_signal'] = 0.0
            result_df['macd_histogram'] = 0.0

        # Bollinger Bands (20-day, 2 standard deviations) - with error handling
        try:
            window = 20
            # Calculate middle band (SMA)
            result_df['bb_middle'] = result_df['price'].rolling(window=window).mean()

            # Calculate standard deviation with min_periods
            result_df['bb_std'] = result_df['price'].rolling(window=window, min_periods=5).std()

            # Replace any zero std dev with a small value to avoid division issues
            result_df['bb_std'] = result_df['bb_std'].replace(0, result_df['price'].mean() * 0.001)

            # Calculate upper and lower bands
            result_df['bb_upper'] = result_df['bb_middle'] + 2 * result_df['bb_std']
            result_df['bb_lower'] = result_df['bb_middle'] - 2 * result_df['bb_std']

            # Calculate width and position with safety checks
            bb_range = result_df['bb_upper'] - result_df['bb_lower']
            result_df['bb_width'] = np.where(
                result_df['bb_middle'] > 0,
                bb_range / result_df['bb_middle'],
                0.0
            )

            result_df['bb_position'] = np.where(
                bb_range > 0,
                (result_df['price'] - result_df['bb_lower']) / bb_range,
                0.5  # Default to middle if no valid range
            )

            # Clip position to 0-1 range (values can exceed bands)
            result_df['bb_position'] = result_df['bb_position'].clip(0, 1)

        except Exception:
            # If BB calculation fails, create empty columns
            result_df['bb_middle'] = result_df['price']
            result_df['bb_std'] = 0
            result_df['bb_upper'] = result_df['price']
            result_df['bb_lower'] = result_df['price']
            result_df['bb_width'] = 0
            result_df['bb_position'] = 0.5

        # Clean up intermediate columns
        if 'ema_fast' in result_df.columns and 'ema_slow' in result_df.columns:
            result_df = result_df.drop(['ema_fast', 'ema_slow'], axis=1)

        # Clean up infinity and NaN values
        for col in result_df.columns:
            if col not in df.columns:  # Only process new columns
                result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                result_df[col] = result_df[col].ffill().fillna(0)

        if drop_na:
            indicator_cols = [col for col in result_df.columns if col not in df.columns]
            result_df = result_df.dropna(subset=indicator_cols)

        return result_df

    def get_selected_features(self, df: pd.DataFrame, feature_set: str = 'minimal') -> List[str]:
        """
        Get a list of important features based on the specified feature set.
        
        Args:
            df: DataFrame with features already added
            feature_set: Which set of features to return ('minimal', 'standard', or 'full')
            
        Returns:
            List of selected feature column names
        """
        available_cols = set(df.columns.tolist())

        # Core features that should always be available
        core_features = ['ma7', 'ma30', 'price_to_ma7', 'price_to_ma30']

        # Check if all core features are available
        if not all(f in available_cols for f in core_features):
            # If core features are missing, something is wrong
            # Return a minimal set of columns that should be available
            safe_columns = [col for col in available_cols if col not in [
                'date', 'price', 'volume', 'market_cap', 'coin_id',
                'future_price', 'target', 'daily_return', 'price_change_pct'
            ]]
            return safe_columns[:10] if len(safe_columns) > 10 else safe_columns

        if feature_set == 'minimal':
            # A small set of proven powerful features
            selected_features = [
                'ma7', 'ma30',
                'price_to_ma7', 'price_to_ma30',
                'price_pct_change_7d', 'price_pct_change_30d',
                'volume_to_ma7',
                'volatility_14d',
                'rsi_14d',
            ]

        elif feature_set == 'standard':
            # A balanced set of diverse features
            selected_features = [
                'ma7', 'ma14', 'ma30', 'ema14',
                'price_to_ma7', 'price_to_ma30',
                'price_pct_change_7d', 'price_pct_change_30d',
                'ma_crossover_7_30',
                'pct_from_high_30d', 'pct_from_low_30d',
                'volume_to_ma7', 'volume_to_ma30',
                'volume_change_7d', 'relative_volume_7d',
                'volatility_14d', 'price_range_14d',
                'rsi_14d', 'momentum_14d', 'roc_14d',
                'macd_histogram', 'macd_line',
                'bb_width', 'bb_position'
            ]

        else:  # 'full'
            # All features except raw/duplicate/auxiliary columns and target-related columns
            # CRITICAL: We must exclude price_change_pct to avoid data leakage
            exclude_cols = [
                'date', 'price', 'volume', 'market_cap', 'coin_id',
                'future_price', 'target', 'daily_return', 'price_change_pct',
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'macd_signal'
            ]
            selected_features = [col for col in available_cols if col not in exclude_cols]

        # Check which features actually exist in the dataframe
        valid_features = [f for f in selected_features if f in available_cols]

        # If we have too few features, add more from what's available
        if len(valid_features) < 5 and feature_set != 'full':
            # Get all potentially useful features
            additional_features = [col for col in available_cols if col not in [
                'date', 'price', 'volume', 'market_cap', 'coin_id',
                'future_price', 'target', 'daily_return', 'price_change_pct'
            ] and col not in valid_features]

            # Add some additional features if needed
            valid_features.extend(additional_features[:10])

        return valid_features
