"""Cryptocurrency analysis and prediction package."""

from .data import CoinGeckoAPIScraper, StablecoinFilter
from .features import CryptoFeatureEngineer, add_crypto_features

__all__ = ['CoinGeckoAPIScraper', 'StablecoinFilter', 'CryptoFeatureEngineer', 'add_crypto_features']

__version__ = '0.1.0'
