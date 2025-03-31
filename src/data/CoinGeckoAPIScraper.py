import requests
import pandas as pd
import time

from logging import Logger
from typing import Dict, Any


class CoinGeckoAPIScraper:
    def __init__(self, api_key: str, output_dir: str, base_api_url: str, logger: Logger) -> None:
        self.api_key = api_key
        self.output_dir = output_dir
        self.base_api_url = base_api_url
        self.logger = logger
        self.headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": api_key
        }

    def _make_request(self, endpoint: str, query_params: Dict[str, Any]) -> Dict:
        """Make a request to the CoinGecko API with rate limiting."""

        url = f"{self.base_api_url}/{endpoint}"

        try:
            response = requests.get(url, params=query_params, headers=self.headers)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 30))
                self.logger.warning(f"Rate limit hit. Waiting for {retry_after} seconds.")
                time.sleep(retry_after)
                return self._make_request(endpoint, query_params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error making request to {url}: {e}")
            if "rate limit" in str(e).lower():
                self.logger.info("Waiting 60 seconds due to rate limit...")
                time.sleep(60)
                return self._make_request(endpoint, query_params)
            raise

    def get_top_coins(self, limit: int = 50, currency: str = "usd") -> dict:
        """Get the top coins by market cap."""
        self.logger.debug(f"Fetching top {limit} coins in {currency} by market cap.")

        query_params = {
            "vs_currency": currency,
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }

        response = self._make_request("coins/markets", query_params)

        coin_list_df = pd.DataFrame(response)
        coin_list_df.to_csv(f"{self.output_dir}/coin_list.csv", index=False)

        self.logger.debug(f"Successfully retrieved {len(response)} coins")
        return response

    def get_historical_data(self, coin_id: str, days: int = 365,
                            currency: str = "usd", interval: str = "daily") -> Dict:
        """Get historical price data for a specific coin."""
        if days < 1:
            raise ValueError("Days must be greater than 0.")

        self.logger.debug(f"Fetching {days} days of historical data for {coin_id}")

        params = {
            "vs_currency": currency,
            "days": days,
            "interval": interval,
            "precision": "full"
        }

        response = self._make_request(f"coins/{coin_id}/market_chart", params)

        self.logger.debug(f"Successfully retrieved historical data for {coin_id}")
        return response

    def process_and_save_historical_data(self, coin_id: str, data: Dict) -> None:
        """Process and save historical data to CSV."""
        # Process price data
        prices_df = pd.DataFrame(data["prices"], columns=["timestamp", "price"]).copy()
        prices_df["date"] = pd.to_datetime(prices_df["timestamp"], unit="ms")

        # Process market cap data
        market_caps_df = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])

        # Process volume data
        volumes_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

        # Merge all data based on timestamp
        merged_df = prices_df.merge(market_caps_df, on="timestamp").merge(volumes_df, on="timestamp")

        # Clean up the dataframe
        final_df = merged_df[["date", "price", "market_cap", "volume"]].sort_values("date")

        # Save to CSV
        output_file = f"{self.output_dir}/coin_{coin_id}.csv"
        final_df.to_csv(output_file, index=False)
        self.logger.info(f"Saved historical data for {coin_id} to {output_file}")

    def collect_all_data(self, number_of_top_coins: int = 50, days: int = 365) -> None:
        """Collect historical data for all top coins and save to CSVs."""
        # Get top IDs by market cap
        top_coins = self.get_top_coins(limit=number_of_top_coins)

        for coin in top_coins:
            coin_id = coin["id"]
            try:
                historical_data = self.get_historical_data(coin_id=coin_id, days=days)
                self.process_and_save_historical_data(coin_id=coin_id, data=historical_data)
                # Pause for 1 second to avoid hitting rate limits
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error collecting historical data for {coin_id}: {e}")
