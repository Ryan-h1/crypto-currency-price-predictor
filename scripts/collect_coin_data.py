import os
import logging
import sys
from dotenv import load_dotenv

NUMBER_OF_TOP_COINS_TO_COLLECT = 20
DAYS_OF_HISTORY_TO_COLLECT = 365

SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_PATH))
sys.path.insert(0, PROJECT_ROOT)

from src import CoinGeckoAPIScraper

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('coin_collector')

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Get API key from environment variables
API_KEY = os.environ.get("COINGECKO_API_KEY")
if not API_KEY:
    logger.error("COINGECKO_API_KEY environment variable not found")
    sys.exit(1)

BASE_API_URL = "https://api.coingecko.com/api/v3"

scraper = CoinGeckoAPIScraper(
    api_key=API_KEY,
    output_dir=DATA_DIR,
    base_api_url=BASE_API_URL,
    logger=logger
)

# Run the collection process
scraper.collect_all_data(number_of_top_coins=NUMBER_OF_TOP_COINS_TO_COLLECT, days=DAYS_OF_HISTORY_TO_COLLECT)
