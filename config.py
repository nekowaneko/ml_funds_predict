import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Stock Configuration ---
SYMBOL_DICT = {
    '3661': '世芯-KY',
    '3017': '奇鋐',
    '2330': '台積電'
}

# --- Scraper Configuration ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

BASE_URL_PATTERN = 'https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={}&stockNo={}&response=html'

# Directory to save raw stock data (CSV files)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'uniblack_stocks')

# --- Discord Configuration ---
# 從環境變數讀取 Token 與 Channel ID，若無則使用預設值 (建議設為環境變數以策安全)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', 'YOUR_DISCORD_BOT_TOKEN')
DISCORD_CHANNEL_ID = os.getenv('DISCORD_CHANNEL_ID', 'YOUR_CHANNEL_ID')
DISCORD_API_URL = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"

# --- Model Configuration ---
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
WINDOW_SIZE = 10
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001