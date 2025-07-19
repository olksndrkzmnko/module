import asyncio
import logging
import time
import aiohttp
from config import load_config

logging.basicConfig(filename='telegram_utils.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CONFIG = load_config()
TELEGRAM_TOKEN = CONFIG.get('telegram_token')
CHAT_ID = CONFIG.get('chat_id')
telegram_buffer = []
last_telegram_flush = time.time()

async def send_telegram_buffered(message: str, force: bool = False) -> bool:
    """Надсилає буферизоване повідомлення в Telegram."""
    global telegram_buffer, last_telegram_flush
    try:
        if not TELEGRAM_TOKEN or not CHAT_ID:
            logging.error("TELEGRAM_TOKEN або CHAT_ID не вказано в config.yaml")
            return False
        telegram_buffer.append(str(message))
        current_time = time.time()
        if force or current_time - last_telegram_flush > CONFIG.get('telegram_flush_interval', 300) or len(telegram_buffer) >= CONFIG.get('telegram_buffer_size', 3):
            message_text = "\n".join(telegram_buffer)
            if len(message_text) > 4000:
                message_text = message_text[:4000] + "..."
            for attempt in range(CONFIG.get('telegram_max_attempts', 3)):
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                        async with session.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            json={"chat_id": CHAT_ID, "text": message_text, "parse_mode": "Markdown"}
                        ) as response:
                            response_data = await response.json()
                            if response.status == 429:
                                retry_after = response_data.get('parameters', {}).get('retry_after', 10)
                                logging.warning(f"RetryAfter: затримка на {retry_after} секунд, спроба {attempt + 1}")
                                await asyncio.sleep(retry_after)
                                continue
                            if response.status != 200:
                                logging.error(f"Помилка Telegram API: HTTP {response.status}, {response_data}")
                                if response.status == 401:
                                    raise ValueError("Некоректний TELEGRAM_TOKEN")
                                if response.status == 403:
                                    raise ValueError("Бот не має доступу до CHAT_ID")
                                if response.status == 404:
                                    raise ValueError("Некоректний URL або параметри запиту Telegram API")
                                continue
                            logging.info(f"Надіслано повідомлення в Telegram: {message_text[:100]}...")
                            telegram_buffer = []
                            last_telegram_flush = current_time
                            return True
                except Exception as e:
                    logging.error(f"Помилка відправки в Telegram, спроба {attempt + 1}: {str(e)}")
                    if attempt == CONFIG.get('telegram_max_attempts', 3) - 1:
                        logging.error("Досягнуто максимум спроб надсилання повідомлення")
                        telegram_buffer = []
                        await asyncio.sleep(5)
                        return False
                    continue
        return False
    except Exception as e:
        logging.error(f"Критична помилка в send_telegram_buffered: {str(e)}")
        telegram_buffer = []
        return False

async def test_telegram() -> bool:
    """Тестує підключення до Telegram API."""
    try:
        success = await send_telegram_buffered("Тестове повідомлення для перевірки Telegram API", force=True)
        if success:
            logging.info("Тестове повідомлення успішно надіслано в Telegram")
            return True
        else:
            logging.error("Не вдалося надіслати тестове повідомлення в Telegram")
            return False
    except Exception as e:
        logging.error(f"Помилка тестування Telegram: {str(e)}")
        return False