import logging
import yaml
import os

# Налаштування логування
logging.basicConfig(
    filename='signals.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Шлях до файлу конфігурації
CONFIG_FILE = 'config.yaml'

def load_config():
    """
    Завантажує конфігурацію з файлу config.yaml і валідує її.

    Returns:
        dict: Словник із параметрами конфігурації.

    Raises:
        FileNotFoundError: Якщо файл config.yaml не знайдено.
        ValueError: Якщо обов’язкові параметри відсутні або некоректні.
    """
    if not os.path.exists(CONFIG_FILE):
        logging.error(f"Файл конфігурації {CONFIG_FILE} не знайдено")
        raise FileNotFoundError(f"Файл конфігурації {CONFIG_FILE} не знайдено")

    try:
        with open(CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        
        # Перевірка обов’язкових параметрів
        required_keys = [
            'telegram_token', 'chat_id', 'binance_api_key', 'binance_api_secret',
            'pairs', 'timeframe', 'max_drawdown_percent', 'trailing_stop_percent_base',
            'model_file', 'model_file_15m', 'signals_file', 'trades_file',
            'max_active_trades', 'simulate_mode', 'reserve_percent', 'daily_loss_limit'
        ]
        for key in required_keys:
            if key not in config:
                logging.error(f"Відсутній обов’язковий параметр '{key}' у {CONFIG_FILE}")
                raise ValueError(f"Відсутній обов’язковий параметр '{key}' у {CONFIG_FILE}")

        # Додаткова валідація
        if not isinstance(config['pairs'], list) or not config['pairs']:
            raise ValueError("'pairs' у config.yaml має бути непорожнім списком")
        if not isinstance(config['max_drawdown_percent'], (int, float)) or config['max_drawdown_percent'] <= 0:
            raise ValueError("'max_drawdown_percent' має бути додатнім числом")
        if not isinstance(config['simulate_mode'], bool):
            raise ValueError("'simulate_mode' має бути булевим значенням")

        logging.info("Конфігурація успішно завантажена")
        return config

    except yaml.YAMLError as e:
        logging.error(f"Помилка парсингу {CONFIG_FILE}: {str(e)}")
        raise ValueError(f"Помилка парсингу {CONFIG_FILE}: {str(e)}")
    except Exception as e:
        logging.error(f"Помилка завантаження конфігурації: {str(e)}")
        raise

def initialize_global_exchange(config):
    """
    Ініціалізація біржі (асинхронна). Поки заглушка, реалізується в trading.py.

    Args:
        config (dict): Конфігурація з load_config.

    Returns:
        None
    """
    # Реалізація буде в trading.py
    logging.info("Ініціалізація біржі (заглушка)")
    return None