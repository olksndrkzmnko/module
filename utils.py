import asyncio
import logging
import pandas as pd
import sqlite3
import ccxt.async_support as ccxt
import os
import pickle
import time
from config import data_cache, exchange

async def save_to_cache(cache_key, data):
    try:
        safe_cache_key = cache_key.replace('/', '_')
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{safe_cache_key}.pkl")
        if not os.access(cache_dir, os.W_OK):
            raise PermissionError(f"Немає прав на запис у папку '{cache_dir}'")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Кеш успішно збережено для {cache_key} за шляхом {cache_path}")
    except PermissionError as e:
        logging.error(f"Помилка прав доступу при збереженні кешу для {cache_key}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Помилка збереження кешу для {cache_key}: {str(e)}")
        raise

async def load_from_cache(cache_key):
    try:
        safe_cache_key = cache_key.replace('/', '_')
        cache_path = f"cache/{safe_cache_key}.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Кеш успішно завантажено для {cache_key}: {len(data)} рядків")
            return data
        else:
            logging.info(f"Кеш не знайдено для {cache_key}")
            return None
    except Exception as e:
        logging.error(f"Помилка завантаження кешу для {cache_key}: {str(e)}")
        return None

def validate_data(df, pair: str, timeframe: str, min_rows: int = 20) -> bool:
    try:
        if df is None or df.empty:
            logging.error(f"Порожні дані для {pair} ({timeframe}): {len(df) if df is not None else 'None'}")
            return False
        if len(df) < min_rows:
            logging.error(f"Недостатньо даних для {pair} ({timeframe}): {len(df)} рядків, потрібно щонайменше {min_rows}")
            return False
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Відсутні необхідні стовпці для {pair} ({timeframe}): {df.columns.tolist()}")
            return False
        if df[required_columns].isna().any().any():
            logging.warning(f"Знайдено NaN у базових стовпцях для {pair} ({timeframe}), заповнюємо")
            df[required_columns] = df[required_columns].ffill().bfill()
            if df[required_columns].isna().any().any():
                logging.error(f"Не вдалося заповнити NaN для {pair} ({timeframe})")
                return False
        return True
    except Exception as e:
        logging.error(f"Помилка валідації даних для {pair} ({timeframe}): {str(e)}")
        return False

def save_to_db(pair, timeframe, df):
    try:
        conn = sqlite3.connect('market_data.db')
        df.to_sql(f"{pair.replace('/', '_')}_{timeframe}", conn, if_exists='replace', index=True)
        conn.commit()
        logging.info(f"Дані збережено в базу для {pair} ({timeframe}): {len(df)} рядків")
    except Exception as e:
        logging.error(f"Помилка збереження даних у базу для {pair} ({timeframe}): {str(e)}")
    finally:
        conn.close()

def load_from_db(pair, timeframe):
    try:
        conn = sqlite3.connect('market_data.db')
        df = pd.read_sql(f"SELECT * FROM {pair.replace('/', '_')}_{timeframe}", conn, index_col='timestamp')
        df.index = pd.to_datetime(df.index)
        logging.info(f"Завантажено {len(df)} рядків з бази для {pair} ({timeframe})")
        return df
    except Exception as e:
        logging.info(f"Дані для {pair} ({timeframe}) не знайдено в базі або помилка: {str(e)}")
        return None
    finally:
        conn.close()

def init_db_table(pair, timeframe):
    try:
        conn = sqlite3.connect('market_data.db')
        cursor = conn.cursor()
        table_name = f"{pair.replace('/', '_')}_{timeframe}"
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        """)
        conn.commit()
        logging.info(f"Таблицю {table_name} ініціалізовано або перевірено")
    except Exception as e:
        logging.error(f"Помилка ініціалізації таблиці для {pair} ({timeframe}): {str(e)}")
    finally:
        conn.close()

async def get_historical_data(pair: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
    global data_cache, exchange
    if exchange is None:
        logging.error("Біржа не ініціалізована")
        return None
    safe_pair = pair.replace('/', '_')
    cache_key = f"{safe_pair}_{timeframe}_{limit}"
    
    cached_data = await load_from_cache(cache_key)
    if cached_data is not None and validate_data(cached_data, pair, timeframe):
        logging.info(f"Використання кешованих даних для {pair} ({timeframe})")
        return cached_data.copy()

    df = load_from_db(pair, timeframe)
    if df is not None and len(df) >= limit and validate_data(df, pair, timeframe):
        logging.info(f"Завантажено {len(df)} рядків з локальної бази для {pair} ({timeframe})")
        await save_to_cache(cache_key, df)
        return df.copy()

    init_db_table(pair, timeframe)

    min_data = 100 if timeframe in ['1m', '5m'] else 50
    all_ohlcv = []
    max_retries = 5
    max_per_request = 1000
    target_limit = 5000 if timeframe in ['1h', '4h'] else limit

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Запит даних для {pair} ({timeframe}), спроба {attempt}, ліміт={target_limit}")
            since = None
            while len(all_ohlcv) < target_limit:
                remaining = target_limit - len(all_ohlcv)
                request_limit = min(max_per_request, remaining)
                ohlcv = await exchange.fetch_ohlcv(pair, timeframe, since=since, limit=request_limit)
                if not ohlcv:
                    logging.warning(f"Порожні дані для {pair} ({timeframe}) на спробі {attempt}")
                    break
                all_ohlcv.extend(ohlcv)
                if len(ohlcv) < request_limit:
                    break
                since = ohlcv[-1][0] + 1

                if all_ohlcv:
                    temp_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
                    temp_df.set_index('timestamp', inplace=True)
                    save_to_db(pair, timeframe, temp_df)

            if not all_ohlcv:
                logging.warning(f"Порожні дані для {pair} ({timeframe}) після запиту")
                await asyncio.sleep(2 ** attempt)
                continue

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            if not validate_data(df, pair, timeframe, min_rows=min_data):
                if timeframe in ['1h', '4h']:
                    lower_timeframe = '5m'
                    multiplier = 12 if timeframe == '1h' else 48
                    lower_limit = target_limit * multiplier
                    logging.info(f"Агрегація даних з {lower_timeframe} для {pair} ({timeframe}), ліміт={lower_limit}")
                    lower_df = await get_historical_data(pair, lower_timeframe, limit=lower_limit)
                    if lower_df is not None and len(lower_df) >= min_data * multiplier // 2:
                        df = lower_df.resample(timeframe).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        logging.info(f"Агреговано {len(df)} рядків для {pair} ({timeframe}) з {lower_timeframe}")

            if not validate_data(df, pair, timeframe, min_rows=min_data):
                logging.error(f"Не вдалося отримати коректні дані для {pair} ({timeframe}): {len(df)} рядків")
                await asyncio.sleep(2 ** attempt)
                continue

            save_to_db(pair, timeframe, df)
            await save_to_cache(cache_key, df)  # Збереження кешу лише один раз
            logging.info(f"Отримано {len(df)} рядків для {pair} ({timeframe}), стовпці={df.columns.tolist()}")
            return df

        except ccxt.NetworkError as e:
            logging.error(f"Мережева помилка для {pair} ({timeframe}) на спробі {attempt}: {str(e)}")
            await asyncio.sleep(2 ** attempt)
        except ccxt.RateLimitExceeded as e:
            retry_after = int(getattr(e, 'response', {}).get('headers', {}).get('Retry-After', 2 ** (attempt + 1)))
            logging.error(f"Перевищено ліміт запитів для {pair} ({timeframe}) на спробі {attempt}, затримка {retry_after} секунд")
            await asyncio.sleep(retry_after)
        except Exception as e:
            logging.error(f"Помилка для {pair} ({timeframe}) на спробі {attempt}: {str(e)}")
            await asyncio.sleep(2 ** attempt)

    logging.error(f"Не вдалося отримати дані для {pair} ({timeframe}) після {max_retries} спроб")
    return None