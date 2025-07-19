import asyncio
import logging
import os
import pickle
import sqlite3
import sys
import time
import ccxt.async_support as ccxt
import aiohttp
from config import load_config
from telegram_utils import send_telegram_buffered

# Налаштування логування
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Завантаження конфігурації
CONFIG = load_config()
PAIRS = CONFIG['pairs']
FEATURES = CONFIG.get('features', [
    'ema20', 'rsi', 'adx', 'macd', 'signal_line', 'bb_upper', 'bb_lower',
    'vwap', 'stoch_k', 'stoch_d', 'obv', 'roc', 'bollinger_width', 'atr',
    'momentum', 'ichimoku_tenkan', 'ichimoku_kijun'
])
LIQUIDITY_THRESHOLDS = CONFIG.get('liquidity_thresholds', {
    'BTC/USDT': 1.5,
    'ETH/USDT': 1.5,
    'SOL/USDT': 1.0,
    'XRP/USDT': 0.8,
    'POL/USDT': 0.3,
    'ADA/USDT': 0.5
})
DATA_CACHE = {}
PAIR_SETTINGS = {
    pair: {
        'signal_threshold_buy': 0.65,
        'signal_threshold_sell': 0.35,
        'min_risk_reward': 1.5 if pair in ['POL/USDT', 'ADA/USDT'] else 2.0,
        'active': True,
        'no_signal_counter': 0,
        'volatility_threshold': 0.0001 if pair in ['SOL/USDT', 'XRP/USDT', 'POL/USDT', 'ADA/USDT'] else 0.0002,
        'max_leverage': 50,
        'min_leverage': 5
    } for pair in PAIRS
}
exchange = None

async def initialize_exchange():
    """
    Ініціалізує асинхронний клієнт Binance.

    Returns:
        ccxt.binance: Ініціалізований об'єкт біржі.
    """
    global exchange
    try:
        exchange = ccxt.binance({
            'apiKey': CONFIG['binance_api_key'],
            'secret': CONFIG['binance_api_secret'],
            'enableRateLimit': True,
            'asyncio_loop': asyncio.get_event_loop()
        })
        await exchange.load_markets()
        logging.info("Біржа Binance успішно ініціалізована")
        return exchange
    except Exception as e:
        logging.error(f"Помилка ініціалізації біржі: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка ініціалізації біржі: {str(e)}", force=True)
        raise

async def get_historical_data(pair: str, timeframe: str, limit: int = 1000) -> 'pandas.DataFrame':
    """
    Отримує історичні дані для торгової пари з біржі.

    Args:
        pair (str): Торгова пара (наприклад, 'BTC/USDT').
        timeframe (str): Таймфрейм ('1m', '5m', '1h' тощо).
        limit (int): Кількість свічок.

    Returns:
        pandas.DataFrame: Дані з колонками ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    import pandas as pd
    global exchange, DATA_CACHE
    try:
        if not exchange:
            exchange = await initialize_exchange()

        cache_key = f"{pair}_{timeframe}_{limit}"
        if cache_key in DATA_CACHE and DATA_CACHE[cache_key]['timestamp'] > time.time() - 1800:
            logging.info(f"Використання кешованих даних для {pair} ({timeframe})")
            return DATA_CACHE[cache_key]['data'].copy()

        ohlcv = await exchange.fetch_ohlcv(pair, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 50:
            logging.error(f"Недостатньо даних для {pair} ({timeframe}): {len(ohlcv)} свічок")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        if df.isna().any().any():
            logging.warning(f"Знайдено NaN у даних для {pair} ({timeframe}), заповнюємо")
            df = df.ffill().bfill()

        DATA_CACHE[cache_key] = {'data': df, 'timestamp': time.time(), 'cache_duration': 1800}
        logging.info(f"Отримано {len(df)} рядків для {pair} ({timeframe})")
        return df
    except Exception as e:
        logging.error(f"Помилка отримання даних для {pair} ({timeframe}): {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка отримання даних для {pair} ({timeframe}): {str(e)}")
        return None

def save_to_cache(key: str, data: 'pandas.DataFrame') -> None:
    """
    Зберігає дані в кеш.

    Args:
        key (str): Ключ кешу.
        data (pandas.DataFrame): Дані для збереження.
    """
    global DATA_CACHE
    try:
        DATA_CACHE[key] = {'data': data.copy(), 'timestamp': time.time(), 'cache_duration': 1800}
        logging.info(f"Дані збережено в кеш: {key}, розмір={len(data)}")
    except Exception as e:
        logging.error(f"Помилка збереження в кеш: {str(e)}")

def load_from_cache(key: str) -> 'pandas.DataFrame':
    """
    Завантажує дані з кешу.

    Args:
        key (str): Ключ кешу.

    Returns:
        pandas.DataFrame: Дані з кешу або None.
    """
    global DATA_CACHE
    try:
        if key in DATA_CACHE and DATA_CACHE[key]['timestamp'] > time.time() - DATA_CACHE[key]['cache_duration']:
            logging.info(f"Завантажено дані з кешу: {key}")
            return DATA_CACHE[key]['data'].copy()
        return None
    except Exception as e:
        logging.error(f"Помилка завантаження з кешу: {str(e)}")
        return None

def validate_data(df: 'pandas.DataFrame', pair: str, timeframe: str, min_rows: int = 100) -> bool:
    """
    Валідує DataFrame з історичними даними.

    Args:
        df (pandas.DataFrame): Дані для перевірки.
        pair (str): Торгова пара.
        timeframe (str): Таймфрейм.
        min_rows (int): Мінімальна кількість рядків.

    Returns:
        bool: True, якщо дані валідні, False інакше.
    """
    import pandas as pd
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < min_rows:
            logging.error(f"Невалідний DataFrame для {pair} ({timeframe}): {len(df) if df is not None else 'None'} рядків")
            return False
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Відсутні необхідні стовпці для {pair} ({timeframe}): {df.columns.tolist()}")
            return False
        if df[required_columns].isna().any().any():
            logging.warning(f"Знайдено NaN у базових стовпцях для {pair} ({timeframe})")
            return False
        return True
    except Exception as e:
        logging.error(f"Помилка валідації даних для {pair} ({timeframe}): {str(e)}")
        return False

def clean_data_cache(max_size: int = 1000, max_age: int = 3600, max_bytes: int = 100_000_000) -> None:
    """
    Очищає кеш даних, видаляючи старі або надмірні записи.

    Args:
        max_size (int): Максимальна кількість записів у кеші.
        max_age (int): Максимальний час життя запису (секунди).
        max_bytes (int): Максимальний розмір кешу в байтах.
    """
    global DATA_CACHE
    current_time = time.time()
    keys_to_remove = []
    total_size = 0

    for key, value in DATA_CACHE.items():
        if current_time - value['timestamp'] > value.get('cache_duration', max_age):
            keys_to_remove.append(key)
        else:
            total_size += sys.getsizeof(value['data'])

    for key in keys_to_remove:
        DATA_CACHE.pop(key, None)

    if len(DATA_CACHE) > max_size or total_size > max_bytes:
        sorted_keys = sorted(DATA_CACHE, key=lambda k: DATA_CACHE[k]['timestamp'])
        for key in sorted_keys[:len(DATA_CACHE) - max_size]:
            DATA_CACHE.pop(key, None)

    logging.info(f"Очищено кеш, залишилося записів: {len(DATA_CACHE)}, розмір: {total_size / 1_000_000:.2f} МБ")

def calculate_rsi(df: 'pandas.DataFrame', periods: int = 14) -> 'pandas.Series':
    """
    Розраховує RSI (Relative Strength Index).

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        periods (int): Період для розрахунку.

    Returns:
        pandas.Series: Значення RSI.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns or len(df) < periods:
            logging.error(f"Некоректний DataFrame для RSI: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index)

        volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1] if len(df) >= 20 else 0.02
        periods = max(10, min(20, int(14 / (1 + volatility * 10))))
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.ffill().bfill().fillna(50.0)
        if rsi.isna().any():
            logging.warning("Знайдено NaN у RSI, заповнюємо 50.0")
            rsi = rsi.fillna(50.0)
        return rsi
    except Exception as e:
        logging.error(f"Помилка в calculate_rsi: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index)

def calculate_stochastic_oscillator(df: 'pandas.DataFrame', k_periods: int = 14, d_periods: int = 3) -> tuple['pandas.Series', 'pandas.Series']:
    """
    Розраховує Stochastic Oscillator (%K і %D).

    Args:
        df (pandas.DataFrame): Дані з колонками ['high', 'low', 'close'].
        k_periods (int): Період для %K.
        d_periods (int): Період для %D.

    Returns:
        tuple: (%K, %D) як pandas.Series.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logging.error(f"Некоректний DataFrame для Stochastic Oscillator: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)

        low_min = df['low'].rolling(window=k_periods, min_periods=1).min()
        high_max = df['high'].rolling(window=k_periods, min_periods=1).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        d = k.rolling(window=d_periods, min_periods=1).mean()
        k = k.ffill().bfill().fillna(50.0)
        d = d.ffill().bfill().fillna(50.0)
        if k.isna().any() or d.isna().any():
            logging.warning("Знайдено NaN у Stochastic Oscillator, заповнюємо 50.0")
            k = k.fillna(50.0)
            d = d.fillna(50.0)
        logging.info(f"Stochastic Oscillator: k={k.iloc[-1]:.4f}, d={d.iloc[-1]:.4f}")
        return k, d
    except Exception as e:
        logging.error(f"Помилка в calculate_stochastic_oscillator: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)

def calculate_obv(df: 'pandas.DataFrame') -> 'pandas.Series':
    """
    Розраховує On-Balance Volume (OBV).

    Args:
        df (pandas.DataFrame): Дані з колонками ['close', 'volume'].

    Returns:
        pandas.Series: Значення OBV.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or not all(col in df.columns for col in ['close', 'volume']):
            logging.error(f"Некоректний DataFrame для OBV: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index)
        direction = df['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        obv = (direction * df['volume']).cumsum()
        obv = obv.ffill().bfill().fillna(0.0)
        return obv
    except Exception as e:
        logging.error(f"Помилка в calculate_obv: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index)

def calculate_adx(df: 'pandas.DataFrame', periods: int = 14) -> 'pandas.Series':
    """
    Розраховує Average Directional Index (ADX).

    Args:
        df (pandas.DataFrame): Дані з колонками ['high', 'low', 'close'].
        periods (int): Період для розрахунку.

    Returns:
        pandas.Series: Значення ADX.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            logging.error(f"Некоректний DataFrame для ADX: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=periods).mean()
        plus_di = 100 * plus_dm.rolling(window=periods).mean() / (atr + 1e-8)
        minus_di = 100 * minus_dm.rolling(window=periods).mean() / (atr + 1e-8)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
        adx = dx.rolling(window=periods).mean()
        adx = adx.ffill().bfill().fillna(0.0)
        return adx
    except Exception as e:
        logging.error(f"Помилка в calculate_adx: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index)

def calculate_macd(df: 'pandas.DataFrame', fast: int = 12, slow: int = 26, signal: int = 9) -> tuple['pandas.Series', 'pandas.Series']:
    """
    Розраховує MACD і сигнальну лінію.

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        fast (int): Період швидкої EMA.
        slow (int): Період повільної EMA.
        signal (int): Період сигнальної лінії.

    Returns:
        tuple: (MACD, signal_line) як pandas.Series.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns:
            logging.error(f"Некоректний DataFrame для MACD: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd = macd.ffill().bfill().fillna(0.0)
        signal_line = signal_line.ffill().bfill().fillna(0.0)
        return macd, signal_line
    except Exception as e:
        logging.error(f"Помилка в calculate_macd: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)

def calculate_ichimoku(df: 'pandas.DataFrame') -> tuple['pandas.Series', 'pandas.Series']:
    """
    Розраховує Ichimoku Cloud (Tenkan-sen і Kijun-sen).

    Args:
        df (pandas.DataFrame): Дані з колонками ['high', 'low'].

    Returns:
        tuple: (tenkan, kijun) як pandas.Series.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or not all(col in df.columns for col in ['high', 'low']):
            logging.error(f"Некоректний DataFrame для Ichimoku: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)
        high_9 = df['high'].rolling(window=9, min_periods=1).max()
        low_9 = df['low'].rolling(window=9, min_periods=1).min()
        tenkan = (high_9 + low_9) / 2
        high_26 = df['high'].rolling(window=26, min_periods=1).max()
        low_26 = df['low'].rolling(window=26, min_periods=1).min()
        kijun = (high_26 + low_26) / 2
        tenkan = tenkan.ffill().bfill().fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        kijun = kijun.ffill().bfill().fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        if tenkan.isna().any() or kijun.isna().any():
            logging.warning("Знайдено NaN у Ichimoku, заповнюємо середнім close")
            tenkan = tenkan.fillna(df['close'].mean() if 'close' in df.columns else 0.0)
            kijun = kijun.fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        logging.info(f"Ichimoku: tenkan={tenkan.iloc[-1]:.4f}, kijun={kijun.iloc[-1]:.4f}")
        return tenkan, kijun
    except Exception as e:
        logging.error(f"Помилка в calculate_ichimoku: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index), pd.Series(np.zeros(len(df)), index=df.index)

def calculate_atr(df: 'pandas.DataFrame', periods: int = 14) -> 'pandas.Series':
    """
    Розраховує Average True Range (ATR).

    Args:
        df (pandas.DataFrame): Дані з колонками ['high', 'low', 'close'].
        periods (int): Період для розрахунку.

    Returns:
        pandas.Series: Значення ATR.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']) or len(df) < periods:
            logging.error(f"Некоректний DataFrame для ATR: {len(df)} рядків, стовпці={df.columns.tolist()}")
            mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.001
            return pd.Series(np.full(len(df), max(0.0001, mean_price * 0.001)), index=df.index)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=periods).mean()
        atr = atr.ffill().bfill()
        low_price_pairs = ['POL/USDT', 'ADA/USDT']
        atr_upper_limit = df['close'].mean() * 0.05 if not df.empty else 0.1
        atr_lower_limit = 0.0001 if df['close'].mean() < 1 else 0.001
        atr = atr.clip(lower=atr_lower_limit, upper=atr_upper_limit)
        if atr.isna().any() or (atr <= 0).any():
            logging.warning(f"ATR містить некоректні значення, заповнюємо мінімальним")
            mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.001
            atr = atr.fillna(max(0.0001, mean_price * 0.001))
        atr = atr.apply(lambda x: round(x, 3))
        logging.info(f"ATR: {atr.iloc[-1]:.3f}")
        return atr
    except Exception as e:
        logging.error(f"Помилка в calculate_atr: {str(e)}")
        mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.001
        return pd.Series(np.full(len(df), max(0.0001, mean_price * 0.001)), index=df.index)

def calculate_bollinger_bands(df: 'pandas.DataFrame', window: int = 20, num_std: int = 2) -> tuple['pandas.Series', 'pandas.Series']:
    """
    Розраховує Bollinger Bands.

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        window (int): Період для розрахунку.
        num_std (int): Кількість стандартних відхилень.

    Returns:
        tuple: (upper_band, lower_band) як pandas.Series.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns or len(df) < window:
            logging.error(f"Некоректний DataFrame для Bollinger Bands: {len(df)} рядків, стовпці={df.columns.tolist()}")
            mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.0
            return pd.Series(np.full(len(df), mean_price), index=df.index), pd.Series(np.full(len(df), mean_price), index=df.index)
        rolling_mean = df['close'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=window, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        upper_band = upper_band.ffill().bfill().fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        lower_band = lower_band.ffill().bfill().fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        if upper_band.isna().any() or lower_band.isna().any():
            logging.warning("Знайдено NaN у Bollinger Bands, заповнюємо середнім close")
            mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.0
            upper_band = upper_band.fillna(mean_price)
            lower_band = lower_band.fillna(mean_price)
        upper_band = upper_band.apply(lambda x: round(x, 3))
        lower_band = lower_band.apply(lambda x: round(x, 3))
        logging.info(f"Bollinger Bands: upper={upper_band.iloc[-1]:.3f}, lower={lower_band.iloc[-1]:.3f}")
        return upper_band, lower_band
    except Exception as e:
        logging.error(f"Помилка в calculate_bollinger_bands: {str(e)}")
        mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.0
        return pd.Series(np.full(len(df), mean_price), index=df.index), pd.Series(np.full(len(df), mean_price), index=df.index)

def calculate_vwap(df: 'pandas.DataFrame') -> 'pandas.Series':
    """
    Розраховує Volume Weighted Average Price (VWAP).

    Args:
        df (pandas.DataFrame): Дані з колонками ['high', 'low', 'close', 'volume'].

    Returns:
        pandas.Series: Значення VWAP.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            logging.error(f"Некоректний DataFrame для VWAP: {len(df)} рядків, стовпці={df.columns.tolist()}")
            mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.0
            return pd.Series(np.full(len(df), mean_price), index=df.index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume_sum = df['volume'].cumsum()
        vwap = (typical_price * df['volume']).cumsum() / (volume_sum + 1e-8)
        vwap = vwap.ffill().bfill()
        mean_close = df['close'].mean() if 'close' in df.columns else 0.0
        vwap = vwap.clip(lower=mean_close * 0.5, upper=mean_close * 1.5)
        if vwap.isna().any():
            logging.warning("Знайдено NaN у VWAP, заповнюємо середнім close")
            vwap = vwap.fillna(mean_close)
        vwap = vwap.apply(lambda x: round(x, 3))
        logging.info(f"VWAP: {vwap.iloc[-1]:.3f}")
        return vwap
    except Exception as e:
        logging.error(f"Помилка в calculate_vwap: {str(e)}")
        mean_price = df['close'].mean() if 'close' in df.columns and not df.empty else 0.0
        return pd.Series(np.full(len(df), mean_price), index=df.index)

def calculate_ema(df: 'pandas.DataFrame', span: int = 20) -> 'pandas.Series':
    """
    Розраховує Exponential Moving Average (EMA).

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        span (int): Період для EMA.

    Returns:
        pandas.Series: Значення EMA.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns or len(df) < span:
            logging.error(f"Некоректний DataFrame для EMA: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.full(len(df), 0.0), index=df.index)
        ema = df['close'].ewm(span=span, adjust=False).mean()
        ema = ema.ffill().bfill().fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        if ema.isna().any():
            logging.warning("Знайдено NaN у EMA, заповнюємо середнім close")
            ema = ema.fillna(df['close'].mean() if 'close' in df.columns else 0.0)
        logging.info(f"EMA{span}: {ema.iloc[-1]:.2f}")
        return ema
    except Exception as e:
        logging.error(f"Помилка в calculate_ema: {str(e)}")
        return pd.Series(np.full(len(df), df['close'].mean() if 'close' in df.columns and not df.empty else 0.0), index=df.index)

def calculate_roc(df: 'pandas.DataFrame', periods: int = 12) -> 'pandas.Series':
    """
    Розраховує Rate of Change (ROC).

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        periods (int): Період для розрахунку.

    Returns:
        pandas.Series: Значення ROC.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns or len(df) < periods:
            logging.error(f"Некоректний DataFrame для ROC: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index)
        roc = ((df['close'] - df['close'].shift(periods)) / (df['close'].shift(periods) + 1e-8)) * 100
        roc = roc.ffill().bfill().fillna(0.0)
        if roc.isna().any():
            logging.warning("Знайдено NaN у ROC, заповнюємо 0.0")
            roc = roc.fillna(0.0)
        return roc
    except Exception as e:
        logging.error(f"Помилка в calculate_roc: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index)

def calculate_bollinger_width(df: 'pandas.DataFrame', window: int = 20) -> 'pandas.Series':
    """
    Розраховує ширину Bollinger Bands.

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        window (int): Період для розрахунку.

    Returns:
        pandas.Series: Ширина Bollinger Bands.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns or len(df) < window:
            logging.error(f"Некоректний DataFrame для Bollinger Width: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.full(len(df), 0.01), index=df.index)
        rolling_mean = df['close'].rolling(window=window, min_periods=1).mean()
        rolling_std = df['close'].rolling(window=window, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        bb_width = (upper_band - lower_band) / (rolling_mean + 1e-8)
        bb_width = bb_width.ffill().bfill().fillna(0.01).clip(lower=0.01)
        logging.info(f"Bollinger Width: {bb_width.iloc[-1]:.4f}")
        return bb_width
    except Exception as e:
        logging.error(f"Помилка в calculate_bollinger_width: {str(e)}")
        return pd.Series(np.full(len(df), 0.01), index=df.index)

def calculate_support_resistance(df: 'pandas.DataFrame', window: int = 20) -> tuple[float, float]:
    """
    Розраховує рівні підтримки та опору.

    Args:
        df (pandas.DataFrame): Дані з колонками ['high', 'low'].
        window (int): Період для розрахунку.

    Returns:
        tuple: (support, resistance) як float.
    """
    import pandas as pd
    try:
        if df.empty or not all(col in df.columns for col in ['high', 'low']):
            logging.error(f"Некоректний DataFrame для Support/Resistance: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return df['close'].iloc[-1] * 0.98 if 'close' in df.columns and not df.empty else 0.0, df['close'].iloc[-1] * 1.02 if 'close' in df.columns and not df.empty else 0.0
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        support = low_min.iloc[-1]
        resistance = high_max.iloc[-1]
        if pd.isna(support) or pd.isna(resistance) or support >= resistance:
            logging.warning(f"Некоректні рівні підтримки/опору: support={support}, resistance={resistance}")
            support = df['close'].iloc[-1] * 0.98 if 'close' in df.columns else 0.0
            resistance = df['close'].iloc[-1] * 1.02 if 'close' in df.columns else 0.0
        return support, resistance
    except Exception as e:
        logging.error(f"Помилка в calculate_support_resistance: {str(e)}")
        return df['close'].iloc[-1] * 0.98 if 'close' in df.columns and not df.empty else 0.0, df['close'].iloc[-1] * 1.02 if 'close' in df.columns and not df.empty else 0.0

def calculate_momentum(df: 'pandas.DataFrame', periods: int = 10) -> 'pandas.Series':
    """
    Розраховує Momentum.

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        periods (int): Період для розрахунку.

    Returns:
        pandas.Series: Значення Momentum.
    """
    import pandas as pd
    import numpy as np
    try:
        if df.empty or 'close' not in df.columns or len(df) < periods:
            logging.error(f"Некоректний DataFrame для Momentum: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return pd.Series(np.zeros(len(df)), index=df.index)
        momentum = df['close'] - df['close'].shift(periods)
        momentum = momentum.ffill().bfill().fillna(0.0)
        if momentum.isna().any():
            logging.warning("Знайдено NaN у Momentum, заповнюємо 0.0")
            momentum = momentum.fillna(0.0)
        logging.info(f"Momentum: {momentum.iloc[-1]:.4f}")
        return momentum
    except Exception as e:
        logging.error(f"Помилка в calculate_momentum: {str(e)}")
        return pd.Series(np.zeros(len(df)), index=df.index)

def calculate_target(df: 'pandas.DataFrame', horizon: int = 5) -> tuple['pandas.Series', 'pandas.Index']:
    """
    Розраховує цільові мітки для машинного навчання.

    Args:
        df (pandas.DataFrame): Дані з колонкою 'close'.
        horizon (int): Горизонт прогнозу (кількість свічок).

    Returns:
        tuple: (y, valid_indices) або (None, None) у разі помилки.
    """
    import pandas as pd
    import numpy as np
    try:
        if df is None or df.empty or len(df) < horizon + 20 or 'close' not in df.columns:
            logging.error(f"Некоректний DataFrame для calculate_target: {len(df) if df is not None else 'None'} рядків, стовпці={df.columns.tolist() if df is not None else []}")
            return None, None
        df = df.copy()
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        valid_indices = df['future_return'].notna()
        if not valid_indices.any():
            logging.error("Немає валідних даних для міток: усі future_return є NaN")
            return None, None
        threshold = CONFIG.get('target_threshold', 0.003)
        volatility = df['close'].pct_change().std() * 100 if len(df) >= 20 else 0.02
        adjusted_threshold = max(0.001, min(0.008, threshold * (1 + volatility)))
        y = (df['future_return'][valid_indices] > adjusted_threshold).astype(int)
        if len(y) < 30:
            logging.warning(f"Недостатньо міток: {len(y)} (мін. 30)")
            return None, None
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logging.warning(f"calculate_target повернув один клас: {unique_classes}, поріг={adjusted_threshold:.4f}")
            adjusted_threshold *= 0.5
            y = (df['future_return'][valid_indices] > adjusted_threshold).astype(int)
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                logging.error(f"Після зниження порогу: один клас: {unique_classes}")
                return None, None
        logging.info(f"Мітки створено: {len(y)} міток, класи={unique_classes}, поріг={adjusted_threshold:.4f}")
        return y, valid_indices
    except Exception as e:
        logging.error(f"Помилка в calculate_target: {str(e)}")
        return None, None

async def calculate_all_indicators(df: 'pandas.DataFrame') -> 'pandas.DataFrame':
    """
    Розраховує всі технічні індикатори для DataFrame.

    Args:
        df (pandas.DataFrame): Дані з колонками ['open', 'high', 'low', 'close', 'volume'].

    Returns:
        pandas.DataFrame: Дані з доданими індикаторами.
    """
    import pandas as pd
    import numpy as np
    global DATA_CACHE, FEATURES
    try:
        if df is None or df.empty or len(df) < 20 or not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logging.error(f"Некоректний DataFrame для calculate_all_indicators: {len(df) if df is not None else 'None'} рядків, стовпці={df.columns.tolist() if df is not None else []}")
            return pd.DataFrame(index=df.index if df is not None else [], columns=FEATURES)
        cache_key = f"indicators_{hash(str(df.to_dict()))}"
        if cache_key in DATA_CACHE and DATA_CACHE[cache_key]['timestamp'] > time.time() - 600:
            logging.info(f"Використання кешованих індикаторів: {cache_key}")
            return DATA_CACHE[cache_key]['data'].copy()
        result_df = df.copy()
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if result_df[required_columns].isna().any().any():
            logging.warning(f"Знайдено NaN у базових стовпцях, заповнюємо")
            result_df[required_columns] = result_df[required_columns].ffill().bfill()
        indicators = [
            ('ema20', lambda df: calculate_ema(df, span=20)),
            ('rsi', calculate_rsi),
            ('adx', calculate_adx),
            ('macd', calculate_macd),
            ('bollinger_bands', calculate_bollinger_bands),
            ('vwap', calculate_vwap),
            ('stochastic_oscillator', calculate_stochastic_oscillator),
            ('obv', calculate_obv),
            ('roc', calculate_roc),
            ('bollinger_width', calculate_bollinger_width),
            ('atr', calculate_atr),
            ('momentum', calculate_momentum),
            ('ichimoku', calculate_ichimoku)
        ]
        for name, indicator in indicators:
            try:
                result = indicator(result_df)
                if name == 'ichimoku':
                    tenkan, kijun = result
                    result_df['ichimoku_tenkan'] = tenkan
                    result_df['ichimoku_kijun'] = kijun
                elif name == 'macd':
                    macd, signal_line = result
                    result_df['macd'] = macd
                    result_df['signal_line'] = signal_line
                elif name == 'bollinger_bands':
                    bb_upper, bb_lower = result
                    result_df['bb_upper'] = bb_upper
                    result_df['bb_lower'] = bb_lower
                elif name == 'stochastic_oscillator':
                    stoch_k, stoch_d = result
                    result_df['stoch_k'] = stoch_k
                    result_df['stoch_d'] = stoch_d
                else:
                    result_df[name] = result
                logging.info(f"Обчислено {name}: останнє значення={result.iloc[-1] if isinstance(result, pd.Series) else result[0].iloc[-1]:.4f}")
            except Exception as e:
                logging.error(f"Помилка обчислення {name}: {str(e)}")
                mean_price = result_df['close'].mean() if not result_df.empty and 'close' in result_df.columns else 0.0
                if name == 'ichimoku':
                    result_df['ichimoku_tenkan'] = pd.Series(np.full(len(result_df), mean_price), index=result_df.index)
                    result_df['ichimoku_kijun'] = pd.Series(np.full(len(result_df), mean_price), index=result_df.index)
                elif name == 'macd':
                    result_df['macd'] = pd.Series(np.zeros(len(result_df)), index=result_df.index)
                    result_df['signal_line'] = pd.Series(np.zeros(len(result_df)), index=result_df.index)
                elif name == 'bollinger_bands':
                    result_df['bb_upper'] = pd.Series(np.full(len(result_df), mean_price), index=result_df.index)
                    result_df['bb_lower'] = pd.Series(np.full(len(result_df), mean_price), index=result_df.index)
                elif name == 'stochastic_oscillator':
                    result_df['stoch_k'] = pd.Series(np.full(len(result_df), 50.0), index=result_df.index)
                    result_df['stoch_d'] = pd.Series(np.full(len(result_df), 50.0), index=result_df.index)
                else:
                    result_df[name] = pd.Series(np.zeros(len(result_df)), index=result_df.index)
        for feature in FEATURES:
            if feature not in result_df.columns:
                logging.warning(f"Відсутній стовпець {feature}, додаємо з нульовими значеннями")
                result_df[feature] = pd.Series(np.zeros(len(result_df)), index=result_df.index)
            if result_df[feature].isna().any():
                logging.warning(f"Знайдено NaN у {feature}, заповнюємо")
                result_df[feature] = result_df[feature].fillna(result_df['close'].mean() if feature in ['bb_upper', 'bb_lower', 'ichimoku_tenkan', 'ichimoku_kijun'] else 0.0)
        if result_df[FEATURES].isna().any().any():
            logging.error(f"NaN залишилися у {FEATURES}, заміна на 0")
            result_df[FEATURES] = result_df[FEATURES].fillna(0.0)
        DATA_CACHE[cache_key] = {'data': result_df, 'timestamp': time.time(), 'cache_duration': 600}
        save_to_cache(cache_key, result_df)
        logging.info(f"Збережено індикатори в кеш: {cache_key}, рядків: {len(result_df)}")
        return result_df
    except Exception as e:
        logging.error(f"Помилка в calculate_all_indicators: {str(e)}")
        return pd.DataFrame(index=df.index if df is not None else [], columns=FEATURES)

def detect_market_regime(df: 'pandas.DataFrame') -> str:
    """
    Визначає ринковий режим (trending, ranging, neutral).

    Args:
        df (pandas.DataFrame): Дані з колонками ['adx', 'bollinger_width'].

    Returns:
        str: Режим ринку ('trending', 'ranging', 'neutral').
    """
    import pandas as pd
    try:
        if df.empty or len(df) < 20 or not all(col in df.columns for col in ['adx', 'bollinger_width']):
            logging.warning(f"Некоректний DataFrame для detect_market_regime: {len(df)} рядків, стовпці={df.columns.tolist()}")
            return 'neutral'
        adx = df['adx'].iloc[-1]
        bb_width = df['bollinger_width'].iloc[-1]
        adx_threshold = CONFIG.get('adx_threshold', 25)
        bb_width_threshold_high = CONFIG.get('bb_width_threshold_high', 0.015)
        bb_width_threshold_low = CONFIG.get('bb_width_threshold_low', 0.01)
        if adx > adx_threshold and bb_width > bb_width_threshold_high:
            regime = 'trending'
        elif bb_width < bb_width_threshold_low:
            regime = 'ranging'
        else:
            regime = 'neutral'
        logging.info(f"Режим ринку: {regime}, ADX={adx:.2f}, bb_width={bb_width:.4f}")
        return regime
    except Exception as e:
        logging.error(f"Помилка в detect_market_regime: {str(e)}")
        return 'neutral'

def save_features_to_file() -> None:
    """
    Зберігає список ознак у файл.
    """
    global FEATURES
    try:
        with open('features.pkl', 'wb') as f:
            pickle.dump(FEATURES, f)
        logging.info("Список ознак збережено у features.pkl")
    except Exception as e:
        logging.error(f"Помилка збереження ознак: {str(e)}")

def load_features_from_file() -> None:
    """
    Завантажує список ознак із файлу.
    """
    global FEATURES
    try:
        if os.path.exists('features.pkl'):
            with open('features.pkl', 'rb') as f:
                FEATURES = pickle.load(f)
            logging.info(f"Список ознак завантажено: {FEATURES}")
        else:
            logging.info("Файл features.pkl не знайдено, використовується стандартний список ознак")
    except Exception as e:
        logging.error(f"Помилка завантаження ознак: {str(e)}")

async def save_to_db(pair: str, timeframe: str, df: 'pandas.DataFrame') -> None:
    """
    Зберігає агреговані дані в SQLite базу даних.

    Args:
        pair (str): Торгова пара.
        timeframe (str): Таймфрейм.
        df (pandas.DataFrame): Дані для збереження.
    """
    import pandas as pd
    try:
        db_path = CONFIG.get('db_path', 'trading_data.db')
        with sqlite3.connect(db_path) as conn:
            table_name = f"{pair.replace('/', '_')}_{timeframe}"
            df.to_sql(table_name, conn, if_exists='replace', index=True)
        logging.info(f"Дані збережено в базу даних: {table_name}, рядків={len(df)}")
    except Exception as e:
        logging.error(f"Помилка збереження даних у базу для {pair} ({timeframe}): {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка збереження даних у базу для {pair} ({timeframe}): {str(e)}")

async def aggregate_data(pair: str, target_timeframe: str, lower_timeframe: str = '5m', limit: int = 2000) -> 'pandas.DataFrame':
    """
    Агрегує дані з нижчого таймфрейму до цільового.

    Args:
        pair (str): Торгова пара.
        target_timeframe (str): Цільовий таймфрейм.
        lower_timeframe (str): Нижчий таймфрейм для агрегації.
        limit (int): Кількість свічок.

    Returns:
        pandas.DataFrame: Агреговані дані або None у разі помилки.
    """
    import pandas as pd
    global DATA_CACHE, PAIR_SETTINGS
    try:
        if not PAIR_SETTINGS.get(pair, {}).get('active', True):
            logging.warning(f"Пара {pair} відключена в pair_settings")
            return None
        multiplier = 12 if target_timeframe == '1h' else 48
        lower_limit = limit * multiplier
        logging.info(f"Запит даних {lower_timeframe} для агрегації в {target_timeframe}, пара={pair}, ліміт={lower_limit}")
        cache_key = f"{pair}_{target_timeframe}_aggregated"
        if cache_key in DATA_CACHE and DATA_CACHE[cache_key]['timestamp'] > time.time() - 1800:
            logging.info(f"Використання кешованих агрегованих даних для {pair} ({target_timeframe})")
            return DATA_CACHE[cache_key]['data'].copy()
        lower_df = await get_historical_data(pair, lower_timeframe, limit=lower_limit)
        if lower_df is None or len(lower_df) < limit * multiplier // 2:
            logging.error(f"Недостатньо даних {lower_timeframe} для {pair}: {len(lower_df) if lower_df is not None else 'None'} рядків")
            return None
        if lower_df.index.has_duplicates or lower_df.index.isna().any():
            logging.warning(f"Пропуски або дублікати в даних {lower_timeframe} для {pair}")
            lower_df = lower_df.drop_duplicates().reindex(
                pd.date_range(start=lower_df.index.min(), end=lower_df.index.max(), freq='5T')
            ).interpolate(method='linear').ffill().bfill()
        df = lower_df.resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        if len(df) < 100:
            logging.warning(f"Недостатньо агрегованих даних для {pair} ({target_timeframe}): {len(df)} рядків")
            return None
        await save_to_db(pair, target_timeframe, df)
        DATA_CACHE[cache_key] = {'data': df, 'timestamp': time.time(), 'cache_duration': 1800}
        logging.info(f"Агреговано {len(df)} рядків для {pair} ({target_timeframe}) з {lower_timeframe}")
        return df
    except Exception as e:
        logging.error(f"Помилка агрегації даних для {pair} ({target_timeframe}): {str(e)}")
        return None

async def check_liquidity(pair: str, df: 'pandas.DataFrame' = None) -> bool:
    """
    Перевіряє ліквідність торгової пари.

    Args:
        pair (str): Торгова пара.
        df (pandas.DataFrame, optional): Дані для оцінки волатильності.

    Returns:
        bool: True, якщо ліквідність достатня, False інакше.
    """
    global exchange
    try:
        if not exchange:
            exchange = await initialize_exchange()
        order_book = await exchange.fetch_order_book(pair, limit=10)
        bid_volume = sum(bid[1] for bid in order_book['bids'])
        ask_volume = sum(ask[1] for ask in order_book['asks'])
        total_volume = bid_volume + ask_volume
        volatility = df['close'].pct_change().rolling(window=20).std().iloc[-1] if df is not None and len(df) >= 20 and 'close' in df.columns else 0.02
        min_volume_threshold = LIQUIDITY_THRESHOLDS.get(pair, 0.5) * (1 - volatility * 0.3)
        if total_volume < min_volume_threshold:
            logging.warning(f"Низька ліквідність для {pair}: total_volume={total_volume:.3f}, поріг={min_volume_threshold:.3f}")
            return False
        logging.info(f"Ліквідність для {pair} достатня: total_volume={total_volume:.3f}, поріг={min_volume_threshold:.3f}")
        return True
    except Exception as e:
        logging.error(f"Помилка перевірки ліквідності для {pair}: {str(e)}")
        return False

async def fetch_social_sentiment(pair: str) -> float:
    """
    Отримує соціальний настрій для торгової пари (заглушка).

    Args:
        pair (str): Торгова пара.

    Returns:
        float: Значення соціального настрою (0.0 за замовчуванням).
    """
    logging.info(f"Соціальний настрій для {pair} відключено, повертаємо 0.0")
    return 0.0

async def fetch_fear_greed_index() -> float:
    """
    Отримує Fear and Greed Index.

    Returns:
        float: Значення індексу (50.0 у разі помилки).
    """
    global DATA_CACHE
    cache_key = "fear_greed_index"
    current_time = time.time()
    if cache_key in DATA_CACHE and current_time - DATA_CACHE[cache_key]['timestamp'] < 600:
        logging.info(f"Використання кешованого Fear and Greed Index: {DATA_CACHE[cache_key]['data']}")
        return DATA_CACHE[cache_key]['data']
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get('https://api.alternative.me/fng/') as response:
                    if response.status != 200:
                        logging.error(f"Помилка отримання Fear and Greed Index: HTTP {response.status}")
                        await asyncio.sleep(2 * (2 ** attempt))
                        continue
                    data = await response.json()
                    if not data.get('data') or not data['data'][0].get('value'):
                        logging.error(f"Некоректні дані Fear and Greed Index: {data}")
                        await send_telegram_buffered(f"⚠️ Некоректні дані Fear and Greed Index: {data}")
                        return 50.0
                    fng_value = float(data['data'][0]['value'])
                    DATA_CACHE[cache_key] = {'data': fng_value, 'timestamp': current_time, 'cache_duration': 600}
                    logging.info(f"Fear and Greed Index: {fng_value}")
                    return fng_value
        except Exception as e:
            logging.error(f"Помилка отримання Fear and Greed Index, спроба {attempt+1}: {str(e)}")
            await asyncio.sleep(2 * (2 ** attempt))
    logging.error("Не вдалося отримати Fear and Greed Index після 3 спроб")
    await send_telegram_buffered("⚠️ Не вдалося отримати Fear and Greed Index")
    return 50.0