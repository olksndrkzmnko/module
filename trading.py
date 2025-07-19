import asyncio
import logging
import uuid
import ccxt.async_support as ccxt
from datetime import datetime
from config import load_config
from data_processing import get_historical_data, validate_data, calculate_all_indicators, detect_market_regime, calculate_atr
from telegram_utils import send_telegram_buffered
from machine_learning import train_ml_model
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application

# Налаштування логування
logging.basicConfig(
    filename='signals.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Завантаження конфігурації
CONFIG = load_config()
PAIRS = CONFIG['pairs']
TIMEFRAMES = [CONFIG['timeframe']]
SWING_TIMEFRAMES = CONFIG['swing_timeframes']
FEATURES = ['ema20', 'rsi', 'adx', 'macd', 'signal_line', 'bb_upper', 'bb_lower', 'vwap', 'stoch_k', 'stoch_d', 'obv', 'roc', 'bollinger_width', 'atr', 'momentum', 'ichimoku_tenkan', 'ichimoku_kijun']
SIMULATE_MODE = CONFIG['simulate_mode']
CHAT_ID = CONFIG['chat_id']
VOLATILITY_THRESHOLD = CONFIG['volatility_threshold']
TRADES_CSV = CONFIG['trades_csv']

# Глобальні змінні
ml_models = {}
pair_settings = {
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
pair_signal_status = {pair: "Немає даних" for pair in PAIRS}
pending_signals = {}
last_signal_time = {}
pair_performance = {pair: {'profit': 0, 'trades': 0, 'win_rate': 0} for pair in PAIRS}
risk_guard = None  # Має бути ініціалізований у main.py
application = None  # Має бути ініціалізований у telegram_bot.py
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

async def calculate_trade_size(pair: str, signal_prob: float, balance: float, price: float, atr: float, leverage: int, timeframe: str = '1h') -> float:
    """
    Розраховує розмір торгової позиції з урахуванням ризик-менеджменту.

    Args:
        pair (str): Торгова пара (наприклад, 'BTC/USDT').
        signal_prob (float): Ймовірність сигналу (0-1).
        balance (float): Доступний баланс у USDT.
        price (float): Поточна ціна активу.
        atr (float): Значення ATR для оцінки волатильності.
        leverage (int): Плече для позиції.
        timeframe (str): Таймфрейм для торгівлі.

    Returns:
        float: Розмір позиції (кількість активу) або 0.0 у разі помилки.
    """
    try:
        if not exchange or pair not in exchange.markets:
            logging.error(f"Пара {pair} недоступна на біржі або біржа не ініціалізована")
            await send_telegram_buffered(f"⚠️ Пара {pair} недоступна на біржі", force=True)
            return 0.0

        # Перевірка вхідних параметрів
        if not all(isinstance(x, (int, float)) and x > 0 for x in [balance, price, signal_prob, atr, leverage]):
            logging.error(f"Некоректні вхідні параметри для {pair}: balance={balance}, price={price}, signal_prob={signal_prob}, atr={atr}, leverage={leverage}")
            return 0.0

        # Перевірка порогу сигналу
        threshold = pair_settings.get(pair, {}).get('signal_threshold_buy', 0.7)
        if signal_prob < threshold:
            logging.info(f"Пропуск розрахунку для {pair}: низька ймовірність сигналу ({signal_prob:.3f} < {threshold:.3f})")
            return 0.0

        # Налаштування ризику залежно від таймфрейму
        risk_per_trade = pair_settings.get(pair, {}).get('risk_per_trade', {'1m': 0.005, '5m': 0.01, '1h': 0.02, '4h': 0.03}).get(timeframe, 0.01)
        available_balance = balance * risk_per_trade

        # Отримання інформації про символ
        symbol_info = exchange.markets.get(pair, {})
        min_amount = symbol_info.get('limits', {}).get('amount', {}).get('min', 0.001)
        min_notional = symbol_info.get('limits', {}).get('cost', {}).get('min', 10.0)
        amount_precision = symbol_info.get('precision', {}).get('amount', 4)

        # Динамічне налаштування Kelly Criterion
        win_rate = pair_settings.get(pair, {}).get('win_rate', 0.5)
        avg_win_loss_ratio = pair_settings.get(pair, {}).get('avg_win_loss_ratio', 2.0)
        volatility = atr / price if price > 0 else 0.0
        kelly_fraction = risk_guard.calculate_kelly_position(signal_prob, win_rate, avg_win_loss_ratio, volatility) if risk_guard else 0.2
        kelly_fraction = min(max(kelly_fraction, 0.1), 0.5)

        # Обчислення розміру позиції
        adjusted_risk = available_balance * kelly_fraction
        trade_size = (adjusted_risk / price) * leverage

        # Перевірка мінімального notional
        notional = trade_size * price
        if notional < min_notional:
            logging.warning(f"Розмір позиції для {pair} нижче мінімального notional: {notional:.2f} < {min_notional:.2f}")
            trade_size = min_notional / price

        # Перевірка мінімального розміру позиції
        if trade_size < min_amount:
            logging.warning(f"Розмір позиції для {pair} замалий: {trade_size:.6f} < {min_amount}, коригуємо до мінімального")
            trade_size = min_amount

        # Обмеження максимального розміру позиції
        max_position = (balance * 0.1 / price) * leverage
        if trade_size > max_position:
            logging.warning(f"Розмір позиції для {pair} перевищує максимум: {trade_size:.6f} > {max_position:.6f}, коригуємо")
            trade_size = max_position

        # Форматування до необхідної точності
        trade_size = round(trade_size, amount_precision)

        # Остаточна перевірка на валідність
        if trade_size <= 0:
            logging.error(f"Розмір позиції для {pair} дорівнює 0 після всіх коригувань")
            return 0.0

        # Перевірка балансу через API
        try:
            balance_info = await exchange.fetch_balance()
            available_usdt = float(balance_info.get('USDT', {}).get('free', 0.0))
            if available_usdt < notional:
                logging.error(f"Недостатньо коштів для {pair}: потрібен {notional:.2f} USDT, доступно {available_usdt:.2f} USDT")
                return 0.0
        except Exception as api_error:
            logging.error(f"Помилка API при перевірці балансу для {pair}: {str(api_error)}")
            return 0.0

        logging.info(f"Розмір угоди для {pair}: {trade_size:.6f} (balance={balance:.2f}, price={price:.2f}, signal_prob={signal_prob:.3f}, kelly={kelly_fraction:.3f}, atr={atr:.4f}, notional={notional:.2f}, timeframe={timeframe})")
        return trade_size

    except Exception as e:
        logging.error(f"Помилка обчислення розміру угоди для {pair}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка обчислення розміру угоди для {pair}: {str(e)}", force=True)
        return 0.0

async def generate_signal(pair: str, df: 'pandas.DataFrame', timeframe: str) -> dict:
    """
    Генерує торговий сигнал на основі даних та моделей машинного навчання.

    Args:
        pair (str): Торгова пара.
        df (pandas.DataFrame): Дані з індикаторами.
        timeframe (str): Таймфрейм для аналізу.

    Returns:
        dict: Словник із параметрами сигналу.
    """
    global ml_models, FEATURES
    try:
        import pandas as pd
        import numpy as np

        if df.empty or not all(f in df.columns for f in FEATURES):
            logging.error(f"Некоректний DataFrame або відсутні ознаки для {pair} ({timeframe}): {df.columns.tolist()}")
            return {
                'signal': 'wait',
                'confidence': 0.0,
                'explanation': 'Некоректні дані або відсутні ознаки',
                'current_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'leverage': 1,
                'position_size': 0.0,
                'trailing_stop': 0.0,
                'callback_rate': 0.0,
                'sentiment': 'neutral',
                'price_prob': 0.0,
                'signal_prob': 0.0,
                'regime': 'neutral'
            }

        # Визначення ринкового режиму
        regime = detect_market_regime(df)
        current_price = float(df['close'].iloc[-1])
        atr = float(df['atr'].iloc[-1]) if 'atr' in df else 0.001 * current_price
        rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df else 50.0
        macd = float(df['macd'].iloc[-1]) if 'macd' in df else 0.0
        signal_line = float(df['signal_line'].iloc[-1]) if 'signal_line' in df else 0.0

        # Перевірка ініціалізації моделей і скейлера
        if not ml_models.get(pair, {}).get(timeframe, {}).get(regime, {}).get('models') or not ml_models[pair][timeframe][regime]['scaler']:
            logging.warning(f"Моделі або скейлер не ініціалізовані для {pair} ({timeframe}, {regime}), запускаємо навчання")
            await train_ml_model([pair], timeframe, regime)
            if not ml_models.get(pair, {}).get(timeframe, {}).get(regime, {}).get('models') or not ml_models[pair][timeframe][regime]['scaler']:
                logging.error(f"Не вдалося ініціалізувати моделі для {pair} ({timeframe}, {regime})")
                return {
                    'signal': 'wait',
                    'confidence': 0.0,
                    'explanation': 'Моделі або скейлер не ініціалізовані',
                    'current_price': round(current_price, 3),
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'leverage': 1,
                    'position_size': 0.0,
                    'trailing_stop': 0.0,
                    'callback_rate': 0.0,
                    'sentiment': regime,
                    'price_prob': 0.0,
                    'signal_prob': 0.0,
                    'regime': regime
                }

        # Підготовка даних для передбачення
        X = df[FEATURES].iloc[-1].values.reshape(1, -1)
        scaler = ml_models[pair][timeframe][regime]['scaler']
        X_scaled = scaler.transform(X)

        # Передбачення
        models = ml_models[pair][timeframe][regime]['models']
        valid_models = [model for model in models if hasattr(model, 'predict_proba')]
        if valid_models:
            signal_prob = np.mean([model.predict_proba(X_scaled)[0][1] for model in valid_models])
            signal_prob = float(signal_prob)
            logging.info(f"Передбачення для {pair} ({timeframe}, {regime}): signal_prob={signal_prob:.2f}")
        else:
            logging.warning(f"Жодна модель для {pair} ({timeframe}, {regime}) не підтримує predict_proba")
            signal_prob = 0.5

        # Логіка сигналів
        signal = 'wait'
        confidence = 0.0
        explanation = ''
        stop_loss = 0.0
        take_profit = 0.0
        leverage = 1
        position_size = 0.0
        trailing_stop = 0.0
        callback_rate = 0.0

        if regime == 'trending':
            if signal_prob > 0.65 and rsi < 70 and macd > signal_line:
                signal = 'buy'
                confidence = min(signal_prob, 0.8)
                stop_loss = current_price - 2 * atr
                take_profit = current_price + 3 * atr
                leverage = 3
                position_size = 0.01
                trailing_stop = atr
                callback_rate = 0.1
                explanation = f"Трендовий ринок: RSI={rsi:.2f}, MACD={macd:.2f}, ATR={atr:.2f}, ML Prob={signal_prob:.2f}"
            elif signal_prob < 0.35 and rsi > 70 and macd < signal_line:
                signal = 'sell'
                confidence = min(1 - signal_prob, 0.8)
                stop_loss = current_price + 2 * atr
                take_profit = current_price - 3 * atr
                leverage = 3
                position_size = 0.01
                trailing_stop = atr
                callback_rate = 0.1
                explanation = f"Трендовий ринок: RSI={rsi:.2f}, MACD={macd:.2f}, ATR={atr:.2f}, ML Prob={signal_prob:.2f}"
        else:  # ranging
            if signal_prob > 0.65 and df['close'].iloc[-1] < df['bb_lower'].iloc[-1] and rsi < 30:
                signal = 'buy'
                confidence = min(signal_prob, 0.7)
                stop_loss = current_price - 1.5 * atr
                take_profit = current_price + 2 * atr
                leverage = 2
                position_size = 0.01
                trailing_stop = atr * 0.5
                callback_rate = 0.05
                explanation = f"Рейндж ринок: ціна нижче Bollinger, RSI={rsi:.2f}, ML Prob={signal_prob:.2f}"
            elif signal_prob < 0.35 and df['close'].iloc[-1] > df['bb_upper'].iloc[-1] and rsi > 70:
                signal = 'sell'
                confidence = min(1 - signal_prob, 0.7)
                stop_loss = current_price + 1.5 * atr
                take_profit = current_price - 2 * atr
                leverage = 2
                position_size = 0.01
                trailing_stop = atr * 0.5
                callback_rate = 0.05
                explanation = f"Рейндж ринок: ціна вище Bollinger, RSI={rsi:.2f}, ML Prob={signal_prob:.2f}"

        # Форматування чисел
        formatted_current_price = round(current_price, 3)
        formatted_stop_loss = round(stop_loss, 3)
        formatted_take_profit = round(take_profit, 3)
        formatted_trailing_stop = round(trailing_stop, 3)
        formatted_confidence = round(confidence, 3)
        formatted_signal_prob = round(signal_prob, 3)
        formatted_position_size = round(position_size, 3)
        formatted_callback_rate = round(callback_rate, 3)

        logging.info(
            f"Сигнал для {pair} ({timeframe}): {signal}, впевненість={formatted_confidence:.3f}, "
            f"режим={regime}, signal_prob={formatted_signal_prob:.3f}, "
            f"price={formatted_current_price:.3f}, SL={formatted_stop_loss:.3f}, "
            f"TP={formatted_take_profit:.3f}, TS={formatted_trailing_stop:.3f}"
        )

        return {
            'signal': signal,
            'confidence': formatted_confidence,
            'explanation': explanation,
            'current_price': formatted_current_price,
            'stop_loss': formatted_stop_loss,
            'take_profit': formatted_take_profit,
            'leverage': int(leverage),
            'position_size': formatted_position_size,
            'trailing_stop': formatted_trailing_stop,
            'callback_rate': formatted_callback_rate,
            'sentiment': regime,
            'price_prob': formatted_signal_prob,
            'signal_prob': formatted_signal_prob,
            'regime': regime
        }
    except Exception as e:
        logging.error(f"Помилка в generate_signal для {pair} ({timeframe}): {str(e)}")
        return {
            'signal': 'wait',
            'confidence': 0.0,
            'explanation': f'Помилка: {str(e)}',
            'current_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'leverage': 1,
            'position_size': 0.0,
            'trailing_stop': 0.0,
            'callback_rate': 0.0,
            'sentiment': 'neutral',
            'price_prob': 0.0,
            'signal_prob': 0.0,
            'regime': 'neutral'
        }

async def send_trade_signal(pair: str, action: str, leverage: int, price: float, stop_loss: float, take_profit: float, signal_prob: float, explanation: str, sentiment: str, price_prob: float, position_size: float, trailing_stop: float, callback_rate: float, timeframe: str = '1m') -> tuple[str, str]:
    """
    Надсилає торговий сигнал у Telegram для підтвердження.

    Args:
        pair (str): Торгова пара.
        action (str): Дія ('buy' або 'sell').
        leverage (int): Плече.
        price (float): Ціна входу.
        stop_loss (float): Стоп-лосс.
        take_profit (float): Тейк-профіт.
        signal_prob (float): Ймовірність сигналу.
        explanation (str): Пояснення сигналу.
        sentiment (str): Режим ринку.
        price_prob (float): Ймовірність ціни.
        position_size (float): Розмір позиції.
        trailing_stop (float): Трейлінг-стоп.
        callback_rate (float): Callback rate для трейлінг-стопу.
        timeframe (str): Таймфрейм.

    Returns:
        tuple: (signal_id, reason) або (None, error_message).
    """
    global risk_guard, pending_signals, last_signal_time, application, exchange

    if not exchange:
        reason = "Біржа не ініціалізована"
        logging.error(reason)
        await send_telegram_buffered(f"⚠️ Помилка: {reason}")
        return None, reason

    # Перевірка частоти сигналів
    signal_key = f"{pair}_{timeframe}"
    current_time = time.time()
    min_signal_interval = 300  # 5 хвилин
    if signal_key in last_signal_time and current_time - last_signal_time[signal_key] < min_signal_interval:
        reason = f"Пропуск сигналу для {pair} ({timeframe}): занадто скоро після попереднього сигналу"
        logging.info(reason)
        return None, reason

    # Перевірка історичної прибутковості
    win_rate = pair_performance.get(pair, {}).get('win_rate', 0.0)
    if win_rate < 0.45 and pair_performance.get(pair, {}).get('trades', 0) >= 10:
        reason = f"Пропуск сигналу для {pair} ({timeframe}): низький win_rate={win_rate:.2f}"
        logging.info(reason)
        return None, reason

    try:
        # Валідація параметрів сигналу
        signal_params = {
            'signal_prob': signal_prob,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'position_size': position_size,
            'leverage': leverage,
            'callback_rate': callback_rate
        }
        is_valid, error_message = validate_signal_params(signal_params)
        if not is_valid:
            reason = f"Некоректні параметри сигналу для {pair} ({timeframe}): {error_message}"
            logging.error(reason)
            await send_telegram_buffered(f"⚠️ Помилка: {reason}")
            return None, reason

        # Форматування чисел
        signal_prob = round(float(signal_prob), 3)
        price = round(float(price), 3)
        stop_loss = round(float(stop_loss), 3)
        take_profit = round(float(take_profit), 3)
        trailing_stop = round(float(trailing_stop), 3)
        position_size = round(float(position_size), 3)
        leverage = int(leverage)
        callback_rate = round(float(callback_rate), 3) if callback_rate is not None else 0.0

        # Перевірка впевненості сигналу
        if signal_prob <= 0.55:
            reason = f"Пропуск сигналу для {pair} ({timeframe}): низька впевненість={signal_prob:.3f}"
            logging.info(reason)
            return None, reason

        # Визначення точності
        low_price_pairs = ['POL/USDT', 'ADA/USDT']
        price_precision = 4
        amount_precision = 4
        if pair in exchange.markets:
            price_precision = int(exchange.markets[pair]['precision']['price']) if isinstance(exchange.markets[pair]['precision']['price'], (int, float)) else 4
            amount_precision = int(exchange.markets[pair]['precision']['amount']) if isinstance(exchange.markets[pair]['precision']['amount'], (int, float)) else 4
        if pair in low_price_pairs:
            price_precision = max(price_precision, 4)

        # Перевірка цін
        if any(price <= 0 for price in [price, stop_loss, take_profit, trailing_stop]):
            ticker = await exchange.fetch_ticker(pair)
            price = round(float(ticker['last']), 3)
            df = await get_historical_data(pair, timeframe, limit=50)
            atr = round(float(calculate_atr(df).iloc[-1]), 3) if df is not None else round(0.002 * price, 3)
            atr_ratio = min(atr / price, 0.5) if pair not in low_price_pairs else min(atr / price, 0.3)
            stop_loss = round(price * (1 - atr_ratio * 2.0), 3) if action == 'buy' else round(price * (1 + atr_ratio * 2.0), 3)
            take_profit = round(price * (1 + atr_ratio * 3.0), 3) if action == 'buy' else round(price * (1 - atr_ratio * 3.0), 3)
            trailing_stop = round(atr, 3)
            explanation += f"\nЦіни виправлено: SL={atr_ratio*2.0:.2%} від ціни, TP={atr_ratio*3.0:.2%} від ціни, Трейлінг-стоп=ATR ({atr:.3f})."
            logging.info(f"Виправлено ціни для {pair} ({timeframe}): price={price:.3f}, stop_loss={stop_loss:.3f}, take_profit={take_profit:.3f}, trailing_stop={trailing_stop:.3f}")

        price = round(price, price_precision)
        stop_loss = round(stop_loss, price_precision)
        take_profit = round(take_profit, price_precision)
        trailing_stop = round(trailing_stop, price_precision)
        position_size = round(position_size, amount_precision)

        # Перевірка співвідношення ризик/прибуток
        min_risk_reward = pair_settings.get(pair, {}).get('min_risk_reward', 1.5)
        risk_reward_ratio = abs(take_profit - price) / max(abs(price - stop_loss), 1e-4)
        if risk_reward_ratio < min_risk_reward:
            reason = f"Пропуск сигналу для {pair} ({timeframe}): низьке співвідношення R:R={risk_reward_ratio:.2f} (мін. {min_risk_reward})"
            logging.info(reason)
            return None, reason

        # Перевірка кореляційних ризиків (заглушка, реалізується в risk_management.py)
        # correlation_matrix = await update_correlation_matrix(PAIRS)
        # if not risk_guard.check_correlated_risk(pair, correlation_matrix):
        #     reason = f"Пропуск сигналу для {pair} ({timeframe}): високий кореляційний ризик"
        #     logging.info(reason)
        #     return None, reason

        # Параметри ордера
        params = {
            'stopLossPrice': str(stop_loss),
            'takeProfitPrice': str(take_profit),
            'leverage': str(leverage)
        }
        if callback_rate is not None and pair in exchange.markets and exchange.markets[pair].get('trailingStop', False):
            params['callbackRate'] = str(callback_rate)

        signal_id = f"{pair}_{uuid.uuid4()}"
        signal_message = (
            f"📢 Торговий сигнал: {signal_id}\n"
            f"Пара: {pair}\n"
            f"Таймфрейм: {timeframe}\n"
            f"Дія: {action.upper()}\n"
            f"Ціна входу: ${price:.{price_precision}f}\n"
            f"Стоп-лосс: ${stop_loss:.{price_precision}f}\n"
            f"Тейк-профіт: ${take_profit:.{price_precision}f}\n"
            f"Трейлінг-стоп: ${trailing_stop:.{price_precision}f} (Callback Rate: {callback_rate:.3f})\n"
            f"Плече: {leverage}x\n"
            f"Розмір позиції: {position_size:.{amount_precision}f}\n"
            f"Ймовірність: {signal_prob:.3f}\n"
            f"Пояснення: {explanation}\n"
            f"Режим ринку: {sentiment}\n"
        )
        keyboard = [
            [
                InlineKeyboardButton("Підтвердити", callback_data=f"confirm_{signal_id}"),
                InlineKeyboardButton("Відхилити", callback_data=f"reject_{signal_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await application.bot.send_message(chat_id=CHAT_ID, text=signal_message, reply_markup=reply_markup)

        trade = {
            'signal_id': signal_id,
            'pair': pair,
            'action': action,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'callback_rate': callback_rate,
            'position_size': position_size,
            'leverage': leverage,
            'open_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confirmed': None,
            'market_regime': sentiment
        }
        pending_signals[signal_id] = trade
        if risk_guard:
            risk_guard.trade_history.append(trade)
        logging.info(f"Сигнал {signal_id} надіслано для {pair} ({timeframe}): {action}, ціна={price:.3f}, розмір={position_size:.3f}, плече={leverage}x, R:R={risk_reward_ratio:.2f}")
        last_signal_time[signal_key] = current_time
        return signal_id, None
    except Exception as e:
        logging.error(f"Помилка надсилання сигналу для {pair} ({timeframe}): {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка надсилання сигналу для {pair} ({timeframe}): {str(e)}")
        return None, str(e)

async def send_close_signal(signal_id: str, pair: str, current_price: float, reason: str, profit: float = 0.0) -> tuple[str, str]:
    """
    Надсилає сигнал закриття угоди в Telegram.

    Args:
        signal_id (str): ID сигналу.
        pair (str): Торгова пара.
        current_price (float): Поточна ціна.
        reason (str): Причина закриття.
        profit (float): Прибуток/збиток.

    Returns:
        tuple: (signal_id, None) або (None, error_message).
    """
    if not risk_guard or signal_id not in risk_guard.active_trades:
        return None, f"Угода {signal_id} не знайдена"

    trade = risk_guard.active_trades[signal_id]
    try:
        close_message = (
            f"📉 Сигнал закриття: {signal_id}\n"
            f"Пара: {pair}\n"
            f"Причина: {reason}\n"
            f"Ціна входу: ${trade['entry_price']:.2f}\n"
            f"Поточна ціна: ${current_price:.2f}\n"
            f"Прибуток/Збиток: ${profit:.2f}\n"
        )
        keyboard = [
            [
                InlineKeyboardButton("Підтвердити закриття", callback_data=f"close_{signal_id}"),
                InlineKeyboardButton("Відхилити", callback_data=f"ignore_close_{signal_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await application.bot.send_message(chat_id=CHAT_ID, text=close_message, reply_markup=reply_markup)
        logging.info(f"Надіслано сигнал закриття для {signal_id}: {reason}, ціна={current_price:.2f}, прибуток={profit:.2f}")
        return signal_id, None
    except Exception as e:
        logging.error(f"Помилка надсилання сигналу закриття для {signal_id}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка надсилання сигналу закриття для {signal_id}: {str(e)}")
        return None, str(e)

async def monitor_signals(pair: str = None, timeframe: str = None) -> None:
    """
    Моніторить сигнали для заданих пар і таймфреймів.

    Args:
        pair (str, optional): Торгова пара. Якщо None, обробляються всі пари.
        timeframe (str, optional): Таймфрейм. Якщо None, обробляються всі таймфрейми.
    """
    from data_processing import clean_data_cache
    clean_data_cache(max_size=1000, max_age=3600)

    # Ініціалізація біржі, якщо не ініціалізована
    global exchange
    if not exchange:
        exchange = await initialize_exchange()

    pairs_to_process = [pair] if pair else PAIRS
    timeframes_to_process = [timeframe] if timeframe else TIMEFRAMES

    for pair in pairs_to_process:
        if not pair_settings.get(pair, {}).get('active', True):
            logging.info(f"Пропуск моніторингу сигналів для {pair}: пара неактивна")
            continue
        for tf in timeframes_to_process:
            try:
                if not hasattr(exchange, 'markets') or pair not in exchange.markets:
                    logging.error(f"Біржа не ініціалізована або пара {pair} недоступна")
                    continue

                df = await get_historical_data(pair, tf, limit=1000 if tf in ['1m', '5m'] else 500)
                if df is None or not validate_data(df, pair, tf, min_rows=500):
                    pair_signal_status[pair] = f"Недостатньо даних: {len(df) if df is not None else 'None'}"
                    loggingondata = len(df) if df is not None else 'None'}"
                    continue

                # Обчислення всіх індикаторів
                df = await calculate_all_indicators(df)
                if df is None or df.empty:
                    pair_signal_status[pair] = "Помилка обчислення індикаторів"
                    logging.warning(f"Помилка обчислення індикаторів для {pair} ({tf})")
                    continue

                if df.isna().any().any():
                    import pandas as pd
                    logging.warning(f"Знайдено NaN у даних для {pair} ({tf}), заповнюємо середніми")
                    df = df.fillna(df.mean(numeric_only=True))
                    if df.isna().any().any():
                        logging.error(f"Не вдалося заповнити NaN для {pair} ({tf})")
                        continue

                # Отримання поточної ціни
                ticker = await exchange.fetch_ticker(pair)
                current_price = float(ticker.get('last', df['close'].iloc[-1]))
                if current_price <= 0:
                    logging.error(f"Некоректна ціна для {pair} ({tf}): {current_price}")
                    pair_signal_status[pair] = f"Помилка: Некоректна ціна {current_price}"
                    continue

                # Отримання балансу
                balance_info = await exchange.fetch_balance()
                balance = float(balance_info.get('USDT', {}).get('free', 0.0))

                # Генерація сигналу
                signal_data = await generate_signal(pair, df, tf)
                if signal_data['signal'] != 'wait':
                    trade_size = await calculate_trade_size(
                        pair=pair,
                        signal_prob=signal_data['signal_prob'],
                        balance=balance,
                        price=current_price,
                        atr=df['atr'].iloc[-1],
                        leverage=signal_data.get('leverage', 1),
                        timeframe=tf
                    )
                    if trade_size <= 0:
                        logging.error(f"Некоректний розмір позиції для {pair} ({tf}): {trade_size}")
                        await send_telegram_buffered(f"⚠️ Помилка: Некоректний розмір позиції для {pair} ({tf}): {trade_size}")
                        continue

                    signal_data['position_size'] = trade_size
                    await process_signal(
                        pair=pair,
                        signal=signal_data['signal'],
                        confidence=signal_data['signal_prob'],
                        explanation=signal_data['explanation'],
                        stop_loss=signal_data['stop_loss'],
                        take_profit=signal_data['take_profit'],
                        atr=df['atr'].iloc[-1],
                        position_size=trade_size,
                        volatility=df['bollinger_width'].iloc[-1],
                        leverage=signal_data['leverage'],
                        current_price=current_price,
                        timestamp=df.index[-1],
                        market_regime=signal_data['regime'],
                        callback_rate=signal_data['callback_rate']
                    )

                pair_signal_status[pair] = (
                    f"Сигнал={signal_data['signal']}, Впевненість={signal_data['signal_prob']:.2f}, "
                    f"RSI={df['rsi'].iloc[-1]:.2f}, ADX={df['adx'].iloc[-1]:.2f}, "
                    f"StochK={df['stoch_k'].iloc[-1]:.2f}, BB_Width={df['bollinger_width'].iloc[-1]:.4f}, "
                    f"Regime={signal_data['regime']}, Price={current_price:.2f}"
                )
            except Exception as e:
                logging.error(f"Помилка моніторингу сигналів для {pair} ({tf}): {str(e)}")
                await send_telegram_buffered(f"⚠️ Помилка моніторингу сигналів для {pair} ({tf}): {str(e)}")
                pair_signal_status[pair] = f"Помилка: {str(e)}"

async def monitor_trades() -> None:
    """
    Моніторить активні угоди та оновлює трейлінг-стопи.
    """
    if not risk_guard:
        logging.error("RiskGuard не ініціалізований")
        return

    for signal_id, trade in list(risk_guard.active_trades.items()):
        try:
            pair = trade['pair']
            df = await get_historical_data(pair, '1m', limit=10)
            if df is None or not validate_data(df, pair, '1m', min_rows=5):
                logging.warning(f"Недостатньо даних для моніторингу угоди {signal_id} ({pair})")
                continue

            current_price = float(df['close'].iloc[-1])
            if current_price <= 0:
                logging.error(f"Некоректна ціна для {pair} (1m): {current_price}")
                continue

            action = trade['action']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            trailing_stop = trade['trailing_stop']
            leverage = trade['leverage']
            position_size = trade['position_size']

            # Оновлення трейлінг-стопу
            if action == 'buy':
                new_trailing_stop = current_price - (entry_price - trailing_stop) if trailing_stop < current_price else trailing_stop
                if new_trailing_stop > trailing_stop:
                    trade['trailing_stop'] = new_trailing_stop
                    risk_guard.active_trades[signal_id]['trailing_stop'] = new_trailing_stop
                    logging.info(f"Оновлено трейлінг-стоп для {signal_id} ({pair}): {new_trailing_stop:.2f}")
            else:  # sell
                new_trailing_stop = current_price + (trailing_stop - entry_price) if trailing_stop > current_price else trailing_stop
                if new_trailing_stop < trailing_stop:
                    trade['trailing_stop'] = new_trailing_stop
                    risk_guard.active_trades[signal_id]['trailing_stop'] = new_trailing_stop
                    logging.info(f"Оновлено трейлінг-стоп для {signal_id} ({pair}): {new_trailing_stop:.2f}")

            # Розрахунок прибутку
            profit = (current_price - entry_price) * position_size * leverage if action == 'buy' else (entry_price - current_price) * position_size * leverage

            # Перевірка умов закриття
            if action == 'buy' and (current_price <= stop_loss or current_price >= take_profit or current_price <= trade['trailing_stop']):
                reason = 'Стоп-лосс' if current_price <= stop_loss else 'Тейк-профіт' if current_price >= take_profit else 'Трейлінг-стоп'
                await send_close_signal(signal_id, pair, current_price, reason, profit)
            elif action == 'sell' and (current_price >= stop_loss or current_price <= take_profit or current_price >= trade['trailing_stop']):
                reason = 'Стоп-лосс' if current_price >= stop_loss else 'Тейк-профіт' if current_price <= take_profit else 'Трейлінг-стоп'
                await send_close_signal(signal_id, pair, current_price, reason, profit)
        except Exception as e:
            logging.error(f"Помилка моніторингу угоди {signal_id} ({pair}): {str(e)}")
            await send_telegram_buffered(f"⚠️ Помилка моніторингу угоди {signal_id} ({pair}): {str(e)}")

async def execute_trade(pair: str, action: str, leverage: int, entry_price: float, stop_loss: float, take_profit: float, position_size: float, trailing_stop: float, callback_rate: float) -> dict:
    """
    Виконує торгову угоду (симуляція або реальне виконання).

    Args:
        pair (str): Торгова пара.
        action (str): Дія ('buy' або 'sell').
        leverage (int): Плече.
        entry_price (float): Ціна входу.
        stop_loss (float): Стоп-лосс.
        take_profit (float): Тейк-профіт.
        position_size (float): Розмір позиції.
        trailing_stop (float): Трейлінг-стоп.
        callback_rate (float): Callback rate для трейлінг-стопу.

    Returns:
        dict: Дані угоди або None у разі помилки.
    """
    global exchange
    try:
        if SIMULATE_MODE:
            logging.info(f"Симуляція угоди для {pair}: {action}, розмір={position_size:.4f}, ціна={entry_price:.2f}, плече={leverage}x")
            order_id = f"tracked_{pair}_{uuid.uuid4()}"
            trade = {
                'order_id': order_id,
                'pair': pair,
                'action': action,
                'entry_price': entry_price,
                'position_size': position_size,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': trailing_stop,
                'callback_rate': callback_rate
            }
            logging.info(f"Угоду {order_id} додано до відстеження: {trade}")
            return trade
        else:
            logging.error("Реальне виконання угод ще не реалізовано")
            await send_telegram_buffered(f"⚠️ Реальне виконання угод для {pair} ще не реалізовано")
            return None
    except Exception as e:
        logging.error(f"Помилка виконання угоди для {pair}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка виконання угоди для {pair}: {str(e)}")
        return None

async def process_signal(pair: str, signal: str, confidence: float, explanation: str, stop_loss: float, take_profit: float, atr: float, position_size: float, volatility: float, leverage: int, current_price: float, timestamp: 'pandas.Timestamp', market_regime: str, is_correlated: bool = False, callback_rate: float = None) -> bool:
    """
    Обробляє торговий сигнал і надсилає його для підтвердження.

    Args:
        pair (str): Торгова пара.
        signal (str): Сигнал ('buy', 'sell' або 'wait').
        confidence (float): Впевненість сигналу.
        explanation (str): Пояснення сигналу.
        stop_loss (float): Стоп-лосс.
        take_profit (float): Тейк-профіт.
        atr (float): ATR.
        position_size (float): Розмір позиції.
        volatility (float): Волатильність (bollinger_width).
        leverage (int): Плече.
        current_price (float): Поточна ціна.
        timestamp (pandas.Timestamp): Час сигналу.
        market_regime (str): Режим ринку.
        is_correlated (bool): Чи є кореляційний ризик.
        callback_rate (float): Callback rate для трейлінг-стопу.

    Returns:
        bool: True, якщо сигнал оброблено успішно, False інакше.
    """
    try:
        if signal == 'wait':
            logging.info(f"Сигнал для {pair}: wait, пропускаємо")
            return False

        signal_id, reason = await send_trade_signal(
            pair=pair,
            action=signal,
            leverage=leverage,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_prob=confidence,
            explanation=explanation,
            sentiment=market_regime,
            price_prob=confidence,
            position_size=position_size,
            trailing_stop=atr,
            callback_rate=callback_rate,
            timeframe='1m'
        )

        if signal_id is None:
            logging.error(f"Не вдалося надіслати сигнал для {pair}: {reason}")
            return False

        logging.info(f"Сигнал {signal_id} надіслано для {pair}: {signal}, ціна={current_price}, розмір={position_size}, плече={leverage}x")
        return True
    except Exception as e:
        logging.error(f"Помилка в process_signal для {pair}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка обробки сигналу для {pair}: {str(e)}")
        return False

def validate_signal_params(params: dict) -> tuple[bool, str]:
    """
    Валідує параметри сигналу.

    Args:
        params (dict): Словник із параметрами сигналу.

    Returns:
        tuple: (is_valid, error_message).
    """
    try:
        required_params = ['signal_prob', 'price', 'stop_loss', 'take_profit', 'trailing_stop', 'position_size', 'leverage']
        for param in required_params:
            if param not in params:
                return False, f"Відсутній параметр: {param}"
            if not isinstance(params[param], (int, float)) or params[param] is None:
                return False, f"Некоректний тип або значення для {param}: {params[param]} (очікується число)"

        if params['signal_prob'] < 0 or params['signal_prob'] > 1:
            return False, f"signal_prob поза допустимим діапазоном: {params['signal_prob']} (очікується 0-1)"
        if params['price'] <= 0:
            return False, f"Некоректна ціна: {params['price']}"
        if params['position_size'] <= 0:
            return False, f"Некоректний розмір позиції: {params['position_size']}"
        if params['leverage'] < 1:
            return False, f"Некоректне плече: {params['leverage']}"

        return True, ""
    except Exception as e:
        return False, f"Помилка валідації параметрів: {str(e)}"