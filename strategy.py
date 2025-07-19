import asyncio
import logging
import pandas as pd
import numpy as np
import os
import pickle
import uuid
import csv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from config import load_config, initialize_global_exchange
from data_processing import get_historical_data, calculate_all_indicators, calculate_rsi, calculate_adx, calculate_atr, calculate_ichimoku, calculate_ema, calculate_macd, calculate_bollinger_bands, calculate_vwap, calculate_stochastic_oscillator, calculate_obv, calculate_roc, calculate_bollinger_width
from machine_learning import train_ml_model, load_saved_models
from trading import send_trade_signal
from telegram_utils import send_telegram_buffered
from risk_management import RiskGuard

logging.basicConfig(filename='strategy.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CONFIG = load_config()
PAIRS = CONFIG['pairs']
TIMEFRAMES = CONFIG['timeframes']
SWING_TIMEFRAMES = CONFIG.get('swing_timeframes', [])
ML_MODELS = {}
FEATURES = CONFIG.get('features', [
    'ema20', 'rsi', 'adx', 'macd', 'signal_line', 'bb_upper', 'bb_lower',
    'vwap', 'stoch_k', 'stoch_d', 'obv', 'roc', 'bollinger_width', 'atr',
    'momentum', 'ichimoku_tenkan', 'ichimoku_kijun'
])
exchange = None
pair_settings = CONFIG.get('pair_settings', {})
pair_performance = CONFIG.get('pair_performance', {})
strategy_state = {'current_strategy': 'swing', 'active_pair': None, 'scalp_end_time': None}
pending_signals = {}
TRADES_CSV = 'trades.csv'

async def initialize():
    """Ініціалізація біржі та завантаження моделей."""
    global exchange, ML_MODELS
    exchange = await initialize_global_exchange()
    if exchange is None:
        logging.error("Не вдалося ініціалізувати біржу")
        return False
    await load_saved_models(PAIRS)
    return True

async def optimize_strategy_parameters(pair: str, df: pd.DataFrame) -> None:
    """Оптимізує параметри стратегії для пари."""
    try:
        if len(df) < 100:
            logging.warning(f"Недостатньо даних для оптимізації {pair}: {len(df)}")
            return
        df = await calculate_all_indicators(df)
        df = df.fillna(method='ffill').fillna(method='bfill').dropna()
        if len(df) < 50:
            logging.warning(f"Замало даних після dropna для {pair}: {len(df)}")
            return
        best_params = {'buy_threshold': 0.6, 'sell_threshold': 0.4, 'volatility_threshold': 0.0002}
        best_sharpe = -float('inf')
        thresholds = [(0.55, 0.45), (0.6, 0.4), (0.65, 0.35)]
        volatility_thresholds = [0.0001, 0.0002, 0.0003]
        for buy_t, sell_t in thresholds:
            for vol_t in volatility_thresholds:
                signals = []
                for i in range(1, len(df)):
                    rsi = df['rsi'].iloc[i]
                    volatility = df['atr'].iloc[i] / df['close'].iloc[i] if df['atr'].iloc[i] else 0.02
                    if volatility < vol_t:
                        signals.append('wait')
                    elif rsi > buy_t * 100:
                        signals.append('buy')
                    elif rsi < sell_t * 100:
                        signals.append('sell')
                    else:
                        signals.append('wait')
                returns = []
                for i in range(len(signals) - 1):
                    if signals[i] == 'buy':
                        ret = (df['close'].iloc[i + 1] - df['close'].iloc[i]) / df['close'].iloc[i]
                        returns.append(ret)
                    elif signals[i] == 'sell':
                        ret = (df['close'].iloc[i] - df['close'].iloc[i + 1]) / df['close'].iloc[i]
                        returns.append(ret)
                if returns:
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if np.std(returns) != 0 else 0
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {'buy_threshold': buy_t, 'sell_threshold': sell_t, 'volatility_threshold': vol_t}
        pair_settings[pair]['signal_threshold_buy'] = best_params['buy_threshold']
        pair_settings[pair]['signal_threshold_sell'] = best_params['sell_threshold']
        pair_settings[pair]['volatility_threshold'] = best_params['volatility_threshold']
        logging.info(f"Оптимізовано параметри для {pair}: buy={best_params['buy_threshold']:.2f}, sell={best_params['sell_threshold']:.2f}, vol={best_params['volatility_threshold']:.4f}, Sharpe={best_sharpe:.2f}")
        await send_telegram_buffered(f"🔄 Оптимізовано параметри для {pair}: buy={best_params['buy_threshold']:.2f}, sell={best_params['sell_threshold']:.2f}, vol={best_params['volatility_threshold']:.4f}")
    except Exception as e:
        logging.error(f"Помилка в optimize_strategy_parameters для {pair}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка оптимізації параметрів для {pair}: {str(e)}")

async def analyze_trade_performance(risk_guard: RiskGuard) -> None:
    """Аналізує продуктивність торгівлі та перенавчає моделі за потреби."""
    for pair in PAIRS:
        recent_trades = [t for t in risk_guard.trade_history[-50:] if t['pair'] == pair]
        if len(recent_trades) < 10:
            continue
        win_rate = sum(1 for t in recent_trades if t['profit'] > 0) / len(recent_trades)
        avg_profit = np.mean([t['profit'] for t in recent_trades])
        if win_rate < 0.4 or avg_profit < 0:
            logging.info(f"Низька продуктивність для {pair}: win_rate={win_rate:.2f}, avg_profit={avg_profit:.2f}")
            pair_settings[pair]['active'] = False
            await send_telegram_buffered(f"⚠️ Торгівлю для {pair} відключено через низьку прибутковість (win_rate={win_rate:.2f})")
            new_data = await get_historical_data(pair, '5m', limit=1000)
            if new_data is not None:
                await train_ml_model(pair, new_data, '5m', force_retrain=True)
                await optimize_strategy_parameters(pair, new_data)
                pair_settings[pair]['active'] = True
                await send_telegram_buffered(f"✅ Торгівлю для {pair} відновлено після перенавчання")

async def run_backtest(pairs: list, initial_balance: float = 1000, commission_rate: float = 0.0004, slippage: float = 0.001, leverage: int = 5) -> dict:
    """Виконує бектест стратегії."""
    results = {}
    temp_risk_guard = RiskGuard(initial_budget=initial_balance, max_drawdown=0.10, reserve_ratio=0.30)
    correlation_matrix = await update_correlation_matrix(pairs) or {}
    for pair in pairs:
        try:
            df = await get_historical_data(pair, '5m', limit=4000)
            if df is None or len(df) < 100:
                logging.error(f"Недостатньо даних для бектесту {pair}: {len(df) if df is not None else 'None'}")
                results[pair] = {'profit': 0, 'trades': 0, 'win_rate': 0}
                continue
            if pair not in ML_MODELS or not any(ML_MODELS[pair].get(tf, {}).get(regime, {}).get('models') for tf in TIMEFRAMES + SWING_TIMEFRAMES for regime in ['trending', 'ranging', 'neutral']):
                logging.error(f"Моделі для {pair} відсутні")
                results[pair] = {'profit': 0, 'trades': 0, 'win_rate': 0}
                continue
            is_correlated = any(abs(correlation_matrix.get(pair, {}).get(other_pair, 0)) > 0.6 for other_pair in correlation_matrix if other_pair != pair)
            if is_correlated and pair not in ['BTC/USDT', 'ETH/USDT']:
                logging.info(f"Пропуск бектесту для {pair} через високу кореляцію")
                results[pair] = {'profit': 0, 'trades': 0, 'win_rate': 0}
                continue
            balance = initial_balance
            trades = 0
            wins = 0
            active_trades = {}
            for i in range(len(df) - 1):
                if temp_risk_guard.emergency_stop or len(active_trades) >= temp_risk_guard.max_active_trades:
                    continue
                market_regime = 'trending' if df['adx'].iloc[i] > 25 else 'ranging'
                models = ML_MODELS[pair][market_regime]['models']
                scaler = ML_MODELS[pair][market_regime]['scaler']
                features = ML_MODELS[pair][market_regime].get('features', FEATURES)
                X = df[features].iloc[i].values.reshape(1, -1)
                X_scaled = scaler.transform(X)
                predictions = [model.predict_proba(X_scaled)[:, 1][0] for model in models]
                avg_pred = np.mean(predictions)
                current_price = df['close'].iloc[i]
                atr = df['atr'].iloc[i]
                price_precision = int(exchange.markets[pair]['precision']['price']) if pair in exchange.markets else 4
                stop_loss = round(current_price * (1 - 1.5 * atr / current_price), price_precision)
                take_profit = round(current_price * (1 + 2.5 * atr / current_price), price_precision)
                position_size = temp_risk_guard.calculate_kelly_position(avg_pred, pair_performance[pair]['win_rate'])
                position_size = min(position_size, temp_risk_guard.available_budget * 0.05 / (current_price * leverage))
                if avg_pred > 0.65 and position_size > 0:
                    balance -= current_price * position_size * (commission_rate + slippage)
                    trade_id = f"backtest_{pair}_{i}"
                    active_trades[trade_id] = {
                        'pair': pair,
                        'entry_price': current_price,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    trades += 1
                elif avg_pred < 0.35 and position_size > 0:
                    balance -= current_price * position_size * (commission_rate + slippage)
                    trade_id = f"backtest_{pair}_{i}"
                    active_trades[trade_id] = {
                        'pair': pair,
                        'entry_price': current_price,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                    trades += 1
                for trade_id, trade in list(active_trades.items()):
                    if trade['entry_price'] < trade['stop_loss'] or trade['entry_price'] > trade['take_profit']:
                        profit = (df['close'].iloc[i+1] - trade['entry_price']) * trade['position_size'] * leverage
                        balance += profit
                        temp_risk_guard.current_budget += profit
                        temp_risk_guard.available_budget += profit
                        if profit > 0:
                            wins += 1
                        del active_trades[trade_id]
            results[pair] = {'profit': balance - initial_balance, 'trades': trades, 'win_rate': wins / trades if trades > 0 else 0}
            logging.info(f"Бектест для {pair}: Прибуток={results[pair]['profit']:.2f}, Торгів={trades}, WinRate={results[pair]['win_rate']:.2f}")
        except Exception as e:
            logging.error(f"Помилка бектесту для {pair}: {str(e)}")
            results[pair] = {'profit': 0, 'trades': 0, 'win_rate': 0}
    return results

async def adjust_strategy(pair: str, recent_trades: list, volatility: float, df: pd.DataFrame) -> None:
    """Адаптує стратегію на основі ринкових умов."""
    try:
        adx = df['adx'].iloc[-1] if 'adx' in df and len(df) > 0 else 20.0
        market_regime = 'trending' if adx > 25 else 'ranging'
        recent_trades_pair = [t for t in recent_trades if t['pair'] == pair and pd.to_datetime(t['open_time']) > pd.Timestamp.now() - pd.Timedelta(hours=24)]
        win_rate = sum(1 for t in recent_trades_pair if t.get('profit', 0) > 0) / len(recent_trades_pair) if recent_trades_pair else 0.5
        avg_profit = sum(t.get('profit', 0) for t in recent_trades_pair) / len(recent_trades_pair) if recent_trades_pair else 0.0
        base_threshold_buy = 0.55 if pair in ['BTC/USDT', 'ETH/USDT'] else 0.50
        base_threshold_sell = 0.45 if pair in ['BTC/USDT', 'ETH/USDT'] else 0.40
        min_risk_reward = 1.8 if pair in ['BTC/USDT', 'ETH/USDT'] else 1.3
        leverage = min(10, max(3, int(10 / (volatility + 0.01))))
        if market_regime == 'trending':
            pair_settings[pair]['signal_threshold_buy'] = base_threshold_buy
            pair_settings[pair]['signal_threshold_sell'] = base_threshold_sell
            pair_settings[pair]['min_risk_reward'] = min_risk_reward
        else:
            pair_settings[pair]['signal_threshold_buy'] = base_threshold_buy - 0.05
            pair_settings[pair]['signal_threshold_sell'] = base_threshold_sell + 0.05
            pair_settings[pair]['min_risk_reward'] = min_risk_reward - 0.2
        if win_rate < 0.4 and len(recent_trades_pair) >= 10:
            pair_settings[pair]['signal_threshold_buy'] += 0.05
            pair_settings[pair]['signal_threshold_sell'] -= 0.05
            pair_settings[pair]['min_risk_reward'] += 0.2
        elif win_rate > 0.6:
            pair_settings[pair]['signal_threshold_buy'] -= 0.05
            pair_settings[pair]['signal_threshold_sell'] += 0.05
            pair_settings[pair]['min_risk_reward'] -= 0.2
        pair_settings[pair]['leverage'] = leverage
        if win_rate < 0.35 and len(recent_trades_pair) >= 15:
            pair_settings[pair]['active'] = False
            pair_settings[pair]['no_signal_counter'] += 1
            await send_telegram_buffered(f"⏸ Пара {pair} відключена через низький win_rate={win_rate:.2f}")
        elif win_rate > 0.55 and not pair_settings[pair]['active']:
            pair_settings[pair]['active'] = True
            pair_settings[pair]['no_signal_counter'] = 0
            await send_telegram_buffered(f"▶️ Пара {pair} увімкнена через високий win_rate={win_rate:.2f}")
        elif pair_settings[pair]['no_signal_counter'] >= 15:
            pair_settings[pair]['active'] = False
            await send_telegram_buffered(f"⏸ Пара {pair} відключена через тривалу відсутність сигналів")
        logging.info(f"Оновлено налаштування для {pair}: режим={market_regime}, buy={pair_settings[pair]['signal_threshold_buy']:.2f}, sell={pair_settings[pair]['signal_threshold_sell']:.2f}, leverage={leverage}, active={pair_settings[pair]['active']}")
    except Exception as e:
        logging.error(f"Помилка в adjust_strategy для {pair}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка налаштування стратегії для {pair}: {str(e)}")

async def analyze_market_with_ai(pair: str, data: pd.DataFrame, timeframe: str = '5m') -> dict:
    """Аналізує ринок за допомогою ML-моделей."""
    try:
        df = data.copy()
        df = await calculate_all_indicators(df)
        if df.empty or df.isna().any().any():
            logging.error(f"Помилка обчислення індикаторів для {pair} ({timeframe})")
            return {'signal': 'wait', 'confidence': 0.0, 'explanation': 'Помилка обчислення індикаторів'}
        regime = 'trending' if df['adx'].iloc[-1] > 25 else 'ranging'
        model_data = ML_MODELS.get(pair, {}).get(timeframe, {}).get(regime, {})
        model = model_data.get('models', [None])[0]
        scaler = model_data.get('scaler')
        features = model_data.get('features', FEATURES)
        if model is None or scaler is None:
            logging.warning(f"Модель або скейлер відсутні для {pair} ({timeframe}, {regime})")
            rsi = float(df['rsi'].iloc[-1])
            macd = float(df['macd'].iloc[-1])
            signal_line = float(df['signal_line'].iloc[-1])
            bb_width = float(df['bollinger_width'].iloc[-1])
            stoch_k = float(df['stoch_k'].iloc[-1])
            adx = float(df['adx'].iloc[-1])
            buy_conditions = rsi < 40 and macd > signal_line and stoch_k < 30 and adx > 20 and bb_width > 0.015
            sell_conditions = rsi > 60 and macd < signal_line and stoch_k > 70 and adx > 20 and bb_width > 0.015
            signal = 'buy' if buy_conditions else 'sell' if sell_conditions else 'wait'
            confidence = 0.75 if signal != 'wait' else 0.0
            explanation = f"Сигнал на основі індикаторів ({regime}): RSI={rsi:.2f}, MACD={macd:.2f}, StochK={stoch_k:.2f}, ADX={adx:.2f}, BB_Width={bb_width:.4f}"
        else:
            X = df[features].iloc[-1:].values
            X_scaled = scaler.transform(X)
            prediction = model.predict_proba(X_scaled)[0]
            confidence = max(prediction)
            signal = 'buy' if prediction[1] > pair_settings[pair]['signal_threshold_buy'] else 'sell' if prediction[0] > pair_settings[pair]['signal_threshold_sell'] else 'wait'
            explanation = f"Сигнал на основі ML ({regime}): RSI={df['rsi'].iloc[-1]:.2f}, MACD={df['macd'].iloc[-1]:.2f}, ADX={df['adx'].iloc[-1]:.2f}, Confidence={confidence:.2f}"
        logging.info(f"Сигнал для {pair} ({timeframe}): {signal}, confidence={confidence:.2f}")
        return {'signal': signal, 'confidence': confidence, 'explanation': explanation}
    except Exception as e:
        logging.error(f"Помилка в analyze_market_with_ai для {pair} ({timeframe}): {str(e)}")
        return {'signal': 'wait', 'confidence': 0.0, 'explanation': f'Помилка: {str(e)}'}

async def process_scalping_trade(pair: str, timeframe: str, signal_count: int) -> tuple:
    """Обробляє скальпінг-торгівлю."""
    try:
        signal_data = await analyze_scalping(pair, timeframe)
        if signal_data['signal'] == 'wait':
            logging.info(f"Пропуск скальпінг-сигналу для {pair} ({timeframe}): signal=wait")
            return None, 0
        signal_id, reason = await send_trade_signal(
            pair=pair,
            action=signal_data['signal'],
            leverage=signal_data['leverage'],
            price=signal_data['current_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            signal_prob=signal_data['confidence'],
            explanation=signal_data['explanation'],
            sentiment=signal_data['market_regime'],
            price_prob=signal_data['confidence'],
            position_size=signal_data['position_size'],
            trailing_stop=signal_data['trailing_stop'],
            callback_rate=signal_data['callback_rate'],
            timeframe=timeframe
        )
        if signal_id is None:
            logging.info(f"Скальпінг-сигнал для {pair} ({timeframe}) відхилено: {reason}")
            return None, 0
        logging.info(f"Скальпінг-сигнал для {pair} ({timeframe}) надіслано: signal_id={signal_id}")
        return signal_id, 1
    except Exception as e:
        logging.error(f"Помилка в process_scalping_trade для {pair} ({timeframe}): {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка скальпінгу для {pair} ({timeframe}): {str(e)}", force=True)
        return None, 0

async def analyze_scalping(pair: str, timeframe: str = '1m') -> dict:
    """Аналізує ринок для скальпінг-стратегії."""
    try:
        df = await get_historical_data(pair, timeframe, limit=200)
        if df is None or len(df) < 100:
            logging.warning(f"Недостатньо даних для скальпінгу {pair} ({timeframe}): {len(df) if df is not None else 'None'}")
            return {'signal': 'wait', 'confidence': 0.0, 'explanation': 'Недостатньо даних', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'atr': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'rsi': 0.0, 'stoch_k': 0.0, 'market_regime': 'neutral'}
        df = await calculate_all_indicators(df)
        if len(df) < 10 or df['close'].isna().any():
            logging.warning(f"Замало даних після індикаторів для {pair} ({timeframe}): {len(df)}")
            return {'signal': 'wait', 'confidence': 0.0, 'explanation': 'Замало даних після індикаторів', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'atr': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'rsi': 0.0, 'stoch_k': 0.0, 'market_regime': 'neutral'}
        df = df.fillna(0.0)
        current_price = float(df['close'].iloc[-1])
        if current_price <= 0:
            ticker = await exchange.fetch_ticker(pair)
            current_price = float(ticker['last'])
        volatility = float(df['close'].pct_change().rolling(window=20).std().iloc[-1]) if len(df) >= 20 else 0.02
        atr = float(max(df['atr'].iloc[-1], 0.0001 * current_price))
        rsi = float(df['rsi'].iloc[-1])
        adx = float(df['adx'].iloc[-1])
        stoch_k = float(df['stoch_k'].iloc[-1])
        bb_width = float(df['bollinger_width'].iloc[-1])
        market_regime = 'trending' if adx > 25 else 'ranging'
        ai_signal, ai_confidence, ai_explanation = await analyze_market_with_ai(pair, df, timeframe)
        ema_fast = df['ema20'].iloc[-1]
        ema_slow = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        low_price_pairs = ['POL/USDT', 'ADA/USDT']
        price_precision = int(exchange.markets[pair]['precision']['price']) if pair in exchange.markets else 4
        amount_precision = int(exchange.markets[pair]['precision']['amount']) if pair in exchange.markets else 4
        volume_threshold = 500 if pair in low_price_pairs else 1000
        volatility_threshold = pair_settings[pair]['volatility_threshold']
        signal = 'wait'
        confidence = 0.0
        explanation = ''
        if volatility < volatility_threshold:
            explanation = f'Низька волатильність: {volatility:.4f} < {volatility_threshold:.4f}'
            pair_settings[pair]['no_signal_counter'] += 1
            if pair_settings[pair]['no_signal_counter'] > 10:
                pair_settings[pair]['volatility_threshold'] = max(0.00005, pair_settings[pair]['volatility_threshold'] - 0.00001)
                await send_telegram_buffered(f"🔄 Знижено поріг волатильності для {pair} до {pair_settings[pair]['volatility_threshold']:.4f}")
        elif (rsi < 35 and stoch_k < 25 and adx < 25 and bb_width > 0.015 and ema_fast > ema_slow and ai_signal == 'buy' and ai_confidence > pair_settings[pair]['signal_threshold_buy']):
            signal = 'buy'
            confidence = 0.85 * ai_confidence
            explanation = f"Скальпінг сигнал: RSI={rsi:.2f}, StochK={stoch_k:.2f}, ADX={adx:.2f}, BB_Width={bb_width:.4f}, Volatility={volatility:.4f}, EMA_Fast>{ema_slow:.2f}, AI_Confidence={ai_confidence:.2f}"
        elif (rsi > 65 and stoch_k > 75 and adx < 25 and bb_width > 0.015 and ema_fast < ema_slow and ai_signal == 'sell' and ai_confidence > pair_settings[pair]['signal_threshold_sell']):
            signal = 'sell'
            confidence = 0.85 * ai_confidence
            explanation = f"Скальпінг сигнал: RSI={rsi:.2f}, StochK={stoch_k:.2f}, ADX={adx:.2f}, BB_Width={bb_width:.4f}, Volatility={volatility:.4f}, EMA_Fast<{ema_slow:.2f}, AI_Confidence={ai_confidence:.2f}"
        else:
            explanation = f"Немає сигналу: RSI={rsi:.2f}, StochK={stoch_k:.2f}, ADX={adx:.2f}, BB_Width={bb_width:.4f}, Volatility={volatility:.4f}, AI_Signal={ai_signal}"
        atr_ratio = min(atr / current_price, 0.5) if pair not in low_price_pairs else min(atr / current_price, 0.3)
        stop_loss = current_price * (1 - atr_ratio * 2.0) if signal == 'buy' else current_price * (1 + atr_ratio * 2.0)
        take_profit = current_price * (1 + atr_ratio * 3.0) if signal == 'buy' else current_price * (1 - atr_ratio * 3.0)
        trailing_stop = current_price * (1 - atr_ratio * 1.5) if signal == 'buy' else current_price * (1 + atr_ratio * 1.5)
        if stop_loss <= 0 or take_profit <= 0 or trailing_stop <= 0:
            stop_loss = round(current_price * 0.99, price_precision)
            take_profit = round(current_price * 1.02, price_precision)
            trailing_stop = round(current_price * 0.995, price_precision)
            explanation += f", Виправлено ціни: SL={stop_loss:.2f}, TP={take_profit:.2f}, TS={trailing_stop:.2f}"
        stop_loss = round(float(stop_loss), price_precision)
        take_profit = round(float(take_profit), price_precision)
        trailing_stop = round(float(trailing_stop), price_precision)
        risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss + 1e-8) if current_price != stop_loss else 1
        min_risk_reward = 1.5 if pair in low_price_pairs else 2.0
        if risk_reward_ratio < min_risk_reward:
            logging.info(f"Пропуск сигналу для {pair} ({timeframe}): R:R={risk_reward_ratio:.2f} < {min_risk_reward}")
            return {'signal': 'wait', 'confidence': 0.0, 'explanation': f'Низьке R:R={risk_reward_ratio:.2f}', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'atr': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'rsi': 0.0, 'stoch_k': 0.0, 'market_regime': 'neutral'}
        if df['volume'].rolling(window=20).mean().iloc[-1] < volume_threshold:
            logging.warning(f"Пропуск сигналу для {pair} ({timeframe}): низький обсяг торгів")
            return {'signal': 'wait', 'confidence': 0.0, 'explanation': 'Низький обсяг торгів', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'atr': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'rsi': 0.0, 'stoch_k': 0.0, 'market_regime': 'neutral'}
        callback_rate = 0.5
        leverage = min(pair_settings[pair]['max_leverage'], max(pair_settings[pair]['min_leverage'], int(20 * confidence * (1 - volatility))))
        position_size = RiskGuard().calculate_kelly_position(confidence, pair_performance[pair]['win_rate'], volatility=volatility)
        position_size = round(float(position_size), amount_precision)
        logging.info(f"Скальпінг для {pair} ({timeframe}): {explanation}, price={current_price:.{price_precision}f}, SL={stop_loss:.{price_precision}f}, TP={take_profit:.{price_precision}f}, TS={trailing_stop:.{price_precision}f}")
        return {
            'signal': signal,
            'confidence': confidence,
            'explanation': explanation,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'leverage': leverage,
            'position_size': position_size,
            'trailing_stop': trailing_stop,
            'callback_rate': callback_rate,
            'rsi': rsi,
            'stoch_k': stoch_k,
            'market_regime': market_regime
        }
    except Exception as e:
        logging.error(f"Помилка в analyze_scalping для {pair} ({timeframe}): {str(e)}")
        return {'signal': 'wait', 'confidence': 0.0, 'explanation': f'Помилка: {str(e)}', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'atr': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'rsi': 0.0, 'stoch_k': 0.0, 'market_regime': 'neutral'}

async def process_swing_trade(pair: str, timeframe: str, signal_count: int) -> tuple:
    """Обробляє свінг-торгівлю."""
    try:
        signal_data = await analyze_swing(pair, timeframe)
        if signal_data['signal'] == 'wait':
            logging.info(f"Пропуск свінг-сигналу для {pair} ({timeframe}): signal=wait")
            return None, 0
        signal_id, reason = await send_trade_signal(
            pair=pair,
            action=signal_data['signal'],
            leverage=signal_data['leverage'],
            price=signal_data['current_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            signal_prob=signal_data['signal_prob'],
            explanation=signal_data['explanation'],
            sentiment=signal_data['sentiment'],
            price_prob=signal_data['price_prob'],
            position_size=signal_data['position_size'],
            trailing_stop=signal_data['trailing_stop'],
            callback_rate=signal_data['callback_rate'],
            timeframe=timeframe
        )
        if signal_id is None:
            logging.info(f"Свінг-сигнал для {pair} ({timeframe}) відхилено: {reason}")
            return None, 0
        logging.info(f"Свінг-сигнал для {pair} ({timeframe}) надіслано: signal_id={signal_id}")
        return signal_id, 1
    except Exception as e:
        logging.error(f"Помилка в process_swing_trade для {pair} ({timeframe}): {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка свінг-трейдингу для {pair} ({timeframe}): {str(e)}", force=True)
        return None, 0

async def analyze_swing(pair: str, timeframe: str = '1h') -> dict:
    """Аналізує ринок для свінг-стратегії."""
    try:
        limit = 1000 if timeframe in SWING_TIMEFRAMES else 500
        df = await get_historical_data(pair, timeframe, limit=limit)
        if df is None or len(df) < 100:
            logging.warning(f"Недостатньо даних для {pair} ({timeframe}): {len(df) if df is not None else 'None'}")
            return {'signal': 'wait', 'signal_prob': 0.0, 'explanation': 'Недостатньо даних', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'sentiment': 'neutral', 'price_prob': 0.0}
        df = await calculate_all_indicators(df)
        if df.empty:
            logging.error(f"Помилка обчислення індикаторів для {pair} ({timeframe})")
            return {'signal': 'wait', 'signal_prob': 0.0, 'explanation': 'Помилка обчислення індикаторів', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'sentiment': 'neutral', 'price_prob': 0.0}
        required_columns = ['close', 'rsi', 'macd', 'signal_line', 'atr', 'adx', 'bollinger_width', 'bb_lower', 'bb_upper']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        current_price = float(df['close'].iloc[-1])
        rsi = float(df['rsi'].iloc[-1])
        macd = float(df['macd'].iloc[-1])
        signal_line = float(df['signal_line'].iloc[-1])
        atr = float(max(df['atr'].iloc[-1], 0.002 * current_price))
        adx = float(df['adx'].iloc[-1])
        bollinger_width = float(df['bollinger_width'].iloc[-1])
        sentiment = 'trending' if adx > 25 else 'ranging' if bollinger_width < 0.05 else 'neutral'
        models_available = bool(ML_MODELS.get(pair, {}).get(timeframe, {}).get(sentiment, {}).get('models', []))
        signal_prob = 0.5
        if models_available:
            features = ML_MODELS[pair][timeframe][sentiment].get('features', FEATURES)
            X = df[features].iloc[-1].values.reshape(1, -1)
            scaler = ML_MODELS[pair][timeframe][sentiment]['scaler']
            X_scaled = scaler.transform(X)
            models = ML_MODELS[pair][timeframe][sentiment]['models']
            valid_models = [model for model in models if hasattr(model, 'predict_proba')]
            if valid_models:
                signal_prob = np.mean([model.predict_proba(X_scaled)[0][1] for model in valid_models])
                signal_prob = float(signal_prob)
        leverage = min(pair_settings[pair]['max_leverage'], max(pair_settings[pair]['min_leverage'], int(15 * signal_prob))) if timeframe in SWING_TIMEFRAMES else min(pair_settings[pair]['max_leverage'], max(pair_settings[pair]['min_leverage'], int(10 * signal_prob)))
        atr_multiplier_sl = 2.0
        atr_multiplier_tp = 3.0
        position_size = RiskGuard().calculate_kelly_position(signal_prob, pair_performance[pair]['win_rate'], volatility=bollinger_width)
        trailing_stop = atr * 1.5
        callback_rate = 0.15 if timeframe in SWING_TIMEFRAMES else 0.05
        confidence_base = 0.85 if models_available else 0.75 if timeframe in SWING_TIMEFRAMES else 0.75 if models_available else 0.65
        signal = 'wait'
        explanation = ''
        stop_loss = 0.0
        take_profit = 0.0
        if sentiment == 'trending':
            if (signal_prob > pair_settings[pair]['signal_threshold_buy'] or (rsi < 65 and macd > signal_line and adx > 20)) and models_available:
                signal = 'buy'
                signal_prob = min(signal_prob, confidence_base)
                stop_loss = current_price - atr_multiplier_sl * atr
                take_profit = current_price + atr_multiplier_tp * atr
                explanation = f"Трендовий ринок: RSI={rsi:.2f}, MACD={macd:.2f}, ADX={adx:.2f}, ML Prob={signal_prob:.2f}"
            elif not models_available and rsi < 65 and macd > signal_line and adx > 20:
                signal = 'buy'
                signal_prob = confidence_base
                stop_loss = current_price - atr_multiplier_sl * atr
                take_profit = current_price + atr_multiplier_tp * atr
                explanation = f"Трендовий ринок (без ML): RSI={rsi:.2f}, MACD={macd:.2f}, ADX={adx:.2f}"
            elif (signal_prob < pair_settings[pair]['signal_threshold_sell'] or (rsi > 75 and macd < signal_line and adx > 20)) and models_available:
                signal = 'sell'
                signal_prob = min(1 - signal_prob, confidence_base)
                stop_loss = current_price + atr_multiplier_sl * atr
                take_profit = current_price - atr_multiplier_tp * atr
                explanation = f"Трендовий ринок: RSI={rsi:.2f}, MACD={macd:.2f}, ADX={adx:.2f}, ML Prob={signal_prob:.2f}"
            elif not models_available and rsi > 75 and macd < signal_line and adx > 20:
                signal = 'sell'
                signal_prob = confidence_base
                stop_loss = current_price + atr_multiplier_sl * atr
                take_profit = current_price - atr_multiplier_tp * atr
                explanation = f"Трендовий ринок (без ML): RSI={rsi:.2f}, MACD={macd:.2f}, ADX={adx:.2f}"
        else:
            if (signal_prob > pair_settings[pair]['signal_threshold_buy'] or (df['close'].iloc[-1] < df['bb_lower'].iloc[-1] and rsi < 35)) and models_available:
                signal = 'buy'
                signal_prob = min(signal_prob, confidence_base)
                stop_loss = current_price - atr_multiplier_sl * atr
                take_profit = current_price + atr_multiplier_tp * atr
                explanation = f"Рейндж ринок: ціна нижче Bollinger, RSI={rsi:.2f}, ML Prob={signal_prob:.2f}"
            elif not models_available and df['close'].iloc[-1] < df['bb_lower'].iloc[-1] and rsi < 35:
                signal = 'buy'
                signal_prob = confidence_base
                stop_loss = current_price - atr_multiplier_sl * atr
                take_profit = current_price + atr_multiplier_tp * atr
                explanation = f"Рейндж ринок (без ML): ціна нижче Bollinger, RSI={rsi:.2f}"
            elif (signal_prob < pair_settings[pair]['signal_threshold_sell'] or (df['close'].iloc[-1] > df['bb_upper'].iloc[-1] and rsi > 65)) and models_available:
                signal = 'sell'
                signal_prob = min(1 - signal_prob, confidence_base)
                stop_loss = current_price + atr_multiplier_sl * atr
                take_profit = current_price - atr_multiplier_tp * atr
                explanation = f"Рейндж ринок: ціна вище Bollinger, RSI={rsi:.2f}, ML Prob={signal_prob:.2f}"
            elif not models_available and df['close'].iloc[-1] > df['bb_upper'].iloc[-1] and rsi > 65:
                signal = 'sell'
                signal_prob = confidence_base
                stop_loss = current_price + atr_multiplier_sl * atr
                take_profit = current_price - atr_multiplier_tp * atr
                explanation = f"Рейндж ринок (без ML): ціна вище Bollinger, RSI={rsi:.2f}"
        price_precision = int(exchange.markets[pair]['precision']['price']) if pair in exchange.markets else 4
        amount_precision = int(exchange.markets[pair]['precision']['amount']) if pair in exchange.markets else 4
        stop_loss = round(float(stop_loss), price_precision)
        take_profit = round(float(take_profit), price_precision)
        trailing_stop = round(float(trailing_stop), price_precision)
        position_size = round(float(position_size), amount_precision)
        logging.info(f"Сигнал для {pair} ({timeframe}): {signal}, prob={signal_prob:.2f}, price={current_price:.{price_precision}f}, SL={stop_loss:.{price_precision}f}, TP={take_profit:.{price_precision}f}")
        return {
            'signal': signal,
            'signal_prob': float(signal_prob),
            'explanation': explanation,
            'current_price': float(current_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'leverage': int(leverage),
            'position_size': float(position_size),
            'trailing_stop': float(trailing_stop),
            'callback_rate': float(callback_rate),
            'sentiment': sentiment,
            'price_prob': float(signal_prob)
        }
    except Exception as e:
        logging.error(f"Помилка в analyze_swing для {pair} ({timeframe}): {str(e)}")
        return {'signal': 'wait', 'signal_prob': 0.0, 'explanation': f'Помилка: {str(e)}', 'current_price': 0.0, 'stop_loss': 0.0, 'take_profit': 0.0, 'leverage': 1, 'position_size': 0.0, 'trailing_stop': 0.0, 'callback_rate': 0.0, 'sentiment': 'neutral', 'price_prob': 0.0}

async def detect_market_activity(pair: str, timeframe: str) -> tuple:
    """Виявляє ринкову активність для переходу до скальпінгу."""
    try:
        df = await get_historical_data(pair, timeframe, limit=1000)
        if df is None or len(df) < 50:
            logging.warning(f"Недостатньо даних для {pair} ({timeframe}) в detect_market_activity: {len(df) if df is not None else 'None'}")
            return False, None
        df = await calculate_all_indicators(df)
        if df.empty:
            logging.error(f"Помилка обчислення індикаторів для {pair} ({timeframe})")
            return False, None
        adx = df['adx'].iloc[-1]
        volume_change = df['volume'].pct_change().iloc[-1]
        bb_width = df['bollinger_width'].iloc[-1]
        is_active = adx > 25 and volume_change > 0.1 and bb_width > 0.015
        if is_active:
            signal_id = f"activity_{pair}_{uuid.uuid4()}"
            signal_message = (
                f"📈 Виявлено ринкову активність для {pair} ({timeframe})\n"
                f"ADX: {adx:.2f}, Volume Change: {volume_change:.2f}, Bollinger Width: {bb_width:.4f}\n"
                f"Активувати скальпінг?"
            )
            keyboard = [[InlineKeyboardButton("Активувати скальпінг", callback_data=f"scalp_{signal_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await CONFIG['application'].bot.send_message(chat_id=CONFIG['chat_id'], text=signal_message, reply_markup=reply_markup)
            pending_signals[signal_id] = {'pair': pair, 'timeframe': timeframe, 'type': 'activity'}
            logging.info(f"Виявлено ринкову активність для {pair} ({timeframe}): is_active={is_active}, signal_id={signal_id}")
            return True, signal_id
        return False, None
    except Exception as e:
        logging.error(f"Помилка в detect_market_activity для {pair} ({timeframe}): {str(e)}")
        return False, None

async def update_correlation_matrix(pairs: list) -> dict:
    """Оновлює кореляційну матрицю для пар."""
    try:
        dfs = {}
        for pair in pairs:
            df = await get_historical_data(pair, '5m', limit=288)
            if df is not None and len(df) >= 50:
                dfs[pair] = df['close']
        if len(dfs) < 2:
            logging.warning("Недостатньо даних для кореляційної матриці")
            return {}
        combined_df = pd.DataFrame(dfs)
        correlation_matrix = combined_df.pct_change().rolling(window=288).corr().iloc[-len(pairs):]
        logging.info("Кореляційну матрицю оновлено")
        return correlation_matrix.to_dict()
    except Exception as e:
        logging.error(f"Помилка оновлення кореляційної матриці: {str(e)}")
        return {}

def save_trade_to_csv(trade: dict) -> None:
    """Зберігає торгівлю в CSV."""
    file_exists = os.path.isfile(TRADES_CSV)
    with open(TRADES_CSV, 'a', newline='') as csvfile:
        fieldnames = ['signal_id', 'pair', 'action', 'entry_price', 'exit_price', 'stop_loss', 'take_profit', 'trailing_stop', 'callback_rate', 'position_size', 'leverage', 'profit', 'open_time', 'close_time', 'reason', 'rsi', 'adx', 'stoch_k', 'market_regime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)

async def trading_loop(prioritize_swing: bool = True) -> None:
    """Основний цикл торгівлі."""
    global strategy_state, PAIRS, TIMEFRAMES, SWING_TIMEFRAMES, ML_MODELS
    if not await initialize():
        logging.error("Не вдалося ініціалізувати торгівельний цикл")
        return
    trained_pairs = set()
    signal_count = 0
    last_signal_reset = time.time()
    MAX_SIGNALS_PER_HOUR = CONFIG.get('max_signals_per_hour', 10)
    while True:
        try:
            current_time = time.time()
            if current_time - last_signal_reset > 3600:
                signal_count = 0
                last_signal_reset = current_time
            if signal_count >= MAX_SIGNALS_PER_HOUR:
                logging.warning(f"Досягнуто ліміт сигналів ({MAX_SIGNALS_PER_HOUR}/год)")
                await asyncio.sleep(300)
                continue
            tasks = []
            for pair in PAIRS:
                if pair not in exchange.markets:
                    logging.error(f"Пара {pair} недоступна на біржі")
                    continue
                models_trained = any(
                    ML_MODELS.get(pair, {}).get(tf, {}).get(regime, {}).get('models')
                    for tf in TIMEFRAMES + SWING_TIMEFRAMES for regime in ['trending', 'ranging', 'neutral']
                )
                if models_trained and pair not in trained_pairs:
                    trained_pairs.add(pair)
                    logging.info(f"Моделі для {pair} готові, дозволяємо торгівлю")
                    await send_telegram_buffered(f"🔄 Моделі для {pair} готові, дозволяємо торгівлю", force=True)
                if prioritize_swing or strategy_state['current_strategy'] == 'swing':
                    for timeframe in SWING_TIMEFRAMES:
                        if pair_settings.get(pair, {}).get('active', True):
                            tasks.append(process_swing_trade(pair, timeframe, signal_count))
                if models_trained and (strategy_state['current_strategy'] == 'scalping' or not prioritize_swing):
                    for timeframe in TIMEFRAMES:
                        if pair_settings.get(pair, {}).get('active', True) and (strategy_state['active_pair'] == pair or strategy_state['active_pair'] is None):
                            tasks.append(process_scalping_trade(pair, timeframe, signal_count))
                if models_trained and strategy_state['current_strategy'] == 'swing':
                    is_active, signal_id = await detect_market_activity(pair, '5m')
                    if is_active:
                        strategy_state['current_strategy'] = 'scalping'
                        strategy_state['active_pair'] = pair
                        strategy_state['scalp_end_time'] = time.time() + 3600
                        logging.info(f"Перехід до скальпінгу для {pair}: signal_id={signal_id}")
                        await send_telegram_buffered(f"📈 Перехід до скальпінгу для {pair}", force=True)
                if strategy_state['current_strategy'] == 'scalping' and strategy_state['scalp_end_time'] and time.time() > strategy_state['scalp_end_time']:
                    strategy_state['current_strategy'] = 'swing'
                    strategy_state['active_pair'] = None
                    strategy_state['scalp_end_time'] = None
                    logging.info(f"Повернення до свінг-трейдингу для {pair}")
                    await send_telegram_buffered(f"🔄 Повернення до свінг-трейдингу для {pair}", force=True)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    signal_id, new_signals = result
                    signal_count += new_signals
            await asyncio.sleep(CONFIG.get('trading_loop_interval', 30))
        except asyncio.CancelledError:
            logging.info("Торговий цикл скасовано")
            break
        except Exception as e:
            logging.error(f"Помилка в trading_loop: {str(e)}")
            await send_telegram_buffered(f"⚠️ Помилка в trading_loop: {str(e)}", force=True)
            await asyncio.sleep(60)