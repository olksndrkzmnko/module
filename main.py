import asyncio
import logging
from config import load_config, initialize_global_exchange
from data_processing import get_historical_data, save_to_db
from machine_learning import train_ml_models_async, load_saved_models
from strategy import trading_loop
from risk_management import RiskGuard
from telegram_utils import send_telegram_buffered, test_telegram
from trading import monitor_signals, monitor_trades

logging.basicConfig(filename='signals.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CONFIG = load_config()
PAIRS = CONFIG['pairs']
TIMEFRAMES = CONFIG['timeframes']
SWING_TIMEFRAMES = CONFIG.get('swing_timeframes', [])
risk_guard = RiskGuard()

async def diagnostics():
    """Виконує діагностику системи перед запуском торгівлі."""
    logging.info("Запуск діагностики системи")
    success = True

    # 1. Перевірка конфігурації
    try:
        if not PAIRS or not TIMEFRAMES:
            logging.error("Помилка: PAIRS або TIMEFRAMES порожні")
            success = False
        else:
            logging.info("Конфігурація успішно завантажена")
    except Exception as e:
        logging.error(f"Помилка завантаження конфігурації: {str(e)}")
        success = False

    # 2. Перевірка ініціалізації біржі
    try:
        await initialize_global_exchange()
        exchange = CONFIG.get('exchange_instance')
        if exchange is None:
            logging.error("Біржа не ініціалізована")
            success = False
        else:
            unavailable_pairs = [pair for pair in PAIRS if pair not in exchange.markets]
            if unavailable_pairs:
                logging.warning(f"Недоступні пари: {unavailable_pairs}")
                await send_telegram_buffered(f"⚠️ Недоступні пари: {unavailable_pairs}", force=True)
                for pair in unavailable_pairs:
                    CONFIG['pair_settings'][pair]['active'] = False
            balance = await exchange.fetch_balance()
            logging.info(f"Баланс: {balance.get('USDT', {}).get('free', 0.0)} USDT")
    except Exception as e:
        logging.error(f"Помилка перевірки біржі: {str(e)}")
        success = False

    # 3. Перевірка Telegram
    try:
        if not await test_telegram():
            logging.error("Помилка Telegram API")
            success = False
        else:
            logging.info("Telegram API працює")
    except Exception as e:
        logging.error(f"Помилка Telegram API: {str(e)}")
        success = False

    # 4. Перевірка моделей
    try:
        loaded_models = await load_saved_models(PAIRS)
        if loaded_models == 0:
            logging.info("Моделі відсутні, запускаємо навчання")
            await train_ml_models_async(PAIRS, TIMEFRAMES + SWING_TIMEFRAMES)
        else:
            logging.info(f"Завантажено {loaded_models} моделей")
    except Exception as e:
        logging.error(f"Помилка перевірки моделей: {str(e)}")
        success = False

    # 5. Перевірка даних та індикаторів
    try:
        test_pair = PAIRS[0]
        test_timeframe = TIMEFRAMES[0]
        data = await get_historical_data(test_pair, test_timeframe, limit=100)
        if data is None or len(data) < 50:
            logging.error(f"Недостатньо даних для {test_pair} ({test_timeframe}): {len(data) if data is not None else 'None'}")
            success = False
        else:
            logging.info(f"Дані для {test_pair} ({test_timeframe}) отримані: {len(data)} рядків")
    except Exception as e:
        logging.error(f"Помилка отримання даних: {str(e)}")
        success = False

    # 6. Перевірка сигналів
    try:
        await monitor_signals(pair=test_pair, timeframe=test_timeframe)
        logging.info(f"Генерація сигналів для {test_pair} ({test_timeframe}) працює")
    except Exception as e:
        logging.error(f"Помилка генерації сигналів: {str(e)}")
        success = False

    # 7. Перевірка RiskGuard
    try:
        test_signal = {'pair': test_pair, 'timeframe': test_timeframe, 'signal_prob': 0.8, 'price': 100000, 'atr': 100}
        result = risk_guard.validate_signal(test_signal)
        logging.info(f"Перевірка RiskGuard: сигнал для {test_pair} ({test_timeframe}) {'підтверджено' if result else 'відхилено'}")
    except Exception as e:
        logging.error(f"Помилка RiskGuard: {str(e)}")
        success = False

    if success:
        logging.info("Діагностика успішно завершена")
        await send_telegram_buffered("✅ Діагностика успішна", force=True)
    else:
        logging.error("Діагностика виявила помилки, перевірте логи")
        await send_telegram_buffered("⚠️ Діагностика виявила помилки, перевірте signals.log", force=True)
    
    return success

async def main():
    """Основна функція запуску торгової системи."""
    try:
        if await diagnostics():
            for pair in PAIRS:
                for timeframe in TIMEFRAMES + SWING_TIMEFRAMES:
                    try:
                        await save_to_db(pair, timeframe, await get_historical_data(pair, timeframe))
                    except Exception as e:
                        logging.error(f"Помилка ініціалізації таблиці для {pair} ({timeframe}): {str(e)}")
                        await send_telegram_buffered(f"⚠️ Помилка ініціалізації таблиці для {pair} ({timeframe}): {str(e)}", force=True)
                        continue
            await load_saved_models(PAIRS)
            tasks = [
                trading_loop(prioritize_swing=True),
                monitor_signals(),
                monitor_trades(),
                train_ml_models_async(PAIRS, TIMEFRAMES + SWING_TIMEFRAMES)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logging.error("Торгівля не запущена через помилки діагностики")
            await send_telegram_buffered("⚠️ Торгівля не запущена через помилки діагностики", force=True)
    except Exception as e:
        logging.error(f"Помилка в основному циклі: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка в основному циклі: {str(e)}", force=True)

if __name__ == "__main__":
    asyncio.run(main())