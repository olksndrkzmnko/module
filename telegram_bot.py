import asyncio
import logging
import time
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from datetime import datetime
from config import TELEGRAM_TOKEN, CHAT_ID, PAIRS, TIMEFRAMES, SWING_TIMEFRAMES, risk_guard, pair_performance, strategy_state, pending_signals, is_trading_running, trading_enabled
from utils import get_historical_data
from telegram_utils import send_telegram_buffered
from machine_learning import train_ml_models_async
from strategy import run_backtest, save_trade_to_csv  # Оновлено імпорт

async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    action, signal_id = data.split('_', 1)

    logging.info(f"Обробка дії {action} для сигналу {signal_id}")

    if action == 'scalp' and signal_id in pending_signals and pending_signals[signal_id]['type'] == 'activity':
        pair = pending_signals[signal_id]['pair']
        strategy_state['current_strategy'] = 'scalping'
        strategy_state['active_pair'] = pair
        strategy_state['scalp_end_time'] = time.time() + strategy_state['scalp_timeout']
        await query.message.reply_text(f"✅ Скальпінг активовано для {pair}")
        logging.info(f"Скальпінг активовано для {pair}, signal_id={signal_id}")
        del pending_signals[signal_id]
        return

    trade = pending_signals.get(signal_id)
    if not trade:
        await query.message.reply_text(f"Сигнал {signal_id} не знайдено або вже оброблено")
        logging.warning(f"Сигнал {signal_id} не знайдено")
        return

    pair = trade['pair']
    try:
        if action == 'confirm':
            trade['confirmed'] = True
            trade['order_id'] = f"tracked_{signal_id}"
            risk_guard.active_trades[signal_id] = trade
            logging.info(f"Угоду {signal_id} підтверджено: {trade}")
            await query.message.reply_text(f"✅ Угоду {signal_id} підтверджено та додано до відстеження")
            save_trade_to_csv(trade)  # Оновлено виклик
            del pending_signals[signal_id]
        elif action == 'reject':
            trade['confirmed'] = False
            logging.info(f"Сигнал {signal_id} відхилено")
            await query.message.reply_text(f"❌ Сигнал {signal_id} відхилено")
            del pending_signals[signal_id]
        elif action == 'close':
            if signal_id in risk_guard.active_trades:
                current_price = (await get_historical_data(pair, '1m', limit=10))['close'].iloc[-1]
                profit = (current_price - trade['entry_price']) * trade['position_size'] * trade['leverage'] if trade['action'] == 'buy' else (trade['entry_price'] - current_price) * trade['position_size'] * trade['leverage']
                trade['exit_price'] = current_price
                trade['profit'] = profit
                trade['close_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                trade['reason'] = 'Ручне закриття'
                save_trade_to_csv(trade)  # Оновлено виклик
                risk_guard.current_budget += profit
                risk_guard.available_budget += profit
                risk_guard.total_profit += profit
                pair_performance[pair]['profit'] += profit
                pair_performance[pair]['trades'] += 1
                pair_performance[pair]['win_rate'] = sum(1 for t in risk_guard.trade_history[-20:] if t['profit'] > 0 and t['pair'] == pair) / min(len([t for t in risk_guard.trade_history if t['pair'] == pair]), 20)
                del risk_guard.active_trades[signal_id]
                del pending_signals[signal_id]
                logging.info(f"Угоду {signal_id} закрито вручну: Прибуток={profit:.2f}, Вихідна ціна={current_price:.2f}")
                await query.message.reply_text(f"✅ Угоду {signal_id} закрито: Прибуток=${profit:.2f}")
            else:
                await query.message.reply_text(f"❌ Угода {signal_id} не знайдена")
                logging.warning(f"Угода {signal_id} не знайдена для закриття")
        elif action == 'ignore_close':
            await query.message.reply_text(f"Сигнал закриття {signal_id} проігноровано")
            logging.info(f"Сигнал закриття {signal_id} проігноровано")
            del pending_signals[signal_id]
    except Exception as e:
        logging.error(f"Помилка обробки підтвердження {signal_id}: {str(e)}")
        await query.message.reply_text(f"⚠️ Помилка обробки {signal_id}: {str(e)}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) запросив довідку")
    help_message = (
        "📖 Доступні команди:\n"
        "/start - Запустити бота\n"
        "/status - Переглянути поточний статус бота та торгівлі\n"
        "/stop - Зупинити торговий цикл\n"
        "/backtest - Запустити бектест для всіх пар\n"
        "/retrain - Перенавчити моделі\n"
        "/reset - Скинути налаштування RiskGuard та статистику пар\n"
        "/help - Показати цю довідку"
    )
    await update.message.reply_text(help_message)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) запустив команду /start")
    try:
        if str(user.id) != str(CHAT_ID):
            await update.message.reply_text("⛔ Ви не авторизовані для використання цього бота")
            logging.warning(f"Неавторизований доступ до /start: user_id={user.id}, CHAT_ID={CHAT_ID}")
            return
        global is_trading_running, trading_enabled
        if is_trading_running:
            await update.message.reply_text("🔔 Бот уже запущено. Використовуйте /help для перегляду команд.")
            logging.info(f"Спроба повторного запуску бота користувачем {user.id}")
            return
        trading_enabled = True
        is_trading_running = True
        await update.message.reply_text(
            f"Вітаю, {user.first_name}! Бот для торгівлі криптовалютами запущено. Використовуйте /help для перегляду доступних команд."
        )
        logging.info(f"Торговий цикл активовано для користувача {user.id}")
        await send_telegram_buffered(f"🔔 Бот запущено користувачем {user.first_name}", force=True)
    except Exception as e:
        logging.error(f"Помилка виконання команди /start для користувача {user.id}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка виконання /start: {str(e)}", force=True)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global is_trading_running
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) зупинив бота")
    try:
        is_trading_running = False
        await update.message.reply_text("🛑 Торговий цикл зупинено.")
    except Exception as e:
        logging.error(f"Помилка виконання команди /stop для користувача {user.id}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка виконання /stop: {str(e)}")

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) запустив бектест")
    try:
        results = await run_backtest(PAIRS)  # Оновлено виклик
        backtest_message = "📊 Результати бектесту:\n" + "\n".join(
            [f"{pair}: Прибуток=${res['profit']:.2f}, Торгів={res['trades']}, WinRate={res['win_rate']:.2f}"
             for pair, res in results.items()]
        )
        await update.message.reply_text(backtest_message)
    except Exception as e:
        logging.error(f"Помилка виконання команди /backtest для користувача {user.id}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка виконання /backtest: {str(e)}")

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) запустив тренування моделей")
    try:
        await train_ml_models_async(PAIRS, ['5m'], force_retrain=True)  # Змінено виклик
        await update.message.reply_text("✅ Моделі натреновано")
    except Exception as e:
        logging.error(f"Помилка виконання /train: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка тренування: {str(e)}")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global risk_guard, pair_performance
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) скинув налаштування")
    try:
        risk_guard.reset()
        pair_performance = {pair: {'profit': 0, 'trades': 0, 'win_rate': 0} for pair in PAIRS}
        await update.message.reply_text("🔄 Налаштування скинуто.")
    except Exception as e:
        logging.error(f"Помилка виконання команди /reset для користувача {user.id}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка виконання /reset: {str(e)}")

async def check_data_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) запросив перевірку даних")
    try:
        message = "📊 Перевірка даних:\n"
        for pair in PAIRS:
            for timeframe in TIMEFRAMES + SWING_TIMEFRAMES:
                df = await get_historical_data(pair, timeframe, limit=1000 if timeframe in ['1m', '5m'] else 500)
                if df is None or df.empty:
                    message += f"{pair} ({timeframe}): Недостатньо даних\n"
                else:
                    message += f"{pair} ({timeframe}): {len(df)} рядків, стовпці={df.columns.tolist()}\n"
        await update.message.reply_text(message)
    except Exception as e:
        logging.error(f"Помилка виконання команди /check_data: {str(e)}")
        await update.message.reply_text(f"⚠️ Помилка виконання /check_data: {str(e)}")

async def swing_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global strategy_state
    user = update.effective_user
    logging.info(f"Користувач {user.id} ({user.username}) запросив повернення до свінг-трейдингу")
    try:
        strategy_state['current_strategy'] = 'swing'
        strategy_state['active_pair'] = None
        strategy_state['scalp_end_time'] = None
        await update.message.reply_text("🔄 Повернення до свінг-трейдингу виконано.")
        logging.info("Повернення до свінг-трейдингу виконано")
    except Exception as e:
        logging.error(f"Помилка виконання команди /swing для користувача {user.id}: {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка виконання /swing: {str(e)}")