import logging
from config import load_config
from datetime import datetime
from telegram_utils import send_telegram_buffered

logging.basicConfig(filename='risk_management.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CONFIG = load_config()
TRADES_CSV = CONFIG.get('trades_csv', 'trades.csv')

class RiskGuard:
    def __init__(self, initial_budget: float = 1000, max_drawdown: float = 0.08, reserve_ratio: float = 0.25):
        """Ініціалізація RiskGuard з агресивними параметрами."""
        self.initial_budget = initial_budget
        self.current_budget = initial_budget
        self.available_budget = initial_budget * (1 - reserve_ratio)
        self.reserve_budget = initial_budget * reserve_ratio
        self.max_drawdown = max_drawdown
        self.active_trades = {}  # {signal_id: trade_info}
        self.trade_history = []
        self.total_profit = 0
        self.daily_loss = 0
        self.consecutive_losses = 0
        self.emergency_stop = False
        self.active = True
        self.max_active_trades = CONFIG.get('max_active_trades', 8)
        self.correlated_pairs = {}
        logging.info("RiskGuard ініціалізовано")

    def reset(self) -> None:
        """Скидає стан RiskGuard."""
        self.daily_loss = 0
        self.consecutive_losses = 0
        self.emergency_stop = False
        self.active = True
        self.correlated_pairs = {}
        self.current_budget = self.initial_budget
        self.available_budget = self.initial_budget * (1 - CONFIG.get('reserve_ratio', 0.25))
        logging.info("RiskGuard скинуто")

    async def update_budget(self) -> bool:
        """Оновлює бюджет і перевіряє просідання."""
        try:
            drawdown = (self.initial_budget - self.current_budget) / self.initial_budget
            logging.info(f"Поточне просідання: {drawdown:.2%}")
            if drawdown > 0.05:
                await send_telegram_buffered(f"⚠️ Попередження: Просідання досягло {drawdown:.2%}")
            if drawdown > self.max_drawdown or self.consecutive_losses >= 5:
                self.emergency_stop = True
                logging.warning(f"Аварійна зупинка: Просідання {drawdown:.2%} або {self.consecutive_losses} послідовних збитків")
                await send_telegram_buffered(f"🚨 Аварійна зупинка: Просідання {drawdown:.2%} або {self.consecutive_losses} збитків")
                return False
            return True
        except Exception as e:
            logging.error(f"Помилка оновлення бюджету: {str(e)}")
            await send_telegram_buffered(f"⚠️ Помилка оновлення бюджету: {str(e)}")
            return False

    def calculate_kelly_position(self, signal_prob: float, win_rate: float, avg_win_loss_ratio: float = 2.5, volatility: float = 0.02) -> float:
        """Розраховує розмір позиції за критерієм Келлі."""
        try:
            kelly_fraction = (signal_prob * (avg_win_loss_ratio + 1) - 1) / avg_win_loss_ratio
            volatility_adjustment = max(0.6, 1.0 - volatility * 4)
            kelly_fraction *= volatility_adjustment
            return max(0.02, min(0.35, kelly_fraction))
        except Exception as e:
            logging.error(f"Помилка розрахунку позиції Келлі: {str(e)}")
            return 0.02

    def check_correlated_risk(self, pair: str, correlation_matrix: dict) -> bool:
        """Перевіряє ризик кореляції між парами."""
        try:
            correlated = [p for p in self.active_trades.values() if abs(correlation_matrix.get(pair, {}).get(p['pair'], 0)) > 0.6]
            if len(correlated) >= 3:
                logging.info(f"Пропуск угоди для {pair}: забагато корельованих угод ({[p['pair'] for p in correlated]})")
                return False
            return True
        except Exception as e:
            logging.error(f"Помилка перевірки кореляційного ризику для {pair}: {str(e)}")
            return False

    def validate_signal(self, signal_params: dict) -> bool:
        """Валідує торговий сигнал."""
        try:
            required_params = ['pair', 'timeframe', 'signal_prob', 'price', 'atr']
            for param in required_params:
                if param not in signal_params:
                    logging.error(f"Відсутній параметр сигналу: {param}")
                    return False
                if param in ['signal_prob', 'price', 'atr'] and (not isinstance(signal_params[param], (int, float)) or signal_params[param] <= 0):
                    logging.error(f"Некоректне значення для {param}: {signal_params[param]}")
                    return False
            if signal_params['signal_prob'] < CONFIG.get('min_signal_prob', 0.55):
                logging.info(f"Сигнал відхилено для {signal_params['pair']} ({signal_params['timeframe']}): низька впевненість {signal_params['signal_prob']:.3f}")
                return False
            if len(self.active_trades) >= self.max_active_trades:
                logging.info(f"Сигнал відхилено для {signal_params['pair']} ({signal_params['timeframe']}): досягнуто максимум активних угод ({self.max_active_trades})")
                return False
            drawdown = (self.initial_budget - self.current_budget) / self.initial_budget
            if drawdown > self.max_drawdown:
                logging.warning(f"Сигнал відхилено для {signal_params['pair']} ({signal_params['timeframe']}): просідання {drawdown:.2%} перевищує максимум {self.max_drawdown}")
                return False
            return True
        except Exception as e:
            logging.error(f"Помилка валідації сигналу для {signal_params.get('pair', 'невідомо')} ({signal_params.get('timeframe', 'невідомо')}): {str(e)}")
            return False

    async def update_trade(self, trade: dict) -> None:
        """Оновлює статус торгівлі."""
        try:
            signal_id = trade.get('signal_id')
            if signal_id not in self.active_trades:
                logging.error(f"Торгівля {signal_id} не знайдена в активних угодах")
                return
            trade_info = self.active_trades[signal_id]
            current_price = trade.get('current_price', trade_info['entry_price'])
            profit = (current_price - trade_info['entry_price']) * trade_info['position_size'] * trade_info['leverage']
            trade_info['profit'] = profit
            trade_info['close_time'] = datetime.now().isoformat()
            if profit < 0:
                self.consecutive_losses += 1
                self.daily_loss += abs(profit)
            else:
                self.consecutive_losses = 0
            self.current_budget += profit
            self.available_budget += profit
            self.total_profit += profit
            self.trade_history.append(trade_info)
            del self.active_trades[signal_id]
            logging.info(f"Торгівлю {signal_id} закрито: Прибуток={profit:.2f}, Бюджет={self.current_budget:.2f}")
            await send_telegram_buffered(f"🔒 Торгівля {signal_id} ({trade_info['pair']}) закрита: Прибуток={profit:.2f}")
            await self.update_budget()
        except Exception as e:
            logging.error(f"Помилка оновлення торгівлі {signal_id}: {str(e)}")
            await send_telegram_buffered(f"⚠️ Помилка оновлення торгівлі {signal_id}: {str(e)}")