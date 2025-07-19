import logging
from prometheus_client import Gauge, start_http_server

def setup_prometheus(port=8000):
    try:
        # Ініціалізація метрик
        metrics = {
            'profit': Gauge('bot_profit', 'Total profit of trading bot', ['pair']),
            'trades': Gauge('bot_trades', 'Total number of trades', ['pair']),
            'win_rate': Gauge('bot_win_rate', 'Win rate of trading bot', ['pair']),
            'drawdown': Gauge('bot_drawdown', 'Current drawdown', ['pair']),
            'active_trades': Gauge('bot_active_trades', 'Number of active trades', ['pair'])
        }
        start_http_server(port)
        logging.info(f"Prometheus-сервер запущено на порту {port}")
        return metrics
    except Exception as e:
        logging.error(f"Помилка налаштування Prometheus: {str(e)}")
        return None