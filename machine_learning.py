import asyncio
import logging
import os
import pickle
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
from config import load_config
from data_processing import get_historical_data, calculate_all_indicators, calculate_target, detect_market_regime
from telegram_utils import send_telegram_buffered
from sklearn.feature_selection import SelectKBest, f_classif

logging.basicConfig(filename='machine_learning.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CONFIG = load_config()
PAIRS = CONFIG['pairs']
TIMEFRAMES = CONFIG['timeframe']
SWING_TIMEFRAMES = CONFIG.get('swing_timeframe', [])
FEATURES = CONFIG.get('features', [
    'ema20', 'rsi', 'adx', 'macd', 'signal_line', 'bb_upper', 'bb_lower',
    'vwap', 'stoch_k', 'stoch_d', 'obv', 'roc', 'bollinger_width', 'atr',
    'momentum', 'ichimoku_tenkan', 'ichimoku_kijun'
])
ML_MODELS = {}

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.access(MODEL_DIR, os.W_OK):
    logging.error(f"Немає прав запису до директорії {MODEL_DIR}")
    raise PermissionError(f"Немає прав запису до {MODEL_DIR}")

def feature_selection(X: pd.DataFrame, y: np.ndarray, features: list, pair: str, timeframe: str, regime: str, n_features: int = 10) -> list:
    """Вибирає найкращі ознаки за допомогою SelectKBest."""
    try:
        if not isinstance(X, pd.DataFrame) or X.empty or len(y) == 0:
            logging.warning(f"Некоректні дані для відбору ознак: {pair} ({timeframe}, {regime})")
            return features
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(features)))
        selector.fit(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [features[i] for i in selected_indices]
        logging.info(f"Вибрано ознаки для {pair} ({timeframe}, {regime}): {selected_features}")
        return selected_features
    except Exception as e:
        logging.error(f"Помилка відбору ознак для {pair} ({timeframe}, {regime}): {str(e)}")
        return features

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, model_type: str, pair: str, timeframe: str, regime: str, n_trials: int = 10) -> dict:
    """Оптимізує гіперпараметри моделі за допомогою Optuna."""
    def objective(trial):
        try:
            if model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            elif model_type == 'gb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 5),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)
            elif model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 5),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'random_state': 42
                }
                model = XGBClassifier(**params)
            elif model_type == 'lr':
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'max_iter': 1000,
                    'random_state': 42
                }
                model = LogisticRegression(**params)
            else:
                raise ValueError(f"Непідтримуваний тип моделі: {model_type}")
            tscv = TimeSeriesSplit(n_splits=3)
            f1_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1')
            return np.mean(f1_scores) if f1_scores.size > 0 else 0.0
        except Exception as e:
            logging.error(f"Помилка в objective для {model_type} ({pair}, {timeframe}, {regime}): {str(e)}")
            return 0.0

    try:
        if not check_data_quality(X, y, pair, timeframe, regime):
            logging.error(f"Некоректні дані для оптимізації {pair} ({timeframe}, {regime})")
            return {
                'rf': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 2, 'random_state': 42},
                'gb': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
                'xgb': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
                'lr': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
            }[model_type]
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=600)
        if not study.trials:
            logging.warning(f"Жодне випробування не завершено для {model_type} ({pair}, {timeframe}, {regime})")
            return {
                'rf': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 2, 'random_state': 42},
                'gb': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
                'xgb': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
                'lr': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
            }[model_type]
        logging.info(f"Найкращі параметри для {model_type} ({pair}, {timeframe}, {regime}): {study.best_params}, F1={study.best_value:.2f}")
        return study.best_params
    except Exception as e:
        logging.error(f"Помилка оптимізації для {model_type} ({pair}, {timeframe}, {regime}): {str(e)}")
        return {
            'rf': {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 2, 'random_state': 42},
            'gb': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
            'xgb': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42},
            'lr': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
        }[model_type]

def check_data_quality(X: pd.DataFrame, y: np.ndarray, pair: str, timeframe: str, regime: str) -> bool:
    """Перевіряє якість даних для навчання моделі."""
    try:
        if X.empty or len(y) == 0:
            logging.error(f"Порожні дані для {pair} ({timeframe}, {regime}): X={len(X)} рядків, y={len(y)} міток")
            return False
        if len(np.unique(y)) < 2:
            logging.warning(f"Недостатньо класів для {pair} ({timeframe}, {regime}): {np.unique(y)}")
            return False
        if X.isna().any().any():
            logging.warning(f"Знайдено NaN у X для {pair} ({timeframe}, {regime}), заповнення...")
            X.fillna(X.mean(numeric_only=True), inplace=True)
            if X.isna().any().any():
                logging.error(f"Не вдалося заповнити NaN для {pair} ({timeframe}, {regime})")
                return False
        if len(X) < 50:
            logging.warning(f"Недостатньо даних для {pair} ({timeframe}, {regime}): {len(X)} рядків")
            return False
        return True
    except Exception as e:
        logging.error(f"Помилка перевірки якості даних для {pair} ({timeframe}, {regime}): {str(e)}")
        return False

async def train_ml_model(pair: str, df: pd.DataFrame, timeframe: str, force_retrain: bool = False) -> None:
    """Навчає моделі машинного навчання для торгової пари та таймфрейму."""
    global ML_MODELS, FEATURES
    try:
        min_data = 500
        if len(df) < min_data:
            logging.warning(f"Пропуск навчання для {pair} ({timeframe}): {len(df)} рядків")
            return
        df = df.copy()
        df = await calculate_all_indicators(df)
        if df is None or df.empty:
            logging.error(f"Помилка обчислення індикаторів для {pair} ({timeframe})")
            return
        regime = detect_market_regime(df)
        logging.info(f"Режим ринку для {pair} ({timeframe}): {regime}")
        y, valid_indices = calculate_target(df)
        if y is None or len(y) < 50:
            logging.warning(f"Недостатньо міток для {pair} ({timeframe}): {len(y) if y is not None else 'None'}")
            return
        X = df.loc[valid_indices, FEATURES]
        if not check_data_quality(X, y, pair, timeframe, regime):
            logging.error(f"Некоректні дані для {pair} ({timeframe}, {regime})")
            return
        smote = SMOTE(random_state=42, k_neighbors=min(3, len(y)-1))
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logging.info(f"SMOTE застосовано для {pair} ({timeframe}, {regime}): {len(X_resampled)} зразків")
        except ValueError as e:
            logging.warning(f"SMOTE не вдалося застосувати для {pair} ({timeframe}, {regime}): {str(e)}")
            X_resampled, y_resampled = X, y
        selected_features = feature_selection(X_resampled, y_resampled, FEATURES, pair, timeframe, regime)
        if not selected_features:
            logging.warning(f"Не вибрано ознак для {pair} ({timeframe}, {regime})")
            return
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled[selected_features])
        models = []
        for model_type in ['rf', 'gb', 'xgb', 'lr']:
            params = await asyncio.get_event_loop().run_in_executor(
                None, lambda: optimize_hyperparameters(X_scaled, y_resampled, model_type, pair, timeframe, regime)
            )
            if model_type == 'rf':
                model = RandomForestClassifier(**params)
            elif model_type == 'gb':
                model = GradientBoostingClassifier(**params)
            elif model_type == 'xgb':
                model = XGBClassifier(**params)
            elif model_type == 'lr':
                model = LogisticRegression(**params)
            model.fit(X_scaled, y_resampled)
            f1 = cross_val_score(model, X_scaled, y_resampled, cv=TimeSeriesSplit(n_splits=3), scoring='f1').mean()
            models.append({'model': model, 'f1_score': f1})
            logging.info(f"Модель {model_type} навчена для {pair} ({timeframe}, {regime}), F1={f1:.2f}")
        ML_MODELS.setdefault(pair, {}).setdefault(timeframe, {})[regime] = {
            'models': models,
            'scaler': scaler,
            'features': selected_features,
            'f1_score': max(model['f1_score'] for model in models)
        }
        model_file = os.path.join(MODEL_DIR, f"model_{pair.replace('/', '_')}_{timeframe}_{regime}.pkl")
        try:
            joblib.dump({'models': models, 'scaler': scaler, 'features': selected_features, 'f1_score': max(model['f1_score'] for model in models)}, model_file)
            logging.info(f"Модель збережено: {model_file}")
        except Exception as e:
            logging.error(f"Помилка збереження моделі {model_file}: {str(e)}")
    except Exception as e:
        logging.error(f"Помилка в train_ml_model для {pair} ({timeframe}): {str(e)}")
        await send_telegram_buffered(f"⚠️ Помилка в train_ml_model: {str(e)}")

async def train_ml_models_async(pairs: list, timeframes: list = None) -> None:
    """Асинхронно навчає моделі для всіх пар і таймфреймів."""
    global ML_MODELS, FEATURES
    timeframes = timeframes or (TIMEFRAMES + SWING_TIMEFRAMES)
    tasks = []
    for pair in pairs:
        for timeframe in timeframes:
            limit = 5000 if timeframe in TIMEFRAMES else 2000
            df = await get_historical_data(pair, timeframe, limit=limit)
            if df is None or len(df) < 500:
                logging.warning(f"Недостатньо даних для {pair} ({timeframe}): {len(df) if df is not None else 'None'}")
                continue
            tasks.append(train_ml_model(pair, df, timeframe))
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info("Навчання всіх моделей завершено")

async def load_saved_models(pairs: list) -> int:
    """Завантажує збережені моделі з файлів."""
    global ML_MODELS, FEATURES
    loaded_models = 0
    for pair in pairs:
        for timeframe in TIMEFRAMES + SWING_TIMEFRAMES:
            for regime in ['trending', 'ranging', 'neutral']:
                model_file = os.path.join(MODEL_DIR, f"model_{pair.replace('/', '_')}_{timeframe}_{regime}.pkl")
                if os.path.exists(model_file):
                    try:
                        if os.path.getsize(model_file) < 100:
                            logging.warning(f"Файл моделі {model_file} замалий, пропускаємо")
                            continue
                        with open(model_file, 'rb') as f:
                            data = pickle.load(f)
                        if not isinstance(data, dict) or 'models' not in data or 'scaler' not in data:
                            logging.error(f"Некоректна структура даних у {model_file}")
                            continue
                        ML_MODELS.setdefault(pair, {}).setdefault(timeframe, {})[regime] = {
                            'models': data['models'],
                            'scaler': data['scaler'],
                            'features': data.get('features', FEATURES),
                            'f1_score': data.get('f1_score', 0.0)
                        }
                        loaded_models += 1
                        logging.info(f"Завантажено модель для {pair} ({timeframe}, {regime})")
                    except Exception as e:
                        logging.error(f"Помилка завантаження моделі {model_file}: {str(e)}")
    if loaded_models == 0:
        logging.warning("Жодну модель не завантажено, запускаємо навчання")
        await train_ml_models_async(pairs, TIMEFRAMES + SWING_TIMEFRAMES)
    else:
        logging.info(f"Завантажено {loaded_models} моделей")
    return loaded_models

async def auto_retrain_models(pairs: list, timeframes: list = None) -> None:
    """Перенавчає моделі, якщо їхня продуктивність нижче порогу."""
    global ML_MODELS, FEATURES
    timeframes = timeframes or (TIMEFRAMES + SWING_TIMEFRAMES)
    for pair in pairs:
        for timeframe in timeframes:
            for regime in ['trending', 'ranging', 'neutral']:
                if ML_MODELS.get(pair, {}).get(timeframe, {}).get(regime, {}).get('f1_score', 0.0) < CONFIG.get('f1_threshold', 0.5):
                    logging.info(f"Продуктивність моделей для {pair} ({timeframe}, {regime}) нижче порогу, перенавчання")
                    df = await get_historical_data(pair, timeframe, limit=5000 if timeframe in TIMEFRAMES else 2000)
                    if df is None or len(df) < 500:
                        logging.warning(f"Недостатньо даних для перенавчання {pair} ({timeframe}): {len(df) if df is not None else 'None'}")
                        continue
                    await train_ml_model(pair, df, timeframe, force_retrain=True)