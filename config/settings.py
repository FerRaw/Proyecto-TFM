"""
Configuración centralizada del proyecto TFM
"""
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración global del proyecto"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Archivos de datos raw
    ELON_POSTS_FILE: str = "elon_posts.csv"
    ELON_QUOTES_FILE: str = "elon_quotes.csv"
    DOGE_DATA_FILE: str = "doge_data.csv"
    TESLA_DATA_FILE: str = "tesla_data.csv"
    
    # Archivos procesados
    PROCESSED_TWEETS_FILE: str = "tweets_processed.parquet"
    PROCESSED_MARKET_FILE: str = "market_features.parquet"
    FINAL_DATASET_FILE: str = "master_dataset.parquet"
    
    # Fechas
    START_DATE: str = "2020-01-01"
    END_DATE: str = "2025-04-15"  # Último tweet de Elon
    
    # Parámetros de preprocesamiento
    MIN_TWEET_LENGTH: int = 10
    MIN_WORD_FREQ: int = 15
    
    # Keywords thresholds
    TESLA_THRESHOLD: int = 20
    DOGE_THRESHOLD: int = 15
    SENTIMENT_THRESHOLD: int = 10
    
    # Lags de Granger (optimizados)
    DOGE_SENTIMENT_LAGS: List[int] = [1, 2, 3]  # Lag óptimo: 3h
    TSLA_SENTIMENT_LAGS: List[int] = [1]        # Lag óptimo: 1h
    MAX_GRANGER_LAG: int = 12
    
    # Binance
    BINANCE_SYMBOL: str = "DOGEUSDT"
    BINANCE_INTERVAL: str = "1h"
    
    # API Keys (desde .env)
    DATABENTO_API_KEY: str = ""
    
    # API FastAPI
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "TFM - Musk Tweets Market Impact API"
    API_VERSION: str = "1.0.0"
    
    # Modelos
    MODELS_VERSION: str = "v1"
    RANDOM_SEED: int = 42
    N_CV_SPLITS: int = 5
    TEST_SIZE_HOURS: int = 24 * 7  # 1 semana para test en CV
    
    # Hiperparámetros XGBoost
    XGB_N_ESTIMATORS: int = 200
    XGB_LEARNING_RATE: float = 0.05
    XGB_MAX_DEPTH: int = 6
    XGB_MIN_CHILD_WEIGHT: int = 3
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    
    # Hiperparámetros LightGBM
    LGB_N_ESTIMATORS: int = 200
    LGB_LEARNING_RATE: float = 0.05
    LGB_MAX_DEPTH: int = 6
    LGB_NUM_LEAVES: int = 31
    
    # Hiperparámetros LSTM
    LSTM_UNITS: int = 64
    LSTM_DROPOUT: float = 0.2
    LSTM_EPOCHS: int = 50
    LSTM_BATCH_SIZE: int = 32
    
    # Hiperparámetros Elastic Net
    EN_ALPHA: float = 0.01
    EN_L1_RATIO: float = 0.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton
settings = Settings()


# Crear directorios si no existen
for directory in [
    settings.DATA_RAW_DIR,
    settings.DATA_PROCESSED_DIR,
    settings.MODELS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)