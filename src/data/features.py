"""
Feature engineering para modelos predictivos
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from config.settings import settings


class FeatureEngineer:
    """Crea features para modelos de ML"""
    
    @staticmethod
    def create_market_features(df_doge: pd.DataFrame, df_tesla: pd.DataFrame) -> pd.DataFrame:
        """
        Merge y feature engineering de datos de mercado
        """
        print("\n Feature engineering de mercado...")
        
        # Merge (left para mantener 24/7 crypto)
        df = df_doge.join(df_tesla, how="left")
        
        # --- TESLA FEATURES ---
        df["tsla_market_open"] = np.where(df["tsla_close"].notna(), 1, 0)
        df["tsla_ret_1h"] = np.log(df["tsla_close"] / df["tsla_close"].shift(1))
        
        # Z-score volumen
        tsla_vol = df['tsla_volume'].fillna(0)
        rolling_tsla = tsla_vol.rolling(window=24)
        df['tsla_vol_zscore'] = (tsla_vol - rolling_tsla.mean()) / rolling_tsla.std()
        
        # Forward fill
        df["tsla_close"] = df["tsla_close"].ffill()
        df["tsla_ret_1h"] = df["tsla_ret_1h"].fillna(0)
        df["tsla_vol_zscore"] = df["tsla_vol_zscore"].fillna(0)
        
        # --- DOGE FEATURES ---
        df["doge_ret_1h"] = np.log(df["doge_close"] / df["doge_close"].shift(1))
        
        # RSI
        df['doge_rsi'] = ta.rsi(df["doge_close"], length=14)
        
        # Z-score volumen
        rolling_doge = df['doge_volume'].rolling(window=24)
        df['doge_vol_zscore'] = (df['doge_volume'] - rolling_doge.mean()) / rolling_doge.std()
        
        # Buy pressure
        df['doge_buy_pressure'] = (
            df['doge_taker_buy'] / df['doge_volume'].replace(0, np.nan)
        ).fillna(0.5)
        
        # --- FEATURES TEMPORALES ---
        hours = df.index.hour
        days = df.index.dayofweek
        
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        df['day_sin'] = np.sin(2 * np.pi * days / 7)
        df['day_cos'] = np.cos(2 * np.pi * days / 7)
        
        # --- TARGETS ---
        df['TARGET_DOGE'] = df['doge_ret_1h'].shift(-1)
        df['TARGET_TSLA'] = df['tsla_ret_1h'].shift(-1)
        
        # Limpiar NaNs
        df.dropna(inplace=True)
        
        # Seleccionar columnas finales
        final_cols = [
            'doge_ret_1h', 'doge_vol_zscore', 'doge_buy_pressure', 'doge_rsi',
            'tsla_ret_1h', 'tsla_market_open', 'tsla_vol_zscore',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'TARGET_DOGE', 'TARGET_TSLA'
        ]
        
        print(f"Features de mercado: {df.shape}")
        
        return df[final_cols]
    
    @staticmethod
    def create_master_dataset(
        df_market: pd.DataFrame,
        df_sentiment: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crea dataset maestro uniendo mercado + sentimiento + lags
        """
        print("\n Creando dataset maestro...")
        
        # Preparar sentimiento
        df_sent = df_sentiment.copy()
        if 'date_h' in df_sent.columns:
            df_sent.set_index('date_h', inplace=True)
        
        cols_needed = ['sentiment_ensemble', 'relevance_score', 'mentions_tesla', 'mentions_doge']
        df_sent = df_sent[cols_needed]
        
        # Merge
        df_master = df_market.join(df_sent, how='left')
        
        # Rellenar huecos (horas sin tweets)
        df_master[cols_needed] = df_master[cols_needed].fillna(0)
        df_master['mentions_tesla'] = df_master['mentions_tesla'].astype(int)
        df_master['mentions_doge'] = df_master['mentions_doge'].astype(int)
        
        # Crear lags
        print("Generando lags...")
        lags = settings.DOGE_SENTIMENT_LAGS  # [1, 2, 3]
        sentiment_features = ['sentiment_ensemble', 'relevance_score']
        
        for col in sentiment_features:
            for lag in lags:
                df_master[f'{col}_lag{lag}'] = df_master[col].shift(lag)
        
        # Limpiar NaNs
        df_master.dropna(inplace=True)
        
        print(f"Dataset maestro: {df_master.shape}")
        print(f"Columnas: {list(df_master.columns)}")
        
        return df_master