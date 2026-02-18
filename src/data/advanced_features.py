"""
Feature Engineering Avanzado
Wavelet transforms, autocorrelación, interacciones complejas
"""

import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pywt


class AdvancedFeatureEngineer:
    """Crea features avanzadas para mejorar poder predictivo"""
    
    @staticmethod
    def create_wavelet_features(series: pd.Series, wavelet: str = 'db4', level: int = 3) -> pd.DataFrame:
        """
        Descomposición Wavelet para separar señal de ruido
        
        Wavelet transform captura patrones en múltiples escalas temporales
        """
        coeffs = pywt.wavedec(series.fillna(method='ffill'), wavelet, level=level)
        
        features = {}
        
        # Approximation (low frequency = trend)
        features['wavelet_trend'] = pywt.upcoef('a', coeffs[0], wavelet, level=level)[:len(series)]
        
        # Detail coefficients (high frequency = volatility)
        for i, coeff in enumerate(coeffs[1:], 1):
            detail = pywt.upcoef('d', coeff, wavelet, level=level, take=len(series))[:len(series)]
            features[f'wavelet_detail_{i}'] = detail
        
        return pd.DataFrame(features, index=series.index)
    
    @staticmethod
    def create_autocorrelation_features(series: pd.Series, lags: list = [1, 6, 12, 24]) -> pd.DataFrame:
        """
        Features de autocorrelación para capturar dependencias temporales
        """
        features = {}
        
        for lag in lags:
            # Autocorrelación
            features[f'autocorr_lag_{lag}'] = series.rolling(lag*2).apply(
                lambda x: x.autocorr(lag=lag) if len(x) >= lag*2 else np.nan
            )
            
            # Partial autocorrelation (aproximado)
            features[f'returns_lag_{lag}'] = series.shift(lag)
        
        return pd.DataFrame(features, index=series.index)
    
    @staticmethod
    def create_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de interacción entre DOGE y TSLA
        
        Captura correlaciones dinámicas entre activos
        """
        features = pd.DataFrame(index=df.index)
        
        # Correlación rolling
        for window in [6, 12, 24]:
            features[f'doge_tsla_corr_{window}h'] = df['doge_ret_1h'].rolling(window).corr(
                df['tsla_ret_1h']
            )
        
        # Ratio de volatilidades
        features['vol_ratio_doge_tsla'] = df['doge_vol_zscore'] / (df['tsla_vol_zscore'].abs() + 1e-6)
        
        # Momentum divergence
        features['momentum_divergence'] = (
            df.get('doge_momentum_6h', 0) - df.get('tsla_momentum_6h', 0)
        )
        
        # Beta rolling (sensibilidad de DOGE a TSLA)
        for window in [12, 24]:
            cov = df['doge_ret_1h'].rolling(window).cov(df['tsla_ret_1h'])
            var = df['tsla_ret_1h'].rolling(window).var()
            features[f'doge_tsla_beta_{window}h'] = cov / (var + 1e-6)
        
        return features
    
    @staticmethod
    def create_sentiment_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Interacciones complejas entre sentimiento y mercado
        """
        features = pd.DataFrame(index=df.index)
        
        # Sentimiento * Volatilidad (amplifica señal en mercados volátiles)
        if 'sentiment_ensemble' in df.columns and 'doge_vol_zscore' in df.columns:
            features['sentiment_x_vol_doge'] = (
                df['sentiment_ensemble'] * df['doge_vol_zscore']
            )
            features['sentiment_x_vol_tsla'] = (
                df['sentiment_ensemble'] * df['tsla_vol_zscore']
            )
        
        # Cambio en sentimiento (aceleración)
        if 'sentiment_ensemble' in df.columns:
            features['sentiment_velocity'] = df['sentiment_ensemble'].diff(1)
            features['sentiment_acceleration'] = features['sentiment_velocity'].diff(1)
        
        # Sentimiento rezagado ponderado (decaying importance)
        if 'sentiment_ensemble' in df.columns:
            weights = np.array([0.5, 0.3, 0.2])  # Más peso a lags recientes
            for i, weight in enumerate(weights, 1):
                lag_col = f'sentiment_ensemble_lag{i}'
                if lag_col in df.columns:
                    if i == 1:
                        features['sentiment_weighted_avg'] = df[lag_col] * weight
                    else:
                        features['sentiment_weighted_avg'] += df[lag_col] * weight
        
        # Relevancia condicionada (solo importa si hay sentimiento fuerte)
        if 'relevance_score' in df.columns and 'sentiment_ensemble' in df.columns:
            features['relevance_conditional'] = (
                df['relevance_score'] * df['sentiment_ensemble'].abs()
            )
        
        return features
    
    @staticmethod
    def create_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta regímenes de mercado (trending, ranging, volatile)
        """
        features = pd.DataFrame(index=df.index)
        
        # ADX (Average Directional Index) - mide fuerza de tendencia
        for col in ['doge_close', 'tsla_close']:
            if col in df.columns:
                # Simplified ADX
                high = df[col].rolling(14).max()
                low = df[col].rolling(14).min()
                close = df[col]
                
                tr = np.maximum(
                    high - low,
                    np.maximum(
                        abs(high - close.shift(1)),
                        abs(low - close.shift(1))
                    )
                )
                
                atr = tr.rolling(14).mean()
                features[f'{col.split("_")[0]}_atr'] = atr
        
        # Volatility regime (high/low)
        if 'doge_vol_zscore' in df.columns:
            features['vol_regime_doge'] = pd.cut(
                df['doge_vol_zscore'],
                bins=[-np.inf, -1, 1, np.inf],
                labels=[0, 1, 2]  # low, normal, high
            ).astype(float)
        
        if 'tsla_vol_zscore' in df.columns:
            features['vol_regime_tsla'] = pd.cut(
                df['tsla_vol_zscore'],
                bins=[-np.inf, -1, 1, np.inf],
                labels=[0, 1, 2]
            ).astype(float)
        
        return features
    
    @staticmethod
    def create_time_encoding_advanced(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encoding temporal más sofisticado
        """
        features = pd.DataFrame(index=df.index)
        
        # Hora del día (no solo sin/cos, sino bins)
        hours = df.index.hour
        
        # Trading sessions
        features['session_asian'] = ((hours >= 0) & (hours < 8)).astype(int)
        features['session_european'] = ((hours >= 8) & (hours < 16)).astype(int)
        features['session_american'] = ((hours >= 16) & (hours < 24)).astype(int)
        
        # Fin de semana (crypto sigue, stocks no)
        features['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        # Principio/fin de mes (comportamientos institucionales)
        features['is_month_start'] = (df.index.day <= 5).astype(int)
        features['is_month_end'] = (df.index.day >= 25).astype(int)
        
        return features
    
    @staticmethod
    def create_all_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de features avanzadas
        """
        print("\nCreando features avanzadas...")
        
        advanced_features = []
        
        # 1. Wavelet decomposition
        if 'doge_ret_1h' in df.columns:
            print("[1/6] Wavelet decomposition (DOGE)...")
            wavelet_doge = AdvancedFeatureEngineer.create_wavelet_features(
                df['doge_ret_1h'], wavelet='db4', level=2
            )
            wavelet_doge.columns = ['doge_' + col for col in wavelet_doge.columns]
            advanced_features.append(wavelet_doge)
        
        if 'tsla_ret_1h' in df.columns:
            print("[2/6] Wavelet decomposition (TSLA)...")
            wavelet_tsla = AdvancedFeatureEngineer.create_wavelet_features(
                df['tsla_ret_1h'], wavelet='db4', level=2
            )
            wavelet_tsla.columns = ['tsla_' + col for col in wavelet_tsla.columns]
            advanced_features.append(wavelet_tsla)
        
        # 2. Autocorrelación
        if 'doge_ret_1h' in df.columns:
            print("[3/6] Autocorrelación features...")
            autocorr_doge = AdvancedFeatureEngineer.create_autocorrelation_features(
                df['doge_ret_1h'], lags=[1, 6, 12, 24]
            )
            autocorr_doge.columns = ['doge_' + col for col in autocorr_doge.columns]
            advanced_features.append(autocorr_doge)
        
        # 3. Cross-asset
        print("[4/6] Interacciones DOGE-TSLA...")
        cross_features = AdvancedFeatureEngineer.create_cross_asset_features(df)
        advanced_features.append(cross_features)
        
        # 4. Sentiment interactions
        print("[5/6] Interacciones sentimiento...")
        sentiment_features = AdvancedFeatureEngineer.create_sentiment_interactions(df)
        advanced_features.append(sentiment_features)
        
        # 5. Market regime
        print("[6/6] Regímenes de mercado...")
        regime_features = AdvancedFeatureEngineer.create_market_regime_features(df)
        advanced_features.append(regime_features)
        
        # Combinar todo
        df_enhanced = df.copy()
        for feat_df in advanced_features:
            df_enhanced = df_enhanced.join(feat_df, how='left')
        
        # Limpiar NaNs
        df_enhanced = df_enhanced.fillna(method='ffill').fillna(0)
        
        print(f"Features avanzadas creadas: {len(df_enhanced.columns)} columnas totales")
        
        return df_enhanced