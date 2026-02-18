"""
Carga de todos los datasets necesarios
"""

from typing import Optional
from datetime import datetime
import pandas as pd
import requests
import time
import databento as db
from config.settings import settings


class TwitterLoader:
    """Carga datos de tweets de Elon Musk"""
    
    @staticmethod
    def load_posts() -> pd.DataFrame:
        """Carga posts directos de Elon"""
        file_path = settings.DATA_RAW_DIR / settings.ELON_POSTS_FILE
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {file_path}\n"
                f"Descarga desde Kaggle: datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march"
            )
        
        df = pd.read_csv(file_path)
        df['createdAt'] = pd.to_datetime(df['createdAt']).dt.tz_localize(None)
        
        # Filtrar por fecha
        cutoff = pd.Timestamp(settings.START_DATE)
        df = df[df['createdAt'] >= cutoff].copy()
        
        print(f"Posts cargados: {len(df):,} desde {df['createdAt'].min()}")
        return df
    
    @staticmethod
    def load_quotes() -> pd.DataFrame:
        """Carga quote tweets de Elon"""
        file_path = settings.DATA_RAW_DIR / settings.ELON_QUOTES_FILE
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        df = pd.read_csv(file_path)
        df['musk_quote_created_at'] = pd.to_datetime(df['musk_quote_created_at']).dt.tz_localize(None)
        
        cutoff = pd.Timestamp(settings.START_DATE)
        df = df[df['musk_quote_created_at'] >= cutoff].copy()
        
        print(f"Quotes cargados: {len(df):,}")
        return df


class BinanceLoader:
    """Carga datos históricos de Binance"""
    
    @staticmethod
    def download_klines(
        symbol: str = "DOGEUSDT",
        interval: str = "1h",
        start_date: str = "2020-01-01"
    ) -> pd.DataFrame:
        """
        Descarga klines (OHLCV) de Binance
        
        Args:
            symbol: Par de trading (ej: DOGEUSDT)
            interval: Intervalo (1h, 1d, etc.)
            start_date: Fecha inicial YYYY-MM-DD
        """
        url = "https://api.binance.com/api/v3/klines"
        
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(settings.END_DATE).timestamp() * 1000)
        
        all_data = []
        
        print(f"Descargando {symbol} desde Binance...")
        
        while start_ts < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "limit": 1000
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                start_ts = data[-1][6] + 1  # Close time + 1ms
                
                last_date = pd.to_datetime(data[-1][0], unit='ms')
                print(f"Descargado hasta: {last_date.strftime('%Y-%m-%d')}")
                
                time.sleep(0.15)  # Rate limit
                
            except Exception as e:
                print(f"Error: {e}. Reintentando...")
                time.sleep(10)
        
        if not all_data:
            raise ValueError("No se obtuvieron datos de Binance")
        
        # Procesar datos
        columns = [
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
            'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
        ]
        
        df = pd.DataFrame(all_data, columns=columns)
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms')
        
        # Convertir a float
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'Taker_Buy_Base_Asset_Volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        df = df.set_index('Open_Time').sort_index()
        
        # Filtrar hasta fecha límite
        df = df[df.index < pd.Timestamp(settings.END_DATE)]
        
        print(f"Descarga completada: {len(df):,} registros")
        return df
    
    @staticmethod
    def load_or_download(force_download: bool = False) -> pd.DataFrame:
        """Carga desde cache o descarga si no existe"""
        file_path = settings.DATA_RAW_DIR / settings.DOGE_DATA_FILE
        
        if file_path.exists() and not force_download:
            print(f"Cargando DOGE desde cache...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"Cargado: {len(df):,} registros")
        else:
            df = BinanceLoader.download_klines(
                settings.BINANCE_SYMBOL,
                settings.BINANCE_INTERVAL,
                settings.START_DATE
            )
            # Guardar cache
            df.to_csv(file_path)
            print(f"Guardado en cache: {file_path}")
        
        return df


class DatabentoLoader:
    """Carga datos de acciones desde Databento"""
    
    @staticmethod
    def download_tesla(api_key: str) -> pd.DataFrame:
        """
        Descarga datos de Tesla desde Databento
        
        Args:
            api_key: API key de Databento
        """
        if not api_key:
            raise ValueError("API key de Databento no configurada")
        
        client = db.Historical(api_key)
        
        print("Descargando TSLA desde Databento ...")
        
        try:
            data = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols="TSLA",
                schema="ohlcv-1h",
                start=f"{settings.START_DATE}T00:00:00",
                end=f"{settings.END_DATE}T23:59:59",
                stype_in="raw_symbol"
            )
            
            df = data.to_df()
            df.index = pd.to_datetime(df.index)
            
            # Seleccionar columnas
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Filtrar fecha límite
            df = df[df.index < pd.Timestamp(settings.END_DATE, tz='UTC')]
            
            print(f"Descarga completada: {len(df):,} registros")
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error descargando de Databento: {e}")
    
    @staticmethod
    def load_or_download(api_key: str, force_download: bool = False) -> pd.DataFrame:
        """Carga desde cache o descarga si no existe"""
        file_path = settings.DATA_RAW_DIR / settings.TESLA_DATA_FILE
        
        if file_path.exists() and not force_download:
            print(f"Cargando TSLA desde cache...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df[df.index < pd.Timestamp(settings.END_DATE, tz='UTC')]
            print(f"Cargado: {len(df):,} registros")
        else:
            df = DatabentoLoader.download_tesla(api_key)
            # Guardar cache
            df.to_csv(file_path)
            print(f"Guardado en cache: {file_path}")
        
        return df


def load_all_raw_data(
    force_download_crypto: bool = False,
    force_download_stocks: bool = False,
    databento_api_key: Optional[str] = None
) -> tuple:
    """
    Carga todos los datos raw necesarios
    
    Returns:
        (df_posts, df_quotes, df_doge, df_tesla)
    """
    print("="*70)
    print("CARGANDO DATOS RAW")
    print("="*70)
    
    # Twitter
    print("\n Twitter Data")
    df_posts = TwitterLoader.load_posts()
    df_quotes = TwitterLoader.load_quotes()
    
    # Binance
    print("\n Crypto Market Data (Binance)")
    df_doge = BinanceLoader.load_or_download(force_download_crypto)
    
    # Databento
    print("\n Stock Market Data (Databento)")
    api_key = databento_api_key or settings.DATABENTO_API_KEY
    df_tesla = DatabentoLoader.load_or_download(api_key, force_download_stocks)
    
    print("\n" + "="*70)
    print("TODOS LOS DATOS CARGADOS")
    print("="*70)
    
    return df_posts, df_quotes, df_doge, df_tesla