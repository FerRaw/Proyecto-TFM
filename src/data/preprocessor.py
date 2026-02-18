"""
Preprocesamiento de tweets y datos de mercado
"""

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config.settings import settings


class TwitterPreprocessor:
    """Preprocesa tweets de Elon Musk"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Limpieza avanzada de tweets
        Mantiene emojis y términos relevantes
        """
        if pd.isna(text):
            return ""
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Eliminar menciones
        text = re.sub(r'@\w+', '', text)
        
        # Reducir caracteres repetidos (goooood -> good)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Eliminar hashtags pero mantener texto
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Eliminar números standalone
        text = re.sub(r'\b\d+\b', '', text)
        
        # Mantener palabras, emojis y signos relevantes
        text = re.sub(r'[^\w\s\U0001F300-\U0001F9FF!?$%]', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lematización
        words = text.split()
        text = " ".join([self.lemmatizer.lemmatize(w) for w in words])
        
        return text
    
    def process_twitter_data(
        self, 
        df_posts: pd.DataFrame, 
        df_quotes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Pipeline completo de procesamiento de Twitter
        
        1. Filtra por fecha
        2. Unifica posts y quotes
        3. Calcula métricas de viralidad
        4. Agrupa por hora
        5. Limpia texto
        
        Returns:
            DataFrame con columnas:
        """
        print("\nProcesando datos de Twitter...")
        
        # --- Procesar Posts ---
        df_p = df_posts[[
            'createdAt', 'fullText', 'likeCount', 'retweetCount',
            'viewCount', 'replyCount', 'quoteCount', 'isReply'
        ]].copy()
        
        df_p.rename(columns={
            'createdAt': 'date',
            'fullText': 'text',
            'likeCount': 'likes',
            'retweetCount': 'retweets',
            'replyCount': 'replies',
            'quoteCount': 'quotes',
            'viewCount': 'views'
        }, inplace=True)
        
        # --- Procesar Quotes ---
        df_q = pd.DataFrame()
        df_q['date'] = df_quotes['musk_quote_created_at']
        df_q['text'] = (
            df_quotes['orig_tweet_text'].fillna('').str.lower() + 
            " " + 
            df_quotes['musk_quote_tweet'].fillna('').str.lower()
        )
        df_q['likes'] = df_quotes['musk_quote_like_count']
        df_q['retweets'] = df_quotes['musk_quote_retweet_count']
        df_q['replies'] = df_quotes['musk_quote_reply_count']
        df_q['quotes'] = df_quotes['musk_quote_quote_count']
        df_q['views'] = df_quotes['musk_quote_view_count']
        df_q['isReply'] = False
        
        # --- Unificar ---
        df_all = pd.concat([df_p, df_q], axis=0, ignore_index=True)
        
        # --- Feature Engineering ---
        df_all['isReply'] = df_all['isReply'].fillna(False).astype(bool).astype(int)
        
        # Viral score
        df_all['viral_score'] = (
            (df_all['views'].fillna(0) * 0.05) +
            (df_all['likes'].fillna(0) * 1.0) +
            (df_all['replies'].fillna(0) * 2.0) +
            (df_all['retweets'].fillna(0) * 3.0) +
            (df_all['quotes'].fillna(0) * 4.0)
        )
        
        # --- Agrupación Horaria ---
        df_all['date_h'] = df_all['date'].dt.floor('H')
        
        df_hourly = df_all.groupby('date_h').agg(
            tweet_count=('text', 'count'),
            is_reply_count=('isReply', 'sum'),
            viral_score=('viral_score', 'sum'),
            text=('text', lambda x: ' [->] '.join(x))
        )
        
        # Log normalization
        df_hourly['score_log'] = np.log1p(df_hourly['viral_score'])
        
        # --- Limpieza de Texto ---
        print("Limpiando textos...")
        df_hourly['text_clean'] = df_hourly['text'].apply(self.clean_text)
        
        # Filtrar tweets muy cortos
        df_clean = df_hourly[df_hourly['text_clean'].str.len() > settings.MIN_TWEET_LENGTH].copy()
        
        print(f"Tweets procesados: {len(df_clean):,} horas con actividad")
        print(f"Rango: {df_clean.index.min()} a {df_clean.index.max()}")
        print(df_clean.head())
        return df_clean


class MarketPreprocessor:
    """Preprocesa datos de mercado (DOGE y TSLA)"""
    
    @staticmethod
    def prepare_dogecoin(df_doge_raw: pd.DataFrame) -> pd.DataFrame:
        """Prepara datos de DOGE"""
        df = df_doge_raw.rename(columns={
            "Open": "doge_open",
            "High": "doge_high",
            "Low": "doge_low",
            "Close": "doge_close",
            "Volume": "doge_volume",
            "Number_of_Trades": "doge_trades",
            "Taker_Buy_Base_Asset_Volume": "doge_taker_buy"
        })
        
        cols = [
            "doge_open", "doge_high", "doge_low", "doge_close",
            "doge_volume", "doge_trades", "doge_taker_buy"
        ]
        df = df[cols]
        
        # Ajustar índice
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_localize(None)
        
        return df
    
    @staticmethod
    def prepare_tesla(df_tesla_raw: pd.DataFrame) -> pd.DataFrame:
        """Prepara datos de TSLA"""
        df = df_tesla_raw.rename(columns={
            "open": "tsla_open",
            "high": "tsla_high",
            "low": "tsla_low",
            "close": "tsla_close",
            "volume": "tsla_volume"
        })
        
        cols = ["tsla_open", "tsla_high", "tsla_low", "tsla_close", "tsla_volume"]
        df = df[cols]
        
        # Ajustar índice
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index).tz_convert('UTC').tz_localize(None)
        
        return df


# ============================================================================= #
# Función principal de preprocesamiento                                         #
# ============================================================================= #

def preprocess_all_data(
    df_posts: pd.DataFrame,
    df_quotes: pd.DataFrame,
    df_doge: pd.DataFrame,
    df_tesla: pd.DataFrame
) -> tuple:
    """
    Preprocesa todos los datos
    
    Returns:
        (df_tweets_clean, df_doge_prep, df_tesla_prep)
    """
    print("="*70)
    print("PREPROCESAMIENTO DE DATOS")
    print("="*70)
    
    # Twitter
    twitter_proc = TwitterPreprocessor()
    df_tweets = twitter_proc.process_twitter_data(df_posts, df_quotes)
    
    # Market
    print("\nProcesando datos de mercado...")
    df_doge_prep = MarketPreprocessor.prepare_dogecoin(df_doge)
    df_tesla_prep = MarketPreprocessor.prepare_tesla(df_tesla)
    
    print("\n" + "="*70)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*70)
    
    return df_tweets, df_doge_prep, df_tesla_prep