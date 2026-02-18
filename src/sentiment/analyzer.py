"""
Análisis de sentimiento 
"""

import numpy as np
import pandas as pd
from typing import Dict
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Analiza sentimiento con múltiples modelos"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Keywords (hardcoded para predicción en tiempo real)
        self.tesla_keywords = ['tesla', 'tsla', 'model', 'car', 'vehicle', 'electric', 'battery', 'autopilot', 'fsd', 'production', 'delivery', 'gigafactory', 'cybertruck', 'roadster', 'supercharger', 'model s', 'model 3', 'model x', 'model y', 'spacex']
        self.doge_keywords = ['doge', 'dogecoin', 'shiba', 'crypto', 'cryptocurrency', 'bitcoin', 'btc', 'coin', 'blockchain', 'hodl', 'moon', 'shib', 'dogefather']
        self.positive_keywords = ['great', 'altcoin', 'memecoin', 'amazing', 'awesome', 'love', 'best', 'excited', 'pump', 'bullish', 'moon', 'rocket', '🚀', '📈']
        self.negative_keywords = ['bad', 'worst', 'hate', 'terrible', 'crash', 'dump', 'bearish', 'scam', 'fraud', 'shitcoin    ']
    
    def analyze_tweet(self, text: str) -> Dict:
        """
        Analiza un ÚNICO tweet (para predicción en tiempo real)
        
        Args:
            text: Texto del tweet
            
        Returns:
            Dict con 'ensemble', 'relevance', 'textblob', 'vader', etc.
        """
        text_lower = text.lower()
        
        # Menciones
        mentions_tesla = any(kw in text_lower for kw in self.tesla_keywords)
        mentions_doge = any(kw in text_lower for kw in self.doge_keywords)
        mentions_any = mentions_tesla or mentions_doge
        
        # Conteo de palabras
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        # 1. TextBlob
        blob = TextBlob(text)
        sentiment_textblob = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # 2. VADER
        sentiment_vader = self.vader.polarity_scores(text)['compound']
        
        # 3. Léxico
        if positive_count + negative_count == 0:
            sentiment_lexicon = 0.0
        else:
            sentiment_lexicon = (positive_count - negative_count) / (positive_count + negative_count)
        
        # 4. Weighted
        base = (sentiment_textblob + sentiment_lexicon) / 2
        if mentions_any:
            boost = (positive_count - negative_count) * 0.1
            sentiment_weighted = np.clip(base + boost, -1, 1)
        else:
            sentiment_weighted = base
        
        # 5. Ensemble final
        sentiment_ensemble = (
            sentiment_textblob * 0.15 +
            sentiment_lexicon * 0.25 +
            sentiment_vader * 0.35 +
            sentiment_weighted * 0.25
        )
        
        # Confianza
        sentiments = [sentiment_textblob, sentiment_lexicon, sentiment_weighted, sentiment_vader]
        sentiment_std = np.std(sentiments)
        confidence_score = 1 - np.clip(sentiment_std / 0.5, 0, 1)
        
        # Relevancia (simplificada sin score_log)
        sentiment_intensity = abs(sentiment_ensemble)
        relevance_raw = (
            (int(mentions_any) * 45) +
            (sentiment_intensity * 30) +
            (confidence_score * 25)
        )
        
        # Normalizar a 0-1
        relevance = relevance_raw / 100.0
        
        return {
            'ensemble': float(sentiment_ensemble),
            'relevance': float(relevance),
            'textblob': float(sentiment_textblob),
            'vader': float(sentiment_vader),
            'lexicon': float(sentiment_lexicon),
            'weighted': float(sentiment_weighted),
            'confidence': float(confidence_score),
            'mentions_tesla': mentions_tesla,
            'mentions_doge': mentions_doge,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def analyze(self, df_clean: pd.DataFrame, keywords_dict: Dict) -> pd.DataFrame:
        """
        Análisis completo de sentimiento
        
        Returns:
            DataFrame con columnas de sentimiento y relevancia
        """
        print("\nAnalizando sentimiento...")
        
        df = df_clean.copy()
        
        # Extraer keywords
        tesla_kw = keywords_dict['tesla_keywords']
        doge_kw = keywords_dict['doge_keywords']
        pos_kw = keywords_dict['positive_keywords']
        neg_kw = keywords_dict['negative_keywords']
        
        # Helper functions
        def contains_kw(text, kw_list):
            text_lower = text.lower()
            return any(kw in text_lower for kw in kw_list)
        
        def count_kw(text, kw_list):
            text_lower = text.lower()
            return sum(1 for kw in kw_list if kw in text_lower)
        
        # Menciones
        df['mentions_tesla'] = df['text_clean'].apply(lambda x: contains_kw(x, tesla_kw))
        df['mentions_doge'] = df['text_clean'].apply(lambda x: contains_kw(x, doge_kw))
        df['mentions_any'] = df['mentions_tesla'] | df['mentions_doge']
        
        # Conteo de palabras de sentimiento
        df['positive_word_count'] = df['text_clean'].apply(lambda x: count_kw(x, pos_kw))
        df['negative_word_count'] = df['text_clean'].apply(lambda x: count_kw(x, neg_kw))
        
        # 1. TextBlob
        print("[1/4] TextBlob...")
        sentiments_tb = df['text_clean'].apply(lambda x: TextBlob(x).sentiment)
        df['sentiment_textblob'] = sentiments_tb.apply(lambda x: x.polarity)
        df['subjectivity'] = sentiments_tb.apply(lambda x: x.subjectivity)
        
        # 2. VADER
        print("[2/4] VADER...")
        df['sentiment_vader'] = df['text_clean'].apply(
            lambda x: self.vader.polarity_scores(x)['compound']
        )
        
        # 3. Léxico
        print("[3/4] Análisis léxico...")
        def lexicon_sentiment(row):
            pos = row['positive_word_count']
            neg = row['negative_word_count']
            if pos + neg == 0:
                return 0.0
            return (pos - neg) / (pos + neg)
        
        df['sentiment_lexicon'] = df.apply(lexicon_sentiment, axis=1)
        
        # 4. Weighted
        print("[4/4] Modelo ponderado...")
        def weighted_sentiment(row):
            base = (row['sentiment_textblob'] + row['sentiment_lexicon']) / 2
            if row['mentions_any']:
                boost = (row['positive_word_count'] - row['negative_word_count']) * 0.1
                return np.clip(base + boost, -1, 1)
            return base
        
        df['sentiment_weighted'] = df.apply(weighted_sentiment, axis=1)
        
        # 5. Ensemble final
        print("Creando ensemble...")
        df['sentiment_ensemble'] = (
            df['sentiment_textblob'] * 0.15 +
            df['sentiment_lexicon'] * 0.25 +
            df['sentiment_vader'] * 0.35 +
            df['sentiment_weighted'] * 0.25
        )
        
        # Categorización
        df['sentiment_category'] = pd.cut(
            df['sentiment_ensemble'],
            bins=[-np.inf, -0.3, -0.05, 0.05, 0.3, np.inf],
            labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        )
        
        # Métricas de confianza
        df['sentiment_std'] = df[[
            'sentiment_textblob', 'sentiment_lexicon',
            'sentiment_weighted', 'sentiment_vader'
        ]].std(axis=1)
        
        df['confidence_score'] = 1 - np.clip(df['sentiment_std'] / 0.5, 0, 1)
        
        # 6. Relevance Score
        print("Calculando relevancia...")
        sentiment_intensity = df['sentiment_ensemble'].abs()
        
        df['relevance_score'] = (
            (df['mentions_any'].astype(int) * 45) +
            (df['score_log'] * 25) +
            (sentiment_intensity * 20) +
            (df['confidence_score'] * 10)
        )
        
        # Normalizar 0-100
        min_val = df['relevance_score'].min()
        max_val = df['relevance_score'].max()
        df['relevance_score'] = 100 * (df['relevance_score'] - min_val) / (max_val - min_val)
        
        print(f"Sentimiento analizado: {len(df):,} tweets")
        
        return df