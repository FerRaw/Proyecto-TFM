"""
Identificación de palabras clave contextuales
"""

import re
import pandas as pd
from collections import Counter
from typing import Dict
from nltk.corpus import stopwords


class KeywordExtractor:
    """Extrae y clasifica keywords de tweets"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_words(self, text: str, remove_stopwords: bool = True):
        """Extrae palabras individuales"""
        words = text.lower().split()
        
        if remove_stopwords:
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        return words
    
    def extract_bigrams(self, text: str, remove_stopwords: bool = True):
        """Extrae bigramas"""
        words = self.extract_words(text, remove_stopwords)
        return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    
    def identify_keywords(
        self,
        df_clean: pd.DataFrame,
        tesla_threshold: int = 20,
        doge_threshold: int = 15,
        sentiment_threshold: int = 10
    ) -> Dict:
        """
        Identifica palabras clave de Tesla, Doge y sentimiento
        """
        print("\nIdentificando keywords contextuales...")
        
        # Contar palabras
        all_words = []
        all_bigrams = []
        
        for text in df_clean['text_clean']:
            all_words.extend(self.extract_words(text))
            all_bigrams.extend(self.extract_bigrams(text))
        
        word_freq = Counter(all_words)
        bigram_freq = Counter(all_bigrams)
        
        # Tesla keywords
        tesla_seeds = [
            'tesla', 'tsla', 'model', 'car', 'vehicle', 'electric',
            'battery', 'autopilot', 'fsd', 'production', 'delivery',
            'gigafactory', 'cybertruck', 'roadster', 'supercharger'
        ]
        
        tesla_kw = []
        for word, freq in word_freq.items():
            if any(seed in word for seed in tesla_seeds) and freq >= tesla_threshold:
                tesla_kw.append(word)
        
        for bigram, freq in bigram_freq.items():
            if any(seed in bigram for seed in tesla_seeds) and freq >= tesla_threshold // 2:
                tesla_kw.append(bigram)
        
        # Doge/Crypto keywords
        crypto_seeds = [
            'doge', 'dogecoin', 'crypto', 'cryptocurrency', 'bitcoin',
            'btc', 'coin', 'blockchain', 'hodl', 'moon', 'shib'
        ]
        
        doge_kw = []
        for word, freq in word_freq.items():
            if any(seed in word for seed in crypto_seeds) and freq >= doge_threshold:
                doge_kw.append(word)
        
        for bigram, freq in bigram_freq.items():
            if any(seed in bigram for seed in crypto_seeds) and freq >= doge_threshold // 2:
                doge_kw.append(bigram)
        
        # Sentiment keywords
        positive_seeds = [
            'great', 'amazing', 'awesome', 'love', 'best', 'excellent',
            'incredible', 'fantastic', 'wonderful', 'perfect', 'happy',
            'success', 'win', 'good', 'better', 'improved', 'progress'
        ]
        
        positive_kw = [
            word for word, freq in word_freq.items()
            if any(seed in word for seed in positive_seeds) and freq >= sentiment_threshold
        ]
        
        negative_seeds = [
            'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible',
            'fail', 'failed', 'problem', 'issue', 'concern', 'worried',
            'fear', 'crisis', 'disaster', 'crash', 'broken', 'wrong'
        ]
        
        negative_kw = [
            word for word, freq in word_freq.items()
            if any(seed in word for seed in negative_seeds) and freq >= sentiment_threshold
        ]
        
        print(f"Tesla: {len(tesla_kw)} términos")
        print(f"Doge/Crypto: {len(doge_kw)} términos")
        print(f"Positivos: {len(positive_kw)} términos")
        print(f"Negativos: {len(negative_kw)} términos")
        
        return {
            'tesla_keywords': list(set(tesla_kw)),
            'doge_keywords': list(set(doge_kw)),
            'positive_keywords': list(set(positive_kw)),
            'negative_keywords': list(set(negative_kw))
        }
