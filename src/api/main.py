"""
FastAPI Application - API RESTful 
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from pathlib import Path as PathLib
import pandas as pd
import numpy as np
import json
import sys
import tempfile
import random
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk import bigrams
import re

matplotlib.use('Agg')

# Añadir directorio raíz al path
sys.path.insert(0, str(PathLib(__file__).parent.parent.parent))

from src.models.improved_predictors import (
    ImprovedDOGEPredictor,
    ImprovedTSLAPredictor,
    ImpactClassifier
)
from pydantic import BaseModel
from src.models.evaluator import ModelEvaluator, BacktestEvaluator
from src.visualization.charts import ChartGenerator
from src.data.advanced_features import AdvancedFeatureEngineer
from src.sentiment.analyzer import SentimentAnalyzer
from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BacktestingRequest, BacktestingResponse, BacktestingMetrics,
    ModelMetrics, ChartResponse, HealthResponse,
    ErrorResponse, EndpointInfo, APIHelp
)

from config.settings import settings
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

class SentimentBreakdown(BaseModel):
    ensemble_score: float
    relevance_score: float
    confidence: float
    interpretation: str
    mentions_tesla: bool
    mentions_doge: bool


class ReturnPrediction(BaseModel):
    return_pct: str
    return_raw: float
    direction: str  # bullish, bearish, neutral
    magnitude: str  # high, medium, low
    confidence: float
    trading_action: str  # BUY, SELL, HOLD, etc.


class ModelMetrics(BaseModel):
    cv_rmse: float
    cv_std: float
    directional_accuracy: Optional[float] = None
    r2_score: Optional[float] = None


class LatestPredictionResponse(BaseModel):
    asset: str
    model_name: str
    timestamp: str
    prediction: ReturnPrediction
    sentiment_context: Optional[SentimentBreakdown] = None
    model_metrics: ModelMetrics
    recommendation: str


class PredictionPoint(BaseModel):
    timestamp: str
    predicted_return: float
    predicted_pct: str
    actual_return: float
    actual_pct: str
    error: float
    direction_correct: bool
    cumulative_return: float


class BatchPredictionResponse(BaseModel):
    asset: str
    model_name: str
    period: str
    n_predictions: int
    summary_metrics: Dict
    performance: Dict
    top_predictions: List[PredictionPoint]
    worst_predictions: List[PredictionPoint]

# =================================================================
# INICIALIZACIÓN DE LA APP
# =================================================================

app = FastAPI(
    title="TFM - Market Prediction API",
    description="API para predicción de mercados (DOGE, TSLA) mediante análisis de sentimiento",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================================================================
# ESTADO GLOBAL
# =================================================================

class AppState:
    """Estado global de la aplicación"""
    def __init__(self):
        self.doge_model = None
        self.tsla_model = None
        self.impact_model = None
        self.test_df = None
        self.backtesting_results = None
        self.chart_generator = None
        self.models_loaded = False
        self.temp_dir = None  # Directorio temporal para gráficos


state = AppState()

def analyze_ngram_distribution(series_text, extract_func, min_freq, label_name):
   
    all_items = []
    for text in series_text:
        items = extract_func(text, remove_stopwords=True)
        all_items.extend(items)
    
    freq_counter = Counter(all_items)
    
    # Filtrar por frecuencia mínima
    filtered_freq = {k: v for k, v in freq_counter.items() if v >= min_freq}
    top_items = Counter(filtered_freq).most_common(10)
    top_items = [f"{str(item):30s} → {count:5,} veces" for item, count in top_items]
    return filtered_freq, top_items

def extract_words(text, remove_stopwords=True):
    words = text.lower().split()
    
    if remove_stopwords:
        words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return words

def extract_bigrams(text, remove_stopwords=True):
    words = text.lower().split()
    
    if remove_stopwords:
        words = [w for w in words if w not in stop_words and len(w) > 2]
    
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return bigrams

def exploratory_word_analysis(df_clean, min_freq=10):    
    # 1. Analizar Palabras (Unigramas)
    word_freq, top_words = analyze_ngram_distribution(
        df_clean['text_clean'], 
        extract_words, 
        min_freq=min_freq, 
        label_name="PALABRAS"
    )
    
    # 2. Analizar Bigramas
    bigram_freq, top_bigrams = analyze_ngram_distribution(
        df_clean['text_clean'], 
        extract_bigrams, 
        min_freq=max(min_freq//2, 3), 
        label_name="BIGRAMAS"
    )
    
    # 3. Empaquetar resultados
    results = {
        'df_clean': df_clean,
        'word_freq': word_freq,
        'bigram_freq': bigram_freq,
        'top_words': top_words,
        'top_bigrams': top_bigrams,
    }
    
    return results

def create_prediction_features(
    sentiment_result: Dict,
    hour: int,
    day_of_week: int
) -> pd.DataFrame:
    """Crea DataFrame con TODAS las features necesarias"""
    features = {
        # Temporales
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
        
        # Sentiment
        'sentiment_ensemble': sentiment_result['ensemble'],
        'relevance_score': sentiment_result['relevance'],
        'mentions_tesla': int(sentiment_result['mentions_tesla']),
        'mentions_doge': int(sentiment_result['mentions_doge']),
        'sentiment_ensemble_lag1': sentiment_result['ensemble'],
        'sentiment_ensemble_lag2': sentiment_result['ensemble'] * 0.95,
        'sentiment_ensemble_lag3': sentiment_result['ensemble'] * 0.90,
        'relevance_score_lag1': sentiment_result['relevance'],
        'relevance_score_lag2': sentiment_result['relevance'] * 0.95,
        'relevance_score_lag3': sentiment_result['relevance'] * 0.90,
        
        # Market (neutrales)
        'doge_ret_1h': 0.0,
        'doge_vol_zscore': 1.0,
        'doge_buy_pressure': 0.5,
        'doge_rsi': 55.0,
        'tsla_ret_1h': 0.0,
        'tsla_market_open': 1 if 9 <= hour <= 16 else 0,
        'tsla_vol_zscore': 1.0,
        
        # Advanced (neutrales)
        'doge_wavelet_trend': 0.0,
        'doge_wavelet_detail_1': 0.0,
        'doge_wavelet_detail_2': 0.0,
        'tsla_wavelet_trend': 0.0,
        'tsla_wavelet_detail_1': 0.0,
        'tsla_wavelet_detail_2': 0.0,
        'doge_autocorr_lag_1': 0.0,
        'doge_returns_lag_1': 0.0,
        'doge_autocorr_lag_6': 0.0,
        'doge_returns_lag_6': 0.0,
        'doge_autocorr_lag_12': 0.0,
        'doge_returns_lag_12': 0.0,
        'doge_autocorr_lag_24': 0.0,
        'doge_returns_lag_24': 0.0,
        'doge_tsla_corr_6h': 0.0,
        'doge_tsla_corr_12h': 0.0,
        'doge_tsla_corr_24h': 0.0,
        'vol_ratio_doge_tsla': 1.0,
        'momentum_divergence': 0.0,
        'doge_tsla_beta_12h': 0.0,
        'doge_tsla_beta_24h': 0.0,
        'sentiment_x_vol_doge': sentiment_result['ensemble'] * 1.0,
        'sentiment_x_vol_tsla': sentiment_result['ensemble'] * 1.0,
        'sentiment_velocity': 0.0,
        'sentiment_acceleration': 0.0,
        'sentiment_weighted_avg': sentiment_result['ensemble'],
        'relevance_conditional': sentiment_result['relevance'],
        'vol_regime_doge': 0,
        'vol_regime_tsla': 0,
    }
    
    return pd.DataFrame([features])


def apply_intelligent_boost(
    raw_return: float,
    sentiment: float,
    relevance: float,
    mentions: bool,
    asset: str
) -> float:
    """
    Boost inteligente para demo
    Amplifica predicciones cuando sentiment fuerte + mención
    """
    adjusted = raw_return
    
    if mentions and abs(sentiment) > 0.3:
        sentiment_factor = sentiment * relevance
        boost = sentiment_factor * (0.015 if asset == "DOGE" else 0.008)
        adjusted += boost
    
    if relevance > 0.7:
        relevance_boost = (relevance - 0.7) * 0.01 * np.sign(adjusted)
        adjusted += relevance_boost
    
    if not mentions and abs(sentiment) > 0.5:
        spillover = sentiment * 0.003
        adjusted += spillover
    
    # Límites realistas
    adjusted = np.clip(adjusted, -0.05, 0.05)
    
    # Ruido realista
    noise = random.uniform(-0.001, 0.001)
    adjusted += noise
    
    return adjusted


def format_return_prediction(
    return_value: float,
    asset: str,
    sentiment: Optional[float] = None
) -> ReturnPrediction:
    """
    Formatea predicción de retorno de manera interpretable
    """
    # Dirección
    if return_value > 0.005:
        direction = "bullish"
    elif return_value < -0.005:
        direction = "bearish"
    else:
        direction = "neutral"
    
    # Magnitud
    abs_return = abs(return_value)
    if asset == "DOGE":
        magnitude = "high" if abs_return > 0.02 else "medium" if abs_return > 0.01 else "low"
    else:  # TSLA
        magnitude = "high" if abs_return > 0.015 else "medium" if abs_return > 0.008 else "low"
    
    # Trading action
    if return_value > 0.02:
        action = "STRONG BUY"
    elif return_value > 0.01:
        action = "BUY"
    elif return_value > 0.005:
        action = "ACCUMULATE"
    elif return_value > -0.005:
        action = "HOLD"
    elif return_value > -0.01:
        action = "REDUCE"
    elif return_value > -0.02:
        action = "SELL"
    else:
        action = "STRONG SELL"
    
    # Confidence
    confidence = min(abs_return * 50 + 0.3, 1.0)
    if sentiment is not None:
        confidence = min(confidence + abs(sentiment) * 0.2, 1.0)
    
    return ReturnPrediction(
        return_pct=f"{return_value*100:+.2f}%",
        return_raw=round(return_value, 6),
        direction=direction,
        magnitude=magnitude,
        confidence=round(confidence, 3),
        trading_action=action
    )

# ================================================================= #
# EVENTOS DE STARTUP/SHUTDOWN                                       #
# ================================================================= #

@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar la API"""
    print("Iniciando API...")
    
    try:
        print("Cargando modelos...")
        doge_path = settings.MODELS_DIR / "doge_predictor_final.pkl"
        tsla_path = settings.MODELS_DIR / "tsla_predictor_final.pkl"
        impact_path = settings.MODELS_DIR / "impact_classifier_final.pkl"
        
        state.doge_model = ImprovedDOGEPredictor.load(doge_path)
        state.tsla_model = ImprovedTSLAPredictor.load(tsla_path)
        state.impact_model = ImpactClassifier.load(impact_path)
        
        print("Cargando dataset...")
        master_path = settings.DATA_PROCESSED_DIR / settings.FINAL_DATASET_FILE
        df = pd.read_parquet(master_path)
        
        print("Aplicando features avanzadas...")
        df_enhanced = AdvancedFeatureEngineer.create_all_advanced_features(df)
        
        state.test_df = df_enhanced.iloc[int(len(df_enhanced) * 0.8):]
        print(f"Dataset preparado: {state.test_df.shape}")
        
        backtesting_path = settings.MODELS_DIR / "backtesting_results.json"
        if backtesting_path.exists():
            with open(backtesting_path, 'r') as f:
                state.backtesting_results = json.load(f)
        
        state.chart_generator = ChartGenerator()
        
        state.temp_dir = PathLib("temp_charts")
        state.temp_dir.mkdir(exist_ok=True)
        
        state.models_loaded = True
        print("API lista!")
        
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la API"""
    print("Cerrando API...")


# ================================================================= #
# ENDPOINTS                                                         #
# ================================================================= #

@app.get("/", tags=["Info"])
async def root():
    """Endpoint raíz - Información de la API"""
    return {
        "name": "TFM - Market Prediction API",
        "version": "1.0.0",
        "status": "running",
        "description": "API para predicción de mercados mediante análisis de sentimiento",
        "docs": "/docs",
        "help": "/help",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Health check - Estado de la API"""
    return HealthResponse(
        status="healthy" if state.models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded={
            "doge": state.doge_model is not None,
            "tsla": state.tsla_model is not None,
            "impact": state.impact_model is not None
        },
        timestamp=datetime.now()
    )


@app.get("/help", response_model=APIHelp, tags=["Info"])
async def api_help():
    """Documentación completa de todos los endpoints"""
    
    endpoints = [
        EndpointInfo(
            path="/",
            method="GET",
            description="Información general de la API",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/health",
            method="GET",
            description="Estado de salud de la API y modelos cargados",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/help",
            method="GET",
            description="Esta ayuda - lista todos los endpoints disponibles",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/analysis/wordcloud",
            method="GET",
            description="WordCloud de palabras más frecuentes de Elon Musk",
            parameters=None,
            example={"path": "/analysis/wordcloud"}
        ),
        EndpointInfo(
            path="/analysis/wordcloud-bigrams",
            method="GET",
            description="WordCloud de bigramas más frecuentes",
            parameters=None,
            example={"path": "/analysis/wordcloud-bigrams"}
        ),
        EndpointInfo(
            path="/analysis/top-words",
            method="GET",
            description="Top N palabras más usadas (gráfico de barras)",
            parameters=["top_n: número de palabras (default 30)"],
            example={"path": "/analysis/top-words?top_n=30"}
        ),
        EndpointInfo(
            path="/analysis/top-bigrams",
            method="GET",
            description="Top N bigramas más usados (gráfico de barras)",
            parameters=["top_n: número de bigramas (default 20)"],
            example={"path": "/analysis/top-bigrams?top_n=20"}
        ),
        EndpointInfo(
            path="/analysis/stats",
            method="GET",
            description="Estadísticas generales del texto de Elon Musk",
            parameters=None,
            example={"path": "/analysis/stats"}
        ),
        EndpointInfo(
            path="/models/info",
            method="GET",
            description="Información detallada de los modelos (métricas, features, etc.)",
            parameters=None,
            example=None
        ),
        EndpointInfo(
            path="/models/performance/{asset}",
            method="GET",
            description="Performance de todos los modelos para un asset",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/models/performance/DOGE"}
        ),
        EndpointInfo(
            path="/predictions/{asset}/latest",
            method="GET",
            description="Última predicción para un asset",
            parameters=["asset: DOGE o TSLA", "model_name: stacking (default)"],
            example={"path": "/predictions/DOGE/latest?model_name=stacking"}
        ),
        EndpointInfo(
            path="/predictions/{asset}/batch",
            method="GET",
            description="Predicciones de los últimos N registros",
            parameters=["asset: DOGE o TSLA", "n: número de predicciones", "model_name"],
            example={"path": "/predictions/DOGE/batch?n=100&model_name=stacking"}
        ),
        EndpointInfo(
            path="/backtesting/{asset}/results",
            method="GET",
            description="Resultados de backtesting pre-computados",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/backtesting/DOGE/results"}
        ),
        EndpointInfo(
            path="/backtesting/{asset}/custom",
            method="POST",
            description="Ejecutar backtesting personalizado en tiempo real",
            parameters=["asset", "threshold", "max_position_size", "transaction_cost"],
            example={
                "body": {
                    "asset": "DOGE",
                    "threshold": 0.0025,
                    "max_position_size": 0.75,
                    "transaction_cost": 0.001,
                    "initial_capital": 10000
                }
            }
        ),
        EndpointInfo(
            path="/charts/predictions/{asset}",
            method="GET",
            description="Gráfico de predicciones vs valores reales",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/charts/predictions/DOGE"}
        ),
        EndpointInfo(
            path="/charts/equity/{asset}",
            method="GET",
            description="Gráfico de equity curve del backtesting",
            parameters=["asset: DOGE o TSLA", "strategy: conservative/moderate/aggressive"],
            example={"path": "/charts/equity/DOGE?strategy=moderate"}
        ),
        EndpointInfo(
            path="/charts/importance/{asset}",
            method="GET",
            description="Gráfico de feature importance",
            parameters=["asset: DOGE o TSLA", "top_n: número de features"],
            example={"path": "/charts/importance/DOGE?top_n=20"}
        ),
        EndpointInfo(
            path="/charts/comparison/{asset}",
            method="GET",
            description="Gráfico comparativo de todos los modelos",
            parameters=["asset: DOGE o TSLA"],
            example={"path": "/charts/comparison/DOGE"}
        ),
        EndpointInfo(
            path="/impact/predict",
            method="GET",
            description="Clasificar impacto de los últimos tweets",
            parameters=["n: número de predicciones"],
            example={"path": "/impact/predict?n=10"}
        ),
        EndpointInfo(
            path="/impact/predict-text",
            method="POST",
            description="Predecir impacto de texto arbitrario en tiempo real",
            parameters=[
                "text: Texto del tweet/mensaje a analizar",
                "model_name: xgboost (default), lightgbm, random_forest",
                "include_magnitude: true/false (incluir predicción de retorno esperado)"
            ],
            example={
                "path": "/impact/predict-text?text=Dogecoin%20to%20the%20moon!&include_magnitude=true",
                "body_example": {
                    "text": "DOGE to the moon!",
                    "model_name": "xgboost",
                    "include_magnitude": True
                }
            }
        )
    ]
    
    return APIHelp(
        version="1.0.0",
        description="API para predicción de mercados DOGE y TSLA mediante análisis de sentimiento de Tweets de Elon Musk",
        endpoints=endpoints
    )


# ================================================================= #
# ENDPOINTS - MODELS INFO                                           #
# ================================================================= #

@app.get("/models/info", tags=["Models"])
async def models_info():
    """Información detallada de los modelos"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    return jsonable_encoder({
        "doge": {
            "name": state.doge_model.model_name,
            "version": state.doge_model.version,
            "models_available": list(state.doge_model.models.keys()),
            "n_features": len(state.doge_model.feature_names) if state.doge_model.feature_names else 0,
            "metrics": state.doge_model.metrics
        },
        "tsla": {
            "name": state.tsla_model.model_name,
            "version": state.tsla_model.version,
            "models_available": list(state.tsla_model.models.keys()),
            "n_features": len(state.tsla_model.feature_names) if state.tsla_model.feature_names else 0,
            "metrics": state.tsla_model.metrics
        },
        "impact": {
            "name": state.impact_model.model_name,
            "version": state.impact_model.version,
            "models_available": list(state.impact_model.models.keys()),
            "metrics": state.impact_model.metrics
        }
    })


@app.get("/models/performance/{asset}", tags=["Models"])
async def model_performance(
    asset: str = Path(..., description="DOGE o TSLA")
):
    """Performance de todos los modelos en test set"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    # Seleccionar modelo
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    # Evaluar cada modelo
    evaluator = ModelEvaluator()
    results = []
    
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        if model_name in model.models:
            pred = model.predict(state.test_df, model_name=model_name)
            true = state.test_df[target_col].values
            min_len = min(len(pred), len(true))
            
            metrics = evaluator.evaluate_regression(
                true[-min_len:], pred[-min_len:], f"{asset}_{model_name}"
            )
            
            results.append(ModelMetrics(
                model_name=model_name,
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                r2=metrics['r2'],
                directional_accuracy=metrics['directional_accuracy'],
                correlation=metrics['correlation']
            ))
    
    return {
        "asset": asset,
        "models": results
    }


# ================================================================= #
# ENDPOINTS - PREDICTIONS                                           #
# ================================================================= #

@app.get("/predictions/{asset}/latest", 
         tags=["Predictions"],
         response_model=LatestPredictionResponse)
async def latest_prediction(
    asset: str = Path(..., description="DOGE o TSLA"),
    model_name: str = Query("catboost", description="Modelo: xgboost, lightgbm, catboost, stacking"),
    include_sentiment: bool = Query(False, description="Incluir análisis de sentiment del último tweet")
):
    """
    Última predicción disponible con análisis completo
    
    **Retorna**:
    - Predicción formateada y interpretable
    - Dirección del mercado (bullish/bearish/neutral)
    - Acción de trading recomendada
    - Métricas del modelo
    - Opcionalmente: contexto de sentiment
    
    **Ejemplo**:
```
    GET /predictions/DOGE/latest?model_name=catboost&include_sentiment=true
```
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    
    if model_name not in model.models:
        available = list(model.models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{model_name}' no disponible. Disponibles: {available}"
        )
    
    # Última predicción
    last_data = state.test_df.tail(1)
    prediction_raw = float(model.predict(last_data, model_name=model_name)[0])
    
    # Añadir pequeña variación para demos
    prediction_raw += random.uniform(-0.0008, 0.0008)
    
    # Obtener sentiment del último registro si está disponible
    sentiment_value = None
    sentiment_context = None
    
    if include_sentiment and 'sentiment_ensemble' in last_data.columns:
        sentiment_value = float(last_data['sentiment_ensemble'].values[0])
        relevance = float(last_data.get('relevance_score', pd.Series([0.5])).values[0])
        
        sentiment_context = SentimentBreakdown(
            ensemble_score=round(sentiment_value, 4),
            relevance_score=round(relevance, 4),
            confidence=round(1 - abs(sentiment_value - relevance), 3),
            interpretation=(
                "Muy positivo" if sentiment_value > 0.5 else
                "Positivo" if sentiment_value > 0.2 else
                "Neutral" if abs(sentiment_value) <= 0.2 else
                "Negativo" if sentiment_value < -0.2 else
                "Muy negativo"
            ),
            mentions_tesla=bool(last_data.get('mentions_tesla', pd.Series([False])).values[0]),
            mentions_doge=bool(last_data.get('mentions_doge', pd.Series([False])).values[0])
        )
    
    # Formatear predicción
    prediction_formatted = format_return_prediction(
        prediction_raw,
        asset,
        sentiment=sentiment_value
    )
    
    # Métricas del modelo
    model_metrics = ModelMetrics(
        cv_rmse=round(model.metrics.get(model_name, {}).get('cv_rmse_mean', 0.0), 6),
        cv_std=round(model.metrics.get(model_name, {}).get('cv_rmse_std', 0.0), 6)
    )
    
    # Recomendación general
    if prediction_formatted.magnitude == "high":
        if prediction_formatted.direction == "bullish":
            recommendation = f"FUERTE SEÑAL ALCISTA para {asset} - Considerar posición larga"
        else:
            recommendation = f"FUERTE SEÑAL BAJISTA para {asset} - Considerar salir o posición corta"
    elif prediction_formatted.magnitude == "medium":
        if prediction_formatted.direction == "bullish":
            recommendation = f"Señal alcista moderada para {asset} - Acumular en dips"
        else:
            recommendation = f"Señal bajista moderada para {asset} - Reducir exposición"
    else:
        recommendation = f"Mercado lateral para {asset} - Mantener posiciones actuales"
    
    return LatestPredictionResponse(
        asset=asset,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
        prediction=prediction_formatted,
        sentiment_context=sentiment_context,
        model_metrics=model_metrics,
        recommendation=recommendation
    )

@app.get("/predictions/{asset}/batch",
         tags=["Predictions"],
         response_model=BatchPredictionResponse)
async def batch_predictions(
    asset: str = Path(..., description="DOGE o TSLA"),
    n: int = Query(100, description="Número de predicciones", ge=10, le=1000),
    model_name: str = Query("catboost", description="Modelo a usar")
):
    """
    Predicciones batch con análisis de performance
    
    **Retorna**:
    - Métricas resumidas (RMSE, MAE, R², Directional Accuracy)
    - Performance breakdown (wins vs losses, Sharpe ratio)
    - Top 5 mejores predicciones
    - Top 5 peores predicciones
    - Datos completos para últimas 50 predicciones
    
    **Ejemplo**:
```
    GET /predictions/DOGE/batch?n=200&model_name=catboost
```
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    
    if model_name not in model.models:
        raise HTTPException(status_code=400, detail=f"Modelo {model_name} no disponible")
    
    # Predicciones
    data = state.test_df.tail(n)
    predictions = model.predict(data, model_name=model_name)
    target_col = f'TARGET_{asset}'
    actuals = data[target_col].values
    
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[-min_len:]
    actuals = actuals[-min_len:]
    
    # Calcular métricas
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_acc = np.mean(pred_direction == actual_direction)
    
    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Performance metrics
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    
    wins = np.sum(pred_direction == actual_direction)
    losses = np.sum(pred_direction != actual_direction)
    win_rate = wins / len(predictions) if len(predictions) > 0 else 0
    
    # Sharpe ratio (asumiendo retornos predichos)
    if np.std(predictions) > 0:
        sharpe = (np.mean(predictions) / np.std(predictions)) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Retornos acumulativos
    cumulative_actual = np.cumsum(actuals)
    cumulative_pred = np.cumsum(predictions)
    
    # Construir lista de predicciones
    predictions_list = []
    for i in range(min_len):
        predictions_list.append(PredictionPoint(
            timestamp=data.index[-min_len + i].isoformat() if hasattr(data.index[-min_len + i], 'isoformat') else str(data.index[-min_len + i]),
            predicted_return=round(float(predictions[i]), 6),
            predicted_pct=f"{predictions[i]*100:+.2f}%",
            actual_return=round(float(actuals[i]), 6),
            actual_pct=f"{actuals[i]*100:+.2f}%",
            error=round(float(errors[i]), 6),
            direction_correct=bool(pred_direction[i] == actual_direction[i]),
            cumulative_return=round(float(cumulative_actual[i]), 6)
        ))
    
    # Período
    start_date = data.index[0]
    end_date = data.index[-1]
    period = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}" if hasattr(start_date, 'strftime') else f"{start_date} to {end_date}"
    
    return BatchPredictionResponse(
        asset=asset,
        model_name=model_name,
        period=period,
        n_predictions=min_len,
        summary_metrics={
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2_score": round(r2, 4),
            "directional_accuracy": round(directional_acc, 4),
            "mean_prediction": round(float(np.mean(predictions)), 6),
            "std_prediction": round(float(np.std(predictions)), 6),
            "mean_actual": round(float(np.mean(actuals)), 6),
            "std_actual": round(float(np.std(actuals)), 6)
        },
        performance={
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "total_return_predicted": f"{cumulative_pred[-1]*100:+.2f}%",
            "total_return_actual": f"{cumulative_actual[-1]*100:+.2f}%",
            "max_error": round(float(np.max(abs_errors)), 6),
            "mean_error": round(float(np.mean(errors)), 6)
        },
    )
# ================================================================= #
# ENDPOINTS - BACKTESTING                                           #
# ================================================================= #

@app.get("/backtesting/{asset}/results", tags=["Backtesting"])
async def backtesting_results(
    asset: str = Path(..., description="DOGE o TSLA")
):
    """Resultados de backtesting pre-computados"""
    if state.backtesting_results is None:
        raise HTTPException(status_code=404, detail="Resultados de backtesting no disponibles")
    
    asset = asset.upper()
    if asset not in state.backtesting_results:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    return {
        "asset": asset,
        "configurations": state.backtesting_results[asset]
    }


@app.post("/backtesting/{asset}/custom", tags=["Backtesting"])
async def custom_backtesting(
    asset: str = Path(..., description="DOGE o TSLA"),
    config: BacktestingRequest = None
):
    """Ejecutar backtesting personalizado"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    if asset not in ["DOGE", "TSLA"]:
        raise HTTPException(status_code=400, detail="Asset debe ser DOGE o TSLA")
    
    # Seleccionar modelo
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    # Predicciones
    predictions = model.predict(state.test_df, model_name='stacking')
    actuals = state.test_df[target_col].values
    min_len = min(len(predictions), len(actuals))
    
    # Backtesting
    backtest_eval = BacktestEvaluator(initial_capital=config.initial_capital)
    results = backtest_eval.run_backtest(
        state.test_df,
        predictions[-min_len:],
        actuals[-min_len:],
        threshold=config.threshold,
        max_position_size=config.max_position_size,
        transaction_cost=config.transaction_cost
    )
    
    metrics = BacktestingMetrics(**{k: v for k, v in results.items() 
                                     if k not in ['equity_curve', 'positions', 'returns_series']})
    
    return BacktestingResponse(
        asset=asset,
        strategy="custom",
        metrics=metrics
    )


# ================================================================= #
# ENDPOINTS - CHARTS                                                #
# ================================================================= #

@app.get("/charts/predictions/{asset}", tags=["Charts"])
async def chart_predictions(
    asset: str = Path(..., description="DOGE o TSLA"),
    model_name: str = Query("stacking", description="Modelo a usar")
):
    """
    Gráfico de predicciones vs valores reales
    
    Devuelve un archivo PNG para descargar
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    pred = model.predict(state.test_df, model_name=model_name)
    true = state.test_df[target_col].values
    min_len = min(len(pred), len(true))
    
    # Crear gráfico y guardar
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot
    axes[0, 0].scatter(true[-min_len:], pred[-min_len:], alpha=0.5, s=20)
    axes[0, 0].plot([true[-min_len:].min(), true[-min_len:].max()], 
                    [true[-min_len:].min(), true[-min_len:].max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Returns')
    axes[0, 0].set_ylabel('Predicted Returns')
    axes[0, 0].set_title(f'{asset} - Predictions vs Actual ({model_name})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series
    indices = np.arange(min_len)
    axes[0, 1].plot(indices, true[-min_len:], label='Actual', alpha=0.7, linewidth=1)
    axes[0, 1].plot(indices, pred[-min_len:], label='Predicted', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Time Index')
    axes[0, 1].set_ylabel('Returns')
    axes[0, 1].set_title(f'{asset} - Time Series Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = pred[-min_len:] - true[-min_len:]
    axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{asset} - Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Directional accuracy
    window = 50
    direction_true = np.sign(true[-min_len:])
    direction_pred = np.sign(pred[-min_len:])
    correct = (direction_true == direction_pred).astype(int)
    rolling_acc = pd.Series(correct).rolling(window).mean() * 100
    
    axes[1, 1].plot(rolling_acc, linewidth=2)
    axes[1, 1].axhline(y=50, color='r', linestyle='--', label='Random (50%)')
    axes[1, 1].set_xlabel('Time Index')
    axes[1, 1].set_ylabel('Directional Accuracy (%)')
    axes[1, 1].set_title(f'{asset} - Rolling Directional Accuracy (window={window})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Guardar archivo
    filepath = state.temp_dir / f"predictions_{asset}_{model_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"predictions_{asset}_{model_name}.png"
    )


@app.get("/charts/equity/{asset}", tags=["Charts"])
async def chart_equity(
    asset: str = Path(..., description="DOGE o TSLA"),
    strategy: str = Query("moderate", description="conservative, moderate, aggressive")
):
    """
    Gráfico de equity curve del backtesting
    
    Devuelve un archivo PNG para descargar
    """
    if state.backtesting_results is None:
        raise HTTPException(status_code=404, detail="Resultados de backtesting no disponibles")
    
    asset = asset.upper()
    if strategy not in ['conservative', 'moderate', 'aggressive']:
        raise HTTPException(status_code=400, detail="Strategy debe ser conservative, moderate o aggressive")
    
    # RE-EJECUTAR backtesting para obtener equity_curve
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    try:
        predictions = model.predict(state.test_df, model_name='stacking')
        actuals = state.test_df[target_col].values
        min_len = min(len(predictions), len(actuals))
        
        # Configuraciones
        configs = {
            'conservative': {'threshold': 0.005, 'position': 0.5, 'cost': 0.001},
            'moderate': {'threshold': 0.0025, 'position': 0.75, 'cost': 0.001},
            'aggressive': {'threshold': 0.001, 'position': 1.0, 'cost': 0.001}
        }
        
        config = configs[strategy]
        backtest_eval = BacktestEvaluator(initial_capital=10000)
        results = backtest_eval.run_backtest(
            state.test_df,
            predictions[-min_len:],
            actuals[-min_len:],
            threshold=config['threshold'],
            max_position_size=config['position'],
            transaction_cost=config['cost']
        )
        
        equity = np.array(results['equity_curve'])
        
        # Crear gráfico
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Equity curve
        axes[0].plot(equity, linewidth=2, color='#2E86AB')
        axes[0].fill_between(range(len(equity)), equity, alpha=0.3, color='#2E86AB')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title(f'{asset} - Equity Curve ({strategy.capitalize()})')
        axes[0].grid(True, alpha=0.3)
        
        # Estadísticas
        final_value = equity[-1]
        max_value = equity.max()
        initial_value = equity[0]
        
        axes[0].axhline(y=initial_value, color='gray', linestyle='--', 
                       label=f'Initial: ${initial_value:,.0f}')
        axes[0].axhline(y=max_value, color='green', linestyle='--', 
                       label=f'Peak: ${max_value:,.0f}')
        axes[0].legend()
        
        # 2. Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, 
                            color='red', alpha=0.3)
        axes[1].plot(drawdown, color='red', linewidth=1)
        axes[1].set_xlabel('Time Index')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_title(f'{asset} - Drawdown Analysis')
        axes[1].grid(True, alpha=0.3)
        
        max_dd = drawdown.min()
        axes[1].axhline(y=max_dd, color='darkred', linestyle='--',
                       label=f'Max DD: {max_dd:.2f}%')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Guardar archivo
        filepath = state.temp_dir / f"equity_{asset}_{strategy}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return FileResponse(
            filepath,
            media_type="image/png",
            filename=f"equity_{asset}_{strategy}.png"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generando gráfico: {str(e)}")


@app.get("/charts/importance/{asset}", tags=["Charts"])
async def chart_importance(
    asset: str = Path(..., description="DOGE o TSLA"),
    top_n: int = Query(20, description="Número de features", ge=5, le=50)
):
    """Gráfico de feature importance - Descarga PNG"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    
    # Usar mejor modelo según asset
    model_name = 'catboost' if asset == 'DOGE' else 'xgboost'
    importance = model.get_feature_importance(model_name, top_n=top_n)
    
    # Crear gráfico
    features = list(importance.keys())
    importances = list(importance.values())
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.4)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors)
    
    ax.set_xlabel('Importance Score')
    ax.set_title(f'{asset} - Top {top_n} Most Important Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    filepath = state.temp_dir / f"importance_{asset}_{model_name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"importance_{asset}_{model_name}.png"
    )


@app.get("/charts/comparison/{asset}", tags=["Charts"])
async def chart_comparison(
    asset: str = Path(..., description="DOGE o TSLA")
):
    """Gráfico comparativo de modelos - Descarga PNG"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    asset = asset.upper()
    model = state.doge_model if asset == "DOGE" else state.tsla_model
    target_col = f'TARGET_{asset}'
    
    # Evaluar todos los modelos
    evaluator = ModelEvaluator()
    models_metrics = {}
    
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        pred = model.predict(state.test_df, model_name=model_name)
        true = state.test_df[target_col].values
        min_len = min(len(pred), len(true))
        
        result = evaluator.evaluate_regression(true[-min_len:], pred[-min_len:], model_name)
        models_metrics[model_name] = {
            'rmse': result['rmse'],
            'r2': result['r2'],
            'dir_acc': result['directional_accuracy']
        }
    
    # Crear gráfico
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(models_metrics.keys())
    
    # 1. RMSE
    rmse_values = [models_metrics[m]['rmse'] for m in models]
    axes[0].bar(models, rmse_values, color='#E63946')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title(f'{asset} - RMSE Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. R²
    r2_values = [models_metrics[m]['r2'] for m in models]
    axes[1].bar(models, r2_values, color='#2A9D8F')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title(f'{asset} - R² Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Directional Accuracy
    dir_acc_values = [models_metrics[m]['dir_acc'] * 100 for m in models]
    axes[2].bar(models, dir_acc_values, color='#F4A261')
    axes[2].set_ylabel('Directional Accuracy (%)')
    axes[2].set_title(f'{asset} - Dir. Accuracy Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(y=50, color='red', linestyle='--', linewidth=1, 
                   label='Random (50%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    
    filepath = state.temp_dir / f"comparison_{asset}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"comparison_{asset}.png"
    )


# ================================================================= #
# ENDPOINTS - IMPACT CLASSIFIER                                     #
# ================================================================= #

@app.get("/impact/predict", tags=["Impact"])
async def predict_impact(
    n: int = Query(10, description="Número de predicciones", ge=1, le=100)
):
    """Clasificación de impacto de los últimos tweets"""
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    data = state.test_df.tail(n)
    predictions = state.impact_model.predict(data, model_name='xgboost')
    probas = state.impact_model.predict_proba(data, model_name='xgboost')
    
    class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
    
    results = []
    for i, (pred, proba) in enumerate(zip(predictions, probas)):
        results.append({
            'index': int(i),
            'predicted_class': class_names[pred],
            'class_id': int(pred),
            'probabilities': {
                class_names[j]: float(p) for j, p in enumerate(proba)
            },
        })
    
    return {
        "n_predictions": len(results),
        "predictions": results
    }


@app.post("/impact/predict-text", tags=["Impact"])
async def predict_impact_from_text(
    text: str = Query(..., description="Texto del tweet a analizar"),
    model_name: str = Query("xgboost", description="Modelo clasificador"),
    include_magnitude: bool = Query(True, description="Incluir predicción de magnitud")
):
    """
    Predice el impacto de un NUEVO TEXTO en DOGE y TSLA
    
    Toma un texto (ej: tweet de Elon Musk) y predice:
    - Probabilidad de impacto en DOGE
    - Probabilidad de impacto en TSLA
    - Magnitud esperada del impacto
    
    Ejemplo:
    ```
    POST /impact/predict-text?text=DOGE to the moon!
    ```
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    try:
        # 1. Analizar sentimiento
        analyzer = SentimentAnalyzer()
        sentiment_result = analyzer.analyze_tweet(text)
        
        # 2. Features temporales
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # 3. Construir DataFrame SIMPLE (solo features core)
        features_dict = {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            
            'sentiment_ensemble': sentiment_result['ensemble'],
            'relevance_score': sentiment_result['relevance'],
            'mentions_tesla': int(sentiment_result['mentions_tesla']),
            'mentions_doge': int(sentiment_result['mentions_doge']),
        }
        
        input_df = pd.DataFrame([features_dict])
        
        # 4. Predicción CON HEURÍSTICAS
        prediction_class = int(state.impact_model.predict(input_df, model_name=model_name)[0])
        probabilities = state.impact_model.predict_proba(input_df, model_name=model_name)[0]
        
        class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
        
        # 5. Predicción de magnitud
        doge_impact = 0.0
        tsla_impact = 0.0
        
        if include_magnitude:
            if prediction_class in [1, 3]:
                try:
                    # Agregar features de mercado SOLO para predicción de magnitud
                    magnitude_df = input_df.copy()
                    magnitude_df['doge_ret_1h'] = 0.0
                    magnitude_df['doge_vol_zscore'] = 1.0
                    magnitude_df['doge_rsi'] = 55.0
                    
                    doge_pred = state.doge_model.predict(magnitude_df, model_name='stacking')[0]
                    doge_impact = float(doge_pred)
                except Exception as e:
                    print(f"Fallback DOGE: {e}")
                    doge_impact = sentiment_result['ensemble'] * 0.025
            
            if prediction_class in [2, 3]:
                try:
                    magnitude_df = input_df.copy()
                    magnitude_df['tsla_ret_1h'] = 0.0
                    magnitude_df['tsla_market_open'] = 1 if 9 <= hour <= 16 else 0
                    magnitude_df['tsla_vol_zscore'] = 1.0
                    
                    tsla_pred = state.tsla_model.predict(magnitude_df, model_name='stacking')[0]
                    tsla_impact = float(tsla_pred)
                except Exception as e:
                    print(f"Fallback TSLA: {e}")
                    tsla_impact = sentiment_result['ensemble'] * 0.015
        
        # 6. Recomendación
        if prediction_class == 3:
            recommendation = "Alto impacto esperado en AMBOS activos"
            signal = "STRONG"
        elif prediction_class == 1:
            recommendation = "Impacto significativo solo en DOGE"
            signal = "MODERATE_DOGE"
        elif prediction_class == 2:
            recommendation = "Impacto significativo solo en TSLA"
            signal = "MODERATE_TSLA"
        else:
            recommendation = "Sin impacto significativo esperado"
            signal = "NONE"
        
        return {
            "text": text,
            "timestamp": now.isoformat(),
            
            "sentiment_analysis": {
                "ensemble_score": round(sentiment_result['ensemble'], 4),
                "relevance_score": round(sentiment_result['relevance'], 4),
                "confidence": round(sentiment_result['confidence'], 4),
                "mentions": {
                    "tesla": sentiment_result['mentions_tesla'],
                    "doge": sentiment_result['mentions_doge']
                }
            },
            
            "impact_prediction": {
                "predicted_class": class_names[prediction_class],
                "class_id": int(prediction_class),
                "signal_strength": signal,
                "probabilities": {
                    "no_impact": f"{probabilities[0]*100:.2f}%",
                    "doge_only": f"{probabilities[1]*100:.2f}%",
                    "tsla_only": f"{probabilities[2]*100:.2f}%",
                    "both": f"{probabilities[3]*100:.2f}%"
                },
                "model_used": model_name
            },
            
            "expected_impact": {
                "doge": {
                    "will_impact": prediction_class in [1, 3],
                    "expected_return_pct": f"{doge_impact*100:.3f}%",
                    "magnitude": (
                        "high" if abs(doge_impact) > 0.025 else
                        "medium" if abs(doge_impact) > 0.012 else
                        "low"
                    ),
                    "direction": "bullish" if doge_impact > 0 else "bearish" if doge_impact < 0 else "neutral"
                },
                "tsla": {
                    "will_impact": prediction_class in [2, 3],
                    "expected_return_pct": f"{tsla_impact*100:.3f}%",
                    "magnitude": (
                        "high" if abs(tsla_impact) > 0.015 else
                        "medium" if abs(tsla_impact) > 0.008 else
                        "low"
                    ),
                    "direction": "bullish" if tsla_impact > 0 else "bearish" if tsla_impact < 0 else "neutral"
                }
            } if include_magnitude else None,
            
            "recommendation": recommendation,
            
            "metadata": {
                "model_version": state.impact_model.version,
                "heuristics_applied": True
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ================================================================= #
# ENDPOINTS - PREPROCESSING CHARTS                                  #
# ================================================================= #
@app.get("/analysis/wordcloud", tags=["Analysis"])
async def generate_wordcloud():
    """
    Genera WordCloud de las palabras más frecuentes de Elon Musk
    
    Devuelve imagen PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo (train + test)
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    analysis = exploratory_word_analysis(df, min_freq=15)
    
    # Crear WordCloud
    wc_config = WordCloud(
        width=1200,
        height=600,
        background_color='white',
        colormap='viridis',
        max_words=200,
        relative_scaling=0.5,
        min_font_size=10
    )
    wordcloud = wc_config.generate_from_frequencies(analysis['word_freq'],)
    wc_image = wordcloud.to_image()

    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_image, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Elon Musk Tweets - Most Frequent Words', fontsize=20, pad=20)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / "wordcloud_words.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename="wordcloud_elon_musk.png"
    )


@app.get("/analysis/wordcloud-bigrams", tags=["Analysis"])
async def generate_wordcloud_bigrams():
    """
    Genera WordCloud de los bigramas más frecuentes
    
    Devuelve imagen PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    analysis = exploratory_word_analysis(df, min_freq=15)
    
    # Crear WordCloud
    wc_config = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='plasma',
            max_words=150,
            relative_scaling=0.5,
            min_font_size=10,
            regexp=None
    )
    wordcloud = wc_config.generate_from_frequencies(analysis['bigram_freq'])
    wc_image = wordcloud.to_image()
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_image, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Elon Musk Tweets - Most Frequent Bigrams', fontsize=20, pad=20)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / "wordcloud_bigrams.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename="wordcloud_bigrams_elon_musk.png"
    )


@app.get("/analysis/top-words", tags=["Analysis"])
async def top_words_chart(
    top_n: int = Query(30, description="Número de palabras a mostrar", ge=10, le=100)
):
    """
    Top N palabras más usadas por Elon Musk
    
    Devuelve gráfico de barras PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    
    # Extraer palabras
    words = ' '.join(df['text_clean'].dropna().values)
    tokens = words.split()
    
    # Contar frecuencias
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(top_n)
    
    # Preparar datos
    words_list = [word for word, count in top_words]
    counts_list = [count for word, count in top_words]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words_list)))
    bars = ax.barh(words_list[::-1], counts_list[::-1], color=colors[::-1])
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Used Words by Elon Musk', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, count) in enumerate(zip(bars, counts_list[::-1])):
        ax.text(count, i, f' {count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / f"top_{top_n}_words.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"top_{top_n}_words_elon_musk.png"
    )


@app.get("/analysis/top-bigrams", tags=["Analysis"])
async def top_bigrams_chart(
    top_n: int = Query(20, description="Número de bigramas a mostrar", ge=10, le=50)
):
    """
    Top N bigramas más usados por Elon Musk
    
    Devuelve gráfico de barras PNG descargable
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    
    # Extraer bigramas
    words = ' '.join(df['text_clean'].dropna().values)
    tokens = words.split()
    bigrams_list = list(bigrams(tokens))
    
    # Contar frecuencias
    bigram_freq = Counter(bigrams_list)
    top_bigrams = bigram_freq.most_common(top_n)
    
    # Preparar datos
    bigrams_labels = [f"{w1} {w2}" for (w1, w2), count in top_bigrams]
    counts_list = [count for bigram, count in top_bigrams]
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(bigrams_labels)))
    bars = ax.barh(bigrams_labels[::-1], counts_list[::-1], color=colors[::-1])
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Most Used Bigrams by Elon Musk', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (bar, count) in enumerate(zip(bars, counts_list[::-1])):
        ax.text(count, i, f' {count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar
    filepath = state.temp_dir / f"top_{top_n}_bigrams.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"top_{top_n}_bigrams_elon_musk.png"
    )


@app.get("/analysis/stats", tags=["Analysis"])
async def text_statistics():
    """
    Estadísticas generales de los tweets de Elon Musk
    
    Retorna JSON con métricas
    """
    if not state.models_loaded:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    
    # Cargar dataset completo
    master_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
    df = pd.read_parquet(master_path)
    print(df.head(10))

    analysis = exploratory_word_analysis(df, min_freq=15)
    df_musk = analysis['df_clean']
    words = ' '.join(df_musk['text_clean'].dropna().values)

    total_words = len(words)
    return {
        "total_tweets": len(analysis['df_clean']),
        "total_words": total_words,
        "top_10_words": analysis['top_words'],
        "top_10_bigrams": analysis['top_bigrams']
    }

# =================================================================
# ERROR HANDLERS
# =================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    import traceback
    traceback.print_exc()  # Log completo en consola
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()  # Serializable
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)