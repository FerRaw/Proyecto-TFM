"""
Pydantic Schemas para la API

Define los modelos de request/response
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


# =================================================================
# REQUEST SCHEMAS
# =================================================================

class PredictionRequest(BaseModel):
    """Request para hacer predicciones"""
    asset: str = Field(..., description="DOGE o TSLA")
    model_name: str = Field(default="stacking", description="xgboost, lightgbm, catboost, stacking")
    
    class Config:
        schema_extra = {
            "example": {
                "asset": "DOGE",
                "model_name": "stacking"
            }
        }


class BacktestingRequest(BaseModel):
    """Request para backtesting personalizado"""
    asset: str = Field(..., description="DOGE o TSLA")
    threshold: float = Field(default=0.0025, description="Umbral mínimo para operar")
    max_position_size: float = Field(default=0.75, description="Tamaño máximo de posición (0-1)")
    transaction_cost: float = Field(default=0.001, description="Costo de transacción")
    initial_capital: float = Field(default=10000, description="Capital inicial")
    
    class Config:
        schema_extra = {
            "example": {
                "asset": "DOGE",
                "threshold": 0.0025,
                "max_position_size": 0.75,
                "transaction_cost": 0.001,
                "initial_capital": 10000
            }
        }


# =================================================================
# RESPONSE SCHEMAS
# =================================================================

class PredictionResponse(BaseModel):
    """Response con predicción"""
    asset: str
    model_name: str
    prediction: float
    timestamp: datetime
    confidence: Optional[float] = None


class ModelMetrics(BaseModel):
    """Métricas de un modelo"""
    model_name: str
    rmse: float
    mae: float
    r2: float
    directional_accuracy: float
    correlation: float


class BacktestingMetrics(BaseModel):
    """Métricas de backtesting"""
    initial_capital: float
    final_capital: float
    total_return_pct: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float


class BacktestingResponse(BaseModel):
    """Response de backtesting"""
    asset: str
    strategy: str
    metrics: BacktestingMetrics


class ChartResponse(BaseModel):
    """Response con gráfico en base64"""
    chart_type: str
    asset: Optional[str] = None
    image_base64: str


class HealthResponse(BaseModel):
    """Response de health check"""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Response de error"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime


# =================================================================
# INFO SCHEMAS
# =================================================================

class EndpointInfo(BaseModel):
    """Información de un endpoint"""
    path: str
    method: str
    description: str
    parameters: Optional[List[str]] = None
    example: Optional[Dict] = None


class APIHelp(BaseModel):
    """Documentación completa de la API"""
    version: str
    description: str
    endpoints: List[EndpointInfo]