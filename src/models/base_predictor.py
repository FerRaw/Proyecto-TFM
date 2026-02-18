"""
Clase base abstracta para todos los modelos predictivos
Versión FINAL
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import joblib
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class BasePredictor(ABC):
    """Clase base para modelos de predicción"""
    
    def __init__(self, model_name: str, version: str = "final"):
        self.model_name = model_name
        self.version = version
        self.models = {}
        self.metrics = {}
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.ensemble_weights = {}
        
        self.metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'trained_at': None,
            'training_samples': None,
            'features_count': None,
            'cv_splits': None
        }
    
    @abstractmethod
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepara features y target para el modelo"""
        pass
    
    @abstractmethod
    def train(self, df: pd.DataFrame, **kwargs):
        """Entrena todos los modelos"""
        pass
    
    @abstractmethod
    def predict(
        self, 
        df: pd.DataFrame, 
        model_name: str = 'ensemble'
    ) -> np.ndarray:
        """Realiza predicciones"""
        pass
    
    def save(self, path: Path):
        """Guarda modelo y metadata"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo
        joblib.dump(self, path)
        
        # Guardar metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Modelo guardado: {path}")
        print(f"Metadata guardada: {metadata_path}")
    
    @classmethod
    def load(cls, path: Path):
        """Carga modelo desde archivo"""
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        
        print(f"Cargando modelo desde {path}...")
        model = joblib.load(path)
        print(f"Modelo cargado: {model.model_name}")
        return model
    
    def get_feature_importance(
        self, 
        model_name: str = 'xgboost',
        top_n: int = None
    ) -> Dict[str, float]:
        """Obtiene importancia de features"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            if top_n:
                importance_dict = dict(list(importance_dict.items())[:top_n])
            
            return importance_dict
        else:
            print(f"Modelo {model_name} no tiene feature_importances_")
            return {}
    
    def print_feature_importance(self, model_name: str = 'xgboost', top_n: int = 20):
        """Imprime feature importance con barras visuales"""
        importances = self.get_feature_importance(model_name, top_n)
        
        if not importances:
            return
        
        print(f"\nFEATURE IMPORTANCE - {model_name.upper()}")
        print("="*70)
        
        for i, (feature, importance) in enumerate(importances.items(), 1):
            bar_length = int(importance * 50 / max(importances.values()))
            bar = "█" * bar_length
            print(f"{i:2d}. {feature:30s} {importance:8.4f} {bar}")
        
        print("="*70)
    
    def update_metadata(self, **kwargs):
        """Actualiza metadata"""
        self.metadata.update(kwargs)
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Resumen del modelo"""
        return {
            'name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'models_available': list(self.models.keys()),
            'n_models': len(self.models),
            'metrics': self.metrics,
            'ensemble_weights': self.ensemble_weights,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'metadata': self.metadata
        }
    
    def print_model_summary(self):
        """Imprime resumen legible"""
        summary = self.get_model_summary()
        
        print(f"\nRESUMEN DEL MODELO")
        print("="*70)
        print(f"Nombre: {summary['name']}")
        print(f"Versión: {summary['version']}")
        print(f"Entrenado: {'Sí' if summary['is_trained'] else 'No'}")
        print(f"Número de modelos: {summary['n_models']}")
        print(f"Modelos disponibles: {', '.join(summary['models_available'])}")
        print(f"Número de features: {summary['n_features']}")
        
        if summary['ensemble_weights']:
            print(f"\nPesos del Ensemble:")
            for name, weight in summary['ensemble_weights'].items():
                print(f"{name}: {weight:.4f}")
        
        if summary['metrics']:
            print(f"\nMétricas de Training (CV):")
            for model_name, metrics in summary['metrics'].items():
                if isinstance(metrics, dict) and 'cv_rmse_mean' in metrics:
                    print(f"{model_name}: RMSE = {metrics['cv_rmse_mean']:.6f} "
                          f"(±{metrics.get('cv_rmse_std', 0):.6f})")
        
        print("="*70)
    
    def _create_ensemble_weights(self, method: str = 'inverse_rmse'):
        """Crea pesos para ensemble"""
        weights = {}
        
        if method == 'equal':
            n_models = len([k for k in self.metrics.keys() if 'cv_rmse_mean' in self.metrics.get(k, {})])
            for name in self.metrics.keys():
                if 'cv_rmse_mean' in self.metrics.get(name, {}):
                    weights[name] = 1.0 / n_models
        
        elif method == 'inverse_rmse':
            total_inv_rmse = 0
            
            for name, metrics in self.metrics.items():
                if isinstance(metrics, dict) and 'cv_rmse_mean' in metrics:
                    inv_rmse = 1 / metrics['cv_rmse_mean']
                    weights[name] = inv_rmse
                    total_inv_rmse += inv_rmse
            
            if total_inv_rmse > 0:
                weights = {k: v / total_inv_rmse for k, v in weights.items()}
        
        self.ensemble_weights = weights
        
        if weights:
            print(f"\nEnsemble Weights ({method}):")
            for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"{name:20s}: {weight:.4f}")
    
    def _print_cv_results(self, model_name: str, scores: List[float]):
        """Imprime resultados de CV"""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        print(f"\n{model_name.upper()}:")
        print(f"CV RMSE: {mean_score:.6f} ({std_score:.6f})")
        print(f"Min: {min_score:.6f} | Max: {max_score:.6f}")
    
    def compare_cv_metrics(self) -> pd.DataFrame:
        """Compara métricas de CV"""
        comparison = []
        
        for model_name, metrics in self.metrics.items():
            if isinstance(metrics, dict):
                row = {'model': model_name}
                
                for key in ['cv_rmse_mean', 'cv_rmse_std', 'cv_rmse_min', 'cv_rmse_max',
                           'cv_mae_mean', 'accuracy', 'precision', 'recall', 'f1']:
                    if key in metrics:
                        row[key] = metrics[key]
                
                comparison.append(row)
        
        if comparison:
            df = pd.DataFrame(comparison)
            if 'cv_rmse_mean' in df.columns:
                df = df.sort_values('cv_rmse_mean')
            return df
        else:
            return pd.DataFrame()
    
    def get_best_model_name(self, metric: str = 'cv_rmse_mean', minimize: bool = True) -> str:
        """Obtiene el mejor modelo"""
        valid_models = {}
        
        for name, metrics in self.metrics.items():
            if isinstance(metrics, dict) and metric in metrics:
                valid_models[name] = metrics[metric]
        
        if not valid_models:
            return None
        
        if minimize:
            best_name = min(valid_models.items(), key=lambda x: x[1])[0]
        else:
            best_name = max(valid_models.items(), key=lambda x: x[1])[0]
        
        return best_name
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"name='{self.model_name}', "
                f"version='{self.version}', "
                f"trained={self.is_trained}, "
                f"models={len(self.models)})")
                