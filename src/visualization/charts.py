"""
Módulo de Generación de Gráficos para FastAPI

Crea visualizaciones interactivas de:
- Predicciones vs Reales
- Equity Curve de Backtesting
- Feature Importance
- Confusion Matrix del Impact Classifier
- Distribución de errores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import io
import base64


class ChartGenerator:
    """Generador de gráficos para la API"""
    
    def __init__(self):
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def _fig_to_base64(self, fig) -> str:
        """Convierte figura matplotlib a base64 para enviar en API"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        asset_name: str,
        model_name: str = "Stacking"
    ) -> str:
        """
        Gráfico de predicciones vs valores reales
        
        Returns:
            Base64 encoded image
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                        [y_true.min(), y_true.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        axes[0, 0].set_title(f'{asset_name} - Predictions vs Actual ({model_name})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series
        indices = np.arange(len(y_true))
        axes[0, 1].plot(indices, y_true, label='Actual', alpha=0.7, linewidth=1)
        axes[0, 1].plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=1)
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('Returns')
        axes[0, 1].set_title(f'{asset_name} - Time Series Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        errors = y_pred - y_true
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{asset_name} - Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Directional accuracy over time
        window = 50
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        correct = (direction_true == direction_pred).astype(int)
        rolling_acc = pd.Series(correct).rolling(window).mean() * 100
        
        axes[1, 1].plot(rolling_acc, linewidth=2)
        axes[1, 1].axhline(y=50, color='r', linestyle='--', label='Random (50%)')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Directional Accuracy (%)')
        axes[1, 1].set_title(f'{asset_name} - Rolling Directional Accuracy (window={window})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 100])
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_equity_curve(
        self,
        equity_curve: List[float],
        asset_name: str,
        strategy_name: str = "Stacking"
    ) -> str:
        """
        Gráfico de equity curve del backtesting
        
        Returns:
            Base64 encoded image
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        equity = np.array(equity_curve)
        
        # 1. Equity curve
        axes[0].plot(equity, linewidth=2, color='#2E86AB')
        axes[0].fill_between(range(len(equity)), equity, alpha=0.3, color='#2E86AB')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title(f'{asset_name} - Equity Curve ({strategy_name})')
        axes[0].grid(True, alpha=0.3)
        
        # Añadir estadísticas
        final_value = equity[-1]
        max_value = equity.max()
        initial_value = equity[0]
        total_return = (final_value - initial_value) / initial_value * 100
        
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
        axes[1].set_title(f'{asset_name} - Drawdown Analysis')
        axes[1].grid(True, alpha=0.3)
        
        max_dd = drawdown.min()
        axes[1].axhline(y=max_dd, color='darkred', linestyle='--',
                       label=f'Max DD: {max_dd:.2f}%')
        axes[1].legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        asset_name: str,
        top_n: int = 20
    ) -> str:
        """
        Gráfico de feature importance
        
        Returns:
            Base64 encoded image
        """
        # Ordenar y tomar top N
        sorted_features = dict(sorted(feature_importance.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:top_n])
        
        features = list(sorted_features.keys())
        importances = list(sorted_features.values())
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.4)))
        
        # Crear barras horizontales
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax.barh(features, importances, color=colors)
        
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{asset_name} - Top {top_n} Most Important Features')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir valores
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        model_name: str = "XGBoost"
    ) -> str:
        """
        Gráfico de confusion matrix
        
        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Crear heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - Impact Classifier ({model_name})', 
                    fontsize=14)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        asset_name: str
    ) -> str:
        """
        Gráfico de distribución de retornos
        
        Returns:
            Base64 encoded image
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Histogram
        axes[0].hist(returns, bins=50, edgecolor='black', alpha=0.7, 
                    density=True, color='#A23B72')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Returns')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{asset_name} - Returns Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Añadir estadísticas
        mean_ret = returns.mean()
        std_ret = returns.std()
        axes[0].axvline(x=mean_ret, color='green', linestyle='--', 
                       label=f'Mean: {mean_ret:.4f}')
        axes[0].legend()
        
        # 2. Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title(f'{asset_name} - Q-Q Plot (Normality Test)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_performance_comparison(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        asset_name: str
    ) -> str:
        """
        Gráfico comparativo de performance de modelos
        
        Args:
            models_metrics: {model_name: {'rmse': x, 'r2': y, 'dir_acc': z}}
        
        Returns:
            Base64 encoded image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(models_metrics.keys())
        
        # 1. RMSE
        rmse_values = [models_metrics[m]['rmse'] for m in models]
        axes[0].bar(models, rmse_values, color='#E63946')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title(f'{asset_name} - RMSE Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. R²
        r2_values = [models_metrics[m]['r2'] for m in models]
        axes[1].bar(models, r2_values, color='#2A9D8F')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title(f'{asset_name} - R² Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. Directional Accuracy
        dir_acc_values = [models_metrics[m]['dir_acc'] * 100 for m in models]
        axes[2].bar(models, dir_acc_values, color='#F4A261')
        axes[2].set_ylabel('Directional Accuracy (%)')
        axes[2].set_title(f'{asset_name} - Dir. Accuracy Comparison')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].axhline(y=50, color='red', linestyle='--', linewidth=1, 
                       label='Random (50%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_ylim([0, 100])
        
        plt.tight_layout()
        return self._fig_to_base64(fig)


# =================================================================
# FUNCIÓN HELPER PARA GENERAR TODOS LOS GRÁFICOS
# =================================================================
def generate_all_charts(
    doge_model,
    tsla_model,
    impact_model,
    test_df: pd.DataFrame,
    doge_results: Dict,
    tsla_results: Dict
) -> Dict[str, str]:
    """
    Genera todos los gráficos necesarios para la API
    
    Returns:
        Dict con gráficos en base64
    """
    generator = ChartGenerator()
    charts = {}
    
    # 1. DOGE - Predictions vs Actual
    doge_pred = doge_model.predict(test_df, model_name='stacking')
    doge_true = test_df['TARGET_DOGE'].values
    min_len = min(len(doge_pred), len(doge_true))
    charts['doge_predictions'] = generator.plot_predictions_vs_actual(
        doge_true[-min_len:], doge_pred[-min_len:], 'DOGE', 'Stacking'
    )
    
    # 2. TSLA - Predictions vs Actual
    tsla_pred = tsla_model.predict(test_df, model_name='stacking')
    tsla_true = test_df['TARGET_TSLA'].values
    min_len = min(len(tsla_pred), len(tsla_true))
    charts['tsla_predictions'] = generator.plot_predictions_vs_actual(
        tsla_true[-min_len:], tsla_pred[-min_len:], 'TSLA', 'Stacking'
    )
    
    # 3. DOGE - Equity Curve (configuración moderada)
    charts['doge_equity'] = generator.plot_equity_curve(
        doge_results['moderate']['equity_curve'],
        'DOGE',
        'Moderate Strategy'
    )
    
    # 4. TSLA - Equity Curve (configuración moderada)
    charts['tsla_equity'] = generator.plot_equity_curve(
        tsla_results['moderate']['equity_curve'],
        'TSLA',
        'Moderate Strategy'
    )
    
    # 5. DOGE - Feature Importance
    doge_importance = doge_model.get_feature_importance('catboost', top_n=20)
    charts['doge_importance'] = generator.plot_feature_importance(
        doge_importance, 'DOGE', top_n=20
    )
    
    # 6. TSLA - Feature Importance
    tsla_importance = tsla_model.get_feature_importance('xgboost', top_n=20)
    charts['tsla_importance'] = generator.plot_feature_importance(
        tsla_importance, 'TSLA', top_n=20
    )
    
    # 7. Impact Classifier - Confusion Matrix
    metrics = impact_model.get_classification_metrics(test_df, 'xgboost')
    charts['impact_confusion'] = generator.plot_confusion_matrix(
        metrics['confusion_matrix'],
        metrics['class_names'],
        'XGBoost'
    )
    
    # 8. Performance Comparison - DOGE
    from src.models.evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    
    doge_models_metrics = {}
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        pred = doge_model.predict(test_df, model_name=model_name)
        true = test_df['TARGET_DOGE'].values
        min_len = min(len(pred), len(true))
        
        result = evaluator.evaluate_regression(true[-min_len:], pred[-min_len:], f'DOGE_{model_name}')
        doge_models_metrics[model_name] = {
            'rmse': result['rmse'],
            'r2': result['r2'],
            'dir_acc': result['directional_accuracy']
        }
    
    charts['doge_comparison'] = generator.plot_performance_comparison(
        doge_models_metrics, 'DOGE'
    )
    
    # 9. Performance Comparison - TSLA
    tsla_models_metrics = {}
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        pred = tsla_model.predict(test_df, model_name=model_name)
        true = test_df['TARGET_TSLA'].values
        min_len = min(len(pred), len(true))
        
        result = evaluator.evaluate_regression(true[-min_len:], pred[-min_len:], f'TSLA_{model_name}')
        tsla_models_metrics[model_name] = {
            'rmse': result['rmse'],
            'r2': result['r2'],
            'dir_acc': result['directional_accuracy']
        }
    
    charts['tsla_comparison'] = generator.plot_performance_comparison(
        tsla_models_metrics, 'TSLA'
    )
    
    return charts