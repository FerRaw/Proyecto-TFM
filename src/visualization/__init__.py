"""
Módulo de Visualización

Generación de gráficos para FastAPI
"""

from .charts import ChartGenerator, generate_all_charts

__all__ = ['ChartGenerator', 'generate_all_charts']