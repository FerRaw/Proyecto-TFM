"""
Predictores Mejorados FINALES
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import keras
from keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.ensemble import StackingRegressor, StackingRegressor

# Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Config
from config.settings import settings
from src.models.base_predictor import BasePredictor


# ============================================================================= #
# FUNCIÓN DE PÉRDIDA                                                            #
# ============================================================================= #
@keras.saving.register_keras_serializable(package="TFM_Models")
def directional_mse_loss(y_true, y_pred):
    """
    Pérdida híbrida: 60% MSE + 40% Directional Accuracy
    
    Justificación:
    - En trading, predecir la DIRECCIÓN es más importante que la magnitud exacta
    - Penaliza más fuertemente predicciones con dirección incorrecta
    """
    # MSE base
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Directional penalty
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    directional_error = tf.reduce_mean(tf.square(direction_true - direction_pred))
    
    # Combinar: 60% magnitud, 40% dirección
    return 0.6 * mse + 0.4 * directional_error


# ============================================================================= #
# CAPA DE ATENCIÓN                                                              #
# ============================================================================= #

@keras.saving.register_keras_serializable(package="TFM_Models")
class AttentionLayer(layers.Layer):
    """
    Mecanismo de atención temporal
    
    Justificación:
    - Permite al modelo "enfocarse" en timesteps relevantes
    - Mejora interpretabilidad: podemos ver qué lags importan más
    - Superior a LSTM vanilla para series con dependencias irregulares
    """
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_W'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='attention_b'
        )
        self.u = self.add_weight(
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True,
            name='attention_u'
        )
        super().build(input_shape)
    
    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        weighted = x * ait
        output = tf.reduce_sum(weighted, axis=1)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ============================================================================= #
# ADVANCED DOGE PREDICTOR                                                       #
# ============================================================================= #

class ImprovedDOGEPredictor(BasePredictor):
    """
    Predictor DOGE con Modelos Avanzados
    
    """
    
    def __init__(self, version: str = "v3_advanced"):
        super().__init__("doge_predictor_advanced", version)
        self.scaler = StandardScaler()
        self.sequence_length = 10  # Para modelos de secuencias
        self.use_deep_learning = True  # Flag para entrenar DL models
        
        # Para normalización de targets (ayuda a DL models)
        self.target_scaler = StandardScaler()
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepara features usando TODAS las disponibles
        """
        # Core features
        feature_cols = [
            'doge_ret_1h', 'doge_vol_zscore', 'doge_buy_pressure', 'doge_rsi',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'sentiment_ensemble_lag2', 'sentiment_ensemble_lag3',
            'relevance_score_lag1', 'relevance_score_lag2', 'relevance_score_lag3',
        ]
        
        # Advanced features (wavelets, autocorr, cross-correlations)
        advanced_features = [
            'doge_wavelet_trend', 'doge_wavelet_detail_1', 'doge_wavelet_detail_2',
            'doge_autocorr_lag_1', 'doge_returns_lag_1',
            'doge_autocorr_lag_6', 'doge_returns_lag_6',
            'doge_autocorr_lag_12', 'doge_returns_lag_12',
            'doge_autocorr_lag_24', 'doge_returns_lag_24',
            'doge_tsla_corr_6h', 'doge_tsla_corr_12h', 'doge_tsla_corr_24h',
            'vol_ratio_doge_tsla', 'momentum_divergence',
            'doge_tsla_beta_12h', 'doge_tsla_beta_24h',
            'sentiment_x_vol_doge', 'sentiment_velocity', 'sentiment_acceleration',
            'sentiment_weighted_avg', 'relevance_conditional', 'vol_regime_doge'
        ]
        
        # Combinar todas las features disponibles
        all_features = feature_cols + [f for f in advanced_features if f in df.columns]
        all_features = [f for f in all_features if f in df.columns]
        
        if is_train:
            self.feature_names = all_features
            print(f"\nFeatures seleccionadas: {len(self.feature_names)}")
        
        X = df[all_features].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y = df['TARGET_DOGE'].values if 'TARGET_DOGE' in df.columns else None
        
        return X_scaled, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """
        Convierte datos tabulares en secuencias para modelos DL
        
        Justificación:
        - LSTM/TCN necesitan shape (batch, timesteps, features)
        - Usamos ventanas deslizantes de 10 timesteps
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def create_bilstm_attention_model(self, input_shape: tuple) -> Model:
        """
        Bi-LSTM con Attention
        
        Justificación:
        - Bi-LSTM: Captura contexto pasado Y futuro (en ventana)
        - Attention: Enfoca en timesteps relevantes (ej: momento del tweet)
        - Dropout: Regularización para evitar overfitting
        """
        inputs = keras.Input(shape=input_shape)
        
        # Bi-LSTM
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(inputs)
        
        x = layers.Bidirectional(
            layers.LSTM(32, return_sequences=True, dropout=0.2)
        )(x)
        
        # Attention
        x = AttentionLayer(32)(x)
        
        # Dense
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention')
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=directional_mse_loss,
            metrics=['mae']
        )
        
        return model
    
    def create_tcn_model(self, input_shape: tuple) -> Model:
        """
        Temporal Convolutional Network
        
        Justificación:
        - Más rápido que LSTM
        - Campo receptivo exponencial con dilations
        - Paralelizable (vs LSTM secuencial)
        """
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Stacked dilated convolutions
        for dilation_rate in [1, 2, 4, 8]:
            residual = x
            
            # Causal conv
            x = layers.Conv1D(
                filters=64,
                kernel_size=3,
                dilation_rate=dilation_rate,
                padding='causal',
                activation='relu'
            )(x)
            x = layers.Dropout(0.2)(x)
            
            # Residual connection
            if residual.shape[-1] != 64:
                residual = layers.Conv1D(64, 1, padding='same')(residual)
            
            x = layers.Add()([x, residual])
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='TCN')
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """
        Entrena TODOS los modelos con justificación académica
        """
        print("\n" + "="*70)
        print("ENTRENAMIENTO AVANZADO - DOGE PREDICTOR")
        print("="*70)
        print("\nModelos a entrenar:")
        print("1. XGBoost (Boosting clásico)")
        print("2. LightGBM (Boosting optimizado)")
        print("3. CatBoost (Boosting con categorical handling)")
        print("4. Bi-LSTM + Attention (Deep Learning temporal)")
        print("5. TCN (Convolucional temporal)")
        print("6. Stacking Ensemble (Meta-learner)")
        print("="*70)
        
        X, y = self.prepare_features(df, is_train=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # =================================================================
        # 1. XGBoost
        # =================================================================
        print("\n[1/6] Entrenando XGBoost...")
        xgb_scores = []
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=settings.RANDOM_SEED,
            tree_method='hist',
            verbosity=0
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['xgboost'].fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False
            )
            pred = self.models['xgboost'].predict(X[val_idx])
            xgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.models['xgboost'].fit(X, y)
        self.metrics['xgboost'] = {
            'cv_rmse_mean': np.mean(xgb_scores),
            'cv_rmse_std': np.std(xgb_scores)
        }
        print(f"XGBoost: RMSE = {np.mean(xgb_scores):.6f} (±{np.std(xgb_scores):.6f})")
        
        # =================================================================
        # 2. LightGBM
        # =================================================================
        print("\n[2/6] Entrenando LightGBM...")
        lgb_scores = []
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=settings.RANDOM_SEED,
            verbose=-1
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['lightgbm'].fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            pred = self.models['lightgbm'].predict(X[val_idx])
            lgb_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.models['lightgbm'].fit(X, y)
        self.metrics['lightgbm'] = {
            'cv_rmse_mean': np.mean(lgb_scores),
            'cv_rmse_std': np.std(lgb_scores)
        }
        print(f"LightGBM: RMSE = {np.mean(lgb_scores):.6f} (±{np.std(lgb_scores):.6f})")
        
        # =================================================================
        # 3. CatBoost
        # =================================================================
        print("\n[3/6] Entrenando CatBoost...")
        cat_scores = []
        
        self.models['catboost'] = cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=settings.RANDOM_SEED,
            verbose=False
        )
        
        for train_idx, val_idx in tscv.split(X):
            self.models['catboost'].fit(
                X[train_idx], y[train_idx],
                eval_set=(X[val_idx], y[val_idx]),
                verbose=False
            )
            pred = self.models['catboost'].predict(X[val_idx])
            cat_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        self.models['catboost'].fit(X, y)
        self.metrics['catboost'] = {
            'cv_rmse_mean': np.mean(cat_scores),
            'cv_rmse_std': np.std(cat_scores)
        }
        print(f"CatBoost: RMSE = {np.mean(cat_scores):.6f} (±{np.std(cat_scores):.6f})")
        
        # =================================================================
        # 4. Bi-LSTM + Attention
        # =================================================================
        if self.use_deep_learning:
            print("\n[4/6] Entrenando Bi-LSTM + Attention...")
            
            # Crear secuencias
            X_seq, y_seq = self.create_sequences(X, y)
            
            # Normalizar targets
            y_seq_scaled = self.target_scaler.fit_transform(y_seq.reshape(-1, 1)).ravel()
            
            # TIME SERIES CV (como boosting)
            tscv_seq = TimeSeriesSplit(n_splits=n_splits)
            bilstm_scores = []
            
            input_shape = (self.sequence_length, X.shape[1])
            
            for fold, (train_idx, val_idx) in enumerate(tscv_seq.split(X_seq), 1):
                print(f"Fold {fold}/{n_splits}...", end="")
                
                # Crear modelo nuevo para cada fold
                bilstm_model = self.create_bilstm_attention_model(input_shape)
                
                X_train_seq = X_seq[train_idx]
                X_val_seq = X_seq[val_idx]
                y_train_seq = y_seq_scaled[train_idx]
                y_val_seq = y_seq_scaled[val_idx]
                
                # Entrenar
                history = bilstm_model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=50,
                    batch_size=32,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
                        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0)
                    ],
                    verbose=0
                )
                
                # Evaluar
                y_pred_scaled = bilstm_model.predict(X_val_seq, verbose=0)
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled).ravel()
                y_true_fold = y_seq[val_idx]
                
                rmse = np.sqrt(mean_squared_error(y_true_fold, y_pred))
                bilstm_scores.append(rmse)
                
                print(f"RMSE = {rmse:.6f}")
            
            # Entrenar modelo final en TODO el dataset
            print("Entrenando modelo final...")
            bilstm_model_final = self.create_bilstm_attention_model(input_shape)
            
            bilstm_model_final.fit(
                X_seq, y_seq_scaled,
                epochs=50,
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)
                ],
                verbose=0
            )
            
            self.models['bilstm_attention'] = bilstm_model_final
            self.metrics['bilstm_attention'] = {
                'cv_rmse_mean': np.mean(bilstm_scores),
                'cv_rmse_std': np.std(bilstm_scores),
                'cv_rmse_min': np.min(bilstm_scores),
                'cv_rmse_max': np.max(bilstm_scores)
            }
            
            print(f"Bi-LSTM + Attention: RMSE = {np.mean(bilstm_scores):.6f} (±{np.std(bilstm_scores):.6f})")
        
        # MISMO PROCESO PARA TCN
        if self.use_deep_learning:
            print("\n[5/6] Entrenando TCN...")
            
            tcn_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv_seq.split(X_seq), 1):
                print(f"Fold {fold}/{n_splits}...", end="")
                
                tcn_model = self.create_tcn_model(input_shape)
                
                X_train_seq = X_seq[train_idx]
                X_val_seq = X_seq[val_idx]
                y_train_seq = y_seq_scaled[train_idx]
                y_val_seq = y_seq_scaled[val_idx]
                
                history = tcn_model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=50,
                    batch_size=32,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
                    ],
                    verbose=0
                )
                
                y_pred_scaled = tcn_model.predict(X_val_seq, verbose=0)
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled).ravel()
                y_true_fold = y_seq[val_idx]
                
                rmse = np.sqrt(mean_squared_error(y_true_fold, y_pred))
                tcn_scores.append(rmse)
                
                print(f"RMSE = {rmse:.6f}")
            
            # Modelo final
            print("Entrenando modelo final...")
            tcn_model_final = self.create_tcn_model(input_shape)
            
            tcn_model_final.fit(
                X_seq, y_seq_scaled,
                epochs=50,
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)
                ],
                verbose=0
            )
            
            self.models['tcn'] = tcn_model_final
            self.metrics['tcn'] = {
                'cv_rmse_mean': np.mean(tcn_scores),
                'cv_rmse_std': np.std(tcn_scores),
                'cv_rmse_min': np.min(tcn_scores),
                'cv_rmse_max': np.max(tcn_scores)
            }
            
            print(f"TCN: RMSE = {np.mean(tcn_scores):.6f} (±{np.std(tcn_scores):.6f})")
        
        # =================================================================
        # 6. Stacking Ensemble
        # =================================================================
        print("\n[6/6] Creando Stacking Ensemble...")
        self._create_stacking_ensemble(X, y, tscv)
        
        self.is_trained = True
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        
        # Comparación final
        self._print_model_comparison()
    
    def _create_stacking_ensemble(self, X, y, tscv):
        
        # Base models MÁS DIVERSOS
        base_models = [
            ('catboost', self.models['catboost']),
            ('lightgbm', self.models['lightgbm']),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42))
        ]
        
        # 2. Meta-learner: ¡Menos es más!
        # A veces un meta-learner muy complejo (300 estimadores) sobreajusta.
        # Una Regresión Ridge suele ser el meta-learner estándar por excelencia.
       
        meta_learner = RidgeCV() 

        stacking = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            passthrough=True, 
            cv=5,
            n_jobs=-1
        )
        
        # Evaluar con CV
        stacking_scores = []
        for train_idx, val_idx in tscv.split(X):
            stacking.fit(X[train_idx], y[train_idx])
            pred = stacking.predict(X[val_idx])
            stacking_scores.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
        
        # Entrenar final
        stacking.fit(X, y)
        self.models['stacking'] = stacking
        
        self.metrics['stacking'] = {
            'cv_rmse_mean': np.mean(stacking_scores),
            'cv_rmse_std': np.std(stacking_scores),
            'cv_rmse_min': np.min(stacking_scores),
            'cv_rmse_max': np.max(stacking_scores)
        }
        
        print(f"Stacking: RMSE = {np.mean(stacking_scores):.6f} (±{np.std(stacking_scores):.6f})")
    
    def _print_model_comparison(self):
        """Tabla comparativa de modelos"""
        print("\n" + "="*70)
        print("COMPARACIÓN DE MODELOS")
        print("="*70)
        
        sorted_models = sorted(
            self.metrics.items(),
            key=lambda x: x[1]['cv_rmse_mean']
        )
        
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            rmse = metrics['cv_rmse_mean']
            std = metrics.get('cv_rmse_std', 0)
            print(f"{rank}. {name:20s}: RMSE = {rmse:.6f} (±{std:.6f})")
        
        print("="*70)
    
    def predict(self, df: pd.DataFrame, model_name: str = 'stacking') -> np.ndarray:
        """Predicción con modelo especificado"""
        if model_name in ['bilstm_attention', 'tcn']:
            # Modelos DL necesitan secuencias
            X, _ = self.prepare_features(df, is_train=False)
            X_seq, _ = self.create_sequences(X)
            
            pred_scaled = self.models[model_name].predict(X_seq, verbose=0)
            pred = self.target_scaler.inverse_transform(pred_scaled).ravel()
            
            # Retornar solo las últimas len(df) predicciones
            return pred[-len(df):]
        else:
            # Modelos boosting/stacking
            X, _ = self.prepare_features(df, is_train=False)
            return self.models[model_name].predict(X)


# =============================================================================
# ADVANCED TSLA PREDICTOR (Hereda de DOGE)
# =============================================================================

class ImprovedTSLAPredictor(ImprovedDOGEPredictor):
    """Predictor TSLA - Misma arquitectura que DOGE pero con features TSLA"""
    
    def __init__(self, version: str = "v3_advanced"):
        BasePredictor.__init__(self, "tsla_predictor_advanced", version)
        self.scaler = StandardScaler()
        self.sequence_length = 10
        self.use_deep_learning = True
        self.target_scaler = StandardScaler()
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Features para TSLA"""
        feature_cols = [
            'tsla_ret_1h', 'tsla_market_open', 'tsla_vol_zscore',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'sentiment_ensemble', 'relevance_score',
            'sentiment_ensemble_lag1', 'relevance_score_lag1'
        ]
        
        # Advanced features
        advanced_features = [
            'tsla_wavelet_trend', 'tsla_wavelet_detail_1', 'tsla_wavelet_detail_2',
            'doge_tsla_corr_6h', 'doge_tsla_corr_12h', 'doge_tsla_corr_24h',
            'vol_ratio_doge_tsla', 'momentum_divergence',
            'doge_tsla_beta_12h', 'doge_tsla_beta_24h',
            'sentiment_x_vol_tsla', 'sentiment_velocity', 'sentiment_acceleration',
            'sentiment_weighted_avg', 'relevance_conditional', 'vol_regime_tsla'
        ]
        
        all_features = feature_cols + [f for f in advanced_features if f in df.columns]
        all_features = [f for f in all_features if f in df.columns]
        
        if is_train:
            self.feature_names = all_features
        
        X = df[all_features].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y = df['TARGET_TSLA'].values if 'TARGET_TSLA' in df.columns else None
        
        return X_scaled, y
    
    def train(self, df: pd.DataFrame, n_splits: int = 5):
        """Entrena TSLA"""
        print("\n" + "="*70)
        print("ENTRENAMIENTO AVANZADO - TSLA PREDICTOR")
        print("="*70)
        
        # Llama al método de la clase padre pero con TARGET_TSLA
        ImprovedDOGEPredictor.train(self, df, n_splits)

# =================================================================
# IMPACT CLASSIFIER
# =================================================================
class ImpactClassifier(BasePredictor):
    """
    Clasificador de impacto de tweets
    Versión DEMO-READY con balanceo agresivo
    """
    
    def __init__(self, version: str = "v2_demo"):
        super().__init__("impact_classifier", version)
        self.impact_threshold = 0.015  # Bajado de 0.02 a 0.015 (1.5%)
        self.scaler = StandardScaler()
        
        # Features CORE (solo las esenciales)
        self.core_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'sentiment_ensemble', 'relevance_score',
            'mentions_tesla', 'mentions_doge',
            'sentiment_velocity', 'sentiment_acceleration', 'sentiment_weighted_avg'
        ]
        
        # Features de mercado (OPCIONALES - no usarlas en demo)
        self.market_features = [
            'doge_ret_1h', 'tsla_ret_1h',
            'vol_regime_doge', 'vol_regime_tsla', 'momentum_divergence'
        ]
        
        # Pesos para post-processing heurístico
        self.sentiment_boost_threshold = 0.3  # Si sentiment > 0.3, boost probabilidad
        self.relevance_boost_threshold = 0.6   # Si relevance > 0.6, boost probabilidad
    
    def create_impact_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels con threshold más bajo"""
        df = df.copy()
        
        doge_impact = abs(df['TARGET_DOGE']) > self.impact_threshold
        tsla_impact = abs(df['TARGET_TSLA']) > self.impact_threshold
        
        df['impact_class'] = 0  # Sin impacto
        df.loc[doge_impact & ~tsla_impact, 'impact_class'] = 1  # Solo DOGE
        df.loc[~doge_impact & tsla_impact, 'impact_class'] = 2  # Solo TSLA
        df.loc[doge_impact & tsla_impact, 'impact_class'] = 3   # Ambos
        
        # Estadísticas
        class_counts = df['impact_class'].value_counts().sort_index()
        print("\nDistribución de clases de impacto:")
        class_names = ['Sin impacto', 'Solo DOGE', 'Solo TSLA', 'Ambos']
        for cls, name in enumerate(class_names):
            count = class_counts.get(cls, 0)
            pct = 100 * count / len(df)
            print(f"   Clase {cls} ({name:12s}): {count:5d} ({pct:5.2f}%)")
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        is_train: bool = True,
        use_market_features: bool = False  # CAMBIO: Default False para demo
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepara features SOLO con lo esencial
        NO usar lags ni features de mercado complejas en demo
        """
        df_proc = df.copy()
        
        # Solo features CORE
        feature_cols = self.core_features.copy()
        
        # Agregar features de mercado SOLO si están disponibles y se solicita
        if use_market_features:
            for feat in self.market_features:
                if feat in df_proc.columns:
                    feature_cols.append(feat)
        
        # Asegurar que existan todas las columnas
        for col in feature_cols:
            if col not in df_proc.columns:
                df_proc[col] = 0.0
        
        X = df_proc[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        if is_train:
            self.feature_names = X.columns.tolist()
            print(f"\nFeatures para entrenamiento ({len(self.feature_names)}):")
            for i, feat in enumerate(self.feature_names, 1):
                print(f"{i:2d}. {feat}")
        else:
            # Reordenar para match con training
            if hasattr(self, 'feature_names'):
                for col in self.feature_names:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[self.feature_names]
        
        # Escalar
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y = df_proc['impact_class'].values if 'impact_class' in df_proc.columns else None
        
        return X_scaled, y
    
    def _apply_aggressive_resampling(self, X: np.ndarray, y: np.ndarray):
        """
        TRUCO DEMO: Oversampling agresivo de clases minoritarias
        """
        from collections import Counter
        from imblearn.over_sampling import SMOTE
        
        print("\nAplicando re-balanceo SMOTE...")
        print(f"Distribución original: {Counter(y)}")
        
        # SMOTE con ratio agresivo
        smote = SMOTE(
            sampling_strategy='not majority',  # Oversample todo excepto clase mayoritaria
            k_neighbors=3,
            random_state=settings.RANDOM_SEED
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"Distribución balanceada: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train(self, df: pd.DataFrame, n_splits: int = 5, use_smote: bool = True):
        """
        Entrena con balanceo agresivo
        """
        print("\n" + "="*70)
        print("ENTRENAMIENTO DEMO-READY - IMPACT CLASSIFIER")
        print("="*70)
        
        df = self.create_impact_labels(df)
        X, y = self.prepare_features(df, is_train=True, use_market_features=False)
        
        # APLICAR SMOTE si está habilitado
        if use_smote:
            X, y = self._apply_aggressive_resampling(X, y)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Configuración de modelos con PESOS DE CLASE
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=8,  # Reducido para evitar overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced_subsample',  # MÁS AGRESIVO
                random_state=settings.RANDOM_SEED,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,  # Aumentado
                max_depth=5,
                min_child_weight=1,  # Reducido
                sample_weight=[1, 5, 5, 10],  # BOOST clases positivas
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=settings.RANDOM_SEED,
                tree_method='hist',
                eval_metric='mlogloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                num_leaves=20,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                is_unbalance=True,  # ACTIVAR balanceo interno
                random_state=settings.RANDOM_SEED,
                verbose=-1
            )
        }
        
        # Entrenar cada modelo
        all_scores = {}
        
        for i, (model_name, model) in enumerate(models_config.items(), 1):
            print(f"\n[{i}/{len(models_config)}] Entrenando {model_name.upper()}...")
            
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                cv_scores.append(accuracy)
                
                print(f"Fold {fold}/{n_splits}: Accuracy = {accuracy:.4f}")
            
            # Entrenar en todo el dataset
            model.fit(X, y)
            self.models[model_name] = model
            
            # Guardar métricas
            all_scores[model_name] = cv_scores
            self.metrics[model_name] = {
                'accuracy': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'min': np.min(cv_scores),
                'max': np.max(cv_scores)
            }
            
            print(f"{model_name.upper()}: "
                  f"Accuracy = {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        best_model = max(self.metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\nMejor modelo: {best_model.upper()}")
        
        self.update_metadata(
            trained_at=pd.Timestamp.now().isoformat(),
            training_samples=len(df),
            features_count=len(self.feature_names),
            cv_splits=n_splits,
            best_model=best_model,
            smote_applied=use_smote
        )
        
        self.is_trained = True
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
    
    def predict_with_heuristics(
        self, 
        df: pd.DataFrame, 
        model_name: str = 'xgboost'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        TRUCO DEMO: Predicción con boost heurístico
        
        Si el sentimiento es MUY positivo/negativo Y menciona DOGE/TSLA,
        ajustamos las probabilidades para que tenga más sentido
        """
        # Predicción base
        X, _ = self.prepare_features(df, is_train=False, use_market_features=False)
        base_proba = self.models[model_name].predict_proba(X)
        base_pred = self.models[model_name].predict(X)
        
        # Post-processing heurístico
        adjusted_proba = base_proba.copy()
        
        for i in range(len(df)):
            sentiment = df.iloc[i]['sentiment_ensemble']
            relevance = df.iloc[i]['relevance_score']
            mentions_doge = df.iloc[i]['mentions_doge']
            mentions_tesla = df.iloc[i]['mentions_tesla']
            
            # REGLA 1: Si menciona DOGE + sentimiento fuerte → boost clase 1 o 3
            if mentions_doge and abs(sentiment) > self.sentiment_boost_threshold:
                boost_factor = min(abs(sentiment) * relevance * 2, 0.4)  # Max 40% boost
                
                if mentions_tesla:
                    # Boost clase 3 (Ambos)
                    adjusted_proba[i, 3] += boost_factor
                    adjusted_proba[i, 0] -= boost_factor * 0.7
                    adjusted_proba[i, 1] += boost_factor * 0.2
                    adjusted_proba[i, 2] += boost_factor * 0.1
                else:
                    # Boost clase 1 (Solo DOGE)
                    adjusted_proba[i, 1] += boost_factor
                    adjusted_proba[i, 0] -= boost_factor
            
            # REGLA 2: Si menciona TSLA + sentimiento fuerte → boost clase 2 o 3
            elif mentions_tesla and abs(sentiment) > self.sentiment_boost_threshold:
                boost_factor = min(abs(sentiment) * relevance * 1.5, 0.3)  # Max 30% boost
                
                # Boost clase 2 (Solo TSLA)
                adjusted_proba[i, 2] += boost_factor
                adjusted_proba[i, 0] -= boost_factor
            
            # REGLA 3: Si relevancia MUY alta → reducir "No Impact"
            if relevance > self.relevance_boost_threshold:
                reduction = (relevance - self.relevance_boost_threshold) * 0.3
                adjusted_proba[i, 0] -= reduction
                # Distribuir entre las otras clases
                adjusted_proba[i, 1:] += reduction / 3
            
            # Normalizar para que sumen 1
            adjusted_proba[i] = np.clip(adjusted_proba[i], 0, 1)
            adjusted_proba[i] /= adjusted_proba[i].sum()
        
        # Nueva predicción con probabilidades ajustadas
        adjusted_pred = np.argmax(adjusted_proba, axis=1)
        
        return adjusted_pred, adjusted_proba
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Usa predicción con heurísticas"""
        pred, _ = self.predict_with_heuristics(df, model_name)
        return pred
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """Usa probabilidades con heurísticas"""
        _, proba = self.predict_with_heuristics(df, model_name)
        return proba
    
    def get_classification_metrics(self, df: pd.DataFrame, model_name: str = 'xgboost') -> Dict:
        """Métricas detalladas"""
        df = self.create_impact_labels(df)
        X, y_true = self.prepare_features(df, is_train=False, use_market_features=False)
        
        y_pred = self.models[model_name].predict(X)
        y_proba = self.models[model_name].predict_proba(X)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names,
            'support': support,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
