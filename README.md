### Prerrequisitos

- Python 3.10+

### Paso 1: Clonar y navegar

```bash
cd tu_proyecto
```

### Paso 2: Verificar estructura

### Paso 3: Ejecutar .bat o .sh dependiendo del sistema operativo

Asegúrate de tener esta estructura:

```
Proyecto/
│
├── config/
│   ├── __init__.py
│   └── settings.py                   Configuración completa
│
├── data/
│   ├── raw/                          Datos originales
│   │   ├── elon_posts.csv            Descargado de Kaggle
│   │   ├── elon_quotes.csv           Descargado de Kaggle
│   │   ├── doge_data.csv             Auto-descarga Binance
│   │   └── tesla_data.csv            Auto-descarga Databento
│   │
│   └── processed/                    Datos procesados
│       ├── tweets_processed.parquet
│       ├── market_features.parquet
│       ├── master_dataset.parquet
│       └── granger_results.csv
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py                Carga desde APIs o guardado
│   │   ├── preprocessor.py           Limpieza y merge
│   │   └── features.py               Feature engineering
│   │
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── keywords.py               Extracción keywords
│   │   └── analyzer.py               Ensemble sentimiento
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_predictor.py         Clase base abstracta
│   │   ├── predictors.py             DOGE, TSLA, Impact
│   │   └── evaluator.py              Métricas financieras
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                   FastAPI completa
│       └── schemas.py                Pydantic models
│
├── models/                           Modelos entrenados
│   ├── doge_predictor_final.pkl
│   ├── tsla_predictor_final.pkl
│   └── impact_classifier_final.pkl
│
├── scripts/
│   ├── 01_preprocess_data.py         Pipeline preprocesamiento
│   ├── 02_train_models.py            Pipeline entrenamiento
│   ├── 03_run_api.py                 Lanzar API
│
├── requirements.txt                  Dependencias
├── .run_api.bat                      Script preparación proyecto Windows
├── .run_api.sh                       Script preparación proyecto Mac/Linux
└── README.md                         Documentación completa
```

## Ejecución

### Windows

```bash
run_api.bat
```

### Linux/Mac

```bash
chmod +x run_api.sh
./run_api.sh
```

### Manual

```bash
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 Uso de la API

### Acceder a la documentación

Una vez iniciada, abre tu navegador:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Ayuda completa**: http://localhost:8000/help