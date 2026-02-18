@echo off
title TFM Market Prediction API
echo ========================================
echo TFM - Market Prediction API
echo ========================================

if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Instalando dependencias...
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo [INFO] Activando entorno virtual...
    call venv\Scripts\activate.bat
)

echo.
echo Servidor iniciado en: http://localhost:8000
echo Documentacion API:   http://localhost:8000/docs
echo.

python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

pause