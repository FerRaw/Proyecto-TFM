#!/bin/bash

echo "========================================"
echo "TFM - Market Prediction API"
echo "========================================"

if [ ! -d "venv" ]; then
    echo "[INFO] Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[INFO] Instalando dependencias..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[INFO] Activando entorno virtual..."
    source venv/bin/activate
fi

echo -e "\nServidor iniciado en: http://localhost:8000"
echo -e "Documentación API:   http://localhost:8000/docs\n"

python3 -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000