"""
Script para ejecutar la API FastAPI

Uso:
    python scripts/03_run_api.py [--host HOST] [--port PORT] [--reload]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from config.settings import settings


def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Ejecuta API de predicción")
    
    parser.add_argument(
        '--host',
        type=str,
        default=settings.API_HOST,
        help=f'Host (default: {settings.API_HOST})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=settings.API_PORT,
        help=f'Puerto (default: {settings.API_PORT})'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Activar auto-reload'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Nivel de logging'
    )
    
    return parser.parse_args()


def main():
    """Ejecuta la API FastAPI"""
    args = parse_args()
    
    print("="*70)
    print("LANZANDO API DE PREDICCIÓN - TFM")
    print("="*70)
    print(f"Host: {args.host}")
    print(f"Puerto: {args.port}")
    print(f"Auto-reload: {args.reload}")
    print(f"Log level: {args.log_level}")
    print("="*70)
    print(f"\n Documentación interactiva:")
    print(f"http://{args.host}:{args.port}/docs")
    print(f"\nEndpoint raíz:")
    print(f"http://{args.host}:{args.port}/")
    print("\n" + "="*70)
    print("Presiona CTRL+C para detener el servidor")
    print("="*70 + "\n")
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\nAPI detenida por el usuario")
        return 0
    except Exception as e:
        print(f"\nError ejecutando API: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())