"""Script de inicialização do GlobalStat"""
import sys
from pathlib import Path

# Garante que o diretório raiz está no sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import create_app

if __name__ == "__main__":
    server, app = create_app()
    print("\n" + "="*60)
    print("GlobalStat - Carbon Pro Console")
    print("="*60)
    print(f"\nServidor iniciando em http://localhost:8050")
    print(f"Login: admin@atrium.com / admin123")
    print("="*60 + "\n")
    app.run_server(debug=True, host="0.0.0.0", port=8050)