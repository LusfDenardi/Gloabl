"""
Exemplo de como usar o sistema de autenticação modularizado.

Este arquivo demonstra como integrar o AuthManager em uma aplicação Flask/Dash.
"""

import os
from flask import Flask
import dash
from pathlib import Path

# Importa o sistema de autenticação modularizado
from src.core.auth import get_auth_manager

# ===================== CONFIGURAÇÃO BÁSICA =====================

# 1. Cria servidor Flask
server = Flask(__name__)
server.secret_key = os.environ.get("SECRET_KEY", "chave_secreta_troque_em_producao")

# 2. Cria app Dash
ASSETS_DIR = Path(__file__).parent.parent / "assets"
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/app/",
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True,
    title="GlobalStat • Exemplo com Autenticação",
)

# ===================== CONFIGURAÇÃO DE AUTENTICAÇÃO =====================

# 3. Cria gerenciador de autenticação com credenciais temporárias
auth_manager = get_auth_manager(
    default_username="admin@atrium.com",  # Altere aqui para seu usuário
    default_password="admin123"            # Altere aqui para sua senha
)

# 4. Opcional: Adicione credenciais temporárias adicionais
# auth_manager.add_credential("usuario2@teste.com", "senha456")
# auth_manager.add_credential("usuario3@teste.com", "senha789")

# 5. Configura rotas de autenticação
# show_temp_creds=True mostra as credenciais na página de login (útil para desenvolvimento)
auth_manager.setup_routes(server, app, show_temp_creds=True)

# ===================== SEU CÓDIGO DO APP AQUI =====================

# Exemplo de layout simples
app.layout = dash.html.Div([
    dash.html.H1("GlobalStat - Exemplo com Autenticação"),
    dash.html.P("Você está autenticado! Esta é uma página protegida."),
    dash.html.A("Logout", href="/logout"),
])

# ===================== EXECUÇÃO =====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n{'='*60}")
    print("GlobalStat - Exemplo com Autenticação Modularizada")
    print(f"{'='*60}")
    print(f"Servidor iniciando em http://{host}:{port}")
    print(f"Login temporário: admin@atrium.com / admin123")
    print(f"{'='*60}\n")
    
    server.run(host=host, port=port, debug=False)

