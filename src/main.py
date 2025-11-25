import os
import sys
from pathlib import Path
from flask import Flask
import dash
import plotly.io as pio

# ===================== Adicionar diretório raiz ao sys.path =====================
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Core imports
from src.core.auth import get_auth_manager
from src.core.config import Config
from src.themes.carbon_pro import create_plotly_template, COLORS

# ===================== CONFIGURAÇÃO DE AUTENTICAÇÃO =====================
# Cria gerenciador de autenticação com credenciais temporárias
# Altere as credenciais abaixo conforme necessário

auth_manager = get_auth_manager(
    default_username="admin@atrium.com",  # Altere aqui para seu usuário
    default_password="admin123"            # Altere aqui para sua senha
)

# Opcional: Adicione credenciais temporárias adicionais
# auth_manager.add_credential("usuario2@teste.com", "senha456")
# auth_manager.add_credential("usuario3@teste.com", "senha789")

# ===================== FUNÇÃO HELPER PARA CONFIGURAR AUTENTICAÇÃO =====================

def setup_auth(server, app, show_temp_creds: bool = True):
    """
    Configura autenticação em um servidor Flask e app Dash existentes.
    
    Args:
        server: Instância Flask
        app: Instância Dash
        show_temp_creds: Se True, mostra credenciais na página de login
    """
    auth_manager.setup_routes(server, app, show_temp_creds=show_temp_creds)
    return auth_manager

# ===================== RE-EXPORTA =====================
__all__ = ["auth_manager", "get_auth_manager", "setup_auth"]

# ===================== EXEMPLO DE USO (se executado diretamente) =====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GlobalStat - Sistema de Autenticação Modularizado")
    print("="*60)
    print("\nEste módulo fornece o sistema de autenticação.")
    print("Para usar, importe e configure:")
    print("\n  from src.main import setup_auth")
    print("  setup_auth(server, app)")
    print("\nOu use diretamente:")
    print("\n  from src.core.auth import get_auth_manager")
    print("  auth_manager = get_auth_manager('usuario', 'senha')")
    print("  auth_manager.setup_routes(server, app)")
    print("\n" + "="*60)
    print(f"Credenciais padrão: admin@atrium.com / admin123")
    print("="*60 + "\n")

