"""Sistema de autenticação centralizado para GlobalStat."""

import os
from typing import Dict, Optional
from datetime import datetime
from flask import Flask, request, session, redirect, render_template_string
from functools import wraps


class AuthManager:
    """Gerenciador de autenticação unificado."""
    
    def __init__(self, default_username: str = "admin@atrium.com", 
                 default_password: str = "admin123"):
        """
        Inicializa o gerenciador de autenticação.
        
        Args:
            default_username: Usuário padrão (usado se BASIC_AUTH_PAIRS não estiver definido)
            default_password: Senha padrão
        """
        self.default_username = default_username
        self.default_password = default_password
        self._credentials = self._load_credentials()
    
    def _load_credentials(self) -> Dict[str, str]:
        """
        Carrega credenciais de variável de ambiente ou usa padrão.
        
        Formato BASIC_AUTH_PAIRS: "user1:pass1,user2:pass2"
        """
        env_val = os.environ.get("BASIC_AUTH_PAIRS", "").strip()
        pairs = {}
        
        if env_val:
            for item in env_val.split(","):
                if ":" in item:
                    u, p = item.split(":", 1)
                    pairs[u.strip()] = p.strip()
        
        # Fallback para credenciais padrão (apenas desenvolvimento local)
        if not pairs:
            print("[AVISO] Nenhum BASIC_AUTH_PAIRS definido. Usando credenciais padrão (somente ambiente local).")
            print(f"[INFO] Login temporário: {self.default_username} / {self.default_password}")
            pairs = {self.default_username: self.default_password}
        
        return pairs
    
    def add_credential(self, username: str, password: str):
        """Adiciona uma credencial temporariamente."""
        self._credentials[username] = password
        print(f"[AUTH] Credencial adicionada: {username}")
    
    def remove_credential(self, username: str):
        """Remove uma credencial."""
        if username in self._credentials:
            del self._credentials[username]
            print(f"[AUTH] Credencial removida: {username}")
    
    def validate(self, username: str, password: str) -> bool:
        """Valida credenciais."""
        return username in self._credentials and self._credentials[username] == password
    
    def get_credentials(self) -> Dict[str, str]:
        """Retorna dicionário de credenciais (sem senhas para segurança)."""
        return {user: "***" for user in self._credentials.keys()}
    
    def get_login_template(self) -> str:
        """Retorna template HTML do login."""
        return """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"><title>Login - GlobalStat</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#020817;--card:#050816;--text:#E6EDF3;--muted:#8B949E;--accent:#18A0FB;--accent-soft:rgba(24,160,251,0.12);--danger:#EF4444;--border:#151B23;--radius:14px;--shadow:0 18px 45px rgba(0,0,0,.65);--font:system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:
radial-gradient(900px 600px at 0% 0%,rgba(24,160,251,0.08),transparent),
radial-gradient(900px 600px at 100% 0%,rgba(45,212,191,0.06),transparent),#020817;font-family:var(--font);color:var(--text);}
.card{width:100%;max-width:380px;background:radial-gradient(140% 160% at 0% 0%,rgba(24,160,251,0.12),transparent),
radial-gradient(140% 220% at 100% 0%,rgba(45,212,191,0.10),transparent),var(--card);border:1px solid rgba(148,163,253,0.16);
border-radius:var(--radius);padding:26px 24px 22px;box-shadow:var(--shadow);backdrop-filter:blur(18px);}
h1{font-size:21px;margin:0 0 6px 0;font-weight:600;letter-spacing:.01em}
p{margin:0 0 16px 0;color:var(--muted);font-size:13px}
.field{margin-bottom:10px}
label{display:block;font-size:11px;color:var(--muted);margin-bottom:5px;font-weight:500;letter-spacing:.03em;text-transform:uppercase}
input{width:100%;padding:10px 11px;border-radius:9px;border:1px solid rgba(148,163,253,0.24);background:rgba(3,7,18,0.96);color:var(--text);outline:none;font-size:13px;}
input:focus{border-color:var(--accent);box-shadow:0 0 0 1px rgba(24,160,251,0.28),0 10px 30px rgba(15,23,42,0.9);}
.btn{width:100%;padding:10px 12px;margin-top:8px;border-radius:9px;border:0;cursor:pointer;font-weight:600;font-size:13px;letter-spacing:.02em;text-transform:uppercase;background:linear-gradient(90deg,#18A0FB,#22C55E);color:#020817;}
.btn:hover{filter:brightness(1.07)}
.error{color:var(--danger);font-size:11px;margin-top:6px}
.brand{display:flex;align-items:center;justify-content:center;margin-bottom:10px}
.brand img.logo{height:40px;display:block;margin:0 auto 8px auto}
.foot{margin-top:10px;text-align:center;font-size:10px;color:var(--muted)}
.hint{font-size:10px;color:var(--muted);margin-top:8px;text-align:center}
.info-box{background:rgba(24,160,251,0.08);border:1px solid rgba(24,160,251,0.2);border-radius:8px;padding:10px;margin-top:12px;font-size:11px;color:var(--muted)}
.info-box code{background:rgba(0,0,0,0.3);padding:2px 6px;border-radius:4px;font-family:monospace}
</style>
</head>
<body>
  <div class="card">
    <div class="brand"><img src="{{ asset('globalstat_logo.png') }}" alt="GlobalStat" class="logo"></div>
    <h1>GlobalStat Console</h1>
    <p>Acesse o painel exclusivo da Atrium com visão profissional de mercados.</p>
    <form method="POST" action="{{ url_for('login') }}">
      <div class="field"><label>E-mail</label><input name="username" type="email" placeholder="voce@dominio.com" required autofocus></div>
      <div class="field"><label>Senha</label><input name="password" type="password" placeholder="********" required></div>
      <button class="btn" type="submit">Entrar</button>
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </form>
    {% if show_temp_creds %}
    <div class="info-box">
      <strong>Credenciais Temporárias:</strong><br>
      Usuário: <code>{{ temp_username }}</code><br>
      Senha: <code>{{ temp_password }}</code>
    </div>
    {% endif %}
    <div class="hint">Configure via <code>BASIC_AUTH_PAIRS</code> no ambiente.</div>
    <div class="foot">© {{ year }} GlobalStat • Atrium</div>
  </div>
</body></html>"""
    
    def setup_routes(self, server: Flask, dash_app, 
                    show_temp_creds: bool = True,
                    redirect_after_login: str = "/app/"):
        """
        Configura rotas de autenticação no servidor Flask.
        
        Args:
            server: Instância Flask
            dash_app: Instância Dash (para get_asset_url)
            show_temp_creds: Se True, mostra credenciais temporárias na página de login
            redirect_after_login: URL para redirecionar após login bem-sucedido
        """
        @server.route("/login", methods=["GET", "POST"])
        def login():
            if request.method == "POST":
                user = request.form.get("username", "").strip()
                pw = request.form.get("password", "").strip()
                
                if self.validate(user, pw):
                    session["logged_in"] = True
                    session["user"] = user
                    session.pop("last_page", None)
                    resp = redirect(redirect_after_login)
                    resp.set_cookie("dash_pathname", "/", max_age=0)
                    return resp
                
                return render_template_string(
                    self.get_login_template(),
                    error="Credenciais inválidas",
                    asset=dash_app.get_asset_url,
                    year=datetime.now().year,
                    show_temp_creds=False
                )
            
            # GET - mostra página de login
            temp_username = self.default_username if not self._credentials else list(self._credentials.keys())[0]
            temp_password = self._credentials.get(temp_username, "")
            
            return render_template_string(
                self.get_login_template(),
                error=None,
                asset=dash_app.get_asset_url,
                year=datetime.now().year,
                show_temp_creds=show_temp_creds and len(self._credentials) == 1,
                temp_username=temp_username,
                temp_password=temp_password
            )
        
        @server.route("/logout")
        def logout():
            session.clear()
            return redirect("/login")
        
        @server.route("/")
        def root_redirect():
            return redirect("/login")
        
        @server.before_request
        def require_login():
            path = request.path or "/"
            
            # Páginas livres
            if path in {"/login", "/favicon.ico", "/healthz"}:
                return
            
            # Assets e recursos estáticos
            if path.startswith((
                "/assets", "/static", "/app/assets",
                "/_dash", "/app/_dash",
                "/_favicon", "/_dash-component-suites"
            )):
                return
            
            # Conteúdo do app Dash
            if path.startswith("/app/"):
                # Verifica se está logado antes de acessar /app/
                if not session.get("logged_in"):
                    return redirect("/login")
                return
            
            # Outras rotas - requer login
            if not session.get("logged_in"):
                return redirect("/login")


# Instância global (singleton)
_auth_instance: Optional[AuthManager] = None

def get_auth_manager(default_username: str = "admin@atrium.com",
                     default_password: str = "admin123") -> AuthManager:
    """
    Retorna instância global do AuthManager (singleton).
    
    Args:
        default_username: Usuário padrão
        default_password: Senha padrão
        
    Returns:
        Instância do AuthManager
    """
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = AuthManager(
            default_username=default_username,
            default_password=default_password
        )
    return _auth_instance

