"""
Script de teste para validar a reorganização do GlobalStat.
Execute: python test_reorganizacao.py
"""

import sys
from pathlib import Path

# Adiciona raiz ao path
_root = Path(__file__).parent
sys.path.insert(0, str(_root))

def test_imports():
    """Testa se todos os imports funcionam."""
    print("=" * 60)
    print("TESTE 1: Imports dos módulos core")
    print("=" * 60)
    
    try:
        from src.core.config import Config
        print("✅ Config importado com sucesso")
        
        from src.core.cache import CacheManager, get_cache
        print("✅ CacheManager importado com sucesso")
        
        from src.core.data_fetcher import YahooDataFetcher, get_fetcher
        print("✅ YahooDataFetcher importado com sucesso")
        
        from src.themes.carbon_pro import CARBON_PRO_THEME
        print("✅ Tema Carbon Pro importado com sucesso")
        
        from src.utils.data_processing import ensure_series_close
        print("✅ Utilitários importados com sucesso")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao importar: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache():
    """Testa o sistema de cache."""
    print("\n" + "=" * 60)
    print("TESTE 2: Sistema de Cache")
    print("=" * 60)
    
    try:
        from src.core.cache import CacheManager
        
        cache = CacheManager(use_persistent=False)
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        
        assert value == "test_value"
        print("✅ Cache funcionando")
        
        cache.clear()
        return True
    except Exception as e:
        print(f"❌ Erro no Cache: {e}")
        return False


def test_compatibility():
    """Testa compatibilidade com código original."""
    print("\n" + "=" * 60)
    print("TESTE 3: Compatibilidade")
    print("=" * 60)
    
    try:
        from src.modules.macro import mount_macro
        print("✅ mount_macro importado")
        
        from src.modules.beta_alpha import mount_beta
        print("✅ mount_beta importado")
        
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 60)
    print("TESTES DE VALIDAÇÃO - REORGANIZAÇÃO GLOBALSTAT")
    print("=" * 60)
    
    results = [test_imports(), test_cache(), test_compatibility()]
    
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Testes passados: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)