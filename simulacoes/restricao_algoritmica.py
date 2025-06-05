# simulacoes/restricao_algoritmica.py
from typing import Optional

_active_hash_load_factor_override: Optional[float] = None
_active_tree_search_step_limit: Optional[int] = None

def configurar_hash_fator_carga_baixo(fator_carga: Optional[float]) -> None:
    """Configura um fator de carga máximo específico para a HashTable."""
    global _active_hash_load_factor_override
    _active_hash_load_factor_override = fator_carga
    if fator_carga is not None:
        print(f"INFO (Restrição Algorítmica): Fator de carga da HashTable configurado para {fator_carga}.")
    else:
        print("INFO (Restrição Algorítmica): Fator de carga da HashTable revertido para o padrão da estrutura.")

def obter_hash_fator_carga_override() -> Optional[float]:
    return _active_hash_load_factor_override

def configurar_limite_passos_busca_arvore(max_passos: Optional[int]) -> None:
    """Configura o limite de passos para busca em árvores."""
    global _active_tree_search_step_limit
    _active_tree_search_step_limit = max_passos
    if max_passos is not None:
        print(f"INFO (Restrição Algorítmica): Limite de passos de busca em árvore configurado para {max_passos}.")
    else:
        print("INFO (Restrição Algorítmica): Limite de passos de busca em árvore desativado.")

def obter_limite_passos_busca_arvore() -> Optional[int]:
    return _active_tree_search_step_limit

def resetar_restricoes_algoritmicas() -> None:
    global _active_hash_load_factor_override, _active_tree_search_step_limit
    _active_hash_load_factor_override = None
    _active_tree_search_step_limit = None
    # print("INFO (Restrição Algorítmica): Todas as restrições algorítmicas resetadas.")