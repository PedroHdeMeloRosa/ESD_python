# simulacoes/restricao_memoria.py
from typing import Optional, Dict, Any

# --- Configurações Globais para Restrições de Memória ---
# Estas não são "globais" no sentido de módulo Python, mas sim valores que o
# StructureAnalyzer obterá deste módulo.

_active_max_elements: Optional[int] = None
_active_ll_lru_capacity: Optional[int] = None

def configurar_limite_max_elementos(max_elementos: Optional[int]) -> None:
    """Configura o limite máximo de elementos para estruturas que o suportam."""
    global _active_max_elements
    _active_max_elements = max_elementos
    if max_elementos is not None:
        print(f"INFO (Restrição Memória): Limite máximo de elementos configurado para {max_elementos}.")
    else:
        print("INFO (Restrição Memória): Limite máximo de elementos desativado.")

def obter_limite_max_elementos() -> Optional[int]:
    """Retorna o limite máximo de elementos configurado."""
    return _active_max_elements

def configurar_descarte_lru_lista(capacidade_lista: Optional[int]) -> None:
    """Configura a capacidade para a LinkedList com descarte LRU."""
    global _active_ll_lru_capacity
    _active_ll_lru_capacity = capacidade_lista
    if capacidade_lista is not None:
        print(f"INFO (Restrição Memória): Capacidade da Lista LRU configurada para {capacidade_lista}.")
    else:
        print("INFO (Restrição Memória): Capacidade da Lista LRU desativada (sem limite LRU).")

def obter_capacidade_lista_lru() -> Optional[int]:
    """Retorna a capacidade configurada para a Lista LRU."""
    return _active_ll_lru_capacity

def resetar_restricoes_memoria() -> None:
    """Reseta todas as configurações de restrição de memória para o padrão."""
    global _active_max_elements, _active_ll_lru_capacity
    _active_max_elements = None
    _active_ll_lru_capacity = None
    # print("INFO (Restrição Memória): Todas as restrições de memória resetadas.")