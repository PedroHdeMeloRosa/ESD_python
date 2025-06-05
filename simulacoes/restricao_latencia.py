# simulacoes/restricao_latencia.py
import time
from typing import List, Any, Optional  # Para type hint de batch_insert

_simulated_operation_delay_seconds: float = 0.0
_active_batch_insert_config: Optional[dict] = None # Para L2

def configurar_delay_operacao_constante(delay_segundos: Optional[float]):
    global _simulated_operation_delay_seconds
    _simulated_operation_delay_seconds = delay_segundos if delay_segundos is not None else 0.0
    msg = f"Delay de operação constante configurado para {_simulated_operation_delay_seconds*1000:.2f} ms." if _simulated_operation_delay_seconds > 0 else "Delay de operação constante desativado."
    print(f"INFO (Restrição Latência): {msg}")

def aplicar_delay_operacao_se_configurado():
    if _simulated_operation_delay_seconds > 0:
        time.sleep(_simulated_operation_delay_seconds)

def configurar_insercao_lote(tamanho_lote: Optional[int], delay_por_lote_segundos: Optional[float]):
    global _active_batch_insert_config
    if tamanho_lote is not None and delay_por_lote_segundos is not None:
        _active_batch_insert_config = {
            "tamanho_lote": tamanho_lote,
            "delay_por_lote_segundos": delay_por_lote_segundos
        }
        print(f"INFO (Restrição Latência): Inserção em lote configurada (Lote: {tamanho_lote}, Delay/Lote: {delay_por_lote_segundos*1000:.0f}ms).")
    else:
        _active_batch_insert_config = None
        print("INFO (Restrição Latência): Inserção em lote desativada.")

def obter_config_insercao_lote() -> Optional[dict]:
    return _active_batch_insert_config

def resetar_restricoes_latencia() -> None:
    global _simulated_operation_delay_seconds, _active_batch_insert_config
    _simulated_operation_delay_seconds = 0.0
    _active_batch_insert_config = None
    # print("INFO (Restrição Latência): Todas as restrições de latência resetadas.")