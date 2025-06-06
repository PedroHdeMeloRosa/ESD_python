# simulacoes/restricao_processamento.py
import random
import time
from typing import Optional

SIMULATED_EXTRA_COMPUTATION_LOOPS = 0
SIMULATED_OPERATION_FIXED_DELAY_S: float = 0.0  # Novo: Delay fixo em segundos


def configurar_carga_computacional_extra(num_loops_extras: Optional[int] = 0):
    global SIMULATED_EXTRA_COMPUTATION_LOOPS
    SIMULATED_EXTRA_COMPUTATION_LOOPS = num_loops_extras if num_loops_extras is not None and num_loops_extras >= 0 else 0
    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        print(f"INFO (Restrição Processamento): Carga CPU extra com {SIMULATED_EXTRA_COMPUTATION_LOOPS} loops/op.")
    # else: # Não precisa imprimir se está desativando, resetar_restricoes fará isso
    #     print("INFO (Restrição Processamento): Carga CPU extra desativada.")


def configurar_delay_fixo_operacao(delay_s: Optional[float] = 0.0):  # Novo
    """Configura um delay fixo a ser adicionado a cada operação medida."""
    global SIMULATED_OPERATION_FIXED_DELAY_S
    SIMULATED_OPERATION_FIXED_DELAY_S = delay_s if delay_s is not None and delay_s > 0 else 0.0
    if SIMULATED_OPERATION_FIXED_DELAY_S > 0:
        print(
            f"INFO (Restrição Processamento): Delay fixo de {SIMULATED_OPERATION_FIXED_DELAY_S * 1000:.2f} ms/op configurado.")
    # else:
    #     print("INFO (Restrição Processamento): Delay fixo de operação desativado.")


def executar_carga_computacional_extra():
    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        for _ in range(SIMULATED_EXTRA_COMPUTATION_LOOPS):
            _ = pow(random.random() + 0.1, random.random() + 0.1)


def aplicar_delay_fixo_operacao():  # Novo
    """Aplica o delay fixo configurado."""
    if SIMULATED_OPERATION_FIXED_DELAY_S > 0:
        time.sleep(SIMULATED_OPERATION_FIXED_DELAY_S)


def resetar_restricoes_processamento() -> None:
    global SIMULATED_EXTRA_COMPUTATION_LOOPS, SIMULATED_OPERATION_FIXED_DELAY_S
    changed = False
    if SIMULATED_EXTRA_COMPUTATION_LOOPS != 0: changed = True
    if SIMULATED_OPERATION_FIXED_DELAY_S != 0.0: changed = True

    configurar_carga_computacional_extra(0)
    configurar_delay_fixo_operacao(0.0)  # Reseta o novo delay também

    if changed:
        print("INFO (Restrição Processamento): Todas as restrições de processamento resetadas.")