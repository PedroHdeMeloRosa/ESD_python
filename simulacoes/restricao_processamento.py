# simulacoes/restricao_processamento.py
import random
import time  # Mantido caso você adicione delays específicos de processamento no futuro
from typing import Optional

# Apenas SIMULATED_EXTRA_COMPUTATION_LOOPS é usado ativamente pelo PerformanceMetrics
SIMULATED_EXTRA_COMPUTATION_LOOPS = 0


def configurar_carga_computacional_extra(num_loops_extras: Optional[int] = 0):  # Default 0 para desligar
    """Configura um número de loops computacionais extras para simular carga de CPU."""
    global SIMULATED_EXTRA_COMPUTATION_LOOPS
    if num_loops_extras is None or num_loops_extras < 0:
        SIMULATED_EXTRA_COMPUTATION_LOOPS = 0
    else:
        SIMULATED_EXTRA_COMPUTATION_LOOPS = num_loops_extras

    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        print(
            f"INFO (Restrição Processamento): Carga de CPU extra configurada com {SIMULATED_EXTRA_COMPUTATION_LOOPS} loops por operação.")
    else:
        print("INFO (Restrição Processamento): Carga de CPU extra desativada/resetada.")


def executar_carga_computacional_extra():
    """Executa um número configurado de operações 'inúteis' para simular carga."""
    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        # Usar uma semente baseada no tempo pode dar variabilidade, mas para consistência entre chamadas
        # dentro do mesmo teste de restrição, não mudar a semente aqui é melhor.
        # random.seed(time.time_ns()) # Opcional para maior aleatoriedade, mas pode dificultar reprodutibilidade exata.
        for _ in range(SIMULATED_EXTRA_COMPUTATION_LOOPS):
            _ = pow(random.random() + 0.1, random.random() + 0.1)


def resetar_restricoes_processamento() -> None:
    """Reseta as configurações de restrição de processamento para o padrão."""
    configurar_carga_computacional_extra(0)  # Chama a função de configuração com 0
    # print("INFO (Restrição Processamento): Restrições de processamento resetadas.") # Opcional, já impresso por configurar_