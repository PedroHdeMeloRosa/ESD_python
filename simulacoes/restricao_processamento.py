# simulacoes/restricao_processamento.py
import random
import time

SIMULATED_CPU_SLOWDOWN_FACTOR = 0.0 # Para um tipo de lentidão baseado em delay fixo por operação
SIMULATED_EXTRA_COMPUTATION_LOOPS = 0

def configurar_lentidao_cpu_delay(delay_segundos: float = 0.001): # Nome mais específico
    """Configura um DELAY fixo para ser adicionado a operações em PerformanceMetrics."""
    global SIMULATED_CPU_SLOWDOWN_FACTOR # Reutilizando, mas com semântica de delay agora
    SIMULATED_CPU_SLOWDOWN_FACTOR = delay_segundos # Agora é o delay direto em segundos
    if delay_segundos > 0:
        print(f"INFO: Delay de CPU simulado configurado para {delay_segundos*1000:.2f} ms por operação (via PerformanceMetrics).")
    else:
        print("INFO: Delay de CPU simulado desativado.")
    # Esta função só define o fator. PerformanceMetrics precisa usá-lo.

def configurar_carga_computacional_extra(num_loops_extras: int = 10000):
    global SIMULATED_EXTRA_COMPUTATION_LOOPS
    SIMULATED_EXTRA_COMPUTATION_LOOPS = num_loops_extras
    if num_loops_extras > 0:
        print(f"INFO: Carga computacional extra configurada com {num_loops_extras} loops por operação.")
    else:
        print("INFO: Carga computacional extra desativada.")

def executar_carga_computacional_extra():
    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        for _ in range(SIMULATED_EXTRA_COMPUTATION_LOOPS):
            _ = pow(random.random() + 0.1, random.random() + 0.1) # Evitar log(0) ou base 0