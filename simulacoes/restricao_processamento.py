# simulacoes/restricao_processamento.py
import random
import time

# --- DEFINIÇÃO DAS VARIÁVEIS GLOBAIS DO MÓDULO ---
# Estes são os valores padrão quando nenhuma restrição está ativa.
SIMULATED_CPU_SLOWDOWN_FACTOR: float = 0.0  # Representa um fator multiplicador ou um delay adicional.
                                            # Se for multiplicador, 1.0 = normal. > 1.0 = mais lento.
                                            # Se for delay, valor em segundos a adicionar.
                                            # Vamos tratar como um delay fixo pequeno por enquanto.
SIMULATED_EXTRA_COMPUTATION_LOOPS: int = 0

# Função para simular delay fixo, se usada.
# A forma como PerformanceMetrics foi modificada para incluir delay_fixo_operacao é melhor.
# Esta variável pode ser usada por aplicar_delay_fixo_operacao se você definir essa função aqui.
# _simulated_fixed_delay_per_op_seconds = 0.0

def configurar_lentidao_cpu_delay(delay_adicional_segundos: float = 0.001): # Ex: 1ms
    """
    Configura um delay adicional FIXO para cada operação medida para simular CPU mais lenta.
    Este delay será adicionado pelo PerformanceMetrics se a lógica lá for ajustada para usar isso.
    Alternativamente, pode ser usado para configurar SIMULATED_CPU_SLOWDOWN_FACTOR se ele representar um delay.
    """
    global SIMULATED_CPU_SLOWDOWN_FACTOR # Modifica a variável global do módulo
    # Vamos assumir que SIMULATED_CPU_SLOWDOWN_FACTOR agora representa este delay adicional.
    SIMULATED_CPU_SLOWDOWN_FACTOR = max(0.0, delay_adicional_segundos)
    if SIMULATED_CPU_SLOWDOWN_FACTOR > 0:
        print(f"INFO (PROC): Lentidão CPU simulada com delay adicional de {SIMULATED_CPU_SLOWDOWN_FACTOR*1000:.2f}ms por operação.")
    else:
        print("INFO (PROC): Delay adicional da CPU simulada desativado.")

def configurar_carga_computacional_extra(num_loops_extras: int = 0):
    """Configura um número de loops computacionais extras para simular carga."""
    global SIMULATED_EXTRA_COMPUTATION_LOOPS
    SIMULATED_EXTRA_COMPUTATION_LOOPS = max(0, num_loops_extras)
    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        print(f"INFO (PROC): Carga computacional extra configurada com {SIMULATED_EXTRA_COMPUTATION_LOOPS} loops.")
    else:
        print("INFO (PROC): Carga computacional extra desativada.")

def executar_carga_computacional_extra():
    """Executa um número configurado de operações 'inúteis'."""
    if SIMULATED_EXTRA_COMPUTATION_LOOPS > 0:
        # Este loop será executado dentro de PerformanceMetrics.measure
        # antes de medir o tempo da operação real.
        for _ in range(SIMULATED_EXTRA_COMPUTATION_LOOPS):
            _ = (random.random() * random.random()) / (random.random() + 1e-9) # Operação simples

def aplicar_delay_fixo_operacao(): # Chamado por PerformanceMetrics
    """
    Aplica um delay fixo se SIMULATED_CPU_SLOWDOWN_FACTOR (interpretado como delay) estiver configurado.
    """
    # Esta função seria chamada DENTRO do PerformanceMetrics.measure
    # APÓS a operação real, para inflar o tempo medido.
    # No entanto, na sua PerformanceMetrics, você já chama restricao_processamento.aplicar_delay_fixo_operacao()
    # Portanto, esta função PRECISA existir.
    if SIMULATED_CPU_SLOWDOWN_FACTOR > 0:
        time.sleep(SIMULATED_CPU_SLOWDOWN_FACTOR)

def resetar_restricoes_processamento():
    """Reseta as variáveis de simulação de processamento para os padrões."""
    global SIMULATED_CPU_SLOWDOWN_FACTOR, SIMULATED_EXTRA_COMPUTATION_LOOPS
    SIMULATED_CPU_SLOWDOWN_FACTOR = 0.0
    SIMULATED_EXTRA_COMPUTATION_LOOPS = 0
    print("INFO (PROC): Restrições de processamento (lentidão/carga) resetadas.")