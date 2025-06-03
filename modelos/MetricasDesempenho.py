import matplotlib.pyplot as plt
import tracemalloc
import time


class AnalisadorDesempenho:
    @staticmethod
    def gerar_grafico_evolucao(dados):
        """Gráfico de evolução temporal do desempenho"""
        plt.figure(figsize=(12, 6))

        for nome, metricas in dados.items():
            tempos = [m['tempo'] for m in metricas['dados_insercao']]
            plt.plot(tempos, label=nome, marker='o')

        plt.xlabel('Número de Operações')
        plt.ylabel('Tempo (ms)')
        plt.title('Evolução do Tempo de Inserção')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def analisar_memoria(dados):
        """Análise detalhada de uso de memória"""
        plt.figure(figsize=(12, 6))

        for nome, metricas in dados.items():
            memorias = [m['memoria_pico'] for m in metricas['dados_insercao']]
            plt.plot(memorias, label=nome, linestyle='--', marker='x')

        plt.xlabel('Número de Operações')
        plt.ylabel('Memória (KB)')
        plt.title('Uso de Memória Durante Inserções')
        plt.legend()
        plt.grid(True)
        plt.show()