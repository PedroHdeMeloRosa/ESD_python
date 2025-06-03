import os
import sys
import time
import random
import tracemalloc
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from modelos.data_handler import DataHandler
from Estruturas.linked_list import LinkedList
from Estruturas.avl_tree import AVLTree
from Estruturas.hash_table import HashTable
from Estruturas.bloom_filter import BloomFilter
from Estruturas.radix_tree import RadixTree
from ui.menu import menu_estrutura
from modelos.moto import MotoEstatisticas, Moto


class MetricasDesempenho:
    @staticmethod
    def medir_tempo_memoria(funcao, *args):
        """Mede tempo de execução e consumo de memória"""
        tracemalloc.start()
        start_time = time.perf_counter()

        resultado = funcao(*args)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.perf_counter()

        return {
            'tempo': (end_time - start_time) * 1000,  # ms
            'memoria_atual': current / 1024,  # KB
            'memoria_pico': peak / 1024,  # KB
            'resultado': resultado
        }


def benchmark_estruturas(motos: List[Moto], tamanho_amostra: int = 1000) -> None:
    """Executa benchmark comparativo das estruturas de dados"""
    try:
        # Seleciona amostra aleatória
        amostra = random.sample(motos, min(tamanho_amostra, len(motos)))
        moto_teste = random.choice(amostra)

        # Inicializa estruturas
        estruturas = {
            'Lista Encadeada': LinkedList(),
            'Árvore AVL': AVLTree(),
            'Tabela Hash': HashTable(),
            'Radix Tree': RadixTree()
        }

        # Preenche estruturas
        print(f"\n⏳ Preparando benchmark com {len(amostra)} motos...")
        for nome, estrutura in estruturas.items():
            print(f"  Carregando {nome}...", end=' ')
            start = time.perf_counter()
            for moto in amostra:
                estrutura.inserir(moto)
            tempo = time.perf_counter() - start
            print(f"({tempo:.4f}s)")

        # Teste de operações
        resultados = []

        for nome, estrutura in estruturas.items():
            print(f"\n🔍 Testando {nome}:")
            tempos = {'Estrutura': nome, 'Busca': None, 'Inserção': None, 'Remoção': None}

            # Busca
            if hasattr(estrutura, 'buscar'):
                start = time.perf_counter()
                resultado, passos = estrutura.buscar(moto_teste)
                tempo = (time.perf_counter() - start) * 1000  # ms
                tempos['Busca'] = tempo
                print(f"  Busca: {tempo:.6f} ms | Passos: {passos}")

            # Inserção
            nova_moto = Moto(marca="TESTE", nome="MODELO_TESTE",
                             preco=10000, revenda=8000, ano=2023)
            start = time.perf_counter()
            estrutura.inserir(nova_moto)
            tempo = (time.perf_counter() - start) * 1000  # ms
            tempos['Inserção'] = tempo
            print(f"  Inserção: {tempo:.6f} ms")

            # Remoção
            if hasattr(estrutura, 'remover'):
                start = time.perf_counter()
                removido = estrutura.remover(moto_teste)
                tempo = (time.perf_counter() - start) * 1000  # ms
                tempos['Remoção'] = tempo
                print(f"  Remoção: {tempo:.6f} ms | {'Sucesso' if removido else 'Falha'}")

            resultados.append(tempos)

        # Exibe resultados comparativos
        print("\n📊 RESULTADOS COMPARATIVOS")
        print("=" * 65)
        print("{:<20} {:<15} {:<15} {:<15}".format(
            "Estrutura", "Busca (ms)", "Inserção (ms)", "Remoção (ms)"))
        print("-" * 65)

        for res in resultados:
            print("{:<20} {:<15} {:<15} {:<15}".format(
                res['Estrutura'],
                f"{res['Busca']:.6f}" if res['Busca'] is not None else "N/A",
                f"{res['Inserção']:.6f}" if res['Inserção'] is not None else "N/A",
                f"{res['Remoção']:.6f}" if res['Remoção'] is not None else "N/A"))

        # Geração de gráficos (apenas para operações disponíveis)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepara dados para o gráfico
        df = pd.DataFrame(resultados)
        df = df.set_index('Estrutura')

        # Remove colunas com todos valores N/A
        df = df.dropna(axis=1, how='all')

        # Gráfico de barras agrupadas
        df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8, edgecolor='black')

        ax.set_title('Benchmark de Estruturas de Dados')
        ax.set_ylabel('Tempo (ms)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Operação')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Salva e mostra resultados
        plt.savefig('benchmark_estruturas.png', dpi=300, bbox_inches='tight')
        print("\n✅ Gráfico salvo como 'benchmark_estruturas.png'")
        plt.show()

    except Exception as e:
        print(f"\n❌ Erro durante benchmark: {e}")
        import traceback
        traceback.print_exc()

def inicializar_estruturas_com_metricas(motos):
    """Inicializa estruturas com medição de desempenho"""
    metricas = {}
    estruturas = {
        'LinkedList': LinkedList(),
        'AVLTree': AVLTree(),
        'HashTable': HashTable(),
        'BloomFilter': BloomFilter(),
        'RadixTree': RadixTree()
    }

    print("\n⏳ Inicializando estruturas com medição de desempenho...")

    for nome, estrutura in estruturas.items():
        print(f"\nInicializando {nome}:")
        resultados = []

        # Medição de inserção
        for i, moto in enumerate(motos[:1000]):  # Limita a 1000 para teste
            if i % 100 == 0:
                print(f"  Inserindo item {i + 1}/{len(motos[:1000])}...")
            metricas_ins = MetricasDesempenho.medir_tempo_memoria(estrutura.inserir, moto)
            resultados.append(metricas_ins)

        # Cálculo de médias
        metricas[nome] = {
            'estrutura': estrutura,
            'tempo_medio_insercao': sum(m['tempo'] for m in resultados) / len(resultados),
            'memoria_pico_insercao': max(m['memoria_pico'] for m in resultados),
            'dados_insercao': resultados
        }

    return metricas


def gerar_relatorio_desempenho(metricas):
    """Gera relatório completo de desempenho"""
    print("\n📊 RELATÓRIO DE DESEMPENHO DAS ESTRUTURAS")
    print("=" * 70)

    # Tabela de desempenho
    print("\n{:<15} {:<20} {:<20} {:<15}".format(
        "Estrutura", "Tempo Inserção (ms)", "Memória Pico (KB)", "Operações/s"))
    print("-" * 70)

    for nome, dados in metricas.items():
        ops_por_segundo = 1000 / dados['tempo_medio_insercao'] if dados['tempo_medio_insercao'] > 0 else 0
        print("{:<15} {:<20.4f} {:<20.2f} {:<15.2f}".format(
            nome, dados['tempo_medio_insercao'],
            dados['memoria_pico_insercao'], ops_por_segundo))

    # Gráficos
    gerar_graficos_desempenho(metricas)


def gerar_graficos_desempenho(metricas):
    """Gera gráficos comparativos de desempenho"""
    plt.figure(figsize=(15, 10))

    # Gráfico 1: Tempo de inserção
    plt.subplot(2, 2, 1)
    nomes = list(metricas.keys())
    tempos = [metricas[n]['tempo_medio_insercao'] for n in nomes]
    plt.bar(nomes, tempos, color='skyblue')
    plt.title('Tempo Médio de Inserção (ms)')
    plt.ylabel('Milissegundos')

    # Gráfico 2: Memória utilizada
    plt.subplot(2, 2, 2)
    memorias = [metricas[n]['memoria_pico_insercao'] for n in nomes]
    plt.bar(nomes, memorias, color='lightgreen')
    plt.title('Uso Máximo de Memória (KB)')
    plt.ylabel('Kilobytes')

    # Gráfico 3: Operações por segundo
    plt.subplot(2, 2, 3)
    ops = [1000 / metricas[n]['tempo_medio_insercao'] if metricas[n]['tempo_medio_insercao'] > 0 else 0 for n in nomes]
    plt.bar(nomes, ops, color='salmon')
    plt.title('Operações por Segundo')
    plt.ylabel('Ops/s')

    # Gráfico 4: Dispersão Tempo x Memória
    plt.subplot(2, 2, 4)
    plt.scatter(tempos, memorias, s=100, alpha=0.6)
    for i, nome in enumerate(nomes):
        plt.annotate(nome, (tempos[i], memorias[i]))
    plt.xlabel('Tempo (ms)')
    plt.ylabel('Memória (KB)')
    plt.title('Relação Tempo-Memória')

    plt.tight_layout()
    plt.show()


def comparar_operacoes(metricas, motos):
    """Compara diferentes operações entre estruturas"""
    # Seleciona 100 motos aleatórias para teste
    amostra = random.sample(motos, min(100, len(motos)))
    moto_teste = random.choice(amostra)

    resultados = {}

    for nome, dados in metricas.items():
        estrutura = dados['estrutura']
        resultados[nome] = {}

        # Busca
        if hasattr(estrutura, 'buscar'):
            res_busca = MetricasDesempenho.medir_tempo_memoria(estrutura.buscar, moto_teste)
            resultados[nome]['busca'] = res_busca

        # Inserção (já temos os dados)

        # Remoção (onde aplicável)
        if hasattr(estrutura, 'remover'):
            res_remocao = MetricasDesempenho.medir_tempo_memoria(estrutura.remover, moto_teste)
            resultados[nome]['remocao'] = res_remocao

    # Exibir resultados comparativos
    print("\n🔍 COMPARAÇÃO DE OPERAÇÕES")
    print("=" * 90)
    print("{:<15} {:<20} {:<20} {:<20} {:<15}".format(
        "Estrutura", "Tempo Busca (ms)", "Memória Busca (KB)",
        "Tempo Remoção (ms)", "Memória Remoção (KB)"))
    print("-" * 90)

    for nome, res in resultados.items():
        linha = [nome]
        for op in ['busca', 'remocao']:
            if op in res:
                linha.append(f"{res[op]['tempo']:.4f}")
                linha.append(f"{res[op]['memoria_pico']:.2f}")
            else:
                linha.extend(["N/A", "N/A"])
        print("{:<15} {:<20} {:<20} {:<20} {:<15}".format(*linha))

    # Gráficos de comparação
    gerar_graficos_comparacao(resultados)


def gerar_graficos_comparacao(resultados):
    """Gera gráficos comparativos entre operações"""
    plt.figure(figsize=(15, 6))

    # Dados para os gráficos
    nomes = list(resultados.keys())

    # Tempos de busca
    tempos_busca = [res['busca']['tempo'] if 'busca' in res else 0 for nome, res in resultados.items()]

    # Tempos de remoção (onde disponível)
    tempos_remocao = []
    for nome in nomes:
        if 'remocao' in resultados[nome]:
            tempos_remocao.append(resultados[nome]['remocao']['tempo'])
        else:
            tempos_remocao.append(0)

    # Gráfico de barras comparativo
    x = range(len(nomes))
    width = 0.35

    plt.bar(x, tempos_busca, width, label='Busca', color='royalblue')
    plt.bar([i + width for i in x], tempos_remocao, width, label='Remoção', color='indianred')

    plt.xlabel('Estruturas')
    plt.ylabel('Tempo (ms)')
    plt.title('Comparação de Tempo entre Operações')
    plt.xticks([i + width / 2 for i in x], nomes)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Função principal do programa"""
    print("=" * 50)
    print("SISTEMA AVANÇADO DE ANÁLISE DE DESEMPENHO")
    print("=" * 50)

    # Carregar dataset
    motos = DataHandler.ler_dataset(os.path.join('data', 'bike_sales_india.csv'))
    if not motos:
        sys.exit(1)

    # Inicializar estruturas com medição
    metricas = inicializar_estruturas_com_metricas(motos)

    # Menu principal
    while True:
        print("\n" + "=" * 50)
        print("MENU PRINCIPAL")
        print("=" * 50)
        print("1. Lista Encadeada")
        print("2. Árvore AVL")
        print("3. Tabela Hash")
        print("4. Bloom Filter")
        print("5. Radix Tree")
        print("6. Relatório Completo de Desempenho")
        print("7. Comparar Operações entre Estruturas")
        print("8. Estatísticas Gerais")
        print("9. Simular Tendências Futuras")
        print("10. Benchmark de Estruturas")
        print("0. Sair")
        escolha = input("\nEscolha uma opção: ").strip()

        if escolha in ['1', '2', '3', '4', '5']:
            estruturas = {
                '1': ('LinkedList', "LISTA ENCADEADA"),
                '2': ('AVLTree', "ÁRVORE AVL"),
                '3': ('HashTable', "TABELA HASH"),
                '4': ('BloomFilter', "BLOOM FILTER"),
                '5': ('RadixTree', "RADIX TREE")
            }
            nome, desc = estruturas[escolha]
            menu_estrutura(metricas[nome]['estrutura'], desc, motos)

        elif escolha == '6':
            gerar_relatorio_desempenho(metricas)

        elif escolha == '7':
            comparar_operacoes(metricas, motos)

        elif escolha == '8':
            try:
                # Verificação adicional de dados
                if not motos or len(motos) == 0:
                    print("⚠️ Nenhum dado disponível para gerar estatísticas!")
                    return

                print("📊 Gerando gráficos...")
                MotoEstatisticas.gerar_graficos(motos)
            except Exception as e:
                print(f"❌ Erro crítico ao gerar gráficos: {e}")
                import traceback
                traceback.print_exc()
        elif escolha == '9':
            anos = int(input("Quantos anos no futuro para prever? "))
            MotoEstatisticas.prever_tendencias(motos, anos)

        elif escolha == '10':
            tamanho = int(input("Tamanho da amostra (padrão 1000): ") or 1000)
            benchmark_estruturas(motos, tamanho)

        elif escolha == '0':
            print("\nEncerrando sistema...")
            break

        input("\nPressione Enter para continuar...")


if __name__ == "__main__":
    main()