# main.py
import cv2
import os
import sys
import time
import random
import tracemalloc
from typing import List, Dict, Any, Callable, Optional  # Adicionado Callable
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd # Removido se DataHandler não usa mais (ou usa internamente sem expor)

from modelos.data_handler import DataHandler
from Estruturas.linked_list import LinkedList
from Estruturas.avl_tree import AVLTree
from Estruturas.hash_table import HashTable
from Estruturas.bloom_filter import BloomFilter
from Estruturas.radix_tree import RadixTree
from ui.menu import menu_estrutura
from modelos.moto import Moto, MotoEstatisticas  # MotoEstatisticas importado


class PerformanceMetrics:
    """Classe para medição encapsulada de métricas de desempenho de funções."""

    @staticmethod
    def measure(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Mede o tempo de execução e o consumo de memória de uma função.
        :param func: A função a ser executada.
        :param args: Argumentos posicionais para a função.
        :param kwargs: Argumentos nomeados para a função.
        :return: Dicionário com 'time' (ms), 'current_memory' (KB),
                 'peak_memory' (KB) e 'result' (retorno da função).
        """
        tracemalloc.start()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.perf_counter()

        return {
            'time': (end_time - start_time) * 1000,  # ms
            'current_memory': current / (1024),  # KB
            'peak_memory': peak / (1024),  # KB
            'result': result
        }


class StructureAnalyzer:
    """
    Classe central para inicializar, analisar e comparar o desempenho
    de diferentes estruturas de dados com um dataset de motocicletas.
    """

    def __init__(self, motorcycles_dataset: List[Moto]):
        self.motorcycles_full_dataset: List[Moto] = motorcycles_dataset
        self.structures_prototypes: Dict[str, Callable[[], Any]] = {
            'LinkedList': LinkedList,
            'AVLTree': AVLTree,
            'HashTable': lambda: HashTable(capacidade=max(101, len(motorcycles_dataset) // 10)),
            # Capacidade inicial baseada no dataset
            'BloomFilter': lambda: BloomFilter(
                num_itens_esperados=len(motorcycles_dataset) if motorcycles_dataset else 1000),
            'RadixTree': RadixTree
        }
        self.initialized_structures: Dict[str, Any] = {}
        self.performance_results: Dict[str, Dict[str, Any]] = {}  # Armazena todos os resultados de benchmark

    def initialize_all_structures(self, sample_size: Optional[int] = None, verbose: bool = True) -> None:
        """
        Inicializa todas as estruturas de dados definidas em `structures_prototypes`
        com uma amostra do dataset, medindo o desempenho da inserção.

        :param sample_size: Número de motos da amostra para usar na inicialização.
                           Se None, usa todo o dataset.
        :param verbose: Se True, imprime o progresso.
        """
        if not self.motorcycles_full_dataset:
            if verbose: print("Dataset de motocicletas está vazio. Não é possível inicializar estruturas.")
            return

        if sample_size is None:
            sample_to_insert = self.motorcycles_full_dataset
            sample_size = len(self.motorcycles_full_dataset)
        else:
            sample_size = min(sample_size, len(self.motorcycles_full_dataset))
            sample_to_insert = random.sample(self.motorcycles_full_dataset, sample_size)

        if verbose: print(f"\n⏳ Inicializando estruturas com {sample_size} motos e medindo desempenho...")

        for name, structure_constructor in self.structures_prototypes.items():
            if verbose: print(f"\n  Inicializando {name}...")
            structure_instance = structure_constructor()

            insertion_metrics_list = []
            total_insertion_time = 0
            max_peak_memory_during_init = 0

            for i, bike_to_insert in enumerate(sample_to_insert):
                if verbose and (i + 1) % (max(1, sample_size // 10)) == 0:  # Imprime progresso a cada 10%
                    print(f"    Inserindo item {i + 1}/{sample_size} em {name}...")

                # Mede a inserção individual
                # Nota: Algumas estruturas (BloomFilter) podem não ter um "resultado" significativo da inserção.
                metrics = PerformanceMetrics.measure(structure_instance.inserir, bike_to_insert)
                insertion_metrics_list.append({
                    'time': metrics['time'],
                    'peak_memory': metrics['peak_memory']  # Pico de memória daquela inserção específica
                })
                total_insertion_time += metrics['time']
                if metrics['peak_memory'] > max_peak_memory_during_init:
                    max_peak_memory_during_init = metrics['peak_memory']

            avg_insert_time = total_insertion_time / sample_size if sample_size > 0 else 0

            self.initialized_structures[name] = structure_instance
            # Armazena resultados detalhados para gráficos de evolução
            self.performance_results[name] = {
                'initialization': {  # Métricas da inicialização como um todo
                    'sample_size': sample_size,
                    'total_time_ms': total_insertion_time,
                    'avg_insert_time_ms': avg_insert_time,
                    'peak_memory_init_kb': max_peak_memory_during_init,  # Pico durante toda a inicialização
                    'insertion_evolution_data': insertion_metrics_list  # Lista de métricas por inserção
                }
            }
            if verbose: print(
                f"  {name} inicializado. Tempo médio de inserção: {avg_insert_time:.4f} ms. Pico de memória na init: {max_peak_memory_during_init:.2f} KB")

    def run_benchmark_operations(self, num_operations: int = 100, verbose: bool = True) -> None:
        """
        Executa operações de benchmark (busca, nova inserção, remoção)
        nas estruturas já inicializadas.

        :param num_operations: Número de operações de cada tipo a serem testadas.
        :param verbose: Se True, imprime o progresso.
        """
        if not self.initialized_structures:
            if verbose: print("Nenhuma estrutura inicializada. Execute `initialize_all_structures` primeiro.")
            return
        if not self.motorcycles_full_dataset:
            if verbose: print("Dataset vazio, não é possível executar benchmarks de operações.")
            return

        # Prepara amostras para teste
        # Garante que as motos de teste existam nas estruturas (se inicializadas com todo o dataset)
        # ou sejam representativas.
        if len(self.motorcycles_full_dataset) < num_operations:
            if verbose: print(
                f"Dataset tem menos de {num_operations} motos. Ajustando número de operações para {len(self.motorcycles_full_dataset)}.")
            num_operations = len(self.motorcycles_full_dataset)

        if num_operations == 0:
            if verbose: print("Nenhuma operação de benchmark a ser executada.")
            return

        sample_for_search_remove = random.sample(self.motorcycles_full_dataset, num_operations)

        # Motos para nova inserção (diferentes das já existentes, se possível)
        sample_for_new_insertion = []
        existing_bikes_set = set(self.motorcycles_full_dataset)  # Para checagem rápida
        for i in range(num_operations):
            sample_for_new_insertion.append(
                Moto(marca=f"MARCA_NOVA_{i}", nome=f"MODELO_NOVO_{i}", preco=10000 + i, revenda=8000 + i, ano=2025 + i)
            )

        if verbose: print(f"\n⚙️ Executando benchmark de operações ({num_operations} de cada tipo)...")

        for name, structure in self.initialized_structures.items():
            if verbose: print(f"\n  Analisando {name}:")

            op_results = {'search': [], 'new_insertion': [], 'removal': []}

            # Teste de Busca
            if hasattr(structure, 'buscar'):
                for bike_to_search in sample_for_search_remove:
                    metrics = PerformanceMetrics.measure(structure.buscar, bike_to_search)
                    op_results['search'].append({'time': metrics['time'], 'peak_memory': metrics['peak_memory'],
                                                 'found': metrics['result'][0] if isinstance(metrics['result'],
                                                                                             tuple) else metrics[
                                                     'result']})
                avg_search_time = sum(m['time'] for m in op_results['search']) / num_operations
                if verbose: print(f"    Busca: Tempo médio {avg_search_time:.4f} ms")
                self.performance_results[name]['search_avg_time_ms'] = avg_search_time
                self.performance_results[name]['search_peak_memory_kb'] = max(
                    m['peak_memory'] for m in op_results['search']) if op_results['search'] else 0

            # Teste de Nova Inserção
            if hasattr(structure, 'inserir'):
                # Cuidado: isso adiciona permanentemente às estruturas inicializadas.
                # Para um benchmark puro, pode-se querer copiar a estrutura antes ou remover depois.
                # Por simplicidade, vamos inserir.
                for new_bike in sample_for_new_insertion:
                    metrics = PerformanceMetrics.measure(structure.inserir, new_bike)
                    op_results['new_insertion'].append({'time': metrics['time'], 'peak_memory': metrics['peak_memory']})
                avg_new_insert_time = sum(m['time'] for m in op_results['new_insertion']) / num_operations
                if verbose: print(f"    Nova Inserção: Tempo médio {avg_new_insert_time:.4f} ms")
                self.performance_results[name]['new_insertion_avg_time_ms'] = avg_new_insert_time
                self.performance_results[name]['new_insertion_peak_memory_kb'] = max(
                    m['peak_memory'] for m in op_results['new_insertion']) if op_results['new_insertion'] else 0

            # Teste de Remoção
            if hasattr(structure, 'remover'):
                items_to_remove_from_structure = sample_for_search_remove  # Usar a mesma amostra da busca
                # Para Bloom Filter e RadixTree (sem remoção), isso será pulado.
                if name not in ["BloomFilter", "RadixTree"]:  # Assumindo que RadixTree não tem remoção
                    for bike_to_remove in items_to_remove_from_structure:
                        metrics = PerformanceMetrics.measure(structure.remover, bike_to_remove)
                        op_results['removal'].append({'time': metrics['time'], 'peak_memory': metrics['peak_memory'],
                                                      'removed': metrics['result']})
                    avg_removal_time = sum(m['time'] for m in op_results['removal']) / num_operations
                    if verbose: print(f"    Remoção: Tempo médio {avg_removal_time:.4f} ms")
                    self.performance_results[name]['removal_avg_time_ms'] = avg_removal_time
                    self.performance_results[name]['removal_peak_memory_kb'] = max(
                        m['peak_memory'] for m in op_results['removal']) if op_results['removal'] else 0

            # Limpar as motos de "nova inserção" para não poluir as estruturas para próximos benchmarks
            if hasattr(structure, 'remover') and name not in ["BloomFilter", "RadixTree"]:
                for new_bike in sample_for_new_insertion:
                    structure.remover(new_bike)  # Tenta remover

    def _generate_performance_report_table(self) -> None:
        """Gera um relatório textual comparativo do desempenho das estruturas."""
        if not self.performance_results:
            print("Nenhum resultado de performance para gerar relatório. Execute benchmarks primeiro.")
            return

        print("\n\n📊 RELATÓRIO COMPLETO DE DESEMPENHO 📊")
        print("=" * 120)
        header = "{:<15} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(
            "Estrutura", "Init Avg Ins (ms)", "Search Avg (ms)", "New Ins Avg (ms)",
            "Removal Avg (ms)", "Init Peak Mem (KB)")
        print(header)
        print("-" * 120)

        for name, metrics_dict in self.performance_results.items():
            init_metrics = metrics_dict.get('initialization', {})
            print("{:<15} | {:<20.4f} | {:<20.4f} | {:<20.4f} | {:<20.4f} | {:<20.2f}".format(
                name,
                init_metrics.get('avg_insert_time_ms', 0),
                metrics_dict.get('search_avg_time_ms', 0),
                metrics_dict.get('new_insertion_avg_time_ms', 0),
                metrics_dict.get('removal_avg_time_ms', 0),  # Pode ser 0 se não suportado
                init_metrics.get('peak_memory_init_kb', 0)
            ))
        print("=" * 120)

    def _generate_comparison_charts(self) -> None:
        """Gera gráficos comparativos de tempo de operação e uso de memória."""
        if not self.performance_results:
            print("Nenhum resultado de performance para gerar gráficos. Execute benchmarks primeiro.")
            return

        names = list(self.performance_results.keys())
        if not names: return

        plt.style.use('seaborn-v0_8-whitegrid')

        # Gráfico 1: Comparação de Tempos Médios das Operações
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        operations = ['initialization_avg_insert', 'search_avg', 'new_insertion_avg', 'removal_avg']
        op_labels = ['Init Inserção Média', 'Busca Média', 'Nova Inserção Média', 'Remoção Média']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold']

        bar_width = 0.2
        index = np.arange(len(names))

        for i, op_key_suffix in enumerate(operations):
            # Ajuste para pegar o tempo correto
            if op_key_suffix == 'initialization_avg_insert':
                times = [self.performance_results[n].get('initialization', {}).get('avg_insert_time_ms', 0) for n in
                         names]
            else:
                times = [self.performance_results[n].get(f'{op_key_suffix}_time_ms', 0) for n in names]

            ax1.bar(index + i * bar_width, times, bar_width, label=op_labels[i], color=colors[i])

        ax1.set_title('Comparação de Tempos Médios das Operações', fontsize=15)
        ax1.set_ylabel('Tempo Médio (ms)', fontsize=12)
        ax1.set_xlabel('Estrutura de Dados', fontsize=12)
        ax1.set_xticks(index + bar_width * (len(operations) - 1) / 2)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()

        try:
            plt.show()
        except Exception:
            print("Não foi possível exibir o gráfico de tempos. Tente salvar em arquivo.")
        finally:
            plt.close(fig1)

        # Gráfico 2: Uso de Memória de Pico na Inicialização
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        memories = [self.performance_results[n].get('initialization', {}).get('peak_memory_init_kb', 0) for n in names]

        ax2.bar(names, memories, color='mediumpurple', alpha=0.7, edgecolor='black')
        ax2.set_title('Uso de Memória de Pico Durante a Inicialização', fontsize=15)
        ax2.set_ylabel('Memória de Pico (KB)', fontsize=12)
        ax2.set_xlabel('Estrutura de Dados', fontsize=12)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()

        try:
            plt.show()
        except Exception:
            print("Não foi possível exibir o gráfico de memória. Tente salvar em arquivo.")
        finally:
            plt.close(fig2)

    def _generate_insertion_evolution_charts(self) -> None:
        """
        Gera gráficos mostrando a evolução do tempo de inserção e uso de memória
        durante a inicialização das estruturas.
        """
        if not self.performance_results:
            print("Nenhum resultado de inicialização para gerar gráficos de evolução.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        num_structures = len(self.performance_results)
        if num_structures == 0: return

        # Gráfico 1: Evolução do Tempo de Inserção
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        ax1.set_title('Evolução do Tempo de Inserção Individual Durante a Inicialização', fontsize=15)
        ax1.set_xlabel('Número da Operação de Inserção', fontsize=12)
        ax1.set_ylabel('Tempo de Inserção (ms)', fontsize=12)

        for name, metrics in self.performance_results.items():
            init_data = metrics.get('initialization', {}).get('insertion_evolution_data', [])
            if init_data:
                times = [m['time'] for m in init_data]
                ax1.plot(times, label=f'{name} (média: {sum(times) / len(times):.3f} ms)', marker='.', linestyle='-',
                         alpha=0.7, markersize=3)

        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            print("Não foi possível exibir o gráfico de evolução de tempo. Tente salvar.")
        finally:
            plt.close(fig1)

        # Gráfico 2: Evolução do Pico de Memória por Inserção
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.set_title('Evolução do Pico de Memória por Inserção Durante a Inicialização', fontsize=15)
        ax2.set_xlabel('Número da Operação de Inserção', fontsize=12)
        ax2.set_ylabel('Pico de Memória da Inserção (KB)', fontsize=12)

        for name, metrics in self.performance_results.items():
            init_data = metrics.get('initialization', {}).get('insertion_evolution_data', [])
            if init_data:
                memories = [m['peak_memory'] for m in init_data]
                ax2.plot(memories, label=f'{name} (pico max: {max(memories):.2f} KB)', marker='.', linestyle='-',
                         alpha=0.7, markersize=3)

        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            print("Não foi possível exibir o gráfico de evolução de memória. Tente salvar.")
        finally:
            plt.close(fig2)

    def run_full_analysis_suite(self, init_sample_size: Optional[int] = 1000, benchmark_ops_count: int = 100):
        """
        Executa todo o conjunto de análises: inicialização, benchmarks de operações,
        e geração de relatórios e gráficos.
        :param init_sample_size: Tamanho da amostra para inicialização das estruturas.
        :param benchmark_ops_count: Número de operações para benchmarks de busca/inserção/remoção.
        """
        print("\n🚀 INICIANDO SUÍTE COMPLETA DE ANÁLISE DE DESEMPENHO 🚀")
        self.initialize_all_structures(sample_size=init_sample_size)
        self.run_benchmark_operations(num_operations=benchmark_ops_count)

        print("\n📋 Gerando Relatórios e Gráficos...")
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()  # Gráficos de evolução
        print("\n🏁 Análise Completa Concluída! 🏁")


def main_menu_loop(analyzer: StructureAnalyzer, full_dataset: List[Moto]):
    """Loop do menu principal da aplicação."""
    while True:
        print("\n" + "=" * 50)
        print("SISTEMA DE ANÁLISE DE ESTRUTURAS DE DADOS")
        print("=" * 50)
        print("--- GERENCIAR ESTRUTURAS INDIVIDUAIS ---")
        print("1. Lista Encadeada")
        print("2. Árvore AVL")
        print("3. Tabela Hash")
        print("4. Bloom Filter")
        print("5. Radix Tree")
        print("--- ANÁLISE E COMPARAÇÃO ---")
        print("6. Executar Suíte Completa de Análise (Inicialização + Benchmarks + Gráficos)")
        print("7. Gerar Gráficos de Evolução da Inicialização (após inicialização)")
        print("--- ANÁLISE DO DATASET ---")
        print("8. Estatísticas Gerais do Dataset e Gráficos")
        print("9. Simular Tendências Futuras do Dataset")
        print("--- SAIR ---")
        print("0. Sair do Sistema")

        escolha_main = input("\nEscolha uma opção: ").strip()

        if escolha_main in ['1', '2', '3', '4', '5']:
            struct_map = {
                '1': ('LinkedList', "LISTA ENCADEADA"),
                '2': ('AVLTree', "ÁRVORE AVL"),
                '3': ('HashTable', "TABELA HASH"),
                '4': ('BloomFilter', "BLOOM FILTER"),
                '5': ('RadixTree', "RADIX TREE")
            }
            struct_key, struct_name_display = struct_map[escolha_main]

            if not analyzer.initialized_structures.get(struct_key):
                print(f"\nA estrutura {struct_name_display} não foi inicializada ainda.")
                if input("Deseja inicializar todas as estruturas agora com uma amostra? (s/n): ").lower() == 's':
                    try:
                        sample_s = int(
                            input("Tamanho da amostra para inicialização (ex: 1000, deixe vazio para todas): ") or len(
                                full_dataset))
                        analyzer.initialize_all_structures(sample_size=sample_s)
                    except ValueError:
                        print("Tamanho da amostra inválido. Usando padrão.")
                        analyzer.initialize_all_structures(sample_size=1000)

                if not analyzer.initialized_structures.get(struct_key):  # Checa de novo
                    print(
                        f"Falha ao inicializar ou usuário cancelou. Não é possível acessar o menu de {struct_name_display}.")
                    continue  # Volta ao menu principal

            menu_estrutura(analyzer.initialized_structures[struct_key],
                           struct_name_display,
                           analyzer.motorcycles_full_dataset)  # Passa o dataset completo original

        elif escolha_main == '6':
            try:
                init_s = input(
                    "Tamanho da amostra para inicialização (padrão 1000, deixe vazio para todas as motos): ").strip()
                init_sample = int(init_s) if init_s else None  # None para usar todo o dataset

                bench_ops_s = input("Número de operações para benchmarks (padrão 100): ").strip()
                bench_ops = int(bench_ops_s) if bench_ops_s else 100

                analyzer.run_full_analysis_suite(init_sample_size=init_sample, benchmark_ops_count=bench_ops)
            except ValueError:
                print("Entrada inválida para tamanhos. Usando padrões.")
                analyzer.run_full_analysis_suite()


        elif escolha_main == '7':
            if not analyzer.performance_results or \
                    not any(res.get('initialization', {}).get('insertion_evolution_data')
                            for res in analyzer.performance_results.values()):
                print("\nAs estruturas não foram inicializadas ou os dados de evolução não estão disponíveis.")
                print("Execute a inicialização (Opção 6 ou ao tentar acessar uma estrutura).")
            else:
                analyzer._generate_insertion_evolution_charts()

        elif escolha_main == '8':
            if not full_dataset:
                print("\nDataset está vazio. Não há estatísticas para mostrar.")
            else:
                MotoEstatisticas.gerar_graficos(full_dataset)

        elif escolha_main == '9':
            if not full_dataset:
                print("\nDataset está vazio. Não é possível simular tendências.")
            else:
                try:
                    anos_f = int(input("Quantos anos no futuro para prever? "))
                    if anos_f > 0:
                        MotoEstatisticas.prever_tendencias(full_dataset, anos_f)
                    else:
                        print("Número de anos deve ser positivo.")
                except ValueError:
                    print("Entrada inválida para anos.")

        elif escolha_main == '0':
            print("\nEncerrando sistema... Até logo! 👋")
            break
        else:
            print("\n❌ Opção inválida! Por favor, tente novamente.")

        if escolha_main != '0':  # Não pausar antes de sair
            input("\nPressione Enter para continuar...")


def main():
    """Função principal do programa."""
    print("=" * 50)
    print("Bem-vindo ao Sistema Avançado de Análise de Desempenho de Estruturas de Dados!")
    print("=" * 50)

    # Caminho para o dataset - AJUSTE SE NECESSÁRIO
    # Assumindo que o CSV está em uma pasta 'data' no mesmo nível de main.py
    # Se seu main.py está na raiz do projeto, e 'data' é uma subpasta:
    caminho_dataset = os.path.join('data', 'bike_sales_india.csv')
    # Se main.py está DENTRO de uma pasta e 'data' está um nível acima:
    # caminho_dataset = os.path.join('..', 'data', 'bike_sales_india.csv')

    if not os.path.exists(caminho_dataset):
        print(f"❌ ERRO CRÍTICO: Arquivo de dataset não encontrado em '{os.path.abspath(caminho_dataset)}'")
        print("Por favor, verifique o caminho e o nome do arquivo.")
        sys.exit(1)

    print(f"\nCarregando dataset de motocicletas de '{caminho_dataset}'...")
    # Adicione uma verificação aqui para garantir que DataHandler.ler_dataset é chamado corretamente
    # e que 'motos_dataset' recebe os dados esperados.
    motos_dataset = DataHandler.ler_dataset(caminho_dataset)

    if not motos_dataset:
        print("❌ ERRO CRÍTICO: Nenhum dado foi carregado do dataset ou o dataset está vazio.")
        print("O programa não pode continuar sem dados. Verifique o arquivo CSV e o DataHandler.")
        sys.exit(1)

    print(f"Dataset carregado com {len(motos_dataset)} registros.")

    # Inicializar o analisador com o dataset completo
    analyzer = StructureAnalyzer(motos_dataset)

    # Opcional: Inicializar estruturas aqui se desejar que estejam prontas desde o início.
    # print("\nRealizando inicialização prévia das estruturas...")
    # analyzer.initialize_all_structures(sample_size=1000) # Exemplo com amostra

    main_menu_loop(analyzer, motos_dataset)


if __name__ == "__main__":
    # Configurar o backend do matplotlib para evitar problemas em alguns ambientes
    # Tente 'TkAgg', 'Qt5Agg', ou 'Agg' (para não interativo/salvar em arquivo)
    try:
        import matplotlib

        # Tenta usar um backend que geralmente funciona bem
        matplotlib.use('TkAgg')  # Ou 'Qt5Agg'
    except ImportError:
        print("Aviso: Matplotlib.pyplot não pôde configurar backend TkAgg/Qt5Agg.")
    except Exception as e:
        print(f"Aviso: Problema ao configurar backend do Matplotlib: {e}. Os gráficos podem não ser exibidos.")

    main()