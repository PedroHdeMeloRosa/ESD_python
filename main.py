# main.py
import os
import sys
import time
import random
import tracemalloc
from typing import List, Dict, Any, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as mcm  # Para a nova API de colormaps

from modelos.data_handler import DataHandler
from Estruturas.linked_list import LinkedList
from Estruturas.avl_tree import AVLTree
from Estruturas.hash_table import HashTable
from Estruturas.bloom_filter import BloomFilter
from Estruturas.radix_tree import RadixTree
from Estruturas.b_tree_v2 import BTreeV2  # Usando BTreeV2
from ui.menu import menu_estrutura
from modelos.moto import Moto, MotoEstatisticas


class PerformanceMetrics:
    @staticmethod
    def measure(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.perf_counter()
        return {
            'time': (end_time - start_time) * 1000,
            'current_memory': current / 1024,
            'peak_memory': peak / 1024,
            'result': result
        }


class StructureAnalyzer:
    def __init__(self, motorcycles_dataset: List[Moto]):
        self.motorcycles_full_dataset: List[Moto] = motorcycles_dataset
        self.t_btree = 3
        self.structures_prototypes: Dict[str, Callable[[], Any]] = {
            'LinkedList': LinkedList,
            'AVLTree': AVLTree,
            'HashTable': lambda: HashTable(
                capacidade=max(101, len(motorcycles_dataset) // 10 if motorcycles_dataset else 101)),
            'BloomFilter': lambda: BloomFilter(
                num_itens_esperados=len(motorcycles_dataset) if motorcycles_dataset else 1000),
            'RadixTree': RadixTree,
            'BTree': lambda: BTreeV2(t=self.t_btree)  # Usando BTreeV2
        }
        self.initialized_structures: Dict[str, Any] = {}
        self.performance_results: Dict[str, Dict[str, Any]] = {}
        self.last_init_sample_size: Optional[int] = None

    def initialize_all_structures(self, sample_size: Optional[int] = None, verbose: bool = True) -> None:
        if not self.motorcycles_full_dataset:
            if verbose: print("Dataset de motocicletas está vazio. Não é possível inicializar estruturas.")
            return

        if sample_size is None:
            actual_sample_size = len(self.motorcycles_full_dataset)
            sample_to_insert = self.motorcycles_full_dataset
        else:
            actual_sample_size = min(sample_size, len(self.motorcycles_full_dataset))
            sample_to_insert = random.sample(self.motorcycles_full_dataset, actual_sample_size)

        self.last_init_sample_size = actual_sample_size

        if verbose: print(f"\n⏳ Inicializando estruturas com {actual_sample_size} motos e medindo desempenho...")

        self.initialized_structures.clear()
        self.performance_results.clear()

        for name, structure_constructor in self.structures_prototypes.items():
            if verbose: print(f"\n  Inicializando {name}...")
            structure_instance = structure_constructor()

            insertion_metrics_list = []
            total_insertion_time = 0
            max_peak_memory_during_init = 0

            for i, bike_to_insert in enumerate(sample_to_insert):
                if verbose and (i + 1) % (max(1, actual_sample_size // 10)) == 0:
                    print(f"    Inserindo item {i + 1}/{actual_sample_size} em {name}...")
                metrics = PerformanceMetrics.measure(structure_instance.inserir, bike_to_insert)
                insertion_metrics_list.append({'time': metrics['time'], 'peak_memory': metrics['peak_memory']})
                total_insertion_time += metrics['time']
                if metrics['peak_memory'] > max_peak_memory_during_init:
                    max_peak_memory_during_init = metrics['peak_memory']

            avg_insert_time = total_insertion_time / actual_sample_size if actual_sample_size > 0 else 0
            self.initialized_structures[name] = structure_instance
            self.performance_results[name] = {
                'initialization': {
                    'sample_size': actual_sample_size,
                    'total_time_ms': total_insertion_time,
                    'avg_insert_time_ms': avg_insert_time,
                    'peak_memory_init_kb': max_peak_memory_during_init,
                    'insertion_evolution_data': insertion_metrics_list
                }
            }
            if verbose: print(
                f"  {name} inicializado. Tempo médio de inserção: {avg_insert_time:.4f} ms. Pico de memória na init: {max_peak_memory_during_init:.2f} KB")

    def run_benchmark_operations(self, num_operations: int = 100, verbose: bool = True) -> None:
        if not self.initialized_structures:
            if verbose: print("Nenhuma estrutura inicializada. Execute `initialize_all_structures` primeiro.")
            return
        if not self.motorcycles_full_dataset:
            if verbose: print("Dataset vazio, não é possível executar benchmarks de operações.")
            return

        actual_num_operations = min(num_operations, len(self.motorcycles_full_dataset))
        if actual_num_operations == 0:
            if verbose: print("Nenhuma operação de benchmark a ser executada.")
            return

        sample_for_search_remove = random.sample(self.motorcycles_full_dataset, actual_num_operations)
        sample_for_new_insertion = [
            Moto(marca=f"MARCA_NOVA_{i}", nome=f"MODELO_NOVO_{i}", preco=10000 + i, revenda=8000 + i, ano=2025 + i)
            for i in range(actual_num_operations)
        ]

        if verbose: print(f"\n⚙️ Executando benchmark de operações ({actual_num_operations} de cada tipo)...")

        for name, structure in self.initialized_structures.items():
            if verbose: print(f"\n  Analisando {name}:")
            op_results_summary = {}

            if hasattr(structure, 'buscar'):
                search_times, search_mems = [], []
                for bike in sample_for_search_remove:
                    metrics = PerformanceMetrics.measure(structure.buscar, bike)
                    search_times.append(metrics['time'])
                    search_mems.append(metrics['peak_memory'])
                op_results_summary['search_avg_time_ms'] = sum(
                    search_times) / actual_num_operations if actual_num_operations else 0
                op_results_summary['search_peak_memory_kb'] = max(search_mems) if search_mems else 0
                if verbose: print(f"    Busca: Tempo médio {op_results_summary['search_avg_time_ms']:.4f} ms")

            if hasattr(structure, 'inserir'):
                insert_times, insert_mems = [], []
                for bike in sample_for_new_insertion:
                    metrics = PerformanceMetrics.measure(structure.inserir, bike)
                    insert_times.append(metrics['time'])
                    insert_mems.append(metrics['peak_memory'])
                op_results_summary['new_insertion_avg_time_ms'] = sum(
                    insert_times) / actual_num_operations if actual_num_operations else 0
                op_results_summary['new_insertion_peak_memory_kb'] = max(insert_mems) if insert_mems else 0
                if verbose: print(
                    f"    Nova Inserção: Tempo médio {op_results_summary['new_insertion_avg_time_ms']:.4f} ms")

            if hasattr(structure, 'remover') and name not in ["BloomFilter"]:
                remove_times, remove_mems = [], []
                for bike in sample_for_search_remove:
                    metrics = PerformanceMetrics.measure(structure.remover, bike)
                    remove_times.append(metrics['time'])
                    remove_mems.append(metrics['peak_memory'])
                op_results_summary['removal_avg_time_ms'] = sum(
                    remove_times) / actual_num_operations if actual_num_operations else 0
                op_results_summary['removal_peak_memory_kb'] = max(remove_mems) if remove_mems else 0
                if verbose: print(f"    Remoção: Tempo médio {op_results_summary['removal_avg_time_ms']:.4f} ms" +
                                  (" (Nota: Remoção em BTree é placeholder)" if name == "BTree" else ""))

            if name in self.performance_results:
                self.performance_results[name].update(op_results_summary)
            else:
                self.performance_results[name] = {'initialization': {}, **op_results_summary}

            if hasattr(structure, 'remover') and name not in ["BloomFilter", "BTree"]:
                for bike in sample_for_new_insertion:
                    structure.remover(bike)

    def _generate_performance_report_table(self) -> None:
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
        for name, metrics_dict in sorted(self.performance_results.items()):
            init_metrics = metrics_dict.get('initialization', {})
            print("{:<15} | {:<20.4f} | {:<20.4f} | {:<20.4f} | {:<20.4f} | {:<20.2f}".format(
                name,
                init_metrics.get('avg_insert_time_ms', 0),
                metrics_dict.get('search_avg_time_ms', 0),
                metrics_dict.get('new_insertion_avg_time_ms', 0),
                metrics_dict.get('removal_avg_time_ms', 0),
                init_metrics.get('peak_memory_init_kb', 0)
            ))
        print("=" * 120)

    def _generate_comparison_charts(self) -> None:
        if not self.performance_results:
            print("Nenhum resultado de performance para gerar gráficos de comparação.")
            return
        names = list(self.performance_results.keys())
        if not names:
            print("Nomes de estruturas vazios para gráficos de comparação.")
            return

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        # Gráfico 1: Comparação de Tempos Médios das Operações
        fig1 = None  # Inicializa para o bloco finally
        try:
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            operations = ['initialization_avg_insert', 'search_avg', 'new_insertion_avg', 'removal_avg']
            op_labels = ['Init Ins. Média', 'Busca Média', 'Nova Ins. Média', 'Remoção Média']
            num_ops_to_plot = len(operations)

            try:
                cmap = mcm.get_cmap('viridis')
                if hasattr(cmap, 'N') and cmap.N < num_ops_to_plot:
                    cmap = mcm.get_cmap('tab20')
                colors_list = [cmap(i / num_ops_to_plot) for i in range(num_ops_to_plot)]
            except Exception:
                default_colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral', 'cyan', 'magenta', 'yellow']
                colors_list = [default_colors[i % len(default_colors)] for i in range(num_ops_to_plot)]

            bar_width = 0.8 / (num_ops_to_plot + 0.5)
            index = np.arange(len(names))

            for i, op_key_suffix in enumerate(operations):
                if op_key_suffix == 'initialization_avg_insert':
                    times = [self.performance_results[n].get('initialization', {}).get('avg_insert_time_ms', 0) for n in
                             names]
                else:
                    times = [self.performance_results[n].get(f'{op_key_suffix}_time_ms', 0) for n in names]

                bar_position = index - (bar_width * num_ops_to_plot / 2) + (i * bar_width) + (bar_width / 2)
                ax1.bar(bar_position, times, bar_width, label=op_labels[i], color=colors_list[i])

            ax1.set_title('Comparação de Tempos Médios das Operações', fontsize=16)
            ax1.set_ylabel('Tempo Médio (ms)', fontsize=13)
            ax1.set_xlabel('Estrutura de Dados', fontsize=13)
            ax1.set_xticks(index)
            ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
            ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            print("\nExibindo gráfico de comparação de tempos... (Feche a janela para continuar)")
            plt.show()  # BLOQUEANTE
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de comparação de tempos: {e}")
        finally:
            if fig1:
                plt.close(fig1)

        # Gráfico 2: Uso de Memória de Pico na Inicialização
        fig2 = None  # Inicializa para o bloco finally
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            memories = [self.performance_results[n].get('initialization', {}).get('peak_memory_init_kb', 0) for n in
                        names]

            try:
                bar_colors = [mcm.get_cmap('Pastel2')(i / len(names)) for i in range(len(names))]
            except:
                bar_colors = 'mediumpurple'

            ax2.bar(names, memories, color=bar_colors, alpha=0.75, edgecolor='black')
            ax2.set_title('Uso de Memória de Pico Durante a Inicialização', fontsize=16)
            ax2.set_ylabel('Memória de Pico (KB)', fontsize=13)
            ax2.set_xlabel('Estrutura de Dados', fontsize=13)

            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=11)

            ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
            plt.tight_layout()

            print("\nExibindo gráfico de comparação de memória... (Feche a janela para continuar)")
            plt.show()  # BLOQUEANTE
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de comparação de memória: {e}")
        finally:
            if fig2:
                plt.close(fig2)

    def _generate_insertion_evolution_charts(self) -> None:
        if not self.performance_results:
            print("Nenhum resultado de inicialização para gerar gráficos de evolução.")
            return

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        # Gráfico 1: Evolução do Tempo de Inserção
        fig_time = None  # Inicializa para o bloco finally
        try:
            fig_time, ax_time = plt.subplots(figsize=(12, 7))
            ax_time.set_title('Evolução do Tempo de Inserção Individual Durante a Inicialização', fontsize=15)
            ax_time.set_xlabel('Número da Operação de Inserção', fontsize=12)
            ax_time.set_ylabel('Tempo de Inserção (ms)', fontsize=12)

            for name, metrics in sorted(self.performance_results.items()):
                init_data = metrics.get('initialization', {}).get('insertion_evolution_data', [])
                if init_data:
                    times = [m['time'] for m in init_data]
                    avg_t = sum(times) / len(times) if times else 0
                    ax_time.plot(times, label=f'{name} (média: {avg_t:.3f} ms)', marker='.', linestyle='-', alpha=0.6,
                                 markersize=2)

            ax_time.legend(loc='upper right')
            ax_time.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()

            print("\nExibindo gráfico de evolução do tempo de inserção... (Feche a janela para continuar)")
            plt.show()  # BLOQUEANTE
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de evolução de tempo: {e}")
        finally:
            if fig_time:
                plt.close(fig_time)

        # Gráfico 2: Evolução do Pico de Memória por Inserção
        fig_mem = None  # Inicializa para o bloco finally
        try:
            fig_mem, ax_mem = plt.subplots(figsize=(12, 7))
            ax_mem.set_title('Evolução do Pico de Memória por Inserção Durante a Inicialização', fontsize=15)
            ax_mem.set_xlabel('Número da Operação de Inserção', fontsize=12)
            ax_mem.set_ylabel('Pico de Memória da Inserção (KB)', fontsize=12)

            for name, metrics in sorted(self.performance_results.items()):
                init_data = metrics.get('initialization', {}).get('insertion_evolution_data', [])
                if init_data:
                    memories = [m['peak_memory'] for m in init_data]
                    max_m = max(memories) if memories else 0
                    ax_mem.plot(memories, label=f'{name} (pico max: {max_m:.2f} KB)', marker='.', linestyle='-',
                                alpha=0.6, markersize=2)

            ax_mem.legend(loc='upper right')
            ax_mem.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()

            print("\nExibindo gráfico de evolução da memória de inserção... (Feche a janela para continuar)")
            plt.show()  # BLOQUEANTE
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de evolução de memória: {e}")
        finally:
            if fig_mem:
                plt.close(fig_mem)

    def run_full_analysis_suite(self, init_sample_size: Optional[int] = 1000, benchmark_ops_count: int = 100):
        print("\n🚀 INICIANDO SUÍTE COMPLETA DE ANÁLISE DE DESEMPENHO 🚀")
        self.initialize_all_structures(sample_size=init_sample_size)
        self.run_benchmark_operations(num_operations=benchmark_ops_count)
        print("\n📋 Gerando Relatórios e Gráficos...")
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()
        print("\n🏁 Análise Completa Concluída! 🏁")


def main_menu_loop(analyzer: StructureAnalyzer, full_dataset: List[Moto]):
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
        print("6. Árvore B")
        print("--- ANÁLISE E COMPARAÇÃO ---")
        print("7. Executar Suíte Completa de Análise")
        print("8. Gerar Gráficos de Evolução da Inicialização")
        print("--- ANÁLISE DO DATASET ---")
        print("9. Estatísticas Gerais do Dataset e Gráficos")
        print("10. Simular Tendências Futuras do Dataset")
        print("--- SAIR ---")
        print("0. Sair do Sistema")

        escolha_main = input("\nEscolha uma opção: ").strip()

        if escolha_main in ['1', '2', '3', '4', '5', '6']:
            struct_map = {
                '1': ('LinkedList', "LISTA ENCADEADA"),
                '2': ('AVLTree', "ÁRVORE AVL"),
                '3': ('HashTable', "TABELA HASH"),
                '4': ('BloomFilter', "BLOOM FILTER"),
                '5': ('RadixTree', "RADIX TREE"),
                '6': ('BTree', "ÁRVORE B")
            }
            struct_key, struct_name_display = struct_map[escolha_main]

            if not analyzer.initialized_structures.get(struct_key):
                print(f"\nAVISO: A estrutura {struct_name_display} não está inicializada.")
                print("  Execute a 'Suíte Completa de Análise' (opção 7) primeiro, ou")
                default_sample = analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000
                if input(
                        f"  deseja inicializar TODAS as estruturas agora com uma amostra ({default_sample})? (s/n): ").lower() == 's':
                    analyzer.initialize_all_structures(sample_size=default_sample, verbose=True)

                if not analyzer.initialized_structures.get(struct_key):
                    print(f"Estrutura {struct_name_display} ainda não inicializada. Voltando ao menu.")
                    continue

            menu_estrutura(analyzer.initialized_structures[struct_key],
                           struct_name_display,
                           analyzer.motorcycles_full_dataset)

        elif escolha_main == '7':
            try:
                default_init_size = analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000
                init_s_str = input(f"Tamanho da amostra para inicialização (padrão {default_init_size}): ").strip()
                init_sample = int(init_s_str) if init_s_str else default_init_size

                bench_ops_s = input("Número de operações para benchmarks (padrão 100): ").strip()
                bench_ops = int(bench_ops_s) if bench_ops_s else 100
                analyzer.run_full_analysis_suite(init_sample_size=init_sample, benchmark_ops_count=bench_ops)
            except ValueError:
                print("Entrada inválida. Usando padrões para análise.")
                analyzer.run_full_analysis_suite()

        elif escolha_main == '8':
            if not analyzer.performance_results or not any(
                    res.get('initialization', {}).get('insertion_evolution_data') for res in
                    analyzer.performance_results.values()):
                print("\nAs estruturas não foram inicializadas com dados de evolução.")
                print("Execute a 'Suíte Completa de Análise' (opção 7) primeiro.")
            else:
                analyzer._generate_insertion_evolution_charts()

        elif escolha_main == '9':
            if not full_dataset:
                print("\nDataset está vazio.")
            else:
                MotoEstatisticas.gerar_graficos(full_dataset)

        elif escolha_main == '10':
            if not full_dataset:
                print("\nDataset está vazio.")
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

        if escolha_main != '0':
            input("\nPressione Enter para continuar...")


def main():
    print("=" * 50)
    print("Bem-vindo ao Sistema Avançado de Análise de Desempenho de Estruturas de Dados!")
    print("=" * 50)
    caminho_dataset = os.path.join('data', 'bike_sales_india.csv')
    if not os.path.exists(caminho_dataset):
        print(f"❌ ERRO CRÍTICO: Arquivo de dataset não encontrado em '{os.path.abspath(caminho_dataset)}'")
        sys.exit(1)
    print(f"\nCarregando dataset de motocicletas de '{caminho_dataset}'...")
    motos_dataset = DataHandler.ler_dataset(caminho_dataset)
    if not motos_dataset:
        print("❌ ERRO CRÍTICO: Nenhum dado foi carregado do dataset ou o dataset está vazio.")
        sys.exit(1)
    print(f"Dataset carregado com {len(motos_dataset)} registros.")
    analyzer = StructureAnalyzer(motos_dataset)

    if not analyzer.initialized_structures:
        print("\nNenhuma estrutura foi inicializada ainda.")
        if input(
                "Deseja realizar uma inicialização padrão de todas as estruturas com uma amostra (1000 motos) agora? (s/n): ").strip().lower() == 's':
            analyzer.initialize_all_structures(sample_size=1000, verbose=True)

    main_menu_loop(analyzer, motos_dataset)


if __name__ == "__main__":
    try:
        import matplotlib

        matplotlib.use('TkAgg')
    except Exception as e:
        # Permitir que o programa continue mesmo se o backend falhar, com um aviso.
        print(f"Aviso: Problema ao configurar backend 'TkAgg' do Matplotlib: {e}. "
              "Os gráficos podem não ser exibidos interativamente ou podem precisar de configuração manual do backend (ex: MPLBACKEND).")
    main()