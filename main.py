# main.py
import os
import sys
import time
import random
import tracemalloc
from typing import List, Dict, Any, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as mcm
import copy

# Importações de simulações
from simulacoes import restricao_dados
from simulacoes import restricao_processamento

from modelos.data_handler import DataHandler
from Estruturas.linked_list import LinkedList
from Estruturas.avl_tree import AVLTree
from Estruturas.hash_table import HashTable
from Estruturas.bloom_filter import BloomFilter
from Estruturas.radix_tree import RadixTree
from Estruturas.b_tree_v2 import BTreeV2
from ui.menu import menu_estrutura, submenu_testes_restricao
from modelos.moto import Moto, MotoEstatisticas


class PerformanceMetrics:
    @staticmethod
    def measure(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        restricao_processamento.executar_carga_computacional_extra()
        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        measured_time_ms = (time.perf_counter() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'time': measured_time_ms,
            'current_memory': current / 1024,
            'peak_memory': peak / 1024,
            'result': result
        }


class StructureAnalyzer:
    def __init__(self, motorcycles_dataset: List[Moto]):
        self.motorcycles_full_dataset_original: List[Moto] = motorcycles_dataset
        self.current_dataset_for_analysis: List[Moto] = copy.deepcopy(motorcycles_dataset)
        self.t_btree = 3
        self.structures_prototypes: Dict[str, Callable[[], Any]] = {
            'LinkedList': LinkedList, 'AVLTree': AVLTree,
            'HashTable': lambda: HashTable(capacidade=max(101,
                                                          len(self.current_dataset_for_analysis) // 10 if self.current_dataset_for_analysis and len(
                                                              self.current_dataset_for_analysis) > 0 else 101)),
            'BloomFilter': lambda: BloomFilter(
                num_itens_esperados=len(self.current_dataset_for_analysis) if self.current_dataset_for_analysis and len(
                    self.current_dataset_for_analysis) > 0 else 1000),
            'RadixTree': RadixTree, 'BTree': lambda: BTreeV2(t=self.t_btree)
        }
        self.initialized_structures: Dict[str, Any] = {}
        self.performance_results: Dict[str, Dict[str, Any]] = {}
        self.last_init_sample_size: Optional[int] = None
        self.scalability_results: Dict[str, List[Dict[str, Any]]] = {}
        self.active_restriction_name: Optional[str] = None

    def _prepare_dataset_for_analysis(self, restriction_config: Optional[Dict[str, Any]] = None) -> None:
        self.current_dataset_for_analysis = copy.deepcopy(self.motorcycles_full_dataset_original)
        if restriction_config:
            self.active_restriction_name = restriction_config.get("nome", "RestricaoDesconhecida")
            tipo = restriction_config.get("tipo")
            params = restriction_config.get("params", {})
            print(f"\nINFO: Aplicando restrição de dados: {self.active_restriction_name} com params {params}")
            if tipo == "corromper_precos":
                self.current_dataset_for_analysis = restricao_dados.corromper_precos_aleatoriamente(
                    self.current_dataset_for_analysis, **params)
            elif tipo == "anos_anomalos":
                self.current_dataset_for_analysis = restricao_dados.introduzir_anos_anomalos(
                    self.current_dataset_for_analysis, **params)
            else:
                print(f"AVISO: Tipo de restrição de dados '{tipo}' não reconhecido.")
        else:
            self.active_restriction_name = None

    def initialize_all_structures(self, sample_size: Optional[int] = None, verbose: bool = True) -> None:
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual está vazio. Não é possível inicializar."); return
        if sample_size is None:
            actual_sample_size = len(self.current_dataset_for_analysis)
            sample_to_insert = self.current_dataset_for_analysis
        else:
            actual_sample_size = min(sample_size, len(self.current_dataset_for_analysis))
            if actual_sample_size <= 0:
                if verbose: print(
                    f"AVISO: Tamanho da amostra ({actual_sample_size}) inválido. Usando dataset completo ou mínimo se necessário.");
                actual_sample_size = len(self.current_dataset_for_analysis) if len(
                    self.current_dataset_for_analysis) > 0 else 1
                if not self.current_dataset_for_analysis and actual_sample_size == 1:
                    if verbose: print("Dataset de análise realmente vazio. Cancelando inicialização."); return
            sample_to_insert = random.sample(self.current_dataset_for_analysis,
                                             actual_sample_size) if actual_sample_size > 0 else []
        self.last_init_sample_size = actual_sample_size
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(
            f"\n⏳ Inicializando estruturas com {actual_sample_size} motos {dataset_info} e medindo desempenho...")
        self.initialized_structures.clear();
        self.performance_results.clear()
        for name, structure_constructor_factory in self.structures_prototypes.items():
            if verbose: print(f"\n  Inicializando {name}...")
            structure_instance = structure_constructor_factory()
            insertion_metrics_list = [];
            total_insertion_time = 0.0;
            max_peak_memory_during_init = 0.0
            if actual_sample_size > 0 and sample_to_insert:  # Adicionado check para sample_to_insert
                for i, bike_to_insert in enumerate(sample_to_insert):
                    if verbose and (i + 1) % (max(1, actual_sample_size // 10)) == 0:
                        print(f"    Inserindo item {i + 1}/{actual_sample_size} em {name}...")
                    metrics = PerformanceMetrics.measure(structure_instance.inserir, bike_to_insert)
                    insertion_metrics_list.append({'time': metrics['time'], 'peak_memory': metrics['peak_memory']})
                    total_insertion_time += metrics['time']
                    if metrics['peak_memory'] > max_peak_memory_during_init: max_peak_memory_during_init = metrics[
                        'peak_memory']
            avg_insert_time = total_insertion_time / actual_sample_size if actual_sample_size > 0 else 0.0
            self.initialized_structures[name] = structure_instance
            self.performance_results[name] = {'initialization': {
                'sample_size': actual_sample_size, 'total_time_ms': total_insertion_time,
                'avg_insert_time_ms': avg_insert_time, 'peak_memory_init_kb': max_peak_memory_during_init,
                'insertion_evolution_data': insertion_metrics_list}}
            if verbose: print(
                f"  {name} inicializado. Média inserção: {avg_insert_time:.4f} ms. Pico Memória: {max_peak_memory_during_init:.2f} KB")

    def run_benchmark_operations(self, num_operations: int = 100, verbose: bool = True) -> None:
        if not self.initialized_structures:
            if verbose: print("Nenhuma estrutura inicializada."); return
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual vazio."); return
        actual_num_operations = min(num_operations, len(self.current_dataset_for_analysis))
        if actual_num_operations <= 0:
            if verbose: print(f"Nenhuma operação de benchmark a ser executada (n_ops={actual_num_operations})."); return
        sample_for_search_remove = random.sample(self.current_dataset_for_analysis, actual_num_operations)
        sample_for_new_insertion = [Moto(f"MARCA_NOVA_{i}", f"MODELO_NOVO_{i}", 10000 + i, 8000 + i, 2025 + i) for i in
                                    range(actual_num_operations)]
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n⚙️ Executando benchmark de ops ({actual_num_operations} de cada) {dataset_info}...")
        for name, structure in self.initialized_structures.items():
            if verbose: print(f"\n  Analisando {name}:")
            op_results_summary = {}
            if hasattr(structure, 'buscar'):
                s_t, s_m = [], [];
                for b in sample_for_search_remove: m = PerformanceMetrics.measure(structure.buscar, b); s_t.append(
                    m['time']); s_m.append(m['peak_memory'])
                op_results_summary['search_avg_time_ms'] = sum(
                    s_t) / actual_num_operations if actual_num_operations else 0.0;
                op_results_summary['search_peak_memory_kb'] = max(s_m) if s_m else 0.0
                if verbose: print(f"    Busca: Tempo médio {op_results_summary['search_avg_time_ms']:.4f} ms")
            if hasattr(structure, 'inserir'):
                i_t, i_m = [], [];
                for b in sample_for_new_insertion: m = PerformanceMetrics.measure(structure.inserir, b); i_t.append(
                    m['time']); i_m.append(m['peak_memory'])
                op_results_summary['new_insertion_avg_time_ms'] = sum(
                    i_t) / actual_num_operations if actual_num_operations else 0.0;
                op_results_summary['new_insertion_peak_memory_kb'] = max(i_m) if i_m else 0.0
                if verbose: print(
                    f"    Nova Inserção: Tempo médio {op_results_summary['new_insertion_avg_time_ms']:.4f} ms")
            if hasattr(structure, 'remover') and name not in ["BloomFilter"]:
                r_t, r_m = [], [];
                for b in sample_for_search_remove: m = PerformanceMetrics.measure(structure.remover, b); r_t.append(
                    m['time']); r_m.append(m['peak_memory'])
                op_results_summary['removal_avg_time_ms'] = sum(
                    r_t) / actual_num_operations if actual_num_operations else 0.0;
                op_results_summary['removal_peak_memory_kb'] = max(r_m) if r_m else 0.0
                if verbose: print(f"    Remoção: Tempo médio {op_results_summary['removal_avg_time_ms']:.4f} ms" + (
                    " (BTree placeholder)" if name == "BTree" else ""))
            if name == 'HashTable' and hasattr(structure, 'obter_estatisticas_colisao'):
                cs = structure.obter_estatisticas_colisao();
                op_results_summary['HashTable_collision_stats'] = cs
                if verbose: print(
                    f"    Stats Colisão HT: Fator Carga={cs.get('fator_carga_real', 0.0):.2f}, Max Bucket={cs.get('max_comprimento_bucket', 0)}")
            if name not in self.performance_results: self.performance_results[name] = {'initialization': {}}
            self.performance_results[name].update(op_results_summary)
            if hasattr(structure, 'remover') and name not in ["BloomFilter", "BTree"]:
                for b_rem in sample_for_new_insertion: structure.remover(b_rem)

    def run_combined_latency_benchmark(self, num_workloads: int = 50, num_ops_per_workload: int = 3,
                                       verbose: bool = True) -> None:
        if not self.initialized_structures:
            if verbose: print("Nenhuma estrutura inicializada para o benchmark de latência."); return
        min_dataset_size_needed = num_ops_per_workload
        if not self.current_dataset_for_analysis or len(self.current_dataset_for_analysis) < min_dataset_size_needed:
            if verbose: print(
                f"Dataset ({len(self.current_dataset_for_analysis)}) pequeno para {num_ops_per_workload} ops/workload. Cancelando."); return
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n⏱️  BENCHMARK DE LATÊNCIA COMBINADA ({num_workloads} workloads) {dataset_info}...")
        base_price_new = 200000
        workload_bikes = [
            Moto(f"WL_Marca_{i}", f"WL_Modelo_{i}", base_price_new + i, base_price_new * 0.8 + i * 0.8, 2028 + (i % 3))
            for i in range(num_workloads)]
        for name, structure_instance_original in self.initialized_structures.items():
            if verbose: print(f"  Testando latência para: {name}")
            workload_times = [];
            structure_under_test = structure_instance_original
            for i in range(num_workloads):
                bike_to_insert_wl = workload_bikes[i]
                items_for_search_wl = []
                if hasattr(structure_under_test, 'buscar'):
                    search_pool = self.current_dataset_for_analysis
                    num_additional_searches = max(0, num_ops_per_workload - 1)
                    if num_additional_searches > 0 and len(search_pool) >= num_additional_searches:
                        items_for_search_wl.extend(random.sample(search_pool, num_additional_searches))
                    elif len(search_pool) > 0:
                        items_for_search_wl.extend(random.sample(search_pool, len(search_pool)))

                def workload_sequence_runner():
                    if hasattr(structure_under_test, 'inserir'): structure_under_test.inserir(bike_to_insert_wl)
                    if hasattr(structure_under_test, 'buscar'):
                        structure_under_test.buscar(bike_to_insert_wl)
                        for s_bike in items_for_search_wl: structure_under_test.buscar(s_bike)
                    if hasattr(structure_under_test, 'remover') and name not in ["BloomFilter", "BTree"]:
                        structure_under_test.remover(bike_to_insert_wl)

                metrics = PerformanceMetrics.measure(workload_sequence_runner)
                workload_times.append(metrics['time'])
            if workload_times:
                avg_lat_ms = sum(workload_times) / len(workload_times)
                if verbose: print(f"    Latência Média por Workload: {avg_lat_ms:.4f} ms")
                if name not in self.performance_results: self.performance_results[name] = {}
                self.performance_results[name]['combined_latency_avg_ms'] = avg_lat_ms
            else:
                if verbose: print(f"    Nenhum workload de latência executado para {name}.")
        if verbose: print("\n⏱️  Benchmark de Latência Combinada Concluído! ⏱️")

    def run_random_access_benchmark(self, num_accesses: int = 100, verbose: bool = True) -> None:
        if not self.initialized_structures:
            if verbose: print("Nenhuma estrutura inicializada. Execute a inicialização primeiro."); return
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual vazio. Não é possível testar acesso aleatório."); return
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n🔑 INICIANDO BENCHMARK DE ACESSO ALEATÓRIO ({num_accesses} acessos) {dataset_info}...")
        for name, structure in self.initialized_structures.items():
            if not hasattr(structure, 'buscar'):
                if verbose: print(f"  {name}: Busca não suportada, pulando teste de acesso aleatório.")
                if name not in self.performance_results: self.performance_results[name] = {}
                self.performance_results[name]['random_access_avg_time_ms'] = 0.0
                continue
            if verbose: print(f"  Testando acesso aleatório para: {name}")
            if not self.current_dataset_for_analysis:
                if verbose: print(f"    {name}: Dataset de análise atual vazio, não pode selecionar itens."); continue
            actual_num_accesses = min(num_accesses, len(self.current_dataset_for_analysis))
            if actual_num_accesses == 0:
                if verbose: print(
                    f"    {name}: Não há itens suficientes no dataset para testar acesso ({len(self.current_dataset_for_analysis)}).");
                if name not in self.performance_results: self.performance_results[name] = {}
                self.performance_results[name]['random_access_avg_time_ms'] = 0.0
                continue
            items_to_access = random.sample(self.current_dataset_for_analysis, actual_num_accesses)
            access_times = []
            for item_to_access in items_to_access:
                metrics = PerformanceMetrics.measure(structure.buscar, item_to_access)
                access_times.append(metrics['time'])
            avg_access_time_ms = sum(access_times) / actual_num_accesses if actual_num_accesses > 0 else 0.0
            if verbose: print(f"    Tempo Médio de Acesso Aleatório: {avg_access_time_ms:.4f} ms")
            if name not in self.performance_results: self.performance_results[name] = {}
            self.performance_results[name]['random_access_avg_time_ms'] = avg_access_time_ms
        if verbose: print("🔑 Benchmark de Acesso Aleatório Concluído!")

    def _generate_performance_report_table(self) -> None:
        report_title = self.active_restriction_name.upper() if self.active_restriction_name else "BENCHMARKS PADRÃO"
        print(f"\n\n📊 RELATÓRIO DE DESEMPENHO ({report_title}) 📊")
        if not self.performance_results: print("Nenhum resultado para gerar relatório."); return
        table_width = 165
        print("=" * table_width)
        header = "{:<15} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<22} | {:<22}".format(
            "Estrutura", "Init Avg Ins (ms)", "Search Avg (ms)", "New Ins Avg (ms)",
            "Removal Avg (ms)", "Init Peak Mem (KB)", "Avg Workload Lat (ms)", "Avg Random Acc (ms)")
        print(header);
        print("-" * table_width)
        for name, mets in sorted(self.performance_results.items()):
            init_m = mets.get('initialization', {})
            print(
                f"{name:<15} | {init_m.get('avg_insert_time_ms', 0.0):<20.4f} | {mets.get('search_avg_time_ms', 0.0):<20.4f} | "
                f"{mets.get('new_insertion_avg_time_ms', 0.0):<20.4f} | {mets.get('removal_avg_time_ms', 0.0):<20.4f} | "
                f"{init_m.get('peak_memory_init_kb', 0.0):<20.2f} | {mets.get('combined_latency_avg_ms', 0.0):<22.4f} | "
                f"{mets.get('random_access_avg_time_ms', 0.0):<22.4f}")
        print("=" * table_width)
        if 'HashTable' in self.performance_results and 'HashTable_collision_stats' in self.performance_results.get(
                'HashTable', {}):
            ht_s = self.performance_results['HashTable']['HashTable_collision_stats']
            ht_i = self.initialized_structures.get('HashTable');
            cap = ht_i.capacidade if ht_i else "N/A"
            print("\n--- Stats Colisão HashTable ---");
            print(f"  Fator Carga Real: {ht_s.get('fator_carga_real', 0.0):.3f}");
            print(f"  Buckets Vazios: {ht_s.get('num_buckets_vazios', 0)} / {cap}")
            print(
                f"  Buckets c/ Colisão (ocupados): {ht_s.get('num_buckets_com_colisao', 0)}/{ht_s.get('num_buckets_ocupados', 0)} ({ht_s.get('percent_buckets_com_colisao_de_ocupados', 0.0):.2f}%)")
            print(f"  Max Compr Bucket: {ht_s.get('max_comprimento_bucket', 0)}");
            print(f"  Compr Médio (Ocupados): {ht_s.get('avg_comprimento_bucket_ocupado', 0.0):.2f}");
            print("=" * 70)

    def _generate_comparison_charts(self) -> None:
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.performance_results: print("Nenhum resultado para gráficos de comparação."); return
        names = list(self.performance_results.keys());
        if not names: print("Nomes de estruturas vazios para gráficos."); return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        fig1 = None
        try:
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            operations = ['initialization_avg_insert', 'search_avg', 'new_insertion_avg', 'removal_avg',
                          'random_access_avg']  # Adicionada random_access_avg
            op_labels = ['Init Ins. Média', 'Busca (Amostra)', 'Nova Ins. Média', 'Remoção Média',
                         'Acesso Aleatório Médio']  # Adicionado rótulo
            n_ops = len(operations)
            try:
                cmap = mcm.get_cmap('viridis'); colors_list = [cmap(i / n_ops) for i in range(n_ops)]
            except:
                colors_list = ['skyblue', 'lightgreen', 'salmon', 'gold', 'violet'][
                              :n_ops]  # Adicionada cor para 5a operação
            bar_w = 0.8 / (n_ops + 0.5);
            idx = np.arange(len(names))
            for i, op_key in enumerate(operations):
                key_for_results = f'{op_key}_time_ms' if op_key != 'initialization_avg_insert' else 'avg_insert_time_ms'
                if op_key == 'initialization_avg_insert':
                    data_source = {n: d.get('initialization', {}) for n, d in self.performance_results.items()}
                else:
                    data_source = self.performance_results
                times = [data_source.get(n, {}).get(key_for_results, 0.0) for n in names]
                pos = idx - (bar_w * n_ops / 2) + (i * bar_w) + (bar_w / 2);
                ax1.bar(pos, times, bar_w, label=op_labels[i], color=colors_list[i])
            ax1.set_title(f'Comparação de Tempos Médios das Operações{chart_suffix}', fontsize=16);
            ax1.set_ylabel('Tempo Médio (ms)', fontsize=13)
            ax1.set_xlabel('Estrutura de Dados', fontsize=13);
            ax1.set_xticks(idx);
            ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1));
            ax1.grid(True, axis='y', ls=':', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1]);
            print(f"\nExibindo gráfico de comparação de tempos{chart_suffix}... (Feche a janela para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de comparação de tempos: {e}")
        finally:
            if fig1 is not None: plt.close(fig1)
        fig2 = None
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            memories = [self.performance_results.get(n, {}).get('initialization', {}).get('peak_memory_init_kb', 0.0)
                        for n in names]
            try:
                bar_colors = [mcm.get_cmap('Pastel2')(i / len(names)) for i in range(len(names))]
            except:
                bar_colors = 'mediumpurple'
            ax2.bar(names, memories, color=bar_colors, alpha=0.75, edgecolor='black')
            ax2.set_title(f'Uso de Memória de Pico na Inicialização{chart_suffix}', fontsize=16);
            ax2.set_ylabel('Memória (KB)', fontsize=13)
            ax2.set_xlabel('Estrutura', fontsize=13);
            ax2.set_xticks(range(len(names)));
            ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax2.grid(True, axis='y', ls=':', alpha=0.6);
            plt.tight_layout();
            print(f"\nExibindo gráfico de comparação de memória{chart_suffix}... (Feche a janela para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de comparação de memória: {e}")
        finally:
            if fig2 is not None: plt.close(fig2)

    def _generate_insertion_evolution_charts(self) -> None:
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.performance_results: print("Nenhum resultado para gráficos de evolução."); return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        fig_t = None
        try:
            fig_t, ax_t = plt.subplots(figsize=(12, 7));
            ax_t.set_title(f'Evolução do Tempo de Inserção{chart_suffix}', fontsize=15)
            ax_t.set_xlabel('Número da Operação de Inserção', fontsize=12);
            ax_t.set_ylabel('Tempo (ms)', fontsize=12)
            for name, mets in sorted(self.performance_results.items()):
                init_d = mets.get('initialization', {}).get('insertion_evolution_data', []);
                if init_d: times = [m.get('time', 0.0) for m in init_d]; avg_t = sum(times) / len(
                    times) if times else 0.0; ax_t.plot(times, label=f'{name} (média:{avg_t:.3f}ms)', marker='.',
                                                        ls='-', alpha=0.6, ms=2)
            ax_t.legend(loc='upper right');
            ax_t.grid(True, ls=':', alpha=0.7);
            plt.tight_layout();
            print(
                f"\nExibindo gráfico de evolução do tempo de inserção{chart_suffix}... (Feche a janela para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de evolução de tempo: {e}")
        finally:
            if fig_t is not None: plt.close(fig_t)
        fig_m = None
        try:
            fig_m, ax_m = plt.subplots(figsize=(12, 7));
            ax_m.set_title(f'Evolução do Pico de Memória na Inserção{chart_suffix}', fontsize=15)
            ax_m.set_xlabel('Número da Operação de Inserção', fontsize=12);
            ax_m.set_ylabel('Memória (KB)', fontsize=12)
            for name, mets in sorted(self.performance_results.items()):
                init_d = mets.get('initialization', {}).get('insertion_evolution_data', [])
                if init_d: mems = [m.get('peak_memory', 0.0) for m in init_d]; max_m = max(
                    mems) if mems else 0.0; ax_m.plot(mems, label=f'{name} (pico max:{max_m:.2f}KB)', marker='.',
                                                      ls='-', alpha=0.6, ms=2)
            ax_m.legend(loc='upper right');
            ax_m.grid(True, ls=':', alpha=0.7);
            plt.tight_layout();
            print(
                f"\nExibindo gráfico de evolução da memória de inserção{chart_suffix}... (Feche a janela para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro ao gerar/exibir gráfico de evolução de memória: {e}")
        finally:
            if fig_m is not None: plt.close(fig_m)

    def run_scalability_tests(self, sizes_to_test: Optional[List[int]] = None, num_searches_per_size: int = 100,
                              verbose: bool = True) -> None:
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual vazio. Testes de escalabilidade cancelados."); return
        if sizes_to_test is None:
            base_s = [100, 500, 1000, 2500, 5000, 7500];
            max_ds_s = len(self.current_dataset_for_analysis)
            sizes_to_test = [s for s in base_s if s <= max_ds_s]
            if max_ds_s not in sizes_to_test and (
                    not sizes_to_test or max_ds_s > sizes_to_test[-1]): sizes_to_test.append(max_ds_s)
            if not sizes_to_test: sizes_to_test = [max_ds_s] if max_ds_s > 0 else [10]
            sizes_to_test = sorted(list(set(s for s in sizes_to_test if s > 0)))
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n🔬 INICIANDO TESTES DE ESCALABILIDADE {dataset_info} para N = {sizes_to_test} ...")
        self.scalability_results.clear()
        for n_size in sizes_to_test:
            if n_size <= 0: continue
            if n_size > len(self.current_dataset_for_analysis):
                if verbose: print(
                    f"AVISO: N={n_size} > dataset atual ({len(self.current_dataset_for_analysis)}). Pulando."); continue
            if verbose: print(f"\n  --- Testando com N = {n_size} ---")
            curr_sample = random.sample(self.current_dataset_for_analysis, n_size)
            for s_name, constructor_factory in self.structures_prototypes.items():
                if verbose: print(f"    Testando estrutura: {s_name}")
                instance = constructor_factory()
                tracemalloc.start();
                t_start_ins = time.perf_counter()
                for bike in curr_sample: instance.inserir(bike)
                t_total_ins_ms = (time.perf_counter() - t_start_ins) * 1000;
                avg_ins_ms = t_total_ins_ms / n_size if n_size else 0.0
                _, peak_mem_kb = tracemalloc.get_traced_memory();
                tracemalloc.stop();
                peak_mem_kb /= 1024
                if verbose: print(
                    f"      Ins ({n_size}): Total={t_total_ins_ms:.2f}ms, Média={avg_ins_ms:.4f}ms/item, Pico Mem={peak_mem_kb:.2f}KB")
                avg_search_ms = 0.0
                if hasattr(instance, 'buscar'):
                    n_searches = min(num_searches_per_size, n_size)
                    if n_searches > 0:
                        search_samp = random.sample(curr_sample, n_searches);
                        search_t_list = []
                        for b_search in search_samp: t_s = time.perf_counter();instance.buscar(
                            b_search);search_t_list.append((time.perf_counter() - t_s) * 1000)
                        avg_search_ms = sum(search_t_list) / n_searches if n_searches else 0.0
                        if verbose: print(f"      Busca ({n_searches}): Média={avg_search_ms:.4f}ms/item")
                    else:
                        if verbose: print("      Busca: Nenhuma busca executada.")
                else:
                    if verbose: print(f"      Busca: Não suportada por {s_name}.")
                if s_name not in self.scalability_results: self.scalability_results[s_name] = []
                self.scalability_results[s_name].append(
                    {'N': n_size, 'avg_insert_time_ms': avg_ins_ms, 'peak_memory_kb': peak_mem_kb,
                     'avg_search_time_ms': avg_search_ms})
        if verbose: print("\n🔬 Testes de Escalabilidade Concluídos! 🔬")

    def _generate_scalability_charts(self, log_scale_plots: bool = False) -> None:
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.scalability_results: print("Nenhum resultado para gráficos de escalabilidade."); return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        metrics_to_plot = [('avg_insert_time_ms', 'Tempo Médio Inserção (ms) vs. N', 'Tempo Médio (ms)'),
                           ('peak_memory_kb', 'Pico Memória (KB) vs. N', 'Memória (KB)'),
                           ('avg_search_time_ms', 'Tempo Médio Busca (ms) vs. N', 'Tempo Médio (ms)')]
        for metric, title_base, ylabel in metrics_to_plot:
            fig = None
            try:
                title = title_base + chart_suffix;
                fig, ax = plt.subplots(figsize=(12, 7));
                ax.set_title(title, fontsize=15)
                ax.set_xlabel('# Elementos (N)', fontsize=12);
                ax.set_ylabel(ylabel, fontsize=12);
                has_data = False
                for s_name, res_list in sorted(self.scalability_results.items()):
                    if not res_list: continue
                    s_res = sorted(res_list, key=lambda x: x['N']);
                    n_vals = [r['N'] for r in s_res];
                    m_vals = [r[metric] for r in s_res]
                    if not any(v > 1e-5 for v in m_vals) and metric != 'peak_memory_kb':
                        if not (metric == 'avg_search_time_ms' and not hasattr(self.structures_prototypes[s_name](),
                                                                               'buscar')): pass
                        continue
                    has_data = True;
                    ax.plot(n_vals, m_vals, marker='o', ls='-', lw=2, ms=5, label=s_name)
                if not has_data: print(f"Nenhum dado para plotar: {title}");
                if fig and not has_data: plt.close(fig); continue
                if log_scale_plots and "Tempo" in ylabel:
                    # Verifica se todos os valores a serem plotados no eixo log são positivos
                    all_positive = True
                    for s_name, res_list in self.scalability_results.items():
                        if any(r[metric] <= 1e-9 for r in res_list if
                               r[metric] is not None):  # Usar uma pequena tolerância
                            all_positive = False;
                            break
                    if all_positive and any(any(r[metric] > 1e-9 for r in res_list) for res_list in
                                            self.scalability_results.values()):  # E há dados > 0
                        ax.set_yscale('log');
                        ax.set_ylabel(f"{ylabel} (Escala Log)", fontsize=12)
                    # else:
                    # print(f"AVISO: Não foi possível aplicar escala log em '{title}' devido a valores não positivos ou todos zero.")
                ax.legend(loc='best', fontsize=10);
                ax.grid(True, ls=':', alpha=0.7);
                plt.tight_layout()
                print(f"\nExibindo gráfico: {title}... (Feche a janela para continuar)");
                plt.show()
            except Exception as e:
                print(f"Erro gráfico escalabilidade '{title}': {e}")
            finally:
                if fig: plt.close(fig)

    def run_suite_with_restriction(self, restriction_config: Dict[str, Any], init_sample_size: Optional[int] = None,
                                   benchmark_ops_count: int = 100, run_scalability_flag: bool = False,
                                   scalability_sizes: Optional[List[int]] = None,
                                   run_latency_bench_flag: bool = False, num_latency_workloads: int = 50,
                                   num_ops_per_latency_workload: int = 3,
                                   run_random_access_bench_flag: bool = False,  # Adicionado default False
                                   num_random_accesses: int = 100  # Adicionado default
                                   ):
        print(f"\n\n{'=' * 10} EXECUTANDO SUÍTE COM RESTRIÇÃO: {restriction_config.get('nome', 'N/A')} {'=' * 10}")
        self._prepare_dataset_for_analysis(restriction_config)
        orig_cpu_slow = restricao_processamento.SIMULATED_CPU_SLOWDOWN_FACTOR
        orig_xtra_loops = restricao_processamento.SIMULATED_EXTRA_COMPUTATION_LOOPS
        if restriction_config.get("tipo_categoria") == "processamento":
            if restriction_config.get("subtipo") == "cpu_lenta_delay":
                restricao_processamento.configurar_lentidao_cpu(**restriction_config.get("params", {}))
            elif restriction_config.get("subtipo") == "carga_extra":
                restricao_processamento.configurar_carga_computacional_extra(**restriction_config.get("params", {}))
        self.initialize_all_structures(sample_size=init_sample_size, verbose=True)
        self.run_benchmark_operations(num_operations=benchmark_ops_count, verbose=True)
        if run_latency_bench_flag: self.run_combined_latency_benchmark(num_workloads=num_latency_workloads,
                                                                       num_ops_per_workload=num_ops_per_latency_workload,
                                                                       verbose=True)
        if run_random_access_bench_flag: self.run_random_access_benchmark(num_accesses=num_random_accesses,
                                                                          verbose=True)
        print(f"\n📋 Gerando Relatórios e Gráficos para Restrição: {self.active_restriction_name}...")
        self._generate_performance_report_table();
        self._generate_comparison_charts();
        self._generate_insertion_evolution_charts()
        if run_scalability_flag:
            self.run_scalability_tests(sizes_to_test=scalability_sizes, verbose=True)
            print(f"\n📈 Gerando Gráficos de Escalabilidade para Restrição: {self.active_restriction_name}...")
            self._generate_scalability_charts(log_scale_plots=True)
        if restriction_config.get("tipo_categoria") == "processamento":  # Reverte
            restricao_processamento.SIMULATED_CPU_SLOWDOWN_FACTOR = orig_cpu_slow
            restricao_processamento.SIMULATED_EXTRA_COMPUTATION_LOOPS = orig_xtra_loops
            print("INFO: Restrições de processamento revertidas.")
        self.active_restriction_name = None;
        self.current_dataset_for_analysis = self.motorcycles_full_dataset_original
        print(f"\n{'=' * 10} SUÍTE COM RESTRIÇÃO {restriction_config.get('nome', 'N/A')} CONCLUÍDA {'=' * 10}")

    def run_full_analysis_suite(self, init_sample_size: Optional[int] = 1000, benchmark_ops_count: int = 100,
                                run_latency_bench_flag: bool = False, num_latency_workloads: int = 50,
                                num_ops_per_latency_workload: int = 3,
                                run_random_access_bench_flag: bool = False,  # Padrão False para não perguntar sempre
                                num_random_accesses: int = 100
                                ):
        print("\n🚀 SUÍTE DE ANÁLISE PADRÃO (SEM RESTRIÇÕES ESPECÍFICAS DE DADOS/PROC) 🚀")
        self._prepare_dataset_for_analysis(None)
        self.initialize_all_structures(sample_size=init_sample_size)
        self.run_benchmark_operations(num_operations=benchmark_ops_count)
        if run_latency_bench_flag:
            self.run_combined_latency_benchmark(num_workloads=num_latency_workloads,
                                                num_ops_per_workload=num_ops_per_latency_workload, verbose=True)
        if run_random_access_bench_flag:
            self.run_random_access_benchmark(num_accesses=num_random_accesses, verbose=True)
        print("\n📋 Gerando Relatórios e Gráficos Padrão...");
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()
        print("\n🏁 Análise Padrão Concluída! 🏁")


CONFIGURACOES_TESTES_RESTRICAO = {
    "dados_precos_corrompidos_10": {"nome": "Preços Corrompidos (10%)", "categoria": "Dados", "tipo_categoria": "dados",
                                    "tipo": "corromper_precos",
                                    "params": {"percentual_corrompido": 0.1, "fator_outlier": 3.0}},
    "dados_anos_anomalos_5": {"nome": "Anos Anômalos (5%)", "categoria": "Dados", "tipo_categoria": "dados",
                              "tipo": "anos_anomalos", "params": {"percentual_anomalo": 0.05}},
    "proc_carga_leve": {"nome": "CPU com Carga Leve (5k loops)", "categoria": "Processamento",
                        "tipo_categoria": "processamento", "subtipo": "carga_extra",
                        "params": {"num_loops_extras": 5000}},
    "proc_carga_alta": {"nome": "CPU com Carga Alta (50k loops)", "categoria": "Processamento",
                        "tipo_categoria": "processamento", "subtipo": "carga_extra",
                        "params": {"num_loops_extras": 50000}},
}


def main_menu_loop(analyzer: StructureAnalyzer, full_dataset: List[Moto]):
    while True:
        print("\n" + "=" * 50 + "\nSISTEMA DE ANÁLISE DE ESTRUTURAS DE DADOS\n" + "=" * 50)
        print("--- GERENCIAR ESTRUTURAS INDIVIDUAIS ---")
        print("1. Lista Encadeada\n2. Árvore AVL\n3. Tabela Hash")
        print("4. Bloom Filter\n5. Radix Tree\n6. Árvore B")
        print("--- ANÁLISE E COMPARAÇÃO ---")
        print("7. Executar Suíte Completa de Análise (Padrão + Opcionais)")
        print("8. Executar Testes de Escalabilidade e Gerar Gráficos")
        print("9. Executar Testes com Condições Restritivas")
        print("10. Gerar Gráficos de Evolução da Inicialização")
        print("--- ANÁLISE DO DATASET ---")
        print("11. Estatísticas Gerais do Dataset (Inclui Numéricas)")
        print("12. Simular Tendências Futuras do Dataset")
        print("0. Sair do Sistema")
        escolha = input("\nEscolha uma opção: ").strip()

        if escolha in ['1', '2', '3', '4', '5', '6']:
            s_map = {'1': ('LinkedList', "LISTA ENCADEADA"), '2': ('AVLTree', "ÁRVORE AVL"),
                     '3': ('HashTable', "TABELA HASH"),
                     '4': ('BloomFilter', "BLOOM FILTER"), '5': ('RadixTree', "RADIX TREE"), '6': ('BTree', "ÁRVORE B")}
            s_key, s_name = s_map[escolha]
            if not analyzer.initialized_structures.get(s_key):
                print(f"\nAVISO: {s_name} não inicializada.");
                print("  Execute Opção 7 (Suíte Completa) ou 8 (Escalabilidade) para popular as estruturas, ou:")
                default_s = (analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000)
                if input(
                        f"  Deseja inicializar TODAS as estruturas agora com uma amostra ({default_s})? (s/n): ").lower() == 's':
                    analyzer._prepare_dataset_for_analysis(None)
                    analyzer.initialize_all_structures(sample_size=default_s, verbose=True)
                if not analyzer.initialized_structures.get(s_key):
                    print(f"{s_name} ainda não inicializada. Voltando ao menu.");
                    continue
            menu_estrutura(analyzer.initialized_structures[s_key], s_name, analyzer.motorcycles_full_dataset_original)

        elif escolha == '7':
            try:
                default_init_s = analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000
                init_s_str = input(f"Amostra para benchmarks (Padrão {default_init_s}. VAZIO=dataset todo): ").strip()
                init_samp: Optional[int] = None if not init_s_str else int(init_s_str)
                if init_samp is not None and init_samp <= 0: init_samp = None; print(
                    "INFO: Amostra inválida, usando dataset todo.")

                bench_ops_s = input(f"Ops para benchmarks individuais (padrão 100): ").strip()
                bench_ops = int(bench_ops_s) if bench_ops_s else 100
                if bench_ops < 0: bench_ops = 100; print("INFO: Ops inválidas, usando 100.")

                run_lat_input = input("Executar benchmark de latência combinada? (s/n, padrão s): ").strip().lower()
                run_lat_b_flag = not run_lat_input or run_lat_input == 's'
                num_lat_wl_val = 50;
                num_ops_per_wl_val = 3
                if run_lat_b_flag:
                    lat_wl_s = input(f"  Número de workloads para latência (padrão {num_lat_wl_val}): ").strip()
                    if lat_wl_s: num_lat_wl_val = int(lat_wl_s)
                    if num_lat_wl_val <= 0: num_lat_wl_val = 50; print(
                        f"INFO: Workloads latência inválido, usando {50}.")
                    ops_wl_s = input(
                        f"  Número de buscas TOTAIS por workload de latência (padrão {num_ops_per_wl_val}): ").strip()
                    if ops_wl_s: num_ops_per_wl_val = int(ops_wl_s)
                    if num_ops_per_wl_val <= 0: num_ops_per_wl_val = 3; print(
                        f"INFO: Buscas por workload inválido, usando {3}.")

                run_ra_input = input("Executar benchmark de acesso aleatório? (s/n, padrão s): ").strip().lower()
                run_ra_b_flag = not run_ra_input or run_ra_input == 's'
                num_ra_acc_val = 100
                if run_ra_b_flag:
                    ra_acc_s = input(f"  Número de acessos aleatórios (padrão {num_ra_acc_val}): ").strip()
                    if ra_acc_s: num_ra_acc_val = int(ra_acc_s)
                    if num_ra_acc_val <= 0: num_ra_acc_val = 100; print(
                        f"INFO: Acessos aleatórios inválido, usando {100}.")

                analyzer.run_full_analysis_suite(
                    init_sample_size=init_samp, benchmark_ops_count=bench_ops,
                    run_latency_bench_flag=run_lat_b_flag, num_latency_workloads=num_lat_wl_val,
                    num_ops_per_latency_workload=num_ops_per_wl_val,
                    run_random_access_bench_flag=run_ra_b_flag,
                    num_random_accesses=num_ra_acc_val
                )
            except ValueError:
                print("ERRO: Entrada inválida. Executando com padrões.")
                analyzer.run_full_analysis_suite(run_latency_bench_flag=True, run_random_access_bench_flag=True)
            except Exception as e:
                print(f"Ocorreu um erro inesperado: {e}")

        elif escolha == '8':
            try:
                print("\n--- Configurar Testes de Escalabilidade ---")
                sizes_str = input("Tamanhos N (ex:100,500). VAZIO=padrão: ").strip();
                s_test_input: Optional[List[int]] = [int(s.strip()) for s in
                                                     sizes_str.split(',')] if sizes_str else None
                if s_test_input and any(s <= 0 for s in s_test_input): print(
                    "AVISO: Ns positivos. Usando padrão."); s_test_input = None

                num_searches_s = input("Buscas por N (padrão 100): ").strip();
                n_s = int(num_searches_s) if num_searches_s else 100
                if n_s < 0: n_s = 100; print("INFO: # Buscas inválido, usando 100.")

                log_s = input("Escala Log para TEMPO nos gráficos? (s/n, padrão s): ").strip().lower();
                log_sc = not log_s or log_s == 's'

                analyzer._prepare_dataset_for_analysis(None)
                analyzer.run_scalability_tests(sizes_to_test=s_test_input, num_searches_per_size=n_s, verbose=True)
                print("\n📈 Gerando Gráficos Escalabilidade...");
                analyzer._generate_scalability_charts(log_scale_plots=log_sc)
            except ValueError:
                print("ERRO: Entrada inválida.")
            except Exception as e:
                print(f"Erro inesperado: {e}")

        elif escolha == '9':
            submenu_testes_restricao(analyzer, CONFIGURACOES_TESTES_RESTRICAO)

        elif escolha == '10':
            if not analyzer.performance_results and not analyzer.scalability_results:
                print("\nNenhum resultado de init/bench disponível. Execute Opção 7 ou 8.");
            elif not any(isinstance(analyzer.performance_results.get(res_name, {}).get('initialization', {}).get(
                    'insertion_evolution_data'), list) for res_name in analyzer.performance_results):
                print("\nDados de evolução da init não disponíveis (Opção 7).")
            else:
                analyzer._generate_insertion_evolution_charts()

        elif escolha == '11':
            if not full_dataset:
                print("\nDataset está vazio.")
            else:
                print("\n--- Estatísticas Gerais Detalhadas do Dataset ---")
                estatisticas = MotoEstatisticas.calcular_estatisticas(full_dataset)
                print(f"\nPreços (Total: {len(full_dataset)} motos):")
                print(f"  Média: R${estatisticas['preco'].get('media', 0.0):.2f}")
                print(f"  Mediana: R${estatisticas['preco'].get('mediana', 0.0):.2f}")
                print(f"  Desvio Padrão: R${estatisticas['preco'].get('desvio_padrao', 0.0):.2f}")
                print(f"  Variância: R${estatisticas['preco'].get('variancia', 0.0):.2f}")
                print(f"\nRevendas:")
                print(f"  Média: R${estatisticas['revenda'].get('media', 0.0):.2f}")
                print(f"  Mediana: R${estatisticas['revenda'].get('mediana', 0.0):.2f}")
                print(f"  Desvio Padrão: R${estatisticas['revenda'].get('desvio_padrao', 0.0):.2f}")
                print(f"  Variância: R${estatisticas['revenda'].get('variancia', 0.0):.2f}")
                print(f"\nAnos:")
                moda_anos = estatisticas['ano'].get('moda', 'N/A')
                if isinstance(moda_anos, list):
                    print(f"  Moda(s): {', '.join(map(str, moda_anos))}")
                else:
                    print(f"  Moda: {moda_anos}")
                print(f"  Média: {estatisticas['ano'].get('media', 0.0):.1f}")
                print(f"  Mediana: {estatisticas['ano'].get('mediana', 0.0)}")
                print(f"\nDepreciação (Valor Absoluto):")
                print(f"  Média: R${estatisticas['depreciacao'].get('media', 0.0):.2f}")
                print(f"  Mediana: R${estatisticas['depreciacao'].get('mediana', 0.0):.2f}")
                print(f"\nTaxa de Depreciação (%):")
                print(f"  Média: {estatisticas['taxa_depreciacao'].get('media', 0.0):.2f}%")
                print(f"  Mediana: {estatisticas['taxa_depreciacao'].get('mediana', 0.0):.2f}%")
                print("\nGerando gráficos estatísticos do dataset completo...")
                MotoEstatisticas.gerar_graficos(full_dataset)

        elif escolha == '12':
            if not full_dataset:
                print("\nDataset está vazio.")
            else:
                try:
                    anos_f_str = input("Quantos anos no futuro para prever? ")
                    anos_f = int(anos_f_str)
                    if anos_f > 0:
                        MotoEstatisticas.prever_tendencias(full_dataset, anos_f)
                    else:
                        print("Número de anos deve ser positivo.")
                except ValueError:
                    print("Entrada inválida para anos.")

        elif escolha == '0':
            print("\nEncerrando sistema... Até logo! 👋");
            break
        else:
            print("\n❌ Opção inválida! Por favor, tente novamente.")

        if escolha != '0':
            input("\nPressione Enter para continuar...")


def main():
    print("=" * 50 + "\nBem-vindo ao Sistema de Análise de Estruturas de Dados!\n" + "=" * 50)
    d_path = os.path.join('data', 'bike_sales_india.csv')
    if not os.path.exists(d_path):
        print(f"ERRO CRÍTICO: Dataset '{os.path.abspath(d_path)}' não encontrado!");
        sys.exit(1)
    print(f"\nCarregando dataset de motocicletas de '{d_path}'...");
    motos_ds = DataHandler.ler_dataset(d_path)
    if not motos_ds:
        print("ERRO CRÍTICO: Nenhum dado foi carregado do dataset ou o dataset está vazio.");
        sys.exit(1)
    print(f"Dataset carregado com {len(motos_ds)} registros.");
    analyzer = StructureAnalyzer(motos_ds)

    if not analyzer.initialized_structures and not analyzer.scalability_results:
        print("\nDica: Nenhuma estrutura foi inicializada ou testada ainda.")
        print("  - Use a Opção 7 para benchmarks padrão (inclui inicialização).")
        print("  - Use a Opção 8 para testes de escalabilidade (inclui inicialização).")
        print("  - Ao selecionar uma estrutura (1-6), você poderá inicializá-las.")

    main_menu_loop(analyzer, motos_ds)


if __name__ == "__main__":
    try:
        import matplotlib

        matplotlib.use('TkAgg')
    except Exception as e:
        print(f"AVISO: Problema ao configurar backend 'TkAgg' do Matplotlib: {e}. "
              "Os gráficos podem não ser exibidos interativamente ou podem precisar de configuração manual (ex: MPLBACKEND).")
    main()