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
from simulacoes import restricao_latencia
from simulacoes import restricao_memoria
from simulacoes import restricao_algoritmica

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
        restricao_latencia.aplicar_delay_operacao_se_configurado()

        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)  # Executa a função original da estrutura
        measured_time_ms = (time.perf_counter() - start_time) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            'time': measured_time_ms,
            'current_memory': current / 1024.0,
            'peak_memory': peak / 1024.0,
            'result': result  # Importante para saber se a inserção/remoção teve sucesso
        }


class StructureAnalyzer:
    def __init__(self, motorcycles_dataset: List[Moto]):
        self.motorcycles_full_dataset_original: List[Moto] = motorcycles_dataset
        self.current_dataset_for_analysis: List[Moto] = copy.deepcopy(motorcycles_dataset)
        self.t_btree = 3
        self.active_restriction_config: Optional[Dict[str, Any]] = None
        # self.structures_prototypes é uma property para ser dinâmica
        self.initialized_structures: Dict[str, Any] = {}
        self.performance_results: Dict[str, Dict[str, Any]] = {}
        self.last_init_sample_size: Optional[int] = None
        self.scalability_results: Dict[str, List[Dict[str, Any]]] = {}
        self.active_restriction_name: Optional[str] = None  # Para nomear relatórios/gráficos

    @property
    def structures_prototypes(self) -> Dict[str, Callable[[], Any]]:
        """Retorna os protótipos das estruturas, aplicando restrições de construtor dinamicamente."""
        # Obtém parâmetros de restrição dos módulos de simulação
        ht_fator_override = restricao_algoritmica.obter_hash_fator_carga_override()
        ll_capacidade_override = restricao_memoria.obter_capacidade_lista_lru()
        max_elements_override = restricao_memoria.obter_limite_max_elementos()

        dataset_len = len(self.current_dataset_for_analysis) if self.current_dataset_for_analysis else 0
        ht_cap_base = max(101, dataset_len // 10 if dataset_len > 0 else 101)
        bf_items_base = dataset_len if dataset_len > 0 else 1000

        final_ht_fator_carga = ht_fator_override if ht_fator_override is not None else 0.7

        return {
            'LinkedList': lambda: LinkedList(capacidade_maxima=ll_capacidade_override),
            'AVLTree': lambda: AVLTree(max_elements=max_elements_override),
            'HashTable': lambda: HashTable(capacidade=ht_cap_base, fator_carga_max=final_ht_fator_carga),
            'BloomFilter': lambda: BloomFilter(num_itens_esperados=bf_items_base),
            'RadixTree': lambda: RadixTree(max_elements=max_elements_override),  # Assumindo RadixTree foi modificada
            'BTree': lambda: BTreeV2(t=self.t_btree, max_elements=max_elements_override)
        }

    def _apply_instance_restrictions(self, instance: Any, struct_name: str):
        """Aplica restrições que afetam uma instância após a criação."""
        if self.active_restriction_config:
            limit_passos = restricao_algoritmica.obter_limite_passos_busca_arvore()
            if limit_passos is not None:
                if struct_name in ["AVLTree", "BTree"] and hasattr(instance, 'set_search_step_limit'):
                    instance.set_search_step_limit(limit_passos)

    def _revert_instance_restrictions(self, instance: Any, struct_name: str):
        """Reverte restrições de instância."""
        if struct_name in ["AVLTree", "BTree"] and hasattr(instance, 'set_search_step_limit'):
            instance.set_search_step_limit(None)  # Reverte para sem limite

    def _prepare_and_configure_for_restriction(self, restriction_config: Optional[Dict[str, Any]]):
        """Prepara o ambiente para uma suíte de teste (com ou sem restrição)."""
        self.current_dataset_for_analysis = copy.deepcopy(self.motorcycles_full_dataset_original)
        self.active_restriction_name = None
        self.active_restriction_config = None

        restricao_processamento.configurar_carga_computacional_extra(0)
        restricao_latencia.configurar_delay_operacao_constante(None)
        restricao_latencia.configurar_insercao_lote(None, None)
        restricao_memoria.configurar_limite_max_elementos(None)
        restricao_memoria.configurar_descarte_lru_lista(None)
        restricao_algoritmica.configurar_hash_fator_carga_baixo(None)
        restricao_algoritmica.configurar_limite_passos_busca_arvore(None)

        if restriction_config:
            self.active_restriction_config = restriction_config
            self.active_restriction_name = restriction_config.get("nome", "RestricaoDesconhecida")
            cat = restriction_config.get("tipo_categoria")
            tipo_subtipo = restriction_config.get("tipo") or restriction_config.get("subtipo")
            params = restriction_config.get("params", {})
            print(
                f"\nINFO: Configurando para restrição: {self.active_restriction_name} ({cat}/{tipo_subtipo}) c/ params {params}")

            if cat == "dados":
                if tipo_subtipo == "corromper_precos":
                    self.current_dataset_for_analysis = restricao_dados.corromper_precos_aleatoriamente(
                        self.current_dataset_for_analysis, **params)
                elif tipo_subtipo == "anos_anomalos":
                    self.current_dataset_for_analysis = restricao_dados.introduzir_anos_anomalos(
                        self.current_dataset_for_analysis, **params)
            elif cat == "processamento":
                restricao_processamento.configurar_carga_computacional_extra(params.get("num_loops_extras", 0))
            elif cat == "latencia":
                if tipo_subtipo == "delay_operacao_constante":
                    restricao_latencia.configurar_delay_operacao_constante(params.get("delay_segundos"))
                elif tipo_subtipo == "insercao_lote_com_delay":
                    restricao_latencia.configurar_insercao_lote(params.get("tamanho_lote"),
                                                                params.get("delay_por_lote_segundos"))
            elif cat == "memoria":
                if tipo_subtipo == "limite_max_elementos":
                    restricao_memoria.configurar_limite_max_elementos(params.get("max_elementos"))
                elif tipo_subtipo == "descarte_lru_lista":
                    restricao_memoria.configurar_descarte_lru_lista(params.get("capacidade_lista"))
            elif cat == "algoritmica":
                if tipo_subtipo == "hash_fator_carga_baixo":
                    restricao_algoritmica.configurar_hash_fator_carga_baixo(params.get("fator_carga_max"))
                elif tipo_subtipo == "limite_passos_busca_arvore":
                    restricao_algoritmica.configurar_limite_passos_busca_arvore(params.get("max_passos"))
        # A property self.structures_prototypes será chamada em initialize_all_structures e run_scalability_tests

    def initialize_all_structures(self, sample_size: Optional[int] = None, verbose: bool = True) -> None:
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual está vazio."); return

        actual_sample_size_requested = len(self.current_dataset_for_analysis) if sample_size is None \
            else min(sample_size, len(self.current_dataset_for_analysis))

        final_sample_size_for_init = actual_sample_size_requested
        # Restrição M1 (limite_max_elementos) aplicada globalmente pelo módulo de simulação
        # Se a estrutura for criada com max_elements, ela se auto-limitará.
        # O `sample_to_insert` ainda pode ser maior que o limite da estrutura.
        max_elements_restr = restricao_memoria.obter_limite_max_elementos()
        if max_elements_restr is not None:
            final_sample_size_for_init = min(actual_sample_size_requested, max_elements_restr)
            if verbose and final_sample_size_for_init < actual_sample_size_requested:
                print(
                    f"INFO: Amostra para inicialização limitada a {final_sample_size_for_init} por restrição de máx. elementos ({max_elements_restr}).")

        sample_to_insert = []
        if final_sample_size_for_init > 0 and len(self.current_dataset_for_analysis) > 0:
            # Garante que k não é maior que a população
            k_sample = min(final_sample_size_for_init, len(self.current_dataset_for_analysis))
            if k_sample > 0:
                sample_to_insert = random.sample(self.current_dataset_for_analysis, k=k_sample)

        if not sample_to_insert and final_sample_size_for_init > 0:
            if verbose: print(
                f"AVISO: Amostra para inserção ficou vazia (sample_size_for_init: {final_sample_size_for_init}, dataset_len: {len(self.current_dataset_for_analysis)}). Nenhuma inserção.")
            final_sample_size_for_init = 0  # Corrige para 0 se não há o que inserir

        self.last_init_sample_size = final_sample_size_for_init
        ds_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n⏳ Inicializando com até {final_sample_size_for_init} motos {ds_info}...")
        self.initialized_structures.clear();
        self.performance_results.clear()

        for name, constructor_factory in self.structures_prototypes.items():
            if verbose: print(f"\n  Inicializando {name}...")
            structure_instance = constructor_factory()  # Cria instância com configs de restrição
            self._apply_instance_restrictions(structure_instance, name)  # Aplica A1

            ins_metrics_list = [];
            total_t_ins = 0.0;
            max_peak_mem_overall = 0.0;
            items_actually_inserted_in_struct = 0

            batch_config = restricao_latencia.obter_config_insercao_lote();  # L2
            batch_s = batch_config["tamanho_lote"] if batch_config else 1
            delay_batch_s = batch_config["delay_por_lote_segundos"] if batch_config else 0.0

            current_items_in_loop = 0
            if final_sample_size_for_init > 0 and sample_to_insert:
                for i in range(0, len(sample_to_insert), batch_s):
                    current_batch_to_insert = sample_to_insert[i: min(i + batch_s, len(sample_to_insert))]
                    if not current_batch_to_insert: continue

                    t_batch_start_for_timing = time.perf_counter()  # Para medir o tempo do lote
                    for bike in current_batch_to_insert:
                        # A estrutura DEVE se auto-limitar se 'max_elements' foi passado no seu construtor (M1)
                        # ou se é a LinkedList com 'capacidade_maxima' (M2).
                        # A contagem 'items_actually_inserted_in_struct' será feita com len(structure_instance) no final.
                        metrics = PerformanceMetrics.measure(structure_instance.inserir, bike)
                        ins_metrics_list.append({'time': metrics['time'], 'peak_memory': metrics['peak_memory']})
                        if metrics['peak_memory'] > max_peak_mem_overall: max_peak_mem_overall = metrics['peak_memory']

                    batch_t_ms = (time.perf_counter() - t_batch_start_for_timing) * 1000
                    total_t_ins += batch_t_ms

                    if batch_config and delay_batch_s > 0: time.sleep(
                        delay_batch_s); total_t_ins += delay_batch_s * 1000
                    current_items_in_loop += len(current_batch_to_insert)
                    if verbose and (i // batch_s + 1) % (max(1, (len(sample_to_insert) // batch_s) // 10)) == 0: print(
                        f"    Processado lote {(i // batch_s) + 1} ({len(current_batch_to_insert)} itens) em {name}...")

            if hasattr(structure_instance, '__len__'):
                try:
                    items_actually_inserted_in_struct = len(structure_instance)
                except:
                    items_actually_inserted_in_struct = sum(
                        1 for m in ins_metrics_list if m['time'] >= 0)  # Estimativa se __len__ falha
            else:
                items_actually_inserted_in_struct = sum(1 for m in ins_metrics_list if m['time'] >= 0)

            denom = items_actually_inserted_in_struct if items_actually_inserted_in_struct > 0 else (
                1 if total_t_ins > 0 else 0)
            avg_ins_t = total_t_ins / denom if denom > 0 else 0.0

            self.initialized_structures[name] = structure_instance
            self.performance_results[name] = {
                'initialization': {'sample_size': items_actually_inserted_in_struct,
                                   'total_time_ms': total_t_ins, 'avg_insert_time_ms': avg_ins_t,
                                   'peak_memory_init_kb': max_peak_mem_overall,
                                   'insertion_evolution_data': ins_metrics_list}}
            if verbose: print(
                f"  {name} inicializado ({items_actually_inserted_in_struct} itens). Média Ins.: {avg_ins_t:.4f} ms. Pico Mem (max indiv.): {max_peak_mem_overall:.2f} KB")
            # Não reverter _apply_instance_restrictions aqui; será feito após benchmarks

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
                can_insert_more = True
                if hasattr(structure, 'max_elements') and structure.max_elements is not None:
                    if hasattr(structure, '__len__') and len(structure) >= structure.max_elements:
                        can_insert_more = False

                if can_insert_more:
                    items_inserted_bench = 0
                    for bike in sample_for_new_insertion:
                        metrics = PerformanceMetrics.measure(structure.inserir, bike)
                        if metrics.get('result', True) is not False:  # Assume None ou True é sucesso
                            items_inserted_bench += 1
                        insert_times.append(metrics['time'])
                        insert_mems.append(metrics['peak_memory'])
                    op_results_summary['new_insertion_avg_time_ms'] = sum(insert_times) / len(
                        insert_times) if insert_times else 0.0
                    op_results_summary['new_insertion_peak_memory_kb'] = max(insert_mems) if insert_mems else 0.0
                    if verbose: print(
                        f"    Nova Inserção ({items_inserted_bench} itens): Tempo médio {op_results_summary['new_insertion_avg_time_ms']:.4f} ms")
                else:
                    if verbose: print(
                        f"    Nova Inserção: Estrutura {name} cheia (limite M1). Teste de inserção pulado.")
                    op_results_summary['new_insertion_avg_time_ms'] = 0.0
                    op_results_summary['new_insertion_peak_memory_kb'] = 0.0

            if hasattr(structure, 'remover') and name not in ["BloomFilter"]:
                remove_times, remove_mems = [], []
                for bike in sample_for_search_remove:
                    metrics = PerformanceMetrics.measure(structure.remover, bike)
                    remove_times.append(metrics['time'])
                    remove_mems.append(metrics['peak_memory'])
                op_results_summary['removal_avg_time_ms'] = sum(
                    remove_times) / actual_num_operations if actual_num_operations else 0
                op_results_summary['removal_peak_memory_kb'] = max(remove_mems) if remove_mems else 0
                if verbose: print(f"    Remoção: Tempo médio {op_results_summary['removal_avg_time_ms']:.4f} ms" + (
                    " (BTree placeholder)" if name == "BTree" else ""))

            if name == 'HashTable' and hasattr(structure, 'obter_estatisticas_colisao'):
                collision_stats = structure.obter_estatisticas_colisao()
                op_results_summary['HashTable_collision_stats'] = collision_stats
                if verbose: print(
                    f"    Stats Colisão HT: Fator Carga={collision_stats['fator_carga_real']:.2f}, Max Bucket={collision_stats['max_comprimento_bucket']}")

            if name not in self.performance_results: self.performance_results[name] = {'initialization': {}}
            self.performance_results[name].update(op_results_summary)

            if hasattr(structure, 'remover') and name not in ["BloomFilter", "BTree"]:
                for bike in sample_for_new_insertion:
                    structure.remover(bike)  # Tenta remover as motos de teste

    def _generate_performance_report_table(self) -> None:
        # (Implementação completa da sua última versão funcional)
        report_title = self.active_restriction_name.upper() if self.active_restriction_name else "BENCHMARKS PADRÃO"
        print(f"\n\n📊 RELATÓRIO DE DESEMPENHO ({report_title}) 📊")
        if not self.performance_results: print("Nenhum resultado para gerar relatório."); return
        table_width = 120;
        print("=" * table_width)
        header = "{:<15} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(
            "Estrutura", "Init Avg Ins (ms)", "Search Avg (ms)", "New Ins Avg (ms)", "Removal Avg (ms)",
            "Init Peak Mem (KB)")
        print(header);
        print("-" * table_width)
        for name, mets in sorted(self.performance_results.items()):
            init_m = mets.get('initialization', {})
            print(
                f"{name:<15} | {init_m.get('avg_insert_time_ms', 0.0):<20.4f} | {mets.get('search_avg_time_ms', 0.0):<20.4f} | "
                f"{mets.get('new_insertion_avg_time_ms', 0.0):<20.4f} | {mets.get('removal_avg_time_ms', 0.0):<20.4f} | "
                f"{init_m.get('peak_memory_init_kb', 0.0):<20.2f}")
        print("=" * table_width)
        ht_perf_res = self.performance_results.get('HashTable', {})
        if 'HashTable_collision_stats' in ht_perf_res:
            ht_s = ht_perf_res['HashTable_collision_stats'];
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
        # (Implementação completa da sua última versão funcional)
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.performance_results: print("Nenhum resultado para gráficos de comparação."); return
        names = list(self.performance_results.keys())
        if not names: print("Nomes de estruturas vazios para gráficos."); return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        fig1 = None
        try:
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            ops = ['initialization_avg_insert', 'search_avg', 'new_insertion_avg', 'removal_avg']
            op_lbls = ['Init Ins. Média', 'Busca Média', 'Nova Ins. Média', 'Remoção Média']
            n_ops = len(ops)
            try:
                cmap = mcm.get_cmap('viridis'); colors_list = [cmap(i / n_ops) for i in range(n_ops)]
            except:
                colors_list = ['skyblue', 'lightgreen', 'salmon', 'gold'][:n_ops]
            bar_w = 0.8 / (n_ops + 0.5);
            idx = np.arange(len(names))
            for i, op_key in enumerate(ops):
                key_for_results = f'{op_key}_time_ms' if op_key != 'initialization_avg_insert' else 'avg_insert_time_ms'
                if op_key == 'initialization_avg_insert':
                    data_source_dict = {n: d.get('initialization', {}) for n, d in self.performance_results.items()}
                else:
                    data_source_dict = self.performance_results
                times = [data_source_dict.get(n, {}).get(key_for_results, 0.0) for n in names]
                pos = idx - (bar_w * n_ops / 2) + (i * bar_w) + (bar_w / 2);
                ax1.bar(pos, times, bar_w, label=op_lbls[i], color=colors_list[i])
            ax1.set_title(f'Comparação de Tempos Médios das Operações{chart_suffix}', fontsize=16);
            ax1.set_ylabel('Tempo Médio (ms)', fontsize=13)
            ax1.set_xlabel('Estrutura', fontsize=13);
            ax1.set_xticks(idx);
            ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax1.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1));
            ax1.grid(True, axis='y', ls=':', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1]);
            print(f"\nExibindo gráfico Comp. Tempos{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico tempos: {e}")
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
            ax2.set_title(f'Pico de Memória na Inicialização{chart_suffix}', fontsize=16);
            ax2.set_ylabel('Memória (KB)', fontsize=13)
            ax2.set_xlabel('Estrutura', fontsize=13);
            ax2.set_xticks(range(len(names)));
            ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax2.grid(True, axis='y', ls=':', alpha=0.6);
            plt.tight_layout();
            print(f"\nExibindo gráfico Comp. Memória{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico memória: {e}")
        finally:
            if fig2 is not None: plt.close(fig2)

    def _generate_insertion_evolution_charts(self) -> None:
        # (Implementação completa da sua última versão funcional)
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.performance_results: print("Nenhum resultado para gráficos de evolução."); return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        fig_t = None
        try:
            fig_t, ax_t = plt.subplots(figsize=(12, 7));
            ax_t.set_title(f'Evolução Tempo Inserção{chart_suffix}', fontsize=15)
            ax_t.set_xlabel('# Inserção', fontsize=12);
            ax_t.set_ylabel('Tempo (ms)', fontsize=12)
            for name, mets in sorted(self.performance_results.items()):
                init_d = mets.get('initialization', {}).get('insertion_evolution_data', [])
                if init_d: times = [m.get('time', 0.0) for m in init_d];avg_t = sum(times) / len(
                    times) if times else 0; ax_t.plot(times, label=f'{name} (média:{avg_t:.3f}ms)', marker='.', ls='-',
                                                      alpha=0.6, ms=2)
            ax_t.legend(loc='upper right');
            ax_t.grid(True, ls=':', alpha=0.7);
            plt.tight_layout();
            print(f"\nExibindo gráfico Evol. Tempo Ins{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico evol. tempo: {e}")
        finally:
            if fig_t is not None: plt.close(fig_t)
        fig_m = None
        try:
            fig_m, ax_m = plt.subplots(figsize=(12, 7));
            ax_m.set_title(f'Evolução Pico Memória Inserção{chart_suffix}', fontsize=15)
            ax_m.set_xlabel('# Inserção', fontsize=12);
            ax_m.set_ylabel('Memória (KB)', fontsize=12)
            for name, mets in sorted(self.performance_results.items()):
                init_d = mets.get('initialization', {}).get('insertion_evolution_data', [])
                if init_d: mems = [m.get('peak_memory', 0.0) for m in init_d];max_m = max(
                    mems) if mems else 0; ax_m.plot(mems, label=f'{name} (pico max:{max_m:.2f}KB)', marker='.', ls='-',
                                                    alpha=0.6, ms=2)
            ax_m.legend(loc='upper right');
            ax_m.grid(True, ls=':', alpha=0.7);
            plt.tight_layout();
            print(f"\nExibindo gráfico Evol. Memória Ins{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico evol. memória: {e}")
        finally:
            if fig_m is not None: plt.close(fig_m)

    def run_scalability_tests(self, sizes_to_test: Optional[List[int]] = None, num_searches_per_size: int = 100,
                              verbose: bool = True) -> None:
        # (Implementação completa da sua última versão funcional)
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual vazio. Testes de escalabilidade cancelados."); return
        if sizes_to_test is None:
            base_s = [100, 500, 1000, 2500, 5000, 7500];
            max_ds_s = len(self.current_dataset_for_analysis)
            sizes_to_test = [s for s in base_s if s <= max_ds_s]
            if max_ds_s > 0 and max_ds_s not in sizes_to_test and (
                    not sizes_to_test or max_ds_s > sizes_to_test[-1]): sizes_to_test.append(max_ds_s)
            if not sizes_to_test and max_ds_s > 0:
                sizes_to_test = [max_ds_s]
            elif not sizes_to_test:
                sizes_to_test = [10]
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
            # Garante que k (n_size) não é maior que a população (len(self.current_dataset_for_analysis))
            k_sample = min(n_size, len(self.current_dataset_for_analysis))
            if k_sample <= 0:
                if verbose: print(
                    f"      Amostra de tamanho {k_sample} inválida para N={n_size}. Pulando estruturas para este N.")
                continue
            curr_sample = random.sample(self.current_dataset_for_analysis, k_sample)
            for s_name, constructor_factory in self.structures_prototypes.items():
                if verbose: print(f"    Testando {s_name}...")
                instance = constructor_factory()
                self._apply_instance_restrictions(instance, s_name)
                items_inserted = 0;
                tracemalloc.start();
                t_s_ins = time.perf_counter()
                for bike in curr_sample:
                    if instance.inserir(bike) is not False: items_inserted += 1
                t_tot_ins_ms = (time.perf_counter() - t_s_ins) * 1000;
                avg_ins_ms = t_tot_ins_ms / items_inserted if items_inserted else 0.0
                _, peak_mem_kb = tracemalloc.get_traced_memory();
                tracemalloc.stop();
                peak_mem_kb /= 1024
                if verbose: print(
                    f"      Ins ({items_inserted}/{n_size}): Tot={t_tot_ins_ms:.2f}ms, Média={avg_ins_ms:.4f}ms/item, PicoMem={peak_mem_kb:.2f}KB")
                avg_srch_ms = 0.0
                if hasattr(instance, 'buscar'):
                    n_srch_actual = min(num_searches_per_size, items_inserted)
                    if n_srch_actual > 0:
                        s_samp_actual = random.sample(curr_sample[:items_inserted], k=min(n_srch_actual,
                                                                                          items_inserted)) if items_inserted > 0 else []
                        s_t_list = []
                        for b_s in s_samp_actual: t_s_srch = time.perf_counter();instance.buscar(b_s);s_t_list.append(
                            (time.perf_counter() - t_s_srch) * 1000)
                        avg_srch_ms = sum(s_t_list) / n_srch_actual if n_srch_actual else 0.0
                        if verbose: print(f"      Busca ({n_srch_actual}): Média={avg_srch_ms:.4f}ms/item")
                    else:
                        if verbose: print(f"      Busca: Nenhuma.")
                else:
                    if verbose: print(f"      Busca: Não suportada.")
                self._revert_instance_restrictions(instance, s_name)
                if s_name not in self.scalability_results: self.scalability_results[s_name] = []
                self.scalability_results[s_name].append(
                    {'N': n_size, 'items_actually_inserted': items_inserted, 'avg_insert_time_ms': avg_ins_ms,
                     'peak_memory_kb': peak_mem_kb, 'avg_search_time_ms': avg_srch_ms})
        if verbose: print("\n🔬 Testes de Escalabilidade Concluídos! 🔬")

    def _generate_scalability_charts(self, log_scale_plots: bool = False) -> None:
        # (Implementação completa da sua última versão funcional)
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
                title = title_base + chart_suffix
                fig, ax = plt.subplots(figsize=(12, 7));
                ax.set_title(title, fontsize=15)
                ax.set_xlabel('Nº Elementos Inseridos', fontsize=12);
                ax.set_ylabel(ylabel, fontsize=12);
                has_data_for_plot = False
                for s_name, res_list in sorted(self.scalability_results.items()):
                    if not res_list: continue
                    s_res = sorted(res_list, key=lambda x: x['N'])
                    n_vals = [r.get('items_actually_inserted', r['N']) for r in s_res]
                    m_vals = [r.get(metric, 0.0) for r in s_res]
                    if not any(abs(v) > 1e-6 for v in m_vals) and metric != 'peak_memory_kb':
                        if not (metric == 'avg_search_time_ms' and not hasattr(self.structures_prototypes[s_name](),
                                                                               'buscar')): pass
                        continue
                    has_data_for_plot = True;
                    ax.plot(n_vals, m_vals, marker='o', ls='-', lw=2, ms=5, label=s_name)
                if not has_data_for_plot: print(f"Nenhum dado válido para plotar: {title}");
                if fig and not has_data_for_plot: plt.close(fig); continue
                if log_scale_plots and "Tempo" in ylabel:
                    valid_points_for_log = False;
                    ax.clear();
                    ax.set_title(title, fontsize=15)
                    ax.set_xlabel('Nº Elementos Inseridos', fontsize=12);
                    ax.set_ylabel(f"{ylabel} (Escala Log)", fontsize=12)
                    for s_name, res_list in sorted(self.scalability_results.items()):
                        s_res = sorted(res_list, key=lambda x: x['N']);
                        n_vals = [r.get('items_actually_inserted', r['N']) for r in s_res];
                        m_vals = [r.get(metric, 0.0) for r in s_res]
                        log_n = [n for n, m in zip(n_vals, m_vals) if m > 1e-9];
                        log_m = [m for m in m_vals if m > 1e-9]
                        if log_n and log_m: ax.plot(log_n, log_m, marker='o', ls='-', lw=2, ms=5,
                                                    label=s_name); valid_points_for_log = True
                    if valid_points_for_log:
                        ax.set_yscale('log')
                    else:
                        ax.set_ylabel(ylabel, fontsize=12)
                ax.legend(loc='best', fontsize=10);
                ax.grid(True, ls=':', alpha=0.7);
                plt.tight_layout();
                print(f"\nExibindo: {title}... (Feche para continuar)");
                plt.show()
            except Exception as e:
                print(f"Erro gráfico escalabilidade '{title}': {e}")
            finally:
                if fig: plt.close(fig)

    def run_suite_with_restriction(self, restriction_config: Dict[str, Any], init_sample_size: Optional[int] = None,
                                   benchmark_ops_count: int = 100, run_scalability_flag: bool = False,
                                   scalability_sizes: Optional[List[int]] = None):
        print(f"\n\n{'=' * 10} EXECUTANDO SUÍTE COM RESTRIÇÃO: {restriction_config.get('nome', 'N/A')} {'=' * 10}")
        self._prepare_and_configure_for_restriction(restriction_config)

        self.initialize_all_structures(sample_size=init_sample_size, verbose=True)
        # Aplica restrições de instância (A1) ANTES dos benchmarks de operações
        for name, instance in self.initialized_structures.items():
            self._apply_instance_restrictions(instance, name)

        self.run_benchmark_operations(num_operations=benchmark_ops_count, verbose=True)

        # Reverte restrições de instância APÓS os benchmarks de operações
        for name, instance in self.initialized_structures.items():
            self._revert_instance_restrictions(instance, name)

        print(f"\n📋 Gerando Relatórios e Gráficos para Restrição: {self.active_restriction_name}...")
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()

        if run_scalability_flag:
            # _prepare_and_configure_for_restriction já foi chamado e configurou os parâmetros
            # que _get_structure_prototypes e _apply_instance_restrictions usarão dentro de run_scalability_tests
            self.run_scalability_tests(sizes_to_test=scalability_sizes, verbose=True)
            print(f"\n📈 Gerando Gráficos de Escalabilidade para Restrição: {self.active_restriction_name}...")
            self._generate_scalability_charts(log_scale_plots=True)

        self._prepare_and_configure_for_restriction(None)  # Reseta tudo ao final da suíte de restrição
        print(f"\n{'=' * 10} SUÍTE COM RESTRIÇÃO {restriction_config.get('nome', 'N/A')} CONCLUÍDA {'=' * 10}")

    def run_full_analysis_suite(self, init_sample_size: Optional[int] = 1000, benchmark_ops_count: int = 100):
        print("\n🚀 SUÍTE DE ANÁLISE PADRÃO (SEM RESTRIÇÕES) 🚀")
        self._prepare_and_configure_for_restriction(None)
        self.initialize_all_structures(sample_size=init_sample_size)
        # Para benchmarks padrão, normalmente não aplicamos restrições como A1,
        # A menos que a A1 fosse um comportamento padrão desejado para as árvores.
        # Se for específico de uma restrição, _apply_instance_restrictions não é chamado aqui.
        self.run_benchmark_operations(num_operations=benchmark_ops_count)
        print("\n📋 Gerando Relatórios e Gráficos Padrão...");
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()
        print("\n🏁 Análise Padrão Concluída! 🏁")


# (COLE CONFIGURACOES_TESTES_RESTRICAO ATUALIZADO com 10 itens AQUI)
CONFIGURACOES_TESTES_RESTRICAO = {
    "D1_precos_corrompidos": {"nome": "Dados: Preços Corrompidos (10%, Outlier 3x)", "categoria": "Dados",
                              "tipo_categoria": "dados", "tipo": "corromper_precos",
                              "params": {"percentual_corrompido": 0.1, "fator_outlier": 3.0}},
    "D2_anos_anomalos": {"nome": "Dados: Anos Anômalos (5%)", "categoria": "Dados", "tipo_categoria": "dados",
                         "tipo": "anos_anomalos", "params": {"percentual_anomalo": 0.05}},
    "P1_carga_cpu_leve": {"nome": "Processamento: Carga Leve (5k loops/op)", "categoria": "Processamento",
                          "tipo_categoria": "processamento", "subtipo": "carga_extra",
                          "params": {"num_loops_extras": 5000}},
    "P2_carga_cpu_alta": {"nome": "Processamento: Carga Alta (50k loops/op)", "categoria": "Processamento",
                          "tipo_categoria": "processamento", "subtipo": "carga_extra",
                          "params": {"num_loops_extras": 50000}},
    "L1_delay_op_5ms": {"nome": "Latência: Delay 5ms/Operação", "categoria": "Latência", "tipo_categoria": "latencia",
                        "subtipo": "delay_operacao_constante", "params": {"delay_segundos": 0.005}},
    "L2_ins_lote_10_delay_50ms": {"nome": "Latência: Inserção Lote(10), Delay 50ms/lote", "categoria": "Latência",
                                  "tipo_categoria": "latencia", "subtipo": "insercao_lote_com_delay",
                                  "params": {"tamanho_lote": 10, "delay_por_lote_segundos": 0.05}},
    "A1_limite_busca_arvore_5": {"nome": "Algorítmica: Busca Árvore Limitada (5 passos)",
                                 "categoria": "Algorítmica/Estrutural", "tipo_categoria": "algoritmica",
                                 "subtipo": "limite_passos_busca_arvore", "params": {"max_passos": 5}},
    "A2_hash_fator_carga_baixo": {"nome": "Algorítmica: HashTable Fator Carga Baixo (0.3)",
                                  "categoria": "Algorítmica/Estrutural", "tipo_categoria": "algoritmica",
                                  "subtipo": "hash_fator_carga_baixo", "params": {"fator_carga_max": 0.3}},
    "M1_limite_elementos_500": {"nome": "Memória: Limite de 500 Elementos/Estrutura", "categoria": "Memória",
                                "tipo_categoria": "memoria", "subtipo": "limite_max_elementos",
                                "params": {"max_elementos": 500}},
    "M2_lista_descarte_lru_1k": {"nome": "Memória: Lista Encadeada LRU (Capacidade 1k)", "categoria": "Memória",
                                 "tipo_categoria": "memoria", "subtipo": "descarte_lru_lista",
                                 "params": {"capacidade_lista": 1000}}
}


# (COLE main_menu_loop, main, e if __name__ DA ÚLTIMA RESPOSTA COMPLETA AQUI)
def main_menu_loop(analyzer: StructureAnalyzer, full_dataset: List[Moto]):
    while True:
        print("\n" + "=" * 50 + "\nSISTEMA DE ANÁLISE DE ESTRUTURAS DE DADOS\n" + "=" * 50)
        print("--- GERENCIAR ESTRUTURAS INDIVIDUAIS ---")
        print("1. Lista Encadeada\n2. Árvore AVL\n3. Tabela Hash")
        print("4. Bloom Filter\n5. Radix Tree\n6. Árvore B")
        print("--- ANÁLISE E COMPARAÇÃO ---")
        print("7. Executar Suíte Completa de Análise (Benchmarks Padrão)")
        print("8. Executar Testes de Escalabilidade e Gerar Gráficos")
        print("9. Executar Testes com Condições Restritivas")
        print("10. Gerar Gráficos de Evolução da Inicialização")
        print("--- ANÁLISE DO DATASET ---")
        print("11. Estatísticas Gerais do Dataset e Gráficos")
        print("12. Simular Tendências Futuras do Dataset")
        print("0. Sair do Sistema")
        escolha = input("\nEscolha uma opção: ").strip()

        if escolha in ['1', '2', '3', '4', '5', '6']:
            s_map = {'1': ('LinkedList', "LISTA ENCADEADA"), '2': ('AVLTree', "ÁRVORE AVL"),
                     '3': ('HashTable', "TABELA HASH"),
                     '4': ('BloomFilter', "BLOOM FILTER"), '5': ('RadixTree', "RADIX TREE"), '6': ('BTree', "ÁRVORE B")}
            s_key, s_name = s_map[escolha]
            # Garante que, se estivermos em modo de restrição, não tentamos re-inicializar tudo aqui,
            # pois a restrição já foi aplicada. O menu individual deve mostrar o estado atual da restrição.
            if not analyzer.initialized_structures.get(s_key) and analyzer.active_restriction_config is None:
                print(f"\nAVISO: {s_name} não inicializada (ou resetada).")
                print(
                    "  Execute Opção 7 (Suíte Completa) ou 8 (Escalabilidade) para popular as estruturas com dados padrão, ou:")
                default_s = (analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000)
                if input(
                        f"  Deseja inicializar TODAS as estruturas agora com amostra padrão ({default_s}) para visualização? (s/n): ").lower() == 's':
                    analyzer._prepare_and_configure_for_restriction(
                        None)  # Garante que está em modo sem restrição para este init
                    analyzer.initialize_all_structures(sample_size=default_s, verbose=True)

            current_struct_instance = analyzer.initialized_structures.get(s_key)
            if not current_struct_instance:
                print(
                    f"{s_name} não está pronta. Se uma restrição está ativa, ela pode não ter sido inicializada. Tente a Opção 9 primeiro ou a Opção 7 para modo padrão.");
                continue

            analyzer._apply_instance_restrictions(current_struct_instance,
                                                  s_key)  # Aplica restrições de instância se houver
            menu_estrutura(current_struct_instance, s_name,
                           analyzer.motorcycles_full_dataset_original)  # Sempre passa o original para referência
            analyzer._revert_instance_restrictions(current_struct_instance, s_key)  # Reverte para o estado normal

        elif escolha == '7':
            try:
                default_init_s = analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000
                init_s_str = input(
                    f"Amostra para benchmarks padrão (Padrão {default_init_s}. VAZIO para dataset completo): ").strip()
                init_samp: Optional[int] = None
                if not init_s_str:
                    init_samp = None
                else:
                    init_samp = int(init_s_str)
                if init_samp is not None and init_samp <= 0: init_samp = None; print(
                    "INFO: Amostra inválida, usando dataset completo.")

                bench_ops_s = input(f"Número de operações para benchmarks padrão (padrão 100): ").strip()
                bench_ops = int(bench_ops_s) if bench_ops_s and bench_ops_s.isdigit() else 100
                if bench_ops < 0: bench_ops = 100; print("INFO: Número de operações inválido, usando 100.")

                analyzer.run_full_analysis_suite(init_sample_size=init_samp, benchmark_ops_count=bench_ops)
            except ValueError:
                print("ERRO: Entrada inválida. Executando com padrões (Amostra: Dataset Completo, Bench Ops: 100).")
                analyzer.run_full_analysis_suite(init_sample_size=None, benchmark_ops_count=100)
            except Exception as e:
                print(f"Ocorreu um erro inesperado ao executar a suíte de análise: {e}")

        elif escolha == '8':
            try:
                print("\n--- Configurar Testes de Escalabilidade ---")
                sizes_str = input("Digite os tamanhos N (ex: 100,500,1000). VAZIO para padrão: ").strip()
                sizes_to_test_input: Optional[List[int]] = None
                if sizes_str:
                    raw_sizes = [s.strip() for s in sizes_str.split(',')];
                    if all(s.isdigit() and int(s) > 0 for s in raw_sizes if s):
                        sizes_to_test_input = [int(s) for s in raw_sizes if s]
                    else:
                        print("AVISO: Formato de Ns inválido. Usando padrão.")
                else:
                    print("INFO: Usando Ns padrão para escalabilidade.")

                num_searches_str = input("Número de buscas aleatórias por tamanho N (padrão 100): ").strip();
                num_s = int(num_searches_str) if num_searches_str and num_searches_str.isdigit() else 100
                if num_s < 0: num_s = 100; print("INFO: # Buscas inválido, usando 100.")

                log_s = input(
                    "Usar escala logarítmica para eixos Y dos gráficos de TEMPO? (s/n, padrão s): ").strip().lower();
                log_sc = not log_s or log_s == 's'

                analyzer._prepare_and_configure_for_restriction(None)  # Garante modo padrão para escalabilidade
                analyzer.run_scalability_tests(sizes_to_test=sizes_to_test_input, num_searches_per_size=num_s,
                                               verbose=True)
                print("\n📈 Gerando Gráficos Escalabilidade...");
                analyzer._generate_scalability_charts(log_scale_plots=log_sc)
            except ValueError:
                print("ERRO: Entrada inválida para parâmetros de escalabilidade.")
            except Exception as e:
                print(f"Erro inesperado durante os testes de escalabilidade: {e}")

        elif escolha == '9':
            submenu_testes_restricao(analyzer, CONFIGURACOES_TESTES_RESTRICAO)

        elif escolha == '10':
            # Verifica se performance_results e a chave initialization existem e se insertion_evolution_data é uma lista
            if not analyzer.performance_results or \
                    not any(isinstance(analyzer.performance_results.get(res_name, {}).get('initialization', {}).get(
                        'insertion_evolution_data'), list)
                            for res_name in analyzer.performance_results):
                print("\nNenhum dado de evolução da inicialização disponível.")
                print("Execute a Opção 7 (Suíte Completa) ou uma Suíte com Restrição (Opção 9) primeiro.")
            else:
                analyzer._generate_insertion_evolution_charts()


        elif escolha == '11':
            if not full_dataset:
                print("\nDataset está vazio.")
            else:
                MotoEstatisticas.gerar_graficos(full_dataset)

        elif escolha == '12':
            if not full_dataset:
                print("\nDataset está vazio.")
            else:
                try:
                    anos_f_str = input("Anos no futuro para prever? ")
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
    print("=" * 50 + "\nBem-vindo ao Sistema Avançado de Análise de Desempenho de Estruturas de Dados!\n" + "=" * 50)
    d_path = os.path.join('data', 'bike_sales_india.csv')
    if not os.path.exists(d_path):
        print(f"ERRO CRÍTICO: Arquivo de dataset não encontrado em '{os.path.abspath(d_path)}'")
        sys.exit(1)
    print(f"\nCarregando dataset de motocicletas de '{d_path}'...")
    motos_ds = DataHandler.ler_dataset(d_path)
    if not motos_ds:
        print("ERRO CRÍTICO: Nenhum dado foi carregado do dataset ou o dataset está vazio.")
        sys.exit(1)
    print(f"Dataset carregado com {len(motos_ds)} registros.")
    analyzer = StructureAnalyzer(motos_ds)

    if not analyzer.initialized_structures and not analyzer.scalability_results:
        print("\nDica: Nenhuma estrutura foi inicializada ou testada ainda.")
        print("  - Use a Opção 7 para benchmarks padrão.")
        print("  - Use a Opção 8 para testes de escalabilidade.")
        print("  - Use a Opção 9 para testes com restrições.")
        print("  - Ao selecionar uma estrutura individual (1-6), você poderá inicializar todas se desejar.")

    main_menu_loop(analyzer, motos_ds)


if __name__ == "__main__":
    try:
        import matplotlib

        matplotlib.use('TkAgg')
    except Exception as e:
        print(f"AVISO: Problema ao configurar backend 'TkAgg' do Matplotlib: {e}. "
              "Os gráficos podem não ser exibidos interativamente ou podem precisar de configuração manual do backend (ex: MPLBACKEND).")
    main()