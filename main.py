# main.py
import os
import sys
import time
import random
import tracemalloc
from typing import List, Dict, Any, Callable, Optional
import matplotlib  # Importa o módulo base primeiro
from matplotlib import colormaps as mcm  # Para a nova API de colormaps
import numpy as np  # Usado nos gráficos e cálculos
import copy

# Importações de simulações
from simulacoes import restricao_dados
from simulacoes import restricao_processamento
from simulacoes import restricao_memoria
from simulacoes import restricao_latencia
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

# Definir backend ANTES de importar pyplot
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt  # Agora importa pyplot

    print("INFO: Usando backend Matplotlib TkAgg.")
except Exception as e_backend:
    print(f"AVISO: Problema ao configurar backend 'TkAgg' do Matplotlib: {e_backend}. Tentando backend 'Agg'.")
    try:
        matplotlib.use('Agg')  # Backend não interativo (salva em arquivo)
        import matplotlib.pyplot as plt

        print("INFO: Usando backend Matplotlib 'Agg'. Gráficos serão salvos em arquivo, não exibidos.")
    except Exception as e_backend_agg:
        print(f"ERRO CRÍTICO: Falha ao configurar backend do Matplotlib ('TkAgg' e 'Agg'): {e_backend_agg}")
        plt = None  # Define plt como None para checagens posteriores


class PerformanceMetrics:
    @staticmethod
    def measure(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        restricao_processamento.executar_carga_computacional_extra()  # Carga

        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        # Aplica delay ANTES de finalizar a medição do tempo da func, para que seja incluído
        restricao_processamento.aplicar_delay_fixo_operacao()  # Delay fixo (se configurado)
        measured_time_ms = (time.perf_counter() - start_time) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'time': measured_time_ms,  # Agora inclui o delay fixo
            'current_memory': current / 1024,
            'peak_memory': peak / 1024,
            'result': result
        }


class StructureAnalyzer:
    def __init__(self, motorcycles_dataset: List[Moto]):
        self.motorcycles_full_dataset_original: List[Moto] = motorcycles_dataset
        self.current_dataset_for_analysis: List[Moto] = copy.deepcopy(motorcycles_dataset)
        self.t_btree = 3
        self.structures_prototypes_base: Dict[str, Callable[[], Any]] = {
            'LinkedList': LinkedList, 'AVLTree': AVLTree,
            'HashTable': lambda: HashTable(capacidade=max(101,
                                                          len(self.current_dataset_for_analysis) // 10 if self.current_dataset_for_analysis and len(
                                                              self.current_dataset_for_analysis) > 0 else 101)),
            'BloomFilter': lambda: BloomFilter(
                num_itens_esperados=len(self.current_dataset_for_analysis) if self.current_dataset_for_analysis and len(
                    self.current_dataset_for_analysis) > 0 else 1000),
            'RadixTree': RadixTree, 'BTree': lambda: BTreeV2(t=self.t_btree)
        }
        self.active_prototypes: Dict[str, Callable[[], Any]] = self.structures_prototypes_base.copy()
        self.initialized_structures: Dict[str, Any] = {}
        self.performance_results: Dict[str, Dict[str, Any]] = {}  # ÚNICO local para todos os resultados
        self.last_init_sample_size: Optional[int] = None
        self.scalability_results: Dict[str, List[Dict[str, Any]]] = {}
        self.active_restriction_name: Optional[str] = None

    def _prepare_dataset_for_analysis(self, restriction_config: Optional[Dict[str, Any]] = None):
        self.current_dataset_for_analysis = copy.deepcopy(self.motorcycles_full_dataset_original)
        self.active_restriction_name = None
        if restriction_config:
            self.active_restriction_name = restriction_config.get("nome", "RestricaoDesconhecida")
            tipo_cat = restriction_config.get("tipo_categoria")
            tipo_sub = restriction_config.get("tipo") or restriction_config.get("subtipo")
            params = restriction_config.get("params", {})
            if tipo_cat == "dados":
                print(f"\nINFO: Aplicando restrição de dados: {self.active_restriction_name} com params {params}")
                if tipo_sub == "corromper_precos":
                    self.current_dataset_for_analysis = restricao_dados.corromper_precos_aleatoriamente(
                        self.current_dataset_for_analysis, **params)
                elif tipo_sub == "anos_anomalos":
                    self.current_dataset_for_analysis = restricao_dados.introduzir_anos_anomalos(
                        self.current_dataset_for_analysis, **params)
                elif tipo_sub == "reduzir_resolucao_precos":
                    self.current_dataset_for_analysis = restricao_dados.reduzir_resolucao_precos(
                        self.current_dataset_for_analysis, **params if params else {})
                else:
                    print(f"AVISO: Subtipo de restrição de dados '{tipo_sub}' não reconhecido.")

    def _apply_structure_prototypes_overrides(self, restriction_config: Optional[Dict[str, Any]] = None):
        self.active_prototypes = self.structures_prototypes_base.copy()
        if not restriction_config: return
        tipo_cat = restriction_config.get("tipo_categoria");
        subtipo = restriction_config.get("subtipo") or restriction_config.get("tipo");
        params = restriction_config.get("params", {})
        if tipo_cat == "memoria":
            if subtipo == "limite_tamanho_hash" and 'HashTable' in self.active_prototypes:
                max_el = params.get("max_elementos", 500)
                print(f"INFO (MEM): Aplicando limite de {max_el} elementos para HashTable.")
                self.active_prototypes['HashTable'] = restricao_memoria.criar_hashtable_limitada_factory(
                    lambda_base=self.structures_prototypes_base['HashTable'], max_elementos=max_el)
            elif subtipo == "descarte_lru_lista_geral" and 'LinkedList' in self.active_prototypes:
                cap_lista = params.get("capacidade_lista", 1000)
                print(f"INFO (MEM): Aplicando capacidade LRU de {cap_lista} para LinkedList.")
                self.active_prototypes['LinkedList'] = restricao_memoria.criar_linkedlist_lru_factory(
                    capacidade=cap_lista)
        elif tipo_cat == "algoritmica":
            if subtipo == "hash_fator_carga_alto" and 'HashTable' in self.active_prototypes:
                fator_c = params.get("fator_carga", 0.9)
                print(f"INFO (ALGO): Configurando HashTable com fator de carga máx: {fator_c}.")
                base_cap_lambda = self.structures_prototypes_base[
                    'HashTable']  # Para pegar a lógica de capacidade original
                self.active_prototypes['HashTable'] = lambda: HashTable(
                    capacidade=max(101,
                                   len(self.current_dataset_for_analysis) // 10 if self.current_dataset_for_analysis else 101),
                    # Usa a lógica de capacidade
                    fator_carga_max=fator_c)
            elif subtipo == "limitar_passos_busca_arvore":
                max_p = params.get("max_passos", 5);
                restricao_algoritmica.configurar_limite_passos_busca_arvore(max_p)
                print(f"INFO (ALGO): Busca em árvores (AVL, BTree) limitada a {max_p} passos.")
        # É importante que os módulos de restrição (ex: restricao_algoritmica) realmente modifiquem o comportamento
        # das estruturas ou suas chamadas, possivelmente através de wrappers ou monkey patching (com cuidado).

    def initialize_all_structures(self, sample_size: Optional[int] = None, verbose: bool = True):
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual está vazio."); return
        actual_sample_size = sample_size if sample_size is not None else len(self.current_dataset_for_analysis)
        actual_sample_size = min(actual_sample_size, len(self.current_dataset_for_analysis))
        if actual_sample_size <= 0: actual_sample_size = 1 if len(self.current_dataset_for_analysis) > 0 else 0
        sample_to_insert = []
        if actual_sample_size > 0 and self.current_dataset_for_analysis:
            try:
                sample_to_insert = random.sample(self.current_dataset_for_analysis, actual_sample_size)
            except ValueError:
                if verbose: print(f"AVISO: Erro ao criar amostra. Usando dataset completo.");
                actual_sample_size = len(self.current_dataset_for_analysis);
                sample_to_insert = self.current_dataset_for_analysis
        self.last_init_sample_size = actual_sample_size
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n⏳ Inicializando com {actual_sample_size} motos {dataset_info} e medindo...")
        self.initialized_structures.clear();
        self.performance_results.clear()
        for name, structure_constructor_factory in self.active_prototypes.items():
            if verbose: print(f"\n  Inicializando {name}...")
            structure_instance = structure_constructor_factory()
            insertion_metrics_list = [];
            total_insertion_time = 0.0;
            max_peak_memory_during_init = 0.0
            num_actually_inserted_in_struct = 0
            if actual_sample_size > 0 and sample_to_insert:
                for i, bike_to_insert in enumerate(sample_to_insert):
                    if verbose and (i + 1) % (max(1, actual_sample_size // 10)) == 0:
                        print(f"    Inserindo item {i + 1}/{actual_sample_size} em {name}...")
                    len_before = len(structure_instance) if hasattr(structure_instance, '__len__') else -1
                    metrics = PerformanceMetrics.measure(structure_instance.inserir, bike_to_insert)
                    len_after = len(structure_instance) if hasattr(structure_instance, '__len__') else -1
                    insertion_metrics_list.append({'time': metrics['time'], 'peak_memory': metrics['peak_memory']})
                    if metrics['peak_memory'] > max_peak_memory_during_init: max_peak_memory_during_init = metrics[
                        'peak_memory']
                    if len_after == -1 or len_after > len_before or metrics.get('result') is not False:
                        total_insertion_time += metrics['time']
                        if len_after > len_before or len_after == -1: num_actually_inserted_in_struct += 1
            if num_actually_inserted_in_struct == 0 and actual_sample_size > 0:
                if verbose: print(
                    f"    AVISO: Nenhum item parece ter sido inserido em {name}. Média de inserção será 0.")
                avg_insert_time = 0.0
                if not hasattr(structure_instance,
                               '__len__'): avg_insert_time = total_insertion_time / actual_sample_size if actual_sample_size > 0 and total_insertion_time > 0 else 0.0
            else:
                avg_insert_time = total_insertion_time / num_actually_inserted_in_struct if num_actually_inserted_in_struct > 0 else 0.0
            self.initialized_structures[name] = structure_instance
            self.performance_results[name] = {'initialization': {
                'sample_size': actual_sample_size, 'total_time_ms': total_insertion_time,
                'avg_insert_time_ms': avg_insert_time, 'peak_memory_init_kb': max_peak_memory_during_init,
                'insertion_evolution_data': insertion_metrics_list}}
            len_final_struct = len(structure_instance) if hasattr(structure_instance, '__len__') else 'N/A'
            if verbose: print(
                f"  {name} inicializado. Itens na estrutura: {len_final_struct}. Média inserção: {avg_insert_time:.4f} ms. Pico Memória: {max_peak_memory_during_init:.2f} KB")

    # --- MÉTODO run_benchmark_operations CORRETAMENTE POSICIONADO E COMPLETO ---
    def run_benchmark_operations(self, num_operations: int = 100, verbose: bool = True) -> None:
        if not self.initialized_structures:
            if verbose: print("AVISO: Estruturas não inicializadas. Execute a inicialização primeiro.")
            return
        if not self.current_dataset_for_analysis:
            if verbose: print(
                "AVISO: Dataset de análise atual está vazio. Benchmarks de operações não podem ser executados.")
            return
        actual_num_operations = min(num_operations, len(self.current_dataset_for_analysis))
        if actual_num_operations <= 0:
            if verbose: print(f"AVISO: Nenhuma operação de benchmark a ser executada (n_ops={actual_num_operations}).")
            return
        sample_for_search_remove = random.sample(self.current_dataset_for_analysis, actual_num_operations)
        sample_for_new_insertion = [
            Moto(f"MARCA_BENCH_{i}", f"MODELO_BENCH_{i}", 15000 + i * 10, 12000 + i * 8, 2028 + i) for i in
            range(actual_num_operations)]
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})";
        if verbose: print(
            f"\n⚙️ Executando benchmarks de operações ({actual_num_operations} de cada) {dataset_info}...")
        for name, structure in self.initialized_structures.items():
            if verbose: print(f"\n  Analisando {name}:"); op_results_summary = {}
            if hasattr(structure, 'buscar'):
                s_t, s_m = [], [];
                for b in sample_for_search_remove: m = PerformanceMetrics.measure(structure.buscar, b);s_t.append(
                    m['time']);s_m.append(m['peak_memory'])
                op_results_summary['search_avg_time_ms'] = sum(
                    s_t) / actual_num_operations if actual_num_operations else 0.0
                op_results_summary['search_peak_memory_kb'] = max(s_m) if s_m else 0.0
                if verbose: print(f"    Busca: Tempo médio {op_results_summary['search_avg_time_ms']:.4f} ms")
            if hasattr(structure, 'inserir'):
                i_t, i_m = [], [];
                for b in sample_for_new_insertion: m = PerformanceMetrics.measure(structure.inserir, b);i_t.append(
                    m['time']);i_m.append(m['peak_memory'])
                op_results_summary['new_insertion_avg_time_ms'] = sum(
                    i_t) / actual_num_operations if actual_num_operations else 0.0
                op_results_summary['new_insertion_peak_memory_kb'] = max(i_m) if i_m else 0.0
                if verbose: print(
                    f"    Nova Inserção: Tempo médio {op_results_summary['new_insertion_avg_time_ms']:.4f} ms")
            if hasattr(structure, 'remover') and name not in ["BloomFilter"]:
                r_t, r_m = [], [];
                for b in sample_for_search_remove: m = PerformanceMetrics.measure(structure.remover, b);r_t.append(
                    m['time']);r_m.append(m['peak_memory'])
                op_results_summary['removal_avg_time_ms'] = sum(
                    r_t) / actual_num_operations if actual_num_operations else 0.0
                op_results_summary['removal_peak_memory_kb'] = max(r_m) if r_m else 0.0
                if verbose: print(f"    Remoção: Tempo médio {op_results_summary['removal_avg_time_ms']:.4f} ms" + (
                    " (BTree placeholder)" if name == "BTree" else ""))
            if name == 'HashTable' and hasattr(structure, 'obter_estatisticas_colisao'):
                cs = structure.obter_estatisticas_colisao();
                op_results_summary['HashTable_collision_stats'] = cs
                if verbose: print(
                    f"    Stats Colisão HT: Fator Carga={cs.get('fator_carga_real', 0.0):.2f}, Max Bucket={cs.get('max_comprimento_bucket', 0)}")
            # Garante que a entrada para a estrutura existe em performance_results
            self.performance_results.setdefault(name, {'initialization': {}}).update(op_results_summary)
            if hasattr(structure, 'remover') and name not in ["BloomFilter", "BTree"]:
                for b_ins in sample_for_new_insertion: structure.remover(b_ins)

    # --- FIM DO run_benchmark_operations ---

    def run_combined_latency_benchmark(self, num_workloads: int = 50, ops_per_workload: int = 3, verbose: bool = True):
        if not self.initialized_structures:
            if verbose: print("AVISO: Estruturas não inicializadas para latência."); return
        min_data_needed = ops_per_workload
        if not self.current_dataset_for_analysis or len(self.current_dataset_for_analysis) < min_data_needed:
            if verbose: print(
                f"AVISO: Dataset ({len(self.current_dataset_for_analysis)}) insuficiente para latência (min {min_data_needed})."); return
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(
            f"\n⚙️ BENCHMARK DE LATÊNCIA COMBINADA {dataset_info} (Workloads:{num_workloads}, Ops/WL:{ops_per_workload})")
        for s_name, struct in self.initialized_structures.items():
            if verbose: print(f"  Testando {s_name}...")
            if not (hasattr(struct, 'inserir') and hasattr(struct, 'buscar') and
                    (hasattr(struct, 'remover') or s_name in ["BloomFilter", "BTree"])):
                if verbose: print(f"    AVISO: {s_name} não suporta todas as ops. Pulando.");
                self.performance_results.setdefault(s_name, {}).update(
                    {'combined_latency_avg_ms': -1.0, 'notes_lat': 'Ops incompletas'});
                continue
            wl_times = []
            avail_items_sr = random.sample(self.current_dataset_for_analysis,
                                           min(len(self.current_dataset_for_analysis),
                                               num_workloads * ops_per_workload))
            if not avail_items_sr and ops_per_workload > ops_per_workload // 3 + (
            1 if ops_per_workload % 3 > 0 else 0):  # Se precisa buscar/remover mas não tem de onde
                if verbose: print(
                    f"    AVISO: Não há itens suficientes em {s_name} para workload de busca/remoção. Pulando workload para esta estrutura.");
                self.performance_results.setdefault(s_name, {}).update(
                    {'combined_latency_avg_ms': -1.0, 'notes_lat': 'Dataset insuficiente para workload'});
                continue

            for i_wl in range(num_workloads):
                n_ins = ops_per_workload // 3 + (1 if ops_per_workload % 3 > 0 else 0)
                n_search = ops_per_workload // 3 + (1 if ops_per_workload % 3 > 1 else 0)
                n_rem = ops_per_workload // 3
                total_ops_def = n_ins + n_search + n_rem;
                if total_ops_def < ops_per_workload: n_ins += (ops_per_workload - total_ops_def)

                items_ins_wl = [Moto(f"LAT_M{i_wl}_{j}", f"LAT_N{i_wl}_{j}", 1 + j, 1 + j, 2030 + j) for j in
                                range(n_ins)]
                items_search_wl = random.sample(avail_items_sr, min(len(avail_items_sr),
                                                                    n_search)) if avail_items_sr and n_search > 0 else []
                rem_for_rm = [it for it in avail_items_sr if it not in items_search_wl]
                items_rem_wl = random.sample(rem_for_rm,
                                             min(len(rem_for_rm), n_rem)) if rem_for_rm and n_rem > 0 else []

                curr_wl_ops = [('insert', item) for item in items_ins_wl] + [('search', item) for item in
                                                                             items_search_wl] + [('remove', item) for
                                                                                                 item in items_rem_wl]
                random.shuffle(curr_wl_ops)

                t_start_wl = time.perf_counter();
                inserted_this_wl = []
                for op_t, item_d in curr_wl_ops:
                    if op_t == 'insert':
                        PerformanceMetrics.measure(struct.inserir, item_d); inserted_this_wl.append(item_d)
                    elif op_t == 'search':
                        PerformanceMetrics.measure(struct.buscar, item_d)
                    elif op_t == 'remove' and hasattr(struct, 'remover') and s_name not in ["BloomFilter", "BTree"]:
                        PerformanceMetrics.measure(struct.remover, item_d)
                wl_times.append((time.perf_counter() - t_start_wl) * 1000)
                if hasattr(struct, 'remover') and s_name not in ["BloomFilter", "BTree"]:
                    for item_i in inserted_this_wl: struct.remover(item_i)
            avg_wl_t = sum(wl_times) / num_workloads if num_workloads > 0 and wl_times else 0.0
            self.performance_results.setdefault(s_name, {}).update({'combined_latency_avg_ms': avg_wl_t})
            if verbose: print(f"    Latência Combinada: Média Workload = {avg_wl_t:.4f} ms")

    def run_random_access_benchmark(self, num_accesses: int = 100, verbose: bool = True):
        if not self.initialized_structures:
            if verbose: print("AVISO: Estruturas não inicializadas para acesso aleatório."); return
        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if verbose: print(f"\n⚙️ BENCHMARK DE ACESSO ALEATÓRIO {dataset_info} (Acessos: {num_accesses})")
        for s_name, struct in self.initialized_structures.items():
            if verbose: print(f"  Testando {s_name}...")
            if not hasattr(struct, 'buscar'):
                if verbose: print(f"    AVISO: {s_name} não suporta busca. Pulando.");
                self.performance_results.setdefault(s_name, {}).update(
                    {'random_access_avg_time_ms': -1.0, 'notes_ra': 'Busca não suportada'});
                continue
            items_in_struct_approx = []
            if self.current_dataset_for_analysis and self.last_init_sample_size:
                s_from_size = min(self.last_init_sample_size, len(self.current_dataset_for_analysis))
                if s_from_size > 0: items_in_struct_approx = random.sample(self.current_dataset_for_analysis,
                                                                           s_from_size)
            if not items_in_struct_approx:
                if verbose: print(f"    AVISO: {s_name} vazia/pequena. Pulando.");
                self.performance_results.setdefault(s_name, {}).update(
                    {'random_access_avg_time_ms': -1.0, 'notes_ra': 'Estrutura vazia'});
                continue
            actual_n_acc = min(num_accesses, len(items_in_struct_approx))
            if actual_n_acc == 0:
                if verbose: print(f"    AVISO: Insuficiente em {s_name} para acessos. Pulando.");
                self.performance_results.setdefault(s_name, {}).update(
                    {'random_access_avg_time_ms': -1.0, 'notes_ra': 'Insuficientes'});
                continue
            items_to_acc = random.sample(items_in_struct_approx, actual_n_acc);
            acc_times = []
            for item_d in items_to_acc: mets = PerformanceMetrics.measure(struct.buscar, item_d); acc_times.append(
                mets['time'])
            avg_acc_t = sum(acc_times) / actual_n_acc if actual_n_acc > 0 and acc_times else 0.0
            self.performance_results.setdefault(s_name, {}).update({'random_access_avg_time_ms': avg_acc_t})
            if verbose: print(f"    Acesso Aleatório: Média Acesso = {avg_acc_t:.4f} ms")

    def run_benchmark_operations(self, num_operations: int = 100, verbose: bool = True) -> None:
            """
            Executa benchmarks padrão de busca, nova inserção e remoção.
            Usa o self.current_dataset_for_analysis para amostras.
            """
            if not self.initialized_structures:
                if verbose: print("AVISO: Estruturas não inicializadas. Execute a inicialização primeiro.")
                return
            if not self.current_dataset_for_analysis:
                if verbose: print(
                    "AVISO: Dataset de análise atual está vazio. Benchmarks de operações não podem ser executados.")
                return

            actual_num_operations = min(num_operations, len(self.current_dataset_for_analysis))
            if actual_num_operations <= 0:
                if verbose: print(
                    f"AVISO: Nenhuma operação de benchmark a ser executada (n_ops={actual_num_operations}).")
                return

            # Amostras para busca e remoção são retiradas do dataset de análise atual
            sample_for_search_remove = random.sample(self.current_dataset_for_analysis, actual_num_operations)

            # Amostras para nova inserção são geradas artificialmente
            sample_for_new_insertion = [
                Moto(marca=f"MARCA_BENCH_{i}", nome=f"MODELO_BENCH_{i}",
                     preco=15000 + i * 10, revenda=12000 + i * 8, ano=2028 + i)
                for i in range(actual_num_operations)
            ]

            dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
            if verbose: print(
                f"\n⚙️ Executando benchmarks de operações ({actual_num_operations} de cada) {dataset_info}...")

            for name, structure in self.initialized_structures.items():
                if verbose: print(f"\n  Analisando {name}:")
                op_results_summary = {}  # Para armazenar os resultados desta estrutura

                # Teste de Busca
                if hasattr(structure, 'buscar'):
                    search_times, search_mems = [], []
                    for bike_to_search in sample_for_search_remove:
                        metrics = PerformanceMetrics.measure(structure.buscar, bike_to_search)
                        search_times.append(metrics['time'])
                        search_mems.append(metrics['peak_memory'])
                    op_results_summary['search_avg_time_ms'] = sum(
                        search_times) / actual_num_operations if actual_num_operations else 0.0
                    op_results_summary['search_peak_memory_kb'] = max(search_mems) if search_mems else 0.0
                    if verbose: print(f"    Busca: Tempo médio {op_results_summary['search_avg_time_ms']:.4f} ms")

                # Teste de Nova Inserção
                if hasattr(structure, 'inserir'):
                    insert_times, insert_mems = [], []
                    for new_bike in sample_for_new_insertion:
                        metrics = PerformanceMetrics.measure(structure.inserir, new_bike)
                        insert_times.append(metrics['time'])
                        insert_mems.append(metrics['peak_memory'])
                    op_results_summary['new_insertion_avg_time_ms'] = sum(
                        insert_times) / actual_num_operations if actual_num_operations else 0.0
                    op_results_summary['new_insertion_peak_memory_kb'] = max(insert_mems) if insert_mems else 0.0
                    if verbose: print(
                        f"    Nova Inserção: Tempo médio {op_results_summary['new_insertion_avg_time_ms']:.4f} ms")

                # Teste de Remoção
                if hasattr(structure, 'remover') and name not in ["BloomFilter"]:  # BloomFilter não tem remover
                    remove_times, remove_mems = [], []
                    # Tenta remover os mesmos itens que foram usados para a busca
                    for bike_to_remove in sample_for_search_remove:
                        metrics = PerformanceMetrics.measure(structure.remover, bike_to_remove)
                        remove_times.append(metrics['time'])
                        remove_mems.append(metrics['peak_memory'])
                    op_results_summary['removal_avg_time_ms'] = sum(
                        remove_times) / actual_num_operations if actual_num_operations else 0.0
                    op_results_summary['removal_peak_memory_kb'] = max(remove_mems) if remove_mems else 0.0
                    if verbose: print(f"    Remoção: Tempo médio {op_results_summary['removal_avg_time_ms']:.4f} ms" +
                                      (" (Nota: Remoção em BTree é placeholder)" if name == "BTree" else ""))

                # Adiciona estatísticas de colisão para HashTable
                if name == 'HashTable' and hasattr(structure, 'obter_estatisticas_colisao'):
                    collision_stats = structure.obter_estatisticas_colisao()
                    op_results_summary['HashTable_collision_stats'] = collision_stats
                    if verbose:
                        print(f"    Estatísticas de Colisão HashTable:")
                        print(f"      Fator de Carga: {collision_stats.get('fator_carga_real', 0.0):.2f}")
                        print(f"      Max Comprimento Bucket: {collision_stats.get('max_comprimento_bucket', 0)}")
                        print(
                            f"      % Buckets com Colisão (de Ocupados): {collision_stats.get('percent_buckets_com_colisao_de_ocupados', 0.0):.2f}%")

                # Atualiza o dicionário principal de resultados de performance
                if name in self.performance_results:
                    self.performance_results[name].update(op_results_summary)
                else:
                    # Isso pode acontecer se initialize_all_structures não criou a entrada ainda
                    # (embora no fluxo normal, deveria ter criado)
                    self.performance_results[name] = {'initialization': {}, **op_results_summary}

                # Limpa as motos de "nova inserção" que foram adicionadas durante este benchmark
                # para não poluir as estruturas para testes subsequentes.
                if hasattr(structure, 'remover') and name not in ["BloomFilter", "BTree"]:
                    for new_bike_inserted in sample_for_new_insertion:
                        structure.remover(new_bike_inserted)  # Tenta remover

    # --- Funções de Geração de Gráficos CORRIGIDAS ---
    def _generate_performance_report_table(self) -> None:
        # (Como na última versão funcional - apenas adiciona o título da restrição se ativa)
        report_title = self.active_restriction_name.upper() if self.active_restriction_name else "BENCHMARKS PADRÃO"
        print(f"\n\n📊 RELATÓRIO DE DESEMPENHO ({report_title}) 📊");
        if not self.performance_results: print("Nenhum resultado para gerar relatório."); return
        table_width = 165;
        print("=" * table_width)
        header = "{:<15} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<22} | {:<22}".format(
            "Estrutura", "Init Avg Ins (ms)", "Search Avg (ms)", "New Ins Avg (ms)",
            "Removal Avg (ms)", "Init Peak Mem (KB)", "Avg Workload Lat (ms)", "Avg Random Acc (ms)")
        print(header);
        print("-" * table_width)
        for name, mets in sorted(self.performance_results.items()):
            init_m = mets.get('initialization', {});
            print(
                f"{name:<15} | {init_m.get('avg_insert_time_ms', 0.0):<20.4f} | {mets.get('search_avg_time_ms', 0.0):<20.4f} | {mets.get('new_insertion_avg_time_ms', 0.0):<20.4f} | {mets.get('removal_avg_time_ms', 0.0):<20.4f} | {init_m.get('peak_memory_init_kb', 0.0):<20.2f} | {mets.get('combined_latency_avg_ms', 0.0):<22.4f} | {mets.get('random_access_avg_time_ms', 0.0):<22.4f}")
        print("=" * table_width)
        if 'HashTable' in self.performance_results and 'HashTable_collision_stats' in self.performance_results.get(
                'HashTable', {}):
            ht_s = self.performance_results['HashTable']['HashTable_collision_stats'];
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
        if plt is None: print("Matplotlib pyplot não está disponível. Gráficos não podem ser gerados."); return
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.performance_results: print(f"Nenhum resultado para Gráficos de Comparação{chart_suffix}."); return
        names = list(self.performance_results.keys())
        if not names: print(f"Nomes de estruturas vazios para Gráficos de Comparação{chart_suffix}."); return

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        fig1 = None
        try:
            fig1, ax1 = plt.subplots(figsize=(15, 8))
            ops = ['initialization_avg_insert', 'search_avg', 'new_insertion_avg', 'removal_avg', 'random_access_avg',
                   'combined_latency_avg']
            op_lbls = ['Init Ins Média', 'Busca (Amostra)', 'Nova Ins Média', 'Remoção Média', 'Acesso Aleatório Médio',
                       'Latência Workload Média']
            n_ops = len(ops)
            try:
                cmap = mcm.get_cmap('viridis'); colors = [cmap(i / n_ops) for i in range(n_ops)]
            except:
                colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'violet', 'orange'][:n_ops]
            bar_w = 0.8 / (n_ops + 0.5);
            idx = np.arange(len(names))

            for i, op_key in enumerate(ops):
                key_for_results = f'{op_key}_time_ms' if op_key != 'initialization_avg_insert' else 'avg_insert_time_ms'
                if op_key == 'initialization_avg_insert':
                    data_src = {n: d.get('initialization', {}) for n, d in self.performance_results.items()}
                else:
                    data_src = self.performance_results
                times = [data_src.get(n, {}).get(key_for_results, 0.0) for n in names]
                pos = idx - (bar_w * n_ops / 2) + (i * bar_w) + (bar_w / 2);
                ax1.bar(pos, times, bar_w, label=op_lbls[i], color=colors[i])

            ax1.set_title(f'Comparação de Tempos Médios das Operações{chart_suffix}', fontsize=16);
            ax1.set_ylabel('Tempo Médio (ms)', fontsize=13)
            ax1.set_xlabel('Estrutura', fontsize=13);
            ax1.set_xticks(idx);
            ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1));
            ax1.grid(True, axis='y', ls=':', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.83, 1]);
            print(f"\nExibindo Comp. Tempos{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico tempos: {e}")
        finally:
            if fig1 is not None: plt.close(fig1)

        fig2 = None
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            mems = [self.performance_results.get(n, {}).get('initialization', {}).get('peak_memory_init_kb', 0.0) for n
                    in names]
            try:
                bar_cols = [mcm.get_cmap('Pastel2')(i / len(names)) for i in range(len(names))]
            except:
                bar_cols = 'mediumpurple'
            ax2.bar(names, mems, color=bar_cols, alpha=0.75, edgecolor='black')
            ax2.set_title(f'Pico de Memória na Inicialização{chart_suffix}', fontsize=16);
            ax2.set_ylabel('Memória (KB)', fontsize=13)
            ax2.set_xlabel('Estrutura', fontsize=13);
            ax2.set_xticks(range(len(names)));
            ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
            ax2.grid(True, axis='y', ls=':', alpha=0.6);
            plt.tight_layout();
            print(f"\nExibindo Comp. Memória{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico memória: {e}")
        finally:
            if fig2 is not None: plt.close(fig2)

    def _generate_insertion_evolution_charts(self) -> None:
        if plt is None: print("Matplotlib pyplot não disponível."); return
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.performance_results: print(f"Nenhum resultado para Gráficos de Evolução{chart_suffix}."); return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        fig_t, fig_m = None, None  # Init figs
        try:
            fig_t, ax_t = plt.subplots(figsize=(12, 7))
            ax_t.set_title(f'Evolução Tempo Inserção{chart_suffix}', fontsize=15);
            ax_t.set_xlabel('# Inserção', fontsize=12);
            ax_t.set_ylabel('Tempo (ms)', fontsize=12)
            for name, mets in sorted(self.performance_results.items()):
                init_d = mets.get('initialization', {}).get('insertion_evolution_data', [])
                if init_d: times = [m.get('time', 0.0) for m in init_d];avg_t = sum(times) / len(
                    times) if times else 0.0;ax_t.plot(times, label=f'{name} (média:{avg_t:.3f}ms)', marker='.', ls='-',
                                                       alpha=0.6, ms=2)
            ax_t.legend(loc='upper right');
            ax_t.grid(True, ls=':', alpha=0.7);
            plt.tight_layout();
            print(f"\nExibindo Evol. Tempo Ins{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico evol. tempo: {e}")
        finally:
            if fig_t is not None: plt.close(fig_t)

        try:
            fig_m, ax_m = plt.subplots(figsize=(12, 7))
            ax_m.set_title(f'Evolução Pico Memória Inserção{chart_suffix}', fontsize=15);
            ax_m.set_xlabel('# Inserção', fontsize=12);
            ax_m.set_ylabel('Memória (KB)', fontsize=12)
            for name, mets in sorted(self.performance_results.items()):
                init_d = mets.get('initialization', {}).get('insertion_evolution_data', [])
                if init_d: mems = [m.get('peak_memory', 0.0) for m in init_d];max_m = max(
                    mems) if mems else 0.0;ax_m.plot(mems, label=f'{name} (pico max:{max_m:.2f}KB)', marker='.', ls='-',
                                                     alpha=0.6, ms=2)
            ax_m.legend(loc='upper right');
            ax_m.grid(True, ls=':', alpha=0.7);
            plt.tight_layout();
            print(f"\nExibindo Evol. Memória Ins{chart_suffix}... (Feche para continuar)");
            plt.show()
        except Exception as e:
            print(f"Erro gráfico evol. memória: {e}")
        finally:
            if fig_m is not None: plt.close(fig_m)

    def run_scalability_tests(self, sizes_to_test: Optional[List[int]] = None, num_searches_per_size: int = 100,
                              verbose: bool = True) -> None:
        # (Código completo e corrigido de run_scalability_tests da última resposta)
        if not self.current_dataset_for_analysis:
            if verbose: print("Dataset de análise atual vazio. Testes de escalabilidade cancelados."); return
        if sizes_to_test is None:
            base_s = [100, 500, 1000, 2500, 5000, 7500];
            max_ds_s = len(self.current_dataset_for_analysis)
            sizes_to_test = [s for s in base_s if s <= max_ds_s]
            if max_ds_s > 0 and (max_ds_s not in sizes_to_test and (
                    not sizes_to_test or max_ds_s > sizes_to_test[-1])): sizes_to_test.append(max_ds_s)
            if not sizes_to_test: sizes_to_test = [max_ds_s] if max_ds_s > 0 else [
                10 if len(self.motorcycles_full_dataset_original) > 10 else 1]  # Evita sample de lista vazia
            sizes_to_test = sorted(list(set(s for s in sizes_to_test if s > 0 and s <= len(
                self.current_dataset_for_analysis))))  # Garante que N não exceda dataset atual

        dataset_info = f"(Dataset: {self.active_restriction_name or 'Original'})"
        if not sizes_to_test:
            if verbose: print(f"Nenhum tamanho N válido para testes de escalabilidade {dataset_info}."); return
        if verbose: print(f"\n🔬 INICIANDO TESTES DE ESCALABILIDADE {dataset_info} para N = {sizes_to_test} ...")
        self.scalability_results.clear()

        for n_size in sizes_to_test:
            if verbose: print(f"\n  --- Testando com N = {n_size} ---")
            curr_sample = random.sample(self.current_dataset_for_analysis, n_size)
            for s_name, constructor_factory in self.active_prototypes.items():  # Usa active_prototypes
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
                        for b_search in search_samp:
                            t_s = time.perf_counter();
                            instance.buscar(b_search);
                            search_t_list.append((time.perf_counter() - t_s) * 1000)
                        avg_search_ms = sum(search_t_list) / n_searches if n_searches else 0.0
                        if verbose: print(f"      Busca ({n_searches}): Média={avg_search_ms:.4f}ms/item")
                    else:
                        if verbose: print("      Busca: Nenhuma busca executada (N muito pequeno ou num_searches=0).")
                else:
                    if verbose: print(f"      Busca: Não suportada pela estrutura {s_name}.")

                self.scalability_results.setdefault(s_name, []).append(
                    {'N': n_size, 'avg_insert_time_ms': avg_ins_ms, 'peak_memory_kb': peak_mem_kb,
                     'avg_search_time_ms': avg_search_ms})
        if verbose: print("\n🔬 Testes de Escalabilidade Concluídos! 🔬")

    def _generate_scalability_charts(self, log_scale_plots: bool = False) -> None:
        # (Código completo e corrigido de _generate_scalability_charts da última resposta)
        if plt is None: print("Matplotlib pyplot não disponível."); return
        chart_suffix = f" (Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""
        if not self.scalability_results: print(
            f"Nenhum resultado para Gráficos de Escalabilidade{chart_suffix}."); return
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
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.set_title(title, fontsize=15);
                ax.set_xlabel('# Elementos (N)', fontsize=12);
                ax.set_ylabel(ylabel, fontsize=12)
                has_data = False
                for s_name, res_list in sorted(self.scalability_results.items()):
                    if not res_list: continue
                    s_res = sorted(res_list, key=lambda x: x['N']);
                    n_vals = [r['N'] for r in s_res];
                    m_vals = [r.get(metric, 0.0) for r in s_res]
                    if not any(v > 1e-5 for v in m_vals) and metric != 'peak_memory_kb': continue
                    has_data = True;
                    ax.plot(n_vals, m_vals, marker='o', ls='-', lw=2, ms=5, label=s_name)
                if not has_data: print(f"Nenhum dado para plotar: {title}");
                if fig and not has_data: plt.close(fig); continue
                if log_scale_plots and "Tempo" in ylabel:
                    # Apenas aplica escala log se houver dados e todos os valores plotados forem > 0
                    can_log_scale = True
                    for s_name, res_list in self.scalability_results.items():
                        if any(r.get(metric, 0.0) <= 1e-9 for r in res_list for _ in [1] if any(v > 1e-5 for v in [
                            r_inner.get(metric, 0.0) for r_inner in
                            res_list])):  # Verifica se há valores não positivos na série
                            can_log_scale = False;
                            break
                    if can_log_scale and has_data: ax.set_yscale('log'); ax.set_ylabel(f"{ylabel} (Escala Log)",
                                                                                       fontsize=12)
                    # else: if has_data: print(f"AVISO: Não foi possível aplicar escala log em '{title}' (valores não positivos ou todos zero).")
                ax.legend(loc='best', fontsize=10);
                ax.grid(True, ls=':', alpha=0.7);
                plt.tight_layout()
                print(f"\nExibindo: {title}... (Feche para continuar)");
                plt.show()
            except Exception as e:
                print(f"Erro gráfico escalabilidade '{title}': {e}")
            finally:
                if fig: plt.close(fig)

    def run_suite_with_restriction(self, restriction_config: Dict[str, Any], init_sample_size: Optional[int] = None,
                                   benchmark_ops_count: int = 100, run_scalability_flag: bool = False,
                                   scalability_sizes: Optional[List[int]] = None, scalability_log_scale: bool = False,
                                   run_latency_bench_flag: bool = False, num_latency_workloads: int = 50,
                                   num_ops_per_latency_workload: int = 3,
                                   run_random_access_bench_flag: bool = False, num_random_accesses: int = 100):
        print(f"\n\n{'=' * 10} EXECUTANDO SUÍTE COM RESTRIÇÃO: {restriction_config.get('nome', 'N/A')} {'=' * 10}")

        self._prepare_dataset_for_analysis(
            restriction_config)  # Modifica self.current_dataset_for_analysis e self.active_restriction_name
        self._apply_structure_prototypes_overrides(restriction_config)  # Modifica self.active_prototypes

        # Guarda estados originais das simulações globais
        orig_cpu_slow = restricao_processamento.SIMULATED_CPU_SLOWDOWN_FACTOR  # Não existe mais, mas mantendo a estrutura
        orig_xtra_loops = restricao_processamento.SIMULATED_EXTRA_COMPUTATION_LOOPS
        orig_op_delay = restricao_latencia._simulated_operation_delay_seconds
        orig_batch_config = restricao_latencia._active_batch_insert_config
        orig_hash_factor = restricao_algoritmica.obter_hash_fator_carga_override()
        orig_tree_limit = restricao_algoritmica.obter_limite_passos_busca_arvore()
        orig_mem_max_elements = restricao_memoria.obter_limite_max_elementos()
        orig_mem_lru_cap = restricao_memoria.obter_capacidade_lista_lru()

        try:
            # Aplica configurações de restrição globais
            tipo_cat = restriction_config.get("tipo_categoria")
            params = restriction_config.get("params", {})
            subtipo = restriction_config.get("subtipo") or restriction_config.get("tipo")

            if tipo_cat == "processamento":
                if subtipo == "carga_extra": restricao_processamento.configurar_carga_computacional_extra(**params)
            elif tipo_cat == "latencia":
                if subtipo == "delay_operacao_constante":
                    restricao_latencia.configurar_delay_operacao_constante(**params)
                elif subtipo == "insercao_lote":
                    restricao_latencia.configurar_insercao_lote(**params)
            elif tipo_cat == "algoritmica":
                if subtipo == "hash_fator_carga_baixo":
                    restricao_algoritmica.configurar_hash_fator_carga_baixo(**params)
                elif subtipo == "limitar_passos_busca_arvore":
                    restricao_algoritmica.configurar_limite_passos_busca_arvore(**params)
            elif tipo_cat == "memoria":
                if subtipo == "limite_tamanho_geral":
                    restricao_memoria.configurar_limite_max_elementos(**params)  # Geral para todas
                elif subtipo == "descarte_lru_lista_geral":
                    restricao_memoria.configurar_descarte_lru_lista(**params)

            self.initialize_all_structures(sample_size=init_sample_size, verbose=True)
            self.run_benchmark_operations(num_operations=benchmark_ops_count, verbose=True)

            if run_latency_bench_flag:  # Benchmark de latência específica
                # A latência já é aplicada globalmente via PerformanceMetrics se delay_operacao_constante estiver ativo.
                # Se insercao_lote estiver ativa, run_benchmark_operations ou initialize_all_structures precisariam
                # de uma lógica especial para agrupar inserções, o que é complexo.
                # Vamos simplificar: a latência combinada apenas rodará com os delays já configurados.
                self.run_combined_latency_benchmark(num_workloads=num_latency_workloads,
                                                    num_ops_per_workload=num_ops_per_latency_workload, verbose=True)
            if run_random_access_bench_flag:
                self.run_random_access_benchmark(num_accesses=num_random_accesses, verbose=True)

            print(f"\n📋 Gerando Relatórios e Gráficos para Restrição: {self.active_restriction_name}...")
            self._generate_performance_report_table()
            self._generate_comparison_charts()
            self._generate_insertion_evolution_charts()

            if run_scalability_flag:
                self.run_scalability_tests(sizes_to_test=scalability_sizes, verbose=True)
                print(f"\n📈 Gerando Gráficos de Escalabilidade para Restrição: {self.active_restriction_name}...")
                self._generate_scalability_charts(log_scale_plots=scalability_log_scale)

        finally:  # Reverte TODAS as configurações de restrição, independentemente do que foi aplicado
            restricao_processamento.resetar_restricoes_processamento()
            restricao_latencia.resetar_restricoes_latencia()
            restricao_algoritmica.resetar_restricoes_algoritmicas()
            restricao_memoria.resetar_restricoes_memoria()

            self.active_prototypes = self.structures_prototypes_base.copy()  # Restaura protótipos base
            self.active_restriction_name = None
            self.current_dataset_for_analysis = self.motorcycles_full_dataset_original  # Restaura dataset
            print(f"INFO: Configurações de restrição, protótipos e dataset revertidos para o padrão.")

        print(f"\n{'=' * 10} SUÍTE COM RESTRIÇÃO {restriction_config.get('nome', 'N/A')} CONCLUÍDA {'=' * 10}")

    def run_full_analysis_suite(self, init_sample_size: Optional[int] = 1000, benchmark_ops_count: int = 100,
                                run_latency_bench_flag: bool = False, num_latency_workloads: int = 50,
                                num_ops_per_latency_workload: int = 3,
                                run_random_access_bench_flag: bool = False, num_random_accesses: int = 100):
        print("\n🚀 SUÍTE DE ANÁLISE PADRÃO (SEM RESTRIÇÕES DE SIMULAÇÃO) 🚀")
        self._prepare_dataset_for_analysis(None)
        self._apply_structure_prototypes_overrides(None)
        restricao_processamento.resetar_restricoes_processamento()
        restricao_latencia.resetar_restricoes_latencia()
        restricao_algoritmica.resetar_restricoes_algoritmicas()
        restricao_memoria.resetar_restricoes_memoria()

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



    def _generate_extra_benchmarks_report(self) -> None:
        """Gera um relatório para os benchmarks de latência e acesso aleatório."""
        print("\n\n📊 RELATÓRIO DE BENCHMARKS ADICIONAIS 📊")
        restriction_info = f" (Sob Restrição: {self.active_restriction_name})" if self.active_restriction_name else ""

        if self.latency_benchmark_results:
            print(f"\n--- Latência Combinada Média por Workload{restriction_info} ---")
            print("{:<20} | {:<25}".format("Estrutura", "Tempo Médio Workload (ms)"))
            print("-" * 50)
            for name, results in sorted(self.latency_benchmark_results.items()):
                if results.get('avg_workload_time_ms', -1) == -1:
                    print(f"{name:<20} | {results.get('notes', 'N/A'):<25}")
                else:
                    print(f"{name:<20} | {results.get('avg_workload_time_ms', 0.0):<25.4f}")
            print("-" * 50)

        if self.random_access_benchmark_results:
            print(f"\n--- Acesso Aleatório Médio por Busca{restriction_info} ---")
            print("{:<20} | {:<25}".format("Estrutura", "Tempo Médio Acesso (ms)"))
            print("-" * 50)
            for name, results in sorted(self.random_access_benchmark_results.items()):
                if results.get('avg_access_time_ms', -1) == -1:
                    print(f"{name:<20} | {results.get('notes', 'N/A'):<25}")
                else:
                    print(f"{name:<20} | {results.get('avg_access_time_ms', 0.0):<25.4f}")
            print("-" * 50)

        if not self.latency_benchmark_results and not self.random_access_benchmark_results:
            print("Nenhum resultado de benchmark adicional para exibir.")

    def run_full_analysis_suite(self,
                                init_sample_size: Optional[int] = 1000,
                                benchmark_ops_count: int = 100,
                                run_latency_bench: bool = False,  # NOVO
                                latency_workloads: int = 50,  # NOVO
                                latency_ops_per_wl: int = 3,  # NOVO
                                run_random_access_bench: bool = False,  # NOVO
                                random_access_count: int = 100,  # NOVO
                                run_scalability_flag: bool = False,  # Mantido
                                scalability_sizes: Optional[List[int]] = None  # Mantido
                                ):
        print("\n🚀 SUÍTE DE ANÁLISE 🚀")
        self._prepare_dataset_for_analysis(None)  # Garante dataset original para esta suíte padrão

        self.initialize_all_structures(sample_size=init_sample_size)
        self.run_benchmark_operations(num_operations=benchmark_ops_count)  # Benchmarks padrão

        if run_latency_bench:
            self.run_combined_latency_benchmark(num_workloads=latency_workloads, ops_per_workload=latency_ops_per_wl)
        if run_random_access_bench:
            self.run_random_access_benchmark(num_accesses=random_access_count)

        print("\n📋 Gerando Relatórios e Gráficos Padrão...");
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()

        if run_latency_bench or run_random_access_bench:  # Se algum dos novos benchmarks rodou
            self._generate_extra_benchmarks_report()

        if run_scalability_flag:  # Mantido separado
            # Importante: Escalabilidade deve rodar com dataset original e limpo para cada N
            self._prepare_dataset_for_analysis(None)
            self.run_scalability_tests(sizes_to_test=scalability_sizes, verbose=True)
            print("\n📈 Gerando Gráficos de Escalabilidade...")
            self._generate_scalability_charts(log_scale_plots=True)

        print("\n🏁 Análise Concluída! 🏁")

    def run_suite_with_restriction(self, restriction_config: Dict[str, Any],
                                   init_sample_size: Optional[int] = None, benchmark_ops_count: int = 100,
                                   run_latency_bench: bool = False, latency_workloads: int = 50,
                                   latency_ops_per_wl: int = 3,
                                   run_random_access_bench: bool = False, random_access_count: int = 100,
                                   run_scalability_flag: bool = False, scalability_sizes: Optional[List[int]] = None):
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
        if run_latency_bench:
            self.run_combined_latency_benchmark(num_workloads=latency_workloads, ops_per_workload=latency_ops_per_wl,
                                                verbose=True)
        if run_random_access_bench:
            self.run_random_access_benchmark(num_accesses=random_access_count, verbose=True)

        print(f"\n📋 Gerando Relatórios e Gráficos para Restrição: {self.active_restriction_name}...")
        self._generate_performance_report_table()
        self._generate_comparison_charts()
        self._generate_insertion_evolution_charts()
        if run_latency_bench or run_random_access_bench:
            self._generate_extra_benchmarks_report()  # Mostra os resultados dos novos benchmarks

        if run_scalability_flag:
            # Note: Escalabilidade sob restrição usa o dataset já modificado pela restrição
            self.run_scalability_tests(sizes_to_test=scalability_sizes, verbose=True)
            print(f"\n📈 Gerando Gráficos de Escalabilidade para Restrição: {self.active_restriction_name}...")
            self._generate_scalability_charts(log_scale_plots=True)

        if restriction_config.get("tipo_categoria") == "processamento":
            restricao_processamento.SIMULATED_CPU_SLOWDOWN_FACTOR = orig_cpu_slow
            restricao_processamento.SIMULATED_EXTRA_COMPUTATION_LOOPS = orig_xtra_loops
            print("INFO: Restrições de processamento revertidas.")
        self.active_restriction_name = None
        self.current_dataset_for_analysis = self.motorcycles_full_dataset_original
        print(f"\n{'=' * 10} SUÍTE COM RESTRIÇÃO {restriction_config.get('nome', 'N/A')} CONCLUÍDA {'=' * 10}")

CONFIGURACOES_TESTES_RESTRICAO = {
    # --- Categoria 1: Restrição de Memória ---
    "R02_mem_hash_lim_500": {
        "nome":"R2: HashTable Limite 500 Elem.", "categoria":"1. Restrição de Memória",
        "tipo_categoria":"memoria", "subtipo":"limite_tamanho_hash", # Usado em _apply_structure_prototypes_overrides
        "params":{"max_elementos": 500},
        "descricao": "Afeta apenas HashTable. Outras usam dataset completo (ou seu próprio limite se implementado)."
    },
    "R05_mem_lista_lru_1k": {
        "nome":"R5: Lista LRU (Cap. 1k)", "categoria":"1. Restrição de Memória",
        "tipo_categoria":"memoria", "subtipo":"descarte_lru_lista_geral", # Usado em _apply_structure_prototypes_overrides
        "params":{"capacidade_lista": 1000}, # Usado por restricao_memoria.configurar_descarte_lru_lista
        "descricao":"Requer LinkedList com lógica LRU ou um wrapper LinkedListLRU."
    },
    # --- Categoria 2: Restrição de Processamento ---
    "R07_proc_carga_leve": {
        "nome":"R7: CPU com Carga Leve (5k loops)", "categoria":"2. Restrição de Processamento",
        "tipo_categoria":"processamento", "subtipo":"carga_extra",
        "params":{"num_loops_extras":5000}
    },
    "R09_proc_carga_alta": {
        "nome":"R9: CPU com Carga Alta (50k loops)", "categoria":"2. Restrição de Processamento",
        "tipo_categoria":"processamento", "subtipo":"carga_extra",
        "params":{"num_loops_extras":50000}
    },
    # --- Categoria 3: Restrição de Latência ---
    "R12_lat_delay_op_5ms": {
        "nome":"R12: Latência Fixa 5ms/Operação", "categoria":"3. Restrição de Latência",
        "tipo_categoria":"latencia", "subtipo":"delay_operacao_constante",
        "params":{"delay_segundos": 0.005}
    },
    "R13_lat_ins_lote_10_50ms": {
        "nome":"R13: Inserção Lote (10, delay 50ms)", "categoria":"3. Restrição de Latência",
        "tipo_categoria":"latencia", "subtipo":"insercao_lote",
        "params":{"tamanho_lote": 10, "delay_por_lote_segundos": 0.050},
        "descricao":"Simulação: lote inserido como bloco único, com delay antes/depois do bloco (requer lógica de benchmark especial)."
    },
    # --- Categoria 4: Restrição de Dados ---
    "R16_dados_precos_corrupt_10": {
        "nome":"R16: Preços Corrompidos (10%)", "categoria":"4. Restrição de Dados",
        "tipo_categoria":"dados", "tipo":"corromper_precos",
        "params":{"percentual_corrompido":0.1,"fator_outlier":3.0}
    },
    "R18_dados_anos_anomalos_5": {
        "nome":"R18: Anos Anômalos (5%)", "categoria":"4. Restrição de Dados",
        "tipo_categoria":"dados", "tipo":"anos_anomalos",
        "params":{"percentual_anomalo":0.05}
    },
    # --- Categoria 5: Restrição Algorítmica/Estrutural ---
    "R22_algo_lim_busca_arvore_5": {
        "nome":"R22: Limitar Busca Árvore (5 Passos)", "categoria":"5. Restrição Algorítmica/Estrutural",
        "tipo_categoria":"algoritmica", "subtipo":"limitar_passos_busca_arvore",
        "params":{"max_passos": 5},
        "descricao":"Afeta AVL, BTree. Requer que 'buscar' use o limite configurado."
    },
    "R24_algo_hash_fator_carga_0_9": {
        "nome":"R24: HashTable Fator Carga Alto (0.9)", "categoria":"5. Restrição Algorítmica/Estrutural",
        "tipo_categoria":"algoritmica", "subtipo":"hash_fator_carga_baixo", # "baixo" refere-se à eficiência esperada
        "params":{"fator_carga": 0.9},
        "descricao":"Reduz eficiência ao forçar mais colisões."
    }
}

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
        print("6. Árvore B")
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
            s_map = {
                '1': ('LinkedList', "LISTA ENCADEADA"), '2': ('AVLTree', "ÁRVORE AVL"),
                '3': ('HashTable', "TABELA HASH"), '4': ('BloomFilter', "BLOOM FILTER"),
                '5': ('RadixTree', "RADIX TREE"), '6': ('BTree', "ÁRVORE B")
            }
            s_key, s_name = s_map[escolha]
            if not analyzer.initialized_structures.get(s_key):
                print(f"\nAVISO: {s_name} não inicializada.")
                print(
                    "  Execute Opção 7 (Suíte Completa), 8 (Escalabilidade) ou 9 (Restrições) para popular as estruturas, ou:")
                default_s = (analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000)
                if input(
                        f"  Deseja inicializar TODAS as estruturas agora com uma amostra ({default_s})? (s/n): ").lower() == 's':
                    analyzer._prepare_dataset_for_analysis(None)  # Garante dataset original
                    analyzer.initialize_all_structures(sample_size=default_s, verbose=True)

                if not analyzer.initialized_structures.get(s_key):  # Recheck
                    print(f"{s_name} ainda não inicializada. Voltando ao menu.")
                    continue
            menu_estrutura(analyzer.initialized_structures[s_key], s_name, analyzer.motorcycles_full_dataset_original)

        elif escolha == '7':  # Suíte Completa
            try:
                print("\n--- Configurar Suíte Completa de Análise ---")
                default_init_s = analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else 1000
                init_s_str = input(
                    f"Amostra para init/bench padrão (Padrão {default_init_s}. VAZIO=dataset todo): ").strip()
                init_samp = None if not init_s_str else int(init_s_str)
                if init_samp is not None and init_samp <= 0: init_samp = None; print(
                    "INFO: Amostra inválida, usando dataset todo.")

                b_ops_s = input("Ops para bench padrão (padrão 100): ").strip();
                b_ops = int(b_ops_s) if b_ops_s else 100
                if b_ops < 0: b_ops = 100; print("INFO: Ops inválidas, usando 100.")

                # Perguntas para os novos benchmarks
                run_lat_s = input("Executar benchmark de latência combinada? (s/n, padrão s): ").strip().lower()
                run_lat = run_lat_s != 'n'  # Padrão para sim
                lat_wl, lat_ops_wl = 50, 3
                if run_lat:
                    lat_wl_s = input(f"  Número de workloads para latência (padrão {lat_wl}): ").strip()
                    lat_wl = int(lat_wl_s) if lat_wl_s else lat_wl
                    lat_ops_wl_s = input(f"  Operações TOTAIS por workload de latência (padrão {lat_ops_wl}): ").strip()
                    lat_ops_wl = int(lat_ops_wl_s) if lat_ops_wl_s else lat_ops_wl
                    if lat_ops_wl < 2: lat_ops_wl = 2; print(
                        "INFO: Ops por workload de latência deve ser >= 2. Usando 2.")

                run_ra_s = input("Executar benchmark de acesso aleatório? (s/n, padrão s): ").strip().lower()
                run_ra = run_ra_s != 'n'  # Padrão para sim
                ra_count = 100
                if run_ra:
                    ra_count_s = input(f"  Número de acessos aleatórios (padrão {ra_count}): ").strip()
                    ra_count = int(ra_count_s) if ra_count_s else ra_count

                analyzer.run_full_analysis_suite(
                    init_sample_size=init_samp, benchmark_ops_count=b_ops,
                    run_latency_bench=run_lat, latency_workloads=lat_wl, latency_ops_per_wl=lat_ops_wl,
                    run_random_access_bench=run_ra, random_access_count=ra_count,
                    run_scalability_flag=False  # Escalabilidade é uma opção separada (8)
                )
            except ValueError:
                print("ERRO: Entrada inválida. Executando com padrões.")
                analyzer.run_full_analysis_suite(init_sample_size=None, benchmark_ops_count=100, run_latency_bench=True,
                                                 run_random_access_bench=True)
            except Exception as e:
                print(f"Ocorreu um erro inesperado: {e}")

        elif escolha == '8':  # Testes de Escalabilidade
            try:
                print("\n--- Configurar Testes de Escalabilidade ---")
                s_str = input("Tamanhos N para testar (ex: 100,500,1000). Deixe VAZIO para padrão: ").strip()
                s_test_input: Optional[List[int]] = None
                if s_str:
                    raw_sizes = [val.strip() for val in s_str.split(',')]
                    if all(val.isdigit() and int(val) > 0 for val in raw_sizes if
                           val):  # Checa se todos são digitos > 0
                        s_test_input = [int(val) for val in raw_sizes if val]
                    else:
                        print("AVISO: Formato de tamanhos N inválido ou contém não positivos. Usando padrão.")
                else:
                    print("INFO: Usando tamanhos N padrão para escalabilidade.")

                n_s_s = input("Número de buscas aleatórias por tamanho N (padrão 100): ").strip()
                n_s = int(n_s_s) if n_s_s and n_s_s.isdigit() else 100
                if n_s < 0:
                    n_s = 100
                    print("INFO: Número de buscas inválido, usando 100.")

                log_s = input(
                    "Usar escala logarítmica para eixos Y dos gráficos de TEMPO? (s/n, padrão s): ").strip().lower()
                log_sc = True if not log_s or log_s == 's' else False  # Padrão para Sim

                analyzer._prepare_dataset_for_analysis(None)  # Garante dataset original
                analyzer.run_scalability_tests(sizes_to_test=s_test_input, num_searches_per_size=n_s, verbose=True)
                print("\n📈 Gerando Gráficos de Escalabilidade...")
                analyzer._generate_scalability_charts(log_scale_plots=log_sc)

            except ValueError:
                print("ERRO: Entrada numérica inválida para parâmetros de escalabilidade.")
            except Exception as e:
                print(f"Erro inesperado durante os testes de escalabilidade: {e}")

        elif escolha == '9':  # Testes com Condições Restritivas
            submenu_testes_restricao(analyzer, CONFIGURACOES_TESTES_RESTRICAO)

        elif escolha == '10':  # Gerar Gráficos de Evolução da Inicialização
            # Checagem mais robusta para dados de evolução
            has_evolution_data = False
            if analyzer.performance_results:
                for res_name in analyzer.performance_results:
                    init_data = analyzer.performance_results[res_name].get('initialization', {})
                    if isinstance(init_data.get('insertion_evolution_data'), list) and init_data[
                        'insertion_evolution_data']:
                        has_evolution_data = True
                        break

            if not has_evolution_data:
                print("\nDados de evolução da inicialização não disponíveis.")
                print("Execute a Opção 7 (Suíte Completa) ou 9 (Testes com Restrições) que envolvem inicialização.")
            else:
                analyzer._generate_insertion_evolution_charts()

        elif escolha == '11':  # Estatísticas Gerais do Dataset
            if not full_dataset:
                print("\nDataset está vazio.")
            else:
                print("\n--- Estatísticas Gerais Detalhadas do Dataset ---")
                estats = MotoEstatisticas.calcular_estatisticas(full_dataset)
                print(f"\nPreços (Total: {len(full_dataset)} motos):")
                print(f"  Média: ₹{estats['preco'].get('media', 0.0):.2f}")
                print(f"  Mediana: ₹{estats['preco'].get('mediana', 0.0):.2f}")
                print(f"  Desvio Padrão: ₹{estats['preco'].get('desvio_padrao', 0.0):.2f}")
                print(f"  Variância: ₹{estats['preco'].get('variancia', 0.0):.2f}")

                print(f"\nRevendas:")
                print(f"  Média: ₹{estats['revenda'].get('media', 0.0):.2f}")
                print(f"  Mediana: ₹{estats['revenda'].get('mediana', 0.0):.2f}")
                print(f"  Desvio Padrão: ₹{estats['revenda'].get('desvio_padrao', 0.0):.2f}")
                print(f"  Variância: ₹{estats['revenda'].get('variancia', 0.0):.2f}")

                print(f"\nAnos:")
                moda_anos = estats['ano'].get('moda', 'N/A')
                if isinstance(moda_anos, list):
                    print(f"  Moda(s): {', '.join(map(str, moda_anos))}")
                else:
                    print(f"  Moda: {moda_anos}")
                print(f"  Média: {estats['ano'].get('media', 0.0):.1f}")
                print(f"  Mediana: {estats['ano'].get('mediana', 0.0)}")

                print(f"\nDepreciação (Valor Absoluto):")
                print(f"  Média: ₹{estats['depreciacao'].get('media', 0.0):.2f}")
                print(f"  Mediana: ₹{estats['depreciacao'].get('mediana', 0.0):.2f}")

                print(f"\nTaxa de Depreciação (%):")
                print(f"  Média: {estats['taxa_depreciacao'].get('media', 0.0):.2f}%")
                print(f"  Mediana: {estats['taxa_depreciacao'].get('mediana', 0.0):.2f}%")

                print("\nGerando gráficos estatísticos do dataset completo...")
                MotoEstatisticas.gerar_graficos(full_dataset)

        elif escolha == '12':  # Simular Tendências Futuras
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
                    print("Entrada inválida para anos. Por favor, digite um número inteiro.")

        elif escolha == '0':
            print("\nEncerrando sistema... Até logo! 👋")
            break
        else:
            print("\n❌ Opção inválida! Por favor, tente novamente.")

        if escolha != '0':
            input("\nPressione Enter para continuar...")


def main():
    print("=" * 50 + "\nBem-vindo ao Sistema de Análise de Estruturas de Dados!\n" + "=" * 50)
    d_path = os.path.join('data/bike_sales_india.csv')
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
    # Tenta definir o backend ANTES de qualquer importação de pyplot
    try:
        matplotlib.use('TkAgg')
        # A importação de pyplot foi movida para depois de matplotlib.use() no topo do arquivo.
        print("INFO: Usando backend Matplotlib TkAgg.")
    except Exception as e_backend:
        print(f"AVISO: Problema ao configurar backend 'TkAgg' do Matplotlib: {e_backend}.")
        print("INFO: Tentando backend 'Agg' (gráficos serão salvos, não exibidos interativamente).")
        try:
            matplotlib.use('Agg')
            # import matplotlib.pyplot as plt # Já importado no topo após o primeiro try
            print("INFO: Usando backend Matplotlib 'Agg'.")
        except Exception as e_backend_agg:
            print(f"ERRO CRÍTICO: Falha ao configurar qualquer backend do Matplotlib: {e_backend_agg}")
            print("AVISO: Os gráficos podem não funcionar.")
            # Define plt como None para que as funções de plotagem possam checar
            # (embora plt já seja importado globalmente, isso é mais uma precaução)
            if 'plt' not in globals() or globals()['plt'] is None:
                globals()['plt'] = None  # Garante que plt é None se tudo falhou
    main()