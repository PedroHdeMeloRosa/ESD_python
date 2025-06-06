# Estruturas/hash_table.py
from modelos.moto import Moto
from typing import List, Tuple, Dict, Any, Optional  # Adicionado Optional


class HashTable:
    def __init__(self, capacidade: int = 101,
                 fator_carga_max: float = 0.7,
                 max_elements: Optional[int] = None):  # NOVO: max_elements
        if capacidade <= 0:
            raise ValueError("Capacidade deve ser um inteiro positivo.")
        if not (0 < fator_carga_max <= 1):
            raise ValueError("Fator de carga máximo deve estar entre 0 (exclusivo) e 1 (inclusivo).")

        self.capacidade = capacidade
        self.fator_carga_max = fator_carga_max
        self.elementos = 0
        self.table: List[List[Moto]] = [[] for _ in range(capacidade)]
        self.max_elements: Optional[int] = max_elements  # NOVO ATRIBUTO para M1

    def _hash(self, chave_moto: Moto) -> int:
        return hash(chave_moto) % self.capacidade

    def _redimensionar(self) -> None:
        nova_capacidade = self.capacidade * 2 + 1
        old_table = self.table
        self.table = [[] for _ in range(nova_capacidade)]
        self.capacidade = nova_capacidade
        # Elementos serão recontados ao reinserir. Mas precisamos resetar self.elementos
        # ANTES de chamar self.inserir recursivamente, porque self.inserir incrementa.
        # E self.inserir também chama _redimensionar. Isso pode levar a um loop se
        # o fator de carga for atingido durante a reinserção antes de todos os itens serem movidos.
        # Uma maneira mais segura é construir a nova tabela e depois atribuir.

        # Abordagem mais segura para redimensionar:
        temp_table = [[] for _ in range(nova_capacidade)]
        current_elementos = 0  # Contador local para a nova tabela
        # Preserva o max_elements da instância original
        original_max_elements_setting = self.max_elements
        self.max_elements = None  # Desativa temporariamente o limite M1 durante o redimensionamento interno

        for bucket in old_table:
            for item_moto in bucket:
                # Calcula o novo índice diretamente
                idx = hash(item_moto) % nova_capacidade
                # Adiciona sem chamar self.inserir para evitar verificações de redimensionamento/limite M1 aqui
                temp_table[idx].append(item_moto)
                current_elementos += 1

        self.table = temp_table
        self.elementos = current_elementos
        self.max_elements = original_max_elements_setting  # Restaura configuração M1

    def inserir(self, data: Moto) -> bool:  # MODIFICADO: Retorna bool
        if self.max_elements is not None and self.elementos >= self.max_elements:  # NOVO: Checa M1
            # print(f"AVISO (HashTable): Capacidade M1 de {self.max_elements} atingida. Não inserindo {data.nome}.")
            return False  # Não inseriu

        if (self.elementos + 1) / self.capacidade > self.fator_carga_max:
            self._redimensionar()

        idx = self._hash(data)

        for item_existente in self.table[idx]:
            if item_existente == data:
                return False  # Duplicata, não inseriu novo

        self.table[idx].append(data)
        self.elementos += 1
        return True  # Novo item inserido com sucesso

    # ... (remover, buscar, exibir, __len__, obter_estatisticas_colisao como antes) ...
    def remover(self, alvo: Moto) -> bool:
        idx = self._hash(alvo)
        bucket = self.table[idx]
        for i, item in enumerate(bucket):
            if item == alvo: del bucket[i]; self.elementos -= 1; return True
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        idx = self._hash(alvo)
        bucket = self.table[idx]
        passos = 1
        for item in bucket:
            passos += 1
            if item == alvo: return True, passos
        return False, passos

    def exibir(self) -> None:
        print(
            f"\n{'=' * 70}\n=== TABELA HASH (Elementos: {self.elementos}, Capacidade: {self.capacidade}, Fator de Carga: {(self.elementos / self.capacidade if self.capacidade > 0 else 0):.2f}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}\n{'-' * 70}")
        count_displayed = 0
        for bucket in self.table:
            for item in bucket:
                if count_displayed < 50:
                    m = item
                    print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                    count_displayed += 1
                else:
                    break
            if count_displayed >= 50: break
        if self.elementos > count_displayed: print(f"... e mais {self.elementos - count_displayed} motos.")
        print("=" * 70)

    def __len__(self) -> int:
        return self.elementos

    def obter_estatisticas_colisao(self) -> Dict[str, Any]:
        if self.capacidade == 0: return {'fator_carga_real': 0, 'num_buckets_vazios': 0, 'num_buckets_ocupados': 0,
                                         'num_buckets_com_colisao': 0, 'max_comprimento_bucket': 0,
                                         'avg_comprimento_bucket_ocupado': 0.0,
                                         'percent_buckets_com_colisao_de_total': 0.0,
                                         'percent_buckets_com_colisao_de_ocupados': 0.0}
        n_vazios, n_colisao, n_ocupados, max_len, soma_len_ocupados = 0, 0, 0, 0, 0
        for bucket in self.table:
            tam = len(bucket)
            if tam == 0:
                n_vazios += 1
            else:
                n_ocupados += 1
                soma_len_ocupados += tam
                if tam > max_len: max_len = tam
                if tam > 1: n_colisao += 1
        avg_len_ocup = (soma_len_ocupados / n_ocupados) if n_ocupados > 0 else 0.0
        perc_col_total = (n_colisao / self.capacidade * 100) if self.capacidade > 0 else 0.0
        perc_col_ocup = (n_colisao / n_ocupados * 100) if n_ocupados > 0 else 0.0
        return {'fator_carga_real': self.elementos / self.capacidade if self.capacidade > 0 else 0,
                'num_buckets_vazios': n_vazios,
                'num_buckets_ocupados': n_ocupados, 'num_buckets_com_colisao': n_colisao,
                'max_comprimento_bucket': max_len,
                'avg_comprimento_bucket_ocupado': avg_len_ocup, 'percent_buckets_com_colisao_de_total': perc_col_total,
                'percent_buckets_com_colisao_de_ocupados': perc_col_ocup}