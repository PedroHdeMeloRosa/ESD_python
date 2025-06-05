# Estruturas/hash_table.py
from modelos.moto import Moto
from typing import List, Tuple, Dict, Any  # Adicionado Dict, Any


class HashTable:
    """
    Implementação de uma Tabela Hash com tratamento de colisão por encadeamento.
    Utiliza o hash embutido do Python para objetos Moto.
    Realiza redimensionamento quando o fator de carga máximo é atingido.
    """

    def __init__(self, capacidade: int = 101, fator_carga_max: float = 0.7):
        if capacidade <= 0:
            raise ValueError("Capacidade deve ser um inteiro positivo.")
        if not (0 < fator_carga_max <= 1):
            raise ValueError("Fator de carga máximo deve estar entre 0 (exclusivo) e 1 (inclusivo).")

        self.capacidade = capacidade
        self.fator_carga_max = fator_carga_max
        self.elementos = 0
        self.table: List[List[Moto]] = [[] for _ in range(capacidade)]
        # self.numero_colisoes_na_insercao = 0 # Opcional: rastrear colisões na inserção

    def _hash(self, chave_moto: Moto) -> int:
        return hash(chave_moto) % self.capacidade

    def _redimensionar(self) -> None:
        nova_capacidade = self.capacidade * 2 + 1
        # print(f"Redimensionando tabela hash de {self.capacidade} para {nova_capacidade}...") # Opcional

        old_table = self.table
        self.table = [[] for _ in range(nova_capacidade)]
        self.capacidade = nova_capacidade
        self.elementos = 0  # Será recontado durante a reinserção
        # self.numero_colisoes_na_insercao = 0 # Reseta se estiver rastreando

        for bucket in old_table:
            for item_moto in bucket:
                self.inserir(item_moto)  # Reinserir para recalcular hash e tratar colisões na nova tabela

    def inserir(self, data: Moto) -> None:
        if (self.elementos + 1) / self.capacidade > self.fator_carga_max:
            self._redimensionar()

        idx = self._hash(data)

        # Verifica duplicatas e colisões
        bucket_vazio_antes = not self.table[idx]

        for item_existente in self.table[idx]:
            if item_existente == data:
                return  # Item já existe, não insere novamente

        # Se chegou aqui, o item é novo para este bucket.
        self.table[idx].append(data)
        self.elementos += 1

        # if not bucket_vazio_antes: # Se o bucket já tinha algo, é uma colisão (para este slot)
        # self.numero_colisoes_na_insercao += 1 # Opcional

    def remover(self, alvo: Moto) -> bool:
        idx = self._hash(alvo)
        bucket = self.table[idx]
        for i, item in enumerate(bucket):
            if item == alvo:
                del bucket[i]
                self.elementos -= 1
                return True
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        idx = self._hash(alvo)
        bucket = self.table[idx]
        passos = 1  # 1 passo para calcular o hash e acessar o bucket

        for item in bucket:
            passos += 1  # 1 passo para cada comparação no bucket
            if item == alvo:
                return True, passos
        return False, passos

    def exibir(self) -> None:
        # ... (código de exibição como antes) ...
        print(f"\n{'=' * 70}")
        print(
            f"=== TABELA HASH (Elementos: {self.elementos}, Capacidade: {self.capacidade}, Fator de Carga: {self.elementos / self.capacidade:.2f}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)
        # ... (loop de exibição como antes, limitado a 50) ...
        count_displayed = 0
        for i, bucket in enumerate(self.table):
            if bucket:
                for item in bucket:
                    if count_displayed < 50:
                        m = item
                        print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                        count_displayed += 1
                    else:
                        break
            if count_displayed >= 50:
                break
        if self.elementos > count_displayed:
            print(f"... e mais {self.elementos - count_displayed} motos não exibidas.")
        print("=" * 70)

    def __len__(self) -> int:
        return self.elementos

    # --- NOVO MÉTODO PARA ESTATÍSTICAS DE COLISÃO ---
    def obter_estatisticas_colisao(self) -> Dict[str, Any]:
        """
        Calcula e retorna estatísticas sobre a distribuição de itens e colisões na tabela hash.
        """
        if self.capacidade == 0:  # Evita divisão por zero se a tabela for mal inicializada
            return {
                'fator_carga_real': 0,
                'num_buckets_vazios': 0,
                'num_buckets_ocupados': 0,
                'num_buckets_com_colisao': 0,
                'max_comprimento_bucket': 0,
                'avg_comprimento_bucket_ocupado': 0.0,
                'percent_buckets_com_colisao': 0.0,
                # 'colisoes_na_insercao': self.numero_colisoes_na_insercao # Opcional
            }

        num_buckets_vazios = 0
        num_buckets_com_colisao = 0
        num_buckets_ocupados = 0
        max_comprimento_bucket = 0
        soma_comprimentos_ocupados = 0

        for bucket in self.table:
            tam_bucket = len(bucket)
            if tam_bucket == 0:
                num_buckets_vazios += 1
            else:
                num_buckets_ocupados += 1
                soma_comprimentos_ocupados += tam_bucket
                if tam_bucket > max_comprimento_bucket:
                    max_comprimento_bucket = tam_bucket
                if tam_bucket > 1:
                    num_buckets_com_colisao += 1

        avg_comprimento_bucket_ocupado = (soma_comprimentos_ocupados / num_buckets_ocupados) \
            if num_buckets_ocupados > 0 else 0.0

        percent_buckets_com_colisao = (num_buckets_com_colisao / self.capacidade * 100) \
            if self.capacidade > 0 else 0.0

        return {
            'fator_carga_real': self.elementos / self.capacidade if self.capacidade > 0 else 0,
            'num_buckets_vazios': num_buckets_vazios,
            'num_buckets_ocupados': num_buckets_ocupados,
            'num_buckets_com_colisao': num_buckets_com_colisao,
            'max_comprimento_bucket': max_comprimento_bucket,
            'avg_comprimento_bucket_ocupado': avg_comprimento_bucket_ocupado,
            'percent_buckets_com_colisao_de_total': percent_buckets_com_colisao,
            'percent_buckets_com_colisao_de_ocupados': (
                        num_buckets_com_colisao / num_buckets_ocupados * 100) if num_buckets_ocupados > 0 else 0.0
            # 'colisoes_na_insercao': self.numero_colisoes_na_insercao # Opcional
        }