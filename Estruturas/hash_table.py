from modelos.moto import Moto
from typing import List, Tuple


class HashTable:
    def __init__(self, capacidade: int = 101, fator_carga_max: float = 0.7):
        self.capacidade = capacidade
        self.fator_carga_max = fator_carga_max
        self.elementos = 0
        self.table: List[List[Moto]] = [[] for _ in range(capacidade)]

    def _hash(self, chave: Moto) -> int:
        return hash(chave) % self.capacidade

    def _redimensionar(self) -> None:
        nova_capacidade = self.capacidade * 2 + 1
        nova_tabela = [[] for _ in range(nova_capacidade)]

        for bucket in self.table:
            for item in bucket:
                idx = hash(item) % nova_capacidade
                nova_tabela[idx].append(item)

        self.table = nova_tabela
        self.capacidade = nova_capacidade

    def inserir(self, data: Moto) -> None:
        if (self.elementos + 1) / self.capacidade > self.fator_carga_max:
            self._redimensionar()

        idx = self._hash(data)
        self.table[idx].append(data)
        self.elementos += 1

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
        passos = 0

        for item in bucket:
            passos += 1
            if item == alvo:
                return True, passos
        return False, passos

    def exibir(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"=== TABELA HASH ({self.elementos}/{self.capacidade}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)

        for bucket in self.table:
            for item in bucket:
                m = item
                print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
        print("=" * 70)

    def medir_desempenho(self, operacao, *args):
        """Executa operação com medição de desempenho"""
        import time
        import tracemalloc

        tracemalloc.start()
        start_time = time.perf_counter()

        if operacao == 'inserir':
            resultado = self.inserir(*args)
        elif operacao == 'buscar':
            resultado = self.buscar(*args)
        elif operacao == 'remover':
            resultado = self.remover(*args)
        else:
            raise ValueError("Operação inválida")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.perf_counter()

        return {
            'tempo_ms': (end_time - start_time) * 1000,
            'memoria_kb': peak / 1024,
            'resultado': resultado
        }