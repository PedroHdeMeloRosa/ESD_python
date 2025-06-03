import math
from modelos.moto import Moto

class BloomFilter:
    def __init__(self, size: int = 100000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [False] * size

    def _hash(self, data: Moto, seed: int) -> int:
        chave = f"{data.marca}{data.nome}{data.preco}{data.revenda}{data.ano}{seed}"
        return hash(chave) % self.size

    def inserir(self, data: Moto) -> None:
        for i in range(self.num_hashes):
            idx = self._hash(data, i)
            self.bits[idx] = True

    def buscar(self, data: Moto) -> bool:
        for i in range(self.num_hashes):
            idx = self._hash(data, i)
            if not self.bits[idx]:
                return False
        return True

    def exibir(self) -> None:
        count = sum(1 for bit in self.bits if bit)
        print(f"\n{'='*70}")
        print(f"=== BLOOM FILTER ===")
        print(f"Elementos aproximados: {count // self.num_hashes}")
        print(f"Taxa de falsos positivos: {math.pow(1 - math.exp(-self.num_hashes * count / self.size), self.num_hashes):.6f}")
        print("="*70)
