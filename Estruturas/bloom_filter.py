# Estruturas/bloom_filter.py
import math
import mmh3  # Usar murmurhash3 para melhor dispersão e velocidade
from modelos.moto import Moto
from typing import List


class BloomFilter:
    """
    Implementação de um Bloom Filter.
    Estrutura de dados probabilística para testar se um elemento é membro de um conjunto.
    Pode haver falsos positivos, mas não falsos negativos.
    A remoção de elementos não é suportada diretamente em um Bloom Filter padrão.
    """

    def __init__(self, num_itens_esperados: int = 1000, taxa_falso_positivo: float = 0.01):
        """
        Inicializa o Bloom Filter.
        :param num_itens_esperados: Número estimado de itens que serão inseridos.
        :param taxa_falso_positivo: Taxa de falsos positivos desejada (ex: 0.01 para 1%).
        """
        if num_itens_esperados <= 0:
            raise ValueError("Número de itens esperados deve ser positivo.")
        if not (0 < taxa_falso_positivo < 1):
            raise ValueError("Taxa de falso positivo deve estar entre 0 e 1 (exclusivos).")

        self.num_itens_esperados = num_itens_esperados
        self.taxa_falso_positivo = taxa_falso_positivo

        # Calcular tamanho do array de bits (m) e número de funções hash (k)
        # m = - (n * ln(p)) / (ln(2)^2)
        self.size = self._calcular_tamanho_otimo(num_itens_esperados, taxa_falso_positivo)
        # k = (m / n) * ln(2)
        self.num_hashes = self._calcular_num_hashes_otimo(self.size, num_itens_esperados)

        self.bits: List[bool] = [False] * self.size
        self.num_elementos_inseridos = 0  # Para estimar a taxa de FP real

    def _calcular_tamanho_otimo(self, n: int, p: float) -> int:
        """Calcula o tamanho ótimo do array de bits (m)."""
        m = - (n * math.log(p)) / (math.log(2) ** 2)
        return math.ceil(m)

    def _calcular_num_hashes_otimo(self, m: int, n: int) -> int:
        """Calcula o número ótimo de funções hash (k)."""
        if n == 0: return 1  # Evita divisão por zero se n=0
        k = (m / n) * math.log(2)
        return math.ceil(k)

    def _hash_moto_para_string(self, data: Moto) -> str:
        """Converte um objeto Moto em uma string consistente para hashing."""
        # Ordenar os atributos pode ser útil se a ordem de construção do objeto variar
        return f"{data.marca}|{data.nome}|{data.preco}|{data.revenda}|{data.ano}"

    def _get_hashes(self, data_str: str) -> List[int]:
        """
        Gera múltiplos hashes para o dado usando MurmurHash3 com diferentes seeds.
        """
        hashes = []
        for i in range(self.num_hashes):
            # mmh3.hash() produz um hash de 32 bits. Se size for maior, precisa de mais cuidado.
            # mmh3.hash128() pode ser uma alternativa para mais bits.
            # Para este exemplo, um hash de 32 bits (com seed) módulo size é usado.
            # Usar sementes diferentes para simular múltiplas funções hash.
            h = mmh3.hash(data_str.encode('utf-8'), seed=i) % self.size
            hashes.append(h)
        return hashes

    def inserir(self, data: Moto) -> None:
        """
        Insere um objeto Moto no Bloom Filter.
        :param data: Objeto Moto a ser inserido.
        """
        data_str = self._hash_moto_para_string(data)
        ja_presente_provavel = True  # Assumir que pode já estar presente
        for idx in self._get_hashes(data_str):
            if not self.bits[idx]:
                ja_presente_provavel = False  # Encontrou um bit 0, então definitivamente não estava
            self.bits[idx] = True

        if not ja_presente_provavel:  # Só incrementa se havia pelo menos um bit 0
            self.num_elementos_inseridos += 1

    def buscar(self, data: Moto) -> bool:
        """
        Verifica se um objeto Moto pode estar no Bloom Filter.
        :param data: Objeto Moto a ser buscado.
        :return: True se o item POSSIVELMENTE está no filtro (pode ser falso positivo).
                 False se o item DEFINITIVAMENTE não está no filtro.
        """
        data_str = self._hash_moto_para_string(data)
        for idx in self._get_hashes(data_str):
            if not self.bits[idx]:
                return False  # Definitivamente não está
        return True  # Possivelmente está (pode ser falso positivo)

    def taxa_falso_positivo_real_estimada(self) -> float:
        """
        Estima a taxa de falsos positivos atual baseada no número de bits setados.
        p = (1 - e^(-k*n/m))^k
        """
        if self.size == 0 or self.num_elementos_inseridos == 0:
            return 0.0

        # n_estimado pode ser self.num_elementos_inseridos
        # No entanto, a fórmula clássica usa a fração de bits setados para estimar n/m
        # Se x é a fração de bits setados (bits_setados / m), então:
        # x = 1 - (1 - 1/m)^(k*n_estimado)  ~  1 - e^(-k*n_estimado/m)
        # log(1-x) = -k*n_estimado/m
        # n_estimado = - (m/k) * log(1-x)

        bits_setados = self.bits.count(True)
        if bits_setados == self.size:  # Se todos os bits estão setados, FP é próximo de 1
            return 1.0

        try:
            # n_estimado_pela_carga = (-self.size / self.num_hashes) * math.log(1 - (bits_setados / self.size))
            # Usando self.num_elementos_inseridos diretamente para n:
            expoente = - (self.num_hashes * self.num_elementos_inseridos) / self.size
            taxa_fp = (1 - math.exp(expoente)) ** self.num_hashes
            return taxa_fp
        except (ValueError, OverflowError):  # math.log(0) or overflow
            return 1.0  # Pior caso

    def exibir(self) -> None:
        """Exibe informações sobre o Bloom Filter."""
        bits_setados = self.bits.count(True)
        carga_filtro = bits_setados / self.size if self.size > 0 else 0

        print(f"\n{'=' * 70}")
        print(f"=== BLOOM FILTER ===")
        print(f"  Tamanho do array de bits (m): {self.size}")
        print(f"  Número de funções hash (k): {self.num_hashes}")
        print(f"  Número de itens esperados (n): {self.num_itens_esperados}")
        print(f"  Taxa de Falso Positivo Configurada (p): {self.taxa_falso_positivo:.4f}")
        print(f"  Número de elementos únicos inseridos: {self.num_elementos_inseridos}")
        print(f"  Bits setados: {bits_setados} ({carga_filtro * 100:.2f}% de carga)")
        print(f"  Taxa de Falso Positivo Real Estimada: {self.taxa_falso_positivo_real_estimada():.6f}")
        print("=" * 70)

    def __len__(self) -> int:
        """Retorna o número de elementos únicos que foram tentados inserir."""
        return self.num_elementos_inseridos