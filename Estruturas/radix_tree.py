from modelos.moto import Moto
from typing import Dict, List, Optional, Tuple

class RadixTree:
    class Node:
        def __init__(self, prefixo: str = ""):
            self.prefixo = prefixo
            self.filhos: Dict[str, RadixTree.Node] = {}
            self.dados: List[Moto] = []
            self.fim = False

    def __init__(self):
        self.raiz = self.Node()

    def inserir(self, data: Moto) -> None:
        self._inserir(self.raiz, data.nome, data, 0)

    def _inserir(self, node: Node, chave: str, data: Moto, profundidade: int) -> None:
        i = 0
        n = min(len(node.prefixo), len(chave) - profundidade)

        while i < n and node.prefixo[i] == chave[profundidade + i]:
            i += 1

        if i < len(node.prefixo):
            novo_prefixo = node.prefixo[i:]
            node.prefixo = node.prefixo[:i]
            novo_node = self.Node(novo_prefixo)
            novo_node.filhos = node.filhos
            novo_node.dados = node.dados
            novo_node.fim = node.fim

            node.filhos = {novo_prefixo[0]: novo_node} if novo_prefixo else {}
            node.dados = []
            node.fim = False

        profundidade += i

        if profundidade == len(chave):
            if data not in node.dados:
                node.dados.append(data)
            node.fim = True
            return

        proximo_char = chave[profundidade]
        if proximo_char not in node.filhos:
            novo_node = self.Node(chave[profundidade:])
            novo_node.dados.append(data)
            novo_node.fim = True
            node.filhos[proximo_char] = novo_node
            return

        self._inserir(node.filhos[proximo_char], chave, data, profundidade)

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        passos = [0]
        encontrado = self._buscar(self.raiz, alvo.nome, alvo, 0, passos)
        return encontrado, passos[0]

    def _buscar(self, node: Node, chave: str, alvo: Moto, profundidade: int, passos: List[int]) -> bool:
        passos[0] += 1

        # Verificar prefixo
        if profundidade + len(node.prefixo) > len(chave):
            return False

        for i, c in enumerate(node.prefixo):
            if c != chave[profundidade + i]:
                return False

        profundidade += len(node.prefixo)

        if profundidade == len(chave):
            return node.fim and alvo in node.dados

        proximo_char = chave[profundidade]
        if proximo_char not in node.filhos:
            return False

        return self._buscar(node.filhos[proximo_char], chave, alvo, profundidade, passos)

    def exibir(self) -> None:
        print("\n" + "=" * 70)
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)
        self._exibir(self.raiz)
        print("=" * 70)

    def _exibir(self, node: Node) -> None:
        if node.fim:
            for moto in node.dados:
                print(f"{moto.marca:<15}{moto.nome:<20}{moto.preco:<12.2f}{moto.revenda:<15.2f}{moto.ano:<6}")

        for filho in node.filhos.values():
            self._exibir(filho)


def validar_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Erro: Valor inválido! Tente novamente.")


def validar_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Erro: Valor inválido! Tente novamente.")


def obter_dados_moto(busca: bool = False) -> Moto:
    moto = Moto()
    moto.marca = input("Marca: ").strip()
    moto.nome = input("Modelo: ").strip()

    if not busca:
        moto.preco = validar_float("Preço: ")
        moto.revenda = validar_float("Revenda: ")
        moto.ano = validar_int("Ano: ")
    else:
        moto.preco = validar_float("Preço (-1 para qualquer): ")
        moto.revenda = validar_float("Revenda (-1 para qualquer): ")
        moto.ano = validar_int("Ano (-1 para qualquer): ")

    return moto


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