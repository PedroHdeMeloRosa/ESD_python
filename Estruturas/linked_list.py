from modelos.moto import Moto
from typing import Optional, Tuple


class LinkedList:
    class Node:
        def __init__(self, data: Moto):
            self.data = data
            self.next: Optional[LinkedList.Node] = None

    def __init__(self):
        self.head: Optional[LinkedList.Node] = None

    def inserir(self, data: Moto) -> None:
        novo_node = self.Node(data)
        novo_node.next = self.head
        self.head = novo_node

    def remover(self, alvo: Moto) -> bool:
        atual = self.head
        anterior = None

        while atual:
            if atual.data == alvo:
                if anterior:
                    anterior.next = atual.next
                else:
                    self.head = atual.next
                return True
            anterior = atual
            atual = atual.next
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        atual = self.head
        passos = 0

        while atual:
            passos += 1
            if atual.data == alvo:
                return True, passos
            atual = atual.next
        return False, passos

    def exibir(self) -> None:
        if not self.head:
            print("Lista vazia!")
            return

        print("\n" + "=" * 70)
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)

        atual = self.head
        while atual:
            m = atual.data
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
            atual = atual.next
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