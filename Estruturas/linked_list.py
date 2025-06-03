# Estruturas/linked_list.py
from modelos.moto import Moto
from typing import Optional, Tuple


class LinkedList:
    """
    Implementação de uma Lista Encadeada Simples para armazenar objetos Moto.
    A inserção é feita no início da lista (O(1)).
    Busca e remoção são O(n) no pior caso.
    """

    class Node:
        """Nó interno da lista encadeada."""
        def __init__(self, data: Moto):
            self.data = data
            self.next: Optional[LinkedList.Node] = None

    def __init__(self):
        """Inicializa uma lista encadeada vazia."""
        self.head: Optional[LinkedList.Node] = None

    def inserir(self, data: Moto) -> None:
        """
        Insere um novo dado no início da lista.
        :param data: Objeto Moto a ser inserido.
        """
        novo_node = self.Node(data)
        novo_node.next = self.head
        self.head = novo_node

    def remover(self, alvo: Moto) -> bool:
        """
        Remove a primeira ocorrência do dado alvo da lista.
        :param alvo: Objeto Moto a ser removido.
        :return: True se o item foi removido, False caso contrário.
        """
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
        """
        Busca por um dado alvo na lista.
        :param alvo: Objeto Moto a ser buscado.
        :return: Tupla (encontrado: bool, passos: int).
        """
        atual = self.head
        passos = 0

        while atual:
            passos += 1
            if atual.data == alvo:
                return True, passos
            atual = atual.next
        return False, passos

    def exibir(self) -> None:
        """Exibe todos os elementos da lista no console."""
        if not self.head:
            print("Lista vazia!")
            return

        print("\n" + "=" * 70)
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)

        atual = self.head
        count = 0
        while atual and count < 50: # Limita a exibição para não sobrecarregar o console
            m = atual.data
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
            atual = atual.next
            count += 1
        if atual:
            print(f"... e mais {self._contar_restantes(atual)} motos.")
        print("=" * 70)

    def _contar_restantes(self, node: Optional[Node]) -> int:
        count = 0
        while node:
            count += 1
            node = node.next
        return count

    def __len__(self) -> int:
        """Retorna o número de elementos na lista."""
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count