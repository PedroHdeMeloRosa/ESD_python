# Estruturas/linked_list.py
from modelos.moto import Moto
from typing import Optional, Tuple


class LinkedList:
    class Node:
        def __init__(self, data: Moto):
            self.data = data
            self.next: Optional[LinkedList.Node] = None

    def __init__(self, capacidade_maxima: Optional[int] = None):  # NOVO: capacidade
        self.head: Optional[LinkedList.Node] = None
        self.tail: Optional[LinkedList.Node] = None  # Para remoção LRU (do início) O(1)
        self._count = 0
        self.capacidade_maxima = capacidade_maxima  # NOVO

    def inserir(self, data: Moto) -> None:  # Modificado para inserir no FINAL (FIFO para LRU)
        novo_node = self.Node(data)
        if self.capacidade_maxima is not None and self._count >= self.capacidade_maxima:
            if self.head:  # Remove o mais antigo (cabeça) se a capacidade foi atingida
                # print(f"DEBUG: Capacidade {self.capacidade_maxima} atingida. Removendo {self.head.data.nome}")
                self.head = self.head.next
                self._count -= 1
                if self.head is None:  # Lista ficou vazia após remoção
                    self.tail = None

        if self.head is None:  # Lista vazia
            self.head = novo_node
            self.tail = novo_node
        else:  # Insere no final
            if self.tail:  # Deve sempre existir se head não é None
                self.tail.next = novo_node
            self.tail = novo_node
        self._count += 1

    def remover(self, alvo: Moto) -> bool:
        atual = self.head
        anterior = None
        while atual:
            if atual.data == alvo:
                if anterior:
                    anterior.next = atual.next
                    if atual == self.tail:  # Se o removido era o tail
                        self.tail = anterior
                else:  # Removendo a cabeça
                    self.head = atual.next
                    if self.head is None:  # Lista ficou vazia
                        self.tail = None
                self._count -= 1
                return True
            anterior = atual
            atual = atual.next
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        # ... (como antes) ...
        atual = self.head;
        passos = 0
        while atual:
            passos += 1
            if atual.data == alvo: return True, passos
            atual = atual.next
        return False, passos

    def exibir(self) -> None:
        # ... (como antes) ...
        if not self.head: print("Lista vazia!"); return
        print("\n" + "=" * 70);
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço':<12}{'Revenda':<15}{'Ano':<6}");
        print("-" * 70)
        atual = self.head;
        count_disp = 0
        while atual and count_disp < 50:
            m = atual.data;
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
            atual = atual.next;
            count_disp += 1
        if atual: print(f"... e mais {self._count - count_disp} motos.")
        print("=" * 70)

    def __len__(self) -> int:
        return self._count