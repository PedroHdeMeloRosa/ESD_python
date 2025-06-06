# Estruturas/linked_list.py
from modelos.moto import Moto
from typing import Optional, Tuple, Any  # Adicionado Any para o futuro max_elements


class LinkedList:
    class Node:
        def __init__(self, data: Moto):
            self.data = data
            self.next: Optional[LinkedList.Node] = None

    def __init__(self, capacidade_maxima: Optional[int] = None,
                 max_elements: Optional[int] = None):  # Adicionado max_elements para M1
        self.head: Optional[LinkedList.Node] = None
        self.tail: Optional[LinkedList.Node] = None
        self._count = 0

        # Prioridade: capacidade_maxima (LRU/FIFO) tem precedência se ambos forem fornecidos.
        # Se M1 (max_elements) for aplicado à LinkedList, ela não deve descartar, apenas parar de inserir.
        # Se M2 (capacidade_maxima) for aplicado, ela descarta.
        # Para simplificar, vamos ter UMA forma de limitar.
        # Se max_elements for passado (para simulação M1), usamos ele e não descartamos.
        # Se capacidade_maxima for passada (para simulação M2), usamos ela e descartamos.

        self.max_elements_M1: Optional[int] = max_elements
        self.capacidade_maxima_M2: Optional[int] = capacidade_maxima

        if self.max_elements_M1 is not None and self.capacidade_maxima_M2 is not None:
            print("AVISO (LinkedList): Ambas 'max_elements' (M1) e 'capacidade_maxima' (M2) fornecidas. "
                  "A 'capacidade_maxima' com descarte terá precedência se for menor ou igual, "
                  "caso contrário, 'max_elements' sem descarte será o limite efetivo.")
            # A lógica de inserção precisará lidar com essa precedência.
            # Por ora, vamos assumir que apenas uma dessas restrições (M1 ou M2) será ativa por vez
            # através da configuração no StructureAnalyzer.

    def inserir(self, data: Moto) -> bool:
        # Lógica para Restrição M1 (limite_max_elementos) - Não descarta, apenas para de inserir
        if self.max_elements_M1 is not None and self.capacidade_maxima_M2 is None:  # Apenas M1 ativa
            if self._count >= self.max_elements_M1:
                # print(f"DEBUG (LL M1): Limite de {self.max_elements_M1} atingido. Não inserindo {data.nome}")
                return False  # Falha ao inserir, capacidade M1 atingida

        novo_node = self.Node(data)

        # Lógica para Restrição M2 (capacidade_maxima_M2 com descarte LRU/FIFO)
        if self.capacidade_maxima_M2 is not None and self._count >= self.capacidade_maxima_M2:
            if self.head:
                # print(f"DEBUG (LL M2): Capacidade {self.capacidade_maxima_M2} atingida. Removendo {self.head.data.nome}")
                self.head = self.head.next
                self._count -= 1
                if self.head is None:
                    self.tail = None

        # Inserção no final (comportamento FIFO para o descarte M2)
        if self.head is None:
            self.head = novo_node
            self.tail = novo_node
        else:
            if self.tail:
                self.tail.next = novo_node
            self.tail = novo_node
        self._count += 1
        return True  # Inserção sempre bem-sucedida (ou um item foi descartado para dar espaço)

    def remover(self, alvo: Moto) -> bool:
        atual = self.head
        anterior = None
        while atual:
            if atual.data == alvo:
                if anterior:
                    anterior.next = atual.next
                    if atual == self.tail:
                        self.tail = anterior
                else:
                    self.head = atual.next
                    if self.head is None:
                        self.tail = None
                self._count -= 1
                return True
            anterior = atual
            atual = atual.next
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        atual = self.head;
        passos = 0
        while atual:
            passos += 1
            if atual.data == alvo: return True, passos
            atual = atual.next
        return False, passos

    def exibir(self) -> None:
        if not self.head: print("Lista Encadeada vazia!"); return
        # Adicionado nome da estrutura na exibição para clareza
        print(f"\n{'=' * 70}\n=== LISTA ENCADEADA (Elementos: {self._count}" +
              (f", Capacidade M1: {self.max_elements_M1}" if self.max_elements_M1 is not None else "") +
              (f", Capacidade M2 (LRU): {self.capacidade_maxima_M2}" if self.capacidade_maxima_M2 is not None else "") +
              f") ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}\n{'-' * 70}")
        atual = self.head;
        count_disp = 0
        while atual and count_disp < 50:
            m = atual.data;
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
            atual = atual.next;
            count_disp += 1
        if atual: print(f"... e mais {self._count - count_disp} motos não exibidas.")
        print("=" * 70)

    def __len__(self) -> int:
        return self._count  # CORRIGIDO de self._counts