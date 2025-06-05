# Estruturas/avl_tree.py
from typing import Optional, Tuple, Any, List
from modelos.moto import Moto


def _altura_avl(node: Optional['AVLTree.Node']) -> int:
    return node.height if node else 0


def _atualizar_altura_avl(node: 'AVLTree.Node') -> None:
    if node:
        node.height = 1 + max(_altura_avl(node.left), _altura_avl(node.right))


class AVLTree:
    class Node:
        def __init__(self, data: Moto):
            self.data: Moto = data
            self.left: Optional[AVLTree.Node] = None
            self.right: Optional[AVLTree.Node] = None
            self.height: int = 1

    def __init__(self, max_elements: Optional[int] = None):  # MODIFICADO
        self.root: Optional[AVLTree.Node] = None
        self._count: int = 0
        self._search_step_limit: Optional[int] = None
        self.max_elements: Optional[int] = max_elements  # NOVO ATRIBUTO

    def set_search_step_limit(self, limit: Optional[int]):  # NOVO MÉTODO
        self._search_step_limit = limit
        # if limit is not None: print(f"INFO (AVLTree): Limite de busca -> {limit} passos.")
        # else: print("INFO (AVLTree): Limite de busca removido.")

    def _balanceamento(self, node: Optional[Node]) -> int:
        return _altura_avl(node.left) - _altura_avl(node.right) if node else 0

    def _rotacao_direita(self, y: Node) -> Node:  # ... (sem mudanças)
        x = y.left;
        T2 = x.right;
        x.right = y;
        y.left = T2
        _atualizar_altura_avl(y);
        _atualizar_altura_avl(x);
        return x

    def _rotacao_esquerda(self, x: Node) -> Node:  # ... (sem mudanças)
        y = x.right;
        T2 = y.left;
        y.left = x;
        x.right = T2
        _atualizar_altura_avl(x);
        _atualizar_altura_avl(y);
        return y

    def inserir(self, data: Moto) -> bool:
        if self.max_elements is not None and self._count >= self.max_elements:  # MODIFICADO: Checa M1
            return False  # Capacidade máxima atingida

        new_root, inserido_flag = self._inserir(self.root, data)
        self.root = new_root
        if inserido_flag:
            self._count += 1
        return inserido_flag

    def _inserir(self, node: Optional[Node], data: Moto) -> Tuple[Optional[Node], bool]:
        # ... (lógica interna de _inserir como antes, garantindo que não insere duplicatas exatas) ...
        # Retorna (nó_atualizado, True/False se inseriu novo elemento)
        inserido_flag = False
        if not node: return self.Node(data), True
        if data < node.data:
            node.left, inserido_flag = self._inserir(node.left, data)
        elif data > node.data:
            node.right, inserido_flag = self._inserir(node.right, data)
        else:  # data == node.data (ordem), checa igualdade de objeto
            if data == node.data:
                return node, False  # Duplicata
            else:
                node.right, inserido_flag = self._inserir(node.right, data)  # Política para "quase duplicata"
        if not inserido_flag: return node, False
        _atualizar_altura_avl(node);
        balance = self._balanceamento(node)
        if balance > 1:
            if data < node.left.data:
                return self._rotacao_direita(node), True
            else:
                node.left = self._rotacao_esquerda(node.left); return self._rotacao_direita(node), True
        if balance < -1:
            if data > node.right.data:
                return self._rotacao_esquerda(node), True
            else:
                node.right = self._rotacao_direita(node.right); return self._rotacao_esquerda(node), True
        return node, True

    def _min_value_node(self, node: Node) -> Node:  # ... (sem mudanças)
        current = node;
        while current.left: current = current.left; return current

    def remover(self, alvo: Moto) -> bool:  # ... (sem mudanças na lógica de remoção em si)
        new_root, removido = self._remover(self.root, alvo)
        self.root = new_root
        if removido: self._count -= 1
        return removido

    def _remover(self, node: Optional[Node], alvo: Moto) -> Tuple[Optional[Node], bool]:
        # ... (lógica interna de _remover como antes) ...
        if not node: return None, False
        removido_flag = False
        if alvo < node.data:
            node.left, removido_flag = self._remover(node.left, alvo)
        elif alvo > node.data:
            node.right, removido_flag = self._remover(node.right, alvo)
        else:
            if alvo == node.data:
                removido_flag = True
                if not node.left or not node.right:
                    temp = node.left or node.right; node = None; return temp, True
                else:
                    temp = self._min_value_node(node.right); node.data = temp.data; node.right, _ = self._remover(
                        node.right, temp.data)
            else:
                node.right, removido_flag = self._remover(node.right, alvo)  # Busca à direita para "quase duplicata"
        if not node: return node, removido_flag
        if not removido_flag: return node, False
        _atualizar_altura_avl(node);
        balance = self._balanceamento(node)
        if balance > 1:
            if self._balanceamento(node.left) >= 0:
                return self._rotacao_direita(node), True
            else:
                node.left = self._rotacao_esquerda(node.left); return self._rotacao_direita(node), True
        if balance < -1:
            if self._balanceamento(node.right) <= 0:
                return self._rotacao_esquerda(node), True
            else:
                node.right = self._rotacao_direita(node.right); return self._rotacao_esquerda(node), True
        return node, True

    def _buscar_recursive(self, node: Optional[Node], alvo: Moto, passos_ref: List[int]) -> bool:
        if not node: return False
        passos_ref[0] += 1
        if self._search_step_limit is not None and passos_ref[0] > self._search_step_limit:  # MODIFICADO: Checa A1
            return False
        if alvo == node.data: return True
        if alvo < node.data:
            return self._buscar_recursive(node.left, alvo, passos_ref)
        else:
            return self._buscar_recursive(node.right, alvo, passos_ref)

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        passos_container = [0]
        encontrado = self._buscar_recursive(self.root, alvo, passos_container)
        return encontrado, passos_container[0]

    def exibir(self) -> None:  # ... (sem mudanças)
        if not self.root: print("Árvore AVL vazia!"); return
        print(f"\n{'=' * 70}\n=== ÁRVORE AVL (Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}\n{'-' * 70}")
        self._displayed_count = 0;
        self._em_ordem(self.root)
        if self._count > self._displayed_count: print(
            f"... e mais {self._count - self._displayed_count} motos não exibidas.")
        print("=" * 70)

    def _em_ordem(self, node: Optional[Node]) -> None:  # ... (sem mudanças)
        if node and self._displayed_count < 50:
            self._em_ordem(node.left)
            if self._displayed_count < 50:
                m = node.data;
                print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                self._displayed_count += 1
                if self._displayed_count < 50: self._em_ordem(node.right)
            else:
                return
        elif node and self._displayed_count >= 50:
            return

    def __len__(self) -> int:
        return self._count