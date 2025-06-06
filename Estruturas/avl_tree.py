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

    def inserir(self, data: Moto) -> bool:  # (como na última versão corrigida)
        if self.max_elements is not None and self._count >= self.max_elements:
            return False
        new_root, inserido_flag = self._inserir(self.root, data)
        self.root = new_root
        if inserido_flag:
            self._count += 1
        return inserido_flag

    def _inserir(self, node: Optional[Node], data: Moto) -> Tuple[Optional[Node], bool]:
        inserido_flag = False
        if not node:
            return self.Node(data), True

        if data < node.data:
            node.left, inserido_flag = self._inserir(node.left, data)
        elif data > node.data:
            node.right, inserido_flag = self._inserir(node.right, data)
        else:
            if data == node.data:
                return node, False  # Duplicata
            else:
                node.right, inserido_flag = self._inserir(node.right, data)  # Política para "quase duplicata"

        if not inserido_flag: return node, False

        _atualizar_altura_avl(node)
        balance = self._balanceamento(node)

        # --- LÓGICA DE ROTAÇÃO CORRIGIDA ---
        if balance > 1:  # Desbalanceado à Esquerda
            if node.left is None: return node, True  # Segurança, mas indica problema
            if self._balanceamento(node.left) >= 0:  # LL ou L0
                return self._rotacao_direita(node), True
            else:  # LR
                if node.left is not None:
                    node.left = self._rotacao_esquerda(node.left)
                else:
                    return node, True
                return self._rotacao_direita(node), True

        if balance < -1:  # Desbalanceado à Direita
            if node.right is None: return node, True  # Segurança
            if self._balanceamento(node.right) <= 0:  # RR ou R0
                return self._rotacao_esquerda(node), True
            else:  # RL
                if node.right is not None:
                    node.right = self._rotacao_direita(node.right)
                else:
                    return node, True
                return self._rotacao_esquerda(node), True

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
        if not node: return None, False
        removido_flag = False

        if alvo < node.data:
            node.left, removido_flag = self._remover(node.left, alvo)
        elif alvo > node.data:
            node.right, removido_flag = self._remover(node.right, alvo)
        else:  # alvo tem mesma "posição de ordenação" que node.data
            print(
                f"DEBUG REMOVER: Tentando remover {alvo.nome}. Nó atual: {node.data.nome if node else 'NoneNode'}, count: {self._count}")
            if alvo == node.data:  # Objeto exato encontrado
                removido_flag = True
                print(f"DEBUG REMOVER: Objeto exato encontrado: {node.data.nome}")
                if not node.left or not node.right:  # 0 ou 1 filho
                    print(f"DEBUG REMOVER: Nó {node.data.nome} tem 0 ou 1 filho.")
                    temp = node.left if node.left else node.right
                    # node = None # Não faça isso aqui, o chamador ajusta o pai
                    return temp, True
                else:  # 2 filhos
                    print(
                        f"DEBUG REMOVER: Nó {node.data.nome} tem 2 filhos. node.right: {'Existe' if node.right else 'None'}")
                    if node.right is None:  # Segurança extra, não deveria acontecer
                        print(f"ERRO CRÍTICO REMOVER: Nó {node.data.nome} marcado com 2 filhos, mas node.right é None!")
                        return node, False  # Evita crash, mas indica bug

                    temp = self._min_value_node(node.right)
                    print(f"DEBUG REMOVER: Sucessor encontrado: {temp.data.nome if temp else 'NoneTemp'}")
                    if temp is None:  # Segurança extra
                        print(
                            f"ERRO CRÍTICO REMOVER: _min_value_node retornou None para node.right de {node.data.nome}")
                        return node, False

                    print(f"DEBUG REMOVER: Copiando {temp.data.nome} para {node.data.nome}")
                    node.data = temp.data  # Copia dados do sucessor
                    print(f"DEBUG REMOVER: Removendo recursivamente o sucessor {temp.data.nome} da subárvore direita.")
                    node.right, _ = self._remover(node.right, temp.data)

        if not node: return node, removido_flag  # Nó foi removido (caso de 0 ou 1 filho)
        if not removido_flag: return node, False  # Não removeu na sub-árvore, não balancear

        _atualizar_altura_avl(node)
        balance = self._balanceamento(node)

        # Rebalanceamento após remoção (lógica similar à inserção, mas checa balanceamento dos filhos)
        # Caso Esquerdo (Subárvore direita ficou "mais curta")
        if balance > 1:
            if node.left is None:  # Segurança
                # print("ALERTA AVL: node.left é None com balance > 1 em _remover")
                return node, True
            if self._balanceamento(node.left) >= 0:  # LL ou L0 (filho esquerdo balanceado ou pesado à esquerda)
                return self._rotacao_direita(node), True
            else:  # LR (filho esquerdo pesado à direita)
                if node.left is not None:
                    node.left = self._rotacao_esquerda(node.left)
                else:  # print("ALERTA AVL: node.left é None em _remover caso LR");
                    return node, True
                return self._rotacao_direita(node), True

        # Caso Direito (Subárvore esquerda ficou "mais curta")
        if balance < -1:
            if node.right is None:  # Segurança
                # print("ALERTA AVL: node.right é None com balance < -1 em _remover")
                return node, True
            if self._balanceamento(node.right) <= 0:  # RR ou R0
                return self._rotacao_esquerda(node), True
            else:  # RL
                if node.right is not None:
                    node.right = self._rotacao_direita(node.right)
                else:  # print("ALERTA AVL: node.right é None em _remover caso RL");
                    return node, True
                return self._rotacao_esquerda(node), True

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