# Estruturas/avl_tree.py
from typing import Optional, Tuple, Any, List  # Adicionado List
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

    def __init__(self, max_elements: Optional[int] = None):  # NOVO: max_elements
        self.root: Optional[AVLTree.Node] = None
        self._count: int = 0
        self._search_step_limit: Optional[int] = None  # NOVO: para Restrição A1
        self.max_elements: Optional[int] = max_elements  # NOVO: para Restrição M1

    def set_search_step_limit(self, limit: Optional[int]):  # NOVO MÉTODO
        """Define um limite máximo de passos para a operação de busca."""
        self._search_step_limit = limit
        if limit is not None:
            print(f"INFO (AVLTree): Limite de busca definido para {limit} passos.")
        else:
            print("INFO (AVLTree): Limite de busca removido.")

    def _balanceamento(self, node: Optional[Node]) -> int:
        return _altura_avl(node.left) - _altura_avl(node.right) if node else 0

    def _rotacao_direita(self, y: Node) -> Node:
        x = y.left
        if x is None: return y
        T2 = x.right
        x.right = y;
        y.left = T2
        _atualizar_altura_avl(y);
        _atualizar_altura_avl(x)
        return x

    def _rotacao_esquerda(self, x: Node) -> Node:
        y = x.right
        if y is None: return x
        T2 = y.left
        y.left = x;
        x.right = T2
        _atualizar_altura_avl(x);
        _atualizar_altura_avl(y)
        return y

    def inserir(self, data: Moto) -> bool:  # Modificado para retornar bool (inserido ou não)
        if self.max_elements is not None and self._count >= self.max_elements:  # NOVO: Checa M1
            # print(f"AVISO (AVLTree): Capacidade máxima de {self.max_elements} elementos atingida. Não inserindo {data.nome}.")
            return False  # Não inseriu

        # Verifica se a chave já existe para não incrementar _count duas vezes se _inserir retornar False
        # Esta busca preliminar pode adicionar overhead, mas garante a contagem correta de _count.
        # Uma alternativa é _inserir retornar um status mais detalhado (inserido_novo, ja_existia).
        # Por ora, vamos confiar que _inserir não insere duplicatas exatas se data == node.data.

        new_root, inserido_flag = self._inserir(self.root, data)
        self.root = new_root  # Atualiza a raiz
        if inserido_flag:
            self._count += 1
        return inserido_flag

    def _inserir(self, node: Optional[Node], data: Moto) -> Tuple[Optional[Node], bool]:
        inserido_flag = False
        if not node:
            return self.Node(data), True

        if data < node.data:  # Usando __lt__ da Moto
            node.left, inserido_flag = self._inserir(node.left, data)
        elif data > node.data:  # Para tratar o caso de > ou se __lt__ não for estrito
            node.right, inserido_flag = self._inserir(node.right, data)
        else:  # data == node.data (ou nome e preço são iguais)
            if data == node.data:  # Duplicata exata, não insere, não incrementa _count
                return node, False
            else:
                # Se nome e preço são iguais, mas outros atributos são diferentes (baseado no __eq__ da Moto),
                # a política de __lt__ da Moto deve decidir para qual lado ir.
                # Assumindo que __lt__ desempata corretamente (ex: por outro atributo ou trata como igual)
                # Se __lt__ e __gt__ não cobrem, e __eq__ falha, onde vai?
                # A lógica atual: se data não é < nem >, mas não é ==, vai para a direita (pode ser uma política).
                node.right, inserido_flag = self._inserir(node.right, data)

        if not inserido_flag: return node, False

        _atualizar_altura_avl(node)
        balance = self._balanceamento(node)

        # Casos de Rotação
        if balance > 1:  # Desbalanceado à Esquerda
            # Verifica se é Esquerda-Esquerda ou Esquerda-Direita
            if data < node.left.data:  # Inserido na subárvore esquerda do filho esquerdo
                return self._rotacao_direita(node), True
            else:  # Inserido na subárvore direita do filho esquerdo
                node.left = self._rotacao_esquerda(node.left)
                return self._rotacao_direita(node), True

        if balance < -1:  # Desbalanceado à Direita
            # Verifica se é Direita-Direita ou Direita-Esquerda
            if data > node.right.data:  # Inserido na subárvore direita do filho direito
                return self._rotacao_esquerda(node), True
            else:  # Inserido na subárvore esquerda do filho direito
                node.right = self._rotacao_direita(node.right)
                return self._rotacao_esquerda(node), True

        return node, True

    def _min_value_node(self, node: Node) -> Node:
        current = node
        while current.left: current = current.left
        return current

    def remover(self, alvo: Moto) -> bool:
        new_root, removido = self._remover(self.root, alvo)
        self.root = new_root  # Atualiza a raiz
        if removido:
            self._count -= 1
        return removido

    def _remover(self, node: Optional[Node], alvo: Moto) -> Tuple[Optional[Node], bool]:
        if not node: return None, False
        removido_flag = False

        if alvo < node.data:
            node.left, removido_flag = self._remover(node.left, alvo)
        elif alvo > node.data:  # Se __lt__ é implementado, e não é < e não é ==, então é >
            node.right, removido_flag = self._remover(node.right, alvo)
        else:  # alvo == node.data (de acordo com a ordenação, agora checa com __eq__)
            if alvo == node.data:  # Objeto exato encontrado
                removido_flag = True
                if not node.left or not node.right:
                    temp = node.left if node.left else node.right
                    node = None
                    return temp, True
                else:
                    temp = self._min_value_node(node.right)
                    node.data = temp.data
                    node.right, _ = self._remover(node.right, temp.data)  # Passa temp.data
            else:  # Mesma posição na ordenação, mas não o mesmo objeto (raro se __lt__ e __eq__ são consistentes)
                # Tenta buscar na direita, seguindo a política de inserção para "quase duplicatas"
                node.right, removido_flag = self._remover(node.right, alvo)

        if not node: return node, removido_flag
        if not removido_flag: return node, False  # Se não removeu na sub-árvore, não há o que balancear

        _atualizar_altura_avl(node)
        balance = self._balanceamento(node)

        # Rotações após remoção
        if balance > 1:  # Esquerda
            if self._balanceamento(node.left) >= 0:  # Esquerda-Esquerda
                return self._rotacao_direita(node), True
            else:  # Esquerda-Direita
                node.left = self._rotacao_esquerda(node.left)
                return self._rotacao_direita(node), True

        if balance < -1:  # Direita
            if self._balanceamento(node.right) <= 0:  # Direita-Direita
                return self._rotacao_esquerda(node), True
            else:  # Direita-Esquerda
                node.right = self._rotacao_direita(node.right)
                return self._rotacao_esquerda(node), True

        return node, True

    def _buscar_recursive(self, node: Optional[Node], alvo: Moto, passos_ref: List[int]) -> bool:
        if not node:
            return False

        passos_ref[0] += 1
        if self._search_step_limit is not None and passos_ref[0] > self._search_step_limit:
            return False

        if alvo == node.data:
            return True

        if alvo < node.data:
            return self._buscar_recursive(node.left, alvo, passos_ref)
        else:
            return self._buscar_recursive(node.right, alvo, passos_ref)

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        passos_container = [0]
        encontrado = self._buscar_recursive(self.root, alvo, passos_container)
        return encontrado, passos_container[0]

    def exibir(self) -> None:
        if not self.root: print("Árvore AVL vazia!"); return
        print(f"\n{'=' * 70}\n=== ÁRVORE AVL (Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}\n{'-' * 70}")
        self._displayed_count = 0
        self._em_ordem(self.root)
        if self._count > self._displayed_count: print(
            f"... e mais {self._count - self._displayed_count} motos não exibidas.")
        print("=" * 70)

    def _em_ordem(self, node: Optional[Node]) -> None:
        if node and self._displayed_count < 50:
            self._em_ordem(node.left)
            if self._displayed_count < 50:
                m = node.data
                print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                self._displayed_count += 1
                if self._displayed_count < 50:
                    self._em_ordem(node.right)
            else:
                return
        elif node and self._displayed_count >= 50:
            return

    def __len__(self) -> int:
        return self._count