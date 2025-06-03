from typing import Optional, Tuple, List
from modelos.moto import Moto


def _altura(node: Optional['AVLTree.Node']) -> int:
    return node.height if node else 0


def _atualizar_altura(node: 'AVLTree.Node') -> None:
    if node:
        node.height = 1 + max(_altura(node.left), _altura(node.right))


class AVLTree:
    class Node:
        def __init__(self, data: Moto):
            self.data = data
            self.left: Optional[AVLTree.Node] = None
            self.right: Optional[AVLTree.Node] = None
            self.height = 1

    def __init__(self):
        self.root: Optional[AVLTree.Node] = None

    def _balanceamento(self, node: Optional[Node]) -> int:
        return _altura(node.left) - _altura(node.right) if node else 0

    def _rotacao_direita(self, y: Node) -> Node:
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        _atualizar_altura(y)
        _atualizar_altura(x)

        return x

    def _rotacao_esquerda(self, x: Node) -> Node:
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        _atualizar_altura(x)
        _atualizar_altura(y)

        return y

    def inserir(self, data: Moto) -> None:
        self.root = self._inserir(self.root, data)

    def _inserir(self, node: Optional[Node], data: Moto) -> Node:
        if not node:
            return self.Node(data)

        if data.nome < node.data.nome:
            node.left = self._inserir(node.left, data)
        elif data.nome > node.data.nome:
            node.right = self._inserir(node.right, data)
        else:
            # Chaves iguais: comparar por preço
            if data.preco < node.data.preco:
                node.left = self._inserir(node.left, data)
            else:
                node.right = self._inserir(node.right, data)

        _atualizar_altura(node)

        balance = self._balanceamento(node)

        # Casos de rotação
        if balance > 1:
            if data.nome < node.left.data.nome or (
                    data.nome == node.left.data.nome and data.preco < node.left.data.preco):
                return self._rotacao_direita(node)
            else:
                node.left = self._rotacao_esquerda(node.left)
                return self._rotacao_direita(node)

        if balance < -1:
            if data.nome > node.right.data.nome or (
                    data.nome == node.right.data.nome and data.preco > node.right.data.preco):
                return self._rotacao_esquerda(node)
            else:
                node.right = self._rotacao_direita(node.right)
                return self._rotacao_esquerda(node)

        return node

    def _min_value_node(self, node: Node) -> Node:
        current = node
        while current.left:
            current = current.left
        return current

    def remover(self, alvo: Moto) -> bool:
        self.root, removido = self._remover(self.root, alvo)
        return removido

    def _remover(self, node: Optional[Node], alvo: Moto) -> Tuple[Optional[Node], bool]:
        if not node:
            return None, False

        removido = False

        # Buscar o nó
        if alvo.nome < node.data.nome:
            node.left, removido = self._remover(node.left, alvo)
        elif alvo.nome > node.data.nome:
            node.right, removido = self._remover(node.right, alvo)
        else:
            # Nome igual, verificar outros campos
            if alvo == node.data:
                removido = True
                # Caso 1: Sem filhos ou apenas um filho
                if not node.left or not node.right:
                    temp = node.left if node.left else node.right
                    return temp, True
                # Caso 2: Dois filhos
                else:
                    temp = self._min_value_node(node.right)
                    node.data = temp.data
                    node.right, _ = self._remover(node.right, temp.data)
                    removido = True
            else:
                # Nome igual mas objeto diferente
                node.left, removido = self._remover(node.left, alvo)
                if not removido:
                    node.right, removido = self._remover(node.right, alvo)

        if not node:
            return node, removido

        _atualizar_altura(node)

        balance = self._balanceamento(node)

        # Balanceamentos
        if balance > 1:
            if self._balanceamento(node.left) >= 0:
                return self._rotacao_direita(node), removido
            else:
                node.left = self._rotacao_esquerda(node.left)
                return self._rotacao_direita(node), removido

        if balance < -1:
            if self._balanceamento(node.right) <= 0:
                return self._rotacao_esquerda(node), removido
            else:
                node.right = self._rotacao_direita(node.right)
                return self._rotacao_esquerda(node), removido

        return node, removido

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        return self._buscar(self.root, alvo, 0)

    def _buscar(self, node: Optional[Node], alvo: Moto, passos: int) -> Tuple[bool, int]:
        if not node:
            return False, passos

        passos += 1

        if alvo == node.data:
            return True, passos

        if alvo.nome < node.data.nome:
            return self._buscar(node.left, alvo, passos)
        elif alvo.nome > node.data.nome:
            return self._buscar(node.right, alvo, passos)
        else:
            # Nome igual, buscar em ambos os lados
            encontrado, passos = self._buscar(node.left, alvo, passos)
            if not encontrado:
                encontrado, passos = self._buscar(node.right, alvo, passos)
            return encontrado, passos

    def exibir(self) -> None:
        if not self.root:
            print("Árvore vazia!")
            return

        print("\n" + "=" * 70)
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)
        self._em_ordem(self.root)
        print("=" * 70)

    def _em_ordem(self, node: Optional[Node]) -> None:
        if node:
            self._em_ordem(node.left)
            m = node.data
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
            self._em_ordem(node.right)

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