# Estruturas/avl_tree.py
from typing import Optional, Tuple, Any, List
from modelos.moto import Moto  # Garanta que Moto está corretamente definida com __lt__, __eq__


def _altura_avl(node: Optional['AVLTree.Node']) -> int:
    """Retorna a altura de um nó, ou 0 se o nó for None."""
    return node.height if node else 0


def _atualizar_altura_avl(node: Optional['AVLTree.Node']) -> None:
    """Recalcula e atualiza a altura de um nó com base na altura de seus filhos."""
    if node:
        node.height = 1 + max(_altura_avl(node.left), _altura_avl(node.right))


class AVLTree:
    class Node:
        def __init__(self, data: Moto):
            self.data: Moto = data
            self.left: Optional[AVLTree.Node] = None
            self.right: Optional[AVLTree.Node] = None
            self.height: int = 1  # Altura inicial de um novo nó (folha) é 1

        def __str__(self):  # Para debugging
            left_data = self.left.data.nome if self.left and self.left.data else "None"
            right_data = self.right.data.nome if self.right and self.right.data else "None"
            return (f"Node(data='{self.data.nome}', h={self.height}, "
                    f"L='{left_data}', R='{right_data}')")

    def __init__(self, max_elements: Optional[int] = None):
        self.root: Optional[AVLTree.Node] = None
        self._count: int = 0
        self.max_elements: Optional[int] = max_elements
        self._search_step_limit: Optional[int] = None  # Para simulação de restrição A1

    def set_search_step_limit(self, limit: Optional[int]):
        self._search_step_limit = limit

    def _fator_balanceamento(self, node: Optional[Node]) -> int:
        """Calcula o fator de balanceamento de um nó."""
        if not node:
            return 0
        return _altura_avl(node.left) - _altura_avl(node.right)

    def _rotacao_direita(self, y: Node) -> Node:
        # print(f"Rot_Dir em {y.data.nome}")
        x = y.left
        if x is None: return y  # Segurança: não deveria acontecer em chamada válida
        T2 = x.right
        x.right = y
        y.left = T2
        _atualizar_altura_avl(y)
        _atualizar_altura_avl(x)
        return x

    def _rotacao_esquerda(self, x: Node) -> Node:
        # print(f"Rot_Esq em {x.data.nome}")
        y = x.right
        if y is None: return x  # Segurança
        T2 = y.left
        y.left = x
        x.right = T2
        _atualizar_altura_avl(x)
        _atualizar_altura_avl(y)
        return y

    def inserir(self, data: Moto) -> bool:
        if self.max_elements is not None and self._count >= self.max_elements:
            # print(f"INFO (AVL): Limite de {self.max_elements} elementos atingido. {data.nome} não inserido.")
            return False  # Falha na inserção devido ao limite

        # Chama _inserir e atualiza a raiz e a contagem
        nova_raiz, foi_inserido = self._inserir_recursivo(self.root, data)
        self.root = nova_raiz
        if foi_inserido:
            self._count += 1
        return foi_inserido

    def _inserir_recursivo(self, node: Optional[Node], data: Moto) -> Tuple[Optional[Node], bool]:
        if not node:
            return self.Node(data), True  # Novo nó criado, inserção bem-sucedida

        inserido_na_subarvore = False
        if data < node.data:  # Usa __lt__ da Moto (nome, depois preço)
            node.left, inserido_na_subarvore = self._inserir_recursivo(node.left, data)
        elif node.data < data:  # data > node.data
            node.right, inserido_na_subarvore = self._inserir_recursivo(node.right, data)
        else:  # Chaves de ordenação iguais (mesmo nome e preço)
            if data == node.data:  # Objeto Moto exato já existe
                return node, False  # Duplicata, não inseriu
            else:
                # Política para "quase duplicatas" (mesmo nome/preço, mas objeto diferente)
                # Inserir à direita (ou esquerda, mas seja consistente).
                # print(f"DEBUG AVL Inserir: Quase duplicata para {data.nome}, inserindo à direita.")
                node.right, inserido_na_subarvore = self._inserir_recursivo(node.right, data)

        if not inserido_na_subarvore:
            return node, False  # Não houve inserção efetiva na subárvore

        # Atualiza altura do nó atual
        _atualizar_altura_avl(node)

        # Obtém fator de balanceamento e aplica rotações se necessário
        balance = self._fator_balanceamento(node)

        # Caso Esquerda-Esquerda (LL)
        if balance > 1 and node.left and data < node.left.data:
            return self._rotacao_direita(node), True

        # Caso Direita-Direita (RR)
        if balance < -1 and node.right and node.right.data < data:  # data > node.right.data
            return self._rotacao_esquerda(node), True

        # Caso Esquerda-Direita (LR)
        if balance > 1 and node.left and node.left.data < data:  # data > node.left.data
            node.left = self._rotacao_esquerda(node.left)
            return self._rotacao_direita(node), True

        # Caso Direita-Esquerda (RL)
        if balance < -1 and node.right and data < node.right.data:
            node.right = self._rotacao_direita(node.right)
            return self._rotacao_esquerda(node), True

        return node, True  # Nó está balanceado ou tornou-se balanceado, inserção na subárvore foi bem-sucedida

    def _get_min_value_node(self, node: Node) -> Node:  # Argumento não deve ser None aqui
        """Retorna o nó com o menor valor na subárvore dada (o mais à esquerda)."""
        current = node
        while current.left is not None:
            current = current.left
        return current

    def remover(self, alvo: Moto) -> bool:
        # print(f"DEBUG AVL REMOVER: Iniciando remoção de {alvo.nome}, Raiz: {self.root.data.nome if self.root else 'None'}")
        self.root, foi_removido = self._remover_recursivo(self.root, alvo)
        if foi_removido:
            self._count -= 1
        # print(f"DEBUG AVL REMOVER: Remoção de {alvo.nome} {'concluída' if foi_removido else 'falhou'}. Novo count: {self._count}")
        return foi_removido

    def _remover_recursivo(self, node: Optional[Node], alvo: Moto) -> Tuple[Optional[Node], bool]:
        if not node:
            return node, False  # Chave não encontrada

        removido_da_subarvore = False
        node_retorno = node  # Nó a ser retornado (pode mudar após rotações)

        if alvo < node.data:
            node.left, removido_da_subarvore = self._remover_recursivo(node.left, alvo)
            node_retorno = node  # Ainda é o nó original se não houve rotação abaixo
        elif node.data < alvo:  # alvo > node.data
            node.right, removido_da_subarvore = self._remover_recursivo(node.right, alvo)
            node_retorno = node
        else:  # Chaves de ordenação iguais (mesmo nome, mesmo preço)
            if alvo == node.data:  # É o objeto exato a ser removido
                removido_da_subarvore = True  # Marcamos que este nó será tratado
                # Caso 1: Nó com um filho ou nenhum filho
                if node.left is None:
                    # print(f"DEBUG AVL REMOVER: {node.data.nome} - Caso 1 (filho direito ou nenhum)")
                    temp_node = node.right
                    node = None  # Para GC, mas o retorno de temp_node é o que importa
                    return temp_node, True
                elif node.right is None:
                    # print(f"DEBUG AVL REMOVER: {node.data.nome} - Caso 1 (filho esquerdo)")
                    temp_node = node.left
                    node = None
                    return temp_node, True

                # Caso 2: Nó com dois filhos
                # Pega o sucessor em ordem (menor valor na subárvore direita)
                # print(f"DEBUG AVL REMOVER: {node.data.nome} - Caso 2 (dois filhos)")
                # node.right NÃO PODE ser None aqui por causa das condições anteriores.
                temp_sucessor = self._get_min_value_node(node.right)
                # print(f"DEBUG AVL REMOVER: Sucessor de {node.data.nome} é {temp_sucessor.data.nome}")

                node.data = temp_sucessor.data  # Copia os dados do sucessor para este nó
                # Remove o sucessor da subárvore direita
                # print(f"DEBUG AVL REMOVER: Removendo sucessor {temp_sucessor.data.nome} da subárvore direita.")
                node.right, _ = self._remover_recursivo(node.right, temp_sucessor.data)
                node_retorno = node  # O nó atual (com dados do sucessor) é o que permanece
            else:
                # Chaves de ordenação iguais, mas não é o objeto exato.
                # Segue a política de "quase duplicatas" da inserção (geralmente à direita).
                node.right, removido_da_subarvore = self._remover_recursivo(node.right, alvo)
                node_retorno = node

        if not removido_da_subarvore:  # Se não houve remoção efetiva na subárvore
            return node_retorno, False  # Retorna o nó original (ou o que se tornou node_retorno)

        # Se o nó se tornou None após a remoção de um filho (ex: era folha)
        if node_retorno is None:  # Esta verificação pode ser redundante se o caso de 0/1 filho já retorna None
            return None, True

        # Atualizar altura do nó atual (que agora é node_retorno)
        _atualizar_altura_avl(node_retorno)

        # Obter fator de balanceamento e rebalancear
        balance = self._fator_balanceamento(node_retorno)

        # Caso Esquerda-Esquerda (LL) ou Esquerda-Zero (L0)
        if balance > 1 and self._fator_balanceamento(node_retorno.left) >= 0:
            # print(f"DEBUG AVL REMOVER: Rebalanceando LL/L0 em {node_retorno.data.nome}")
            return self._rotacao_direita(node_retorno), True

        # Caso Esquerda-Direita (LR)
        if balance > 1 and self._fator_balanceamento(node_retorno.left) < 0:
            # print(f"DEBUG AVL REMOVER: Rebalanceando LR em {node_retorno.data.nome}")
            if node_retorno.left: node_retorno.left = self._rotacao_esquerda(node_retorno.left)
            return self._rotacao_direita(node_retorno), True

        # Caso Direita-Direita (RR) ou Direita-Zero (R0)
        if balance < -1 and self._fator_balanceamento(node_retorno.right) <= 0:
            # print(f"DEBUG AVL REMOVER: Rebalanceando RR/R0 em {node_retorno.data.nome}")
            return self._rotacao_esquerda(node_retorno), True

        # Caso Direita-Esquerda (RL)
        if balance < -1 and self._fator_balanceamento(node_retorno.right) > 0:
            # print(f"DEBUG AVL REMOVER: Rebalanceando RL em {node_retorno.data.nome}")
            if node_retorno.right: node_retorno.right = self._rotacao_direita(node_retorno.right)
            return self._rotacao_esquerda(node_retorno), True

        return node_retorno, True  # Retorna nó (possivelmente balanceado) e que remoção ocorreu

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        passos_container = [0]  # Usar lista para passar por referência em Python
        encontrado = self._buscar_recursivo_com_limite(self.root, alvo, passos_container)
        return encontrado, passos_container[0]

    def _buscar_recursivo_com_limite(self, node: Optional[Node], alvo: Moto, passos_ref: List[int]) -> bool:
        if not node:
            return False

        passos_ref[0] += 1  # Conta cada nó visitado/comparado

        # Verifica limite de passos para simulação de restrição A1
        if self._search_step_limit is not None and passos_ref[0] > self._search_step_limit:
            # print(f"DEBUG AVL BUSCAR: Limite de {self._search_step_limit} passos atingido.")
            return False  # Limite de passos atingido, considera não encontrado

        if alvo == node.data:  # Comparação exata do objeto Moto
            return True
        elif alvo < node.data:  # Usa __lt__ da Moto
            return self._buscar_recursivo_com_limite(node.left, alvo, passos_ref)
        else:  # alvo > node.data
            return self._buscar_recursivo_com_limite(node.right, alvo, passos_ref)

    def exibir(self) -> None:
        if not self.root: print("Árvore AVL vazia!"); return
        print(f"\n{'=' * 70}\n=== ÁRVORE AVL (Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (₹)':<12}{'Revenda (₹)':<15}{'Ano':<6}\n{'-' * 70}")
        self._displayed_count = 0
        self._exibir_em_ordem(self.root)
        if self._count > self._displayed_count:
            print(
                f"... e mais {self._count - self._displayed_count} motos não exibidas ({self._displayed_count}/{self._count}).")
        print("=" * 70)

    def _exibir_em_ordem(self, node: Optional[Node]) -> None:
        if node and self._displayed_count < 50:
            self._exibir_em_ordem(node.left)
            if self._displayed_count < 50:  # Checa de novo após chamada recursiva à esquerda
                m = node.data
                print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                self._displayed_count += 1
                if self._displayed_count < 50:  # Checa antes de ir para a direita
                    self._exibir_em_ordem(node.right)
            # else: # Se limite atingido após recursão esquerda, não continua
            #     return
        # elif node and self._displayed_count >= 50: # Se já começou com limite atingido
        #     return

    def __len__(self) -> int:
        return self._count