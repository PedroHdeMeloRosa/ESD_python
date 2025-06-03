# Estruturas/avl_tree.py
from typing import Optional, Tuple, Any
from modelos.moto import Moto


def _altura_avl(node: Optional['AVLTree.Node']) -> int:
    """Função auxiliar para obter a altura de um nó (0 se None)."""
    return node.height if node else 0


def _atualizar_altura_avl(node: 'AVLTree.Node') -> None:
    """Função auxiliar para recalcular e atualizar a altura de um nó."""
    if node:
        node.height = 1 + max(_altura_avl(node.left), _altura_avl(node.right))


class AVLTree:
    """
    Implementação de uma Árvore AVL (Adelson-Velsky e Landis).
    É uma árvore de busca binária auto-balanceada.
    A ordenação primária é pelo nome da moto e a secundária pelo preço.
    """

    class Node:
        """Nó interno da Árvore AVL."""

        def __init__(self, data: Moto):
            self.data: Moto = data
            self.left: Optional[AVLTree.Node] = None
            self.right: Optional[AVLTree.Node] = None
            self.height: int = 1  # Altura do nó (folhas têm altura 1)

    def __init__(self):
        """Inicializa uma Árvore AVL vazia."""
        self.root: Optional[AVLTree.Node] = None
        self._count = 0  # Para rastrear o número de elementos

    def _balanceamento(self, node: Optional[Node]) -> int:
        """Calcula o fator de balanceamento de um nó."""
        return _altura_avl(node.left) - _altura_avl(node.right) if node else 0

    def _rotacao_direita(self, y: Node) -> Node:
        """Executa uma rotação simples à direita."""
        # print(f"Rotação Direita em {y.data.nome if y else 'None'}")
        x = y.left
        if x is None: return y  # Segurança, não deveria acontecer em AVL válida
        T2 = x.right

        x.right = y
        y.left = T2

        _atualizar_altura_avl(y)
        _atualizar_altura_avl(x)
        return x

    def _rotacao_esquerda(self, x: Node) -> Node:
        """Executa uma rotação simples à esquerda."""
        # print(f"Rotação Esquerda em {x.data.nome if x else 'None'}")
        y = x.right
        if y is None: return x  # Segurança
        T2 = y.left

        y.left = x
        x.right = T2

        _atualizar_altura_avl(x)
        _atualizar_altura_avl(y)
        return y

    def inserir(self, data: Moto) -> None:
        """
        Insere um objeto Moto na árvore, mantendo as propriedades da AVL.
        :param data: Objeto Moto a ser inserido.
        """
        # Verifica se a moto já existe para evitar duplicatas exatas,
        # embora a lógica de inserção também trate isso.
        # A busca antes da inserção pode adicionar overhead.
        # Para este projeto, vamos confiar na lógica de inserção para não duplicar.
        # if self.buscar(data)[0]:
        # print(f"Debug: Moto {data.nome} já existe, não inserindo novamente.")
        #     return

        self.root, inserido = self._inserir(self.root, data)
        if inserido:
            self._count += 1

    def _inserir(self, node: Optional[Node], data: Moto) -> Tuple[Optional[Node], bool]:
        """Método recursivo auxiliar para inserir um dado e balancear a árvore."""
        inserido_flag = False
        if not node:
            return self.Node(data), True  # Nó criado, inserção bem sucedida

        # Comparação primária por nome, secundária por preço (como em Moto.__lt__)
        if data.nome < node.data.nome:
            node.left, inserido_flag = self._inserir(node.left, data)
        elif data.nome > node.data.nome:
            node.right, inserido_flag = self._inserir(node.right, data)
        else:  # Nomes iguais
            if data.preco < node.data.preco:
                node.left, inserido_flag = self._inserir(node.left, data)
            elif data.preco > node.data.preco:
                node.right, inserido_flag = self._inserir(node.right, data)
            else:  # Moto exatamente igual (nome e preço), ou outros campos se __eq__ for mais complexo
                if data == node.data:  # Evita duplicatas exatas
                    return node, False  # Não inserido, já existe
                else:  # Nomes e preços iguais, mas outros atributos diferentes. Decida uma regra (ex: ir para direita)
                    # Para simplificar, se nome e preço são iguais, mas o objeto não é, insere à direita.
                    # Isso pode criar "duplicatas" se o __eq__ for muito estrito.
                    # A classe Moto tem __eq__ bem definido.
                    node.right, inserido_flag = self._inserir(node.right, data)

        if not inserido_flag:  # Se não foi inserido em sub-árvore, retorna
            return node, False

        _atualizar_altura_avl(node)
        balance = self._balanceamento(node)

        # Casos de Rotação (após inserção)
        # Esquerda-Esquerda
        if balance > 1:
            # Verifica o critério de desempate para o filho esquerdo
            sub_balance_node = node.left
            nome_menor = data.nome < sub_balance_node.data.nome
            nome_igual_preco_menor = (data.nome == sub_balance_node.data.nome and
                                      data.preco < sub_balance_node.data.preco)

            if nome_menor or nome_igual_preco_menor:  # Inserido na subárvore esquerda do filho esquerdo
                return self._rotacao_direita(node), inserido_flag
            else:  # Esquerda-Direita
                node.left = self._rotacao_esquerda(node.left)
                return self._rotacao_direita(node), inserido_flag

        # Direita-Direita
        if balance < -1:
            sub_balance_node = node.right
            nome_maior = data.nome > sub_balance_node.data.nome
            nome_igual_preco_maior = (data.nome == sub_balance_node.data.nome and
                                      data.preco > sub_balance_node.data.preco)

            if nome_maior or nome_igual_preco_maior:  # Inserido na subárvore direita do filho direito
                return self._rotacao_esquerda(node), inserido_flag
            else:  # Direita-Esquerda
                node.right = self._rotacao_direita(node.right)
                return self._rotacao_esquerda(node), inserido_flag

        return node, inserido_flag

    def _min_value_node(self, node: Node) -> Node:
        """Encontra o nó com o menor valor (mais à esquerda) em uma subárvore."""
        current = node
        while current.left:
            current = current.left
        return current

    def remover(self, alvo: Moto) -> bool:
        """
        Remove um objeto Moto da árvore, mantendo as propriedades da AVL.
        :param alvo: Objeto Moto a ser removido.
        :return: True se o item foi removido, False caso contrário.
        """
        self.root, removido = self._remover(self.root, alvo)
        if removido:
            self._count -= 1
        return removido

    def _remover(self, node: Optional[Node], alvo: Moto) -> Tuple[Optional[Node], bool]:
        """Método recursivo auxiliar para remover um dado e balancear a árvore."""
        if not node:
            return None, False  # Nó não encontrado

        removido_flag = False
        # Comparação para encontrar o nó
        if alvo.nome < node.data.nome:
            node.left, removido_flag = self._remover(node.left, alvo)
        elif alvo.nome > node.data.nome:
            node.right, removido_flag = self._remover(node.right, alvo)
        else:  # Nomes iguais
            if alvo.preco < node.data.preco:
                node.left, removido_flag = self._remover(node.left, alvo)
            elif alvo.preco > node.data.preco:
                node.right, removido_flag = self._remover(node.right, alvo)
            else:  # Nomes e preços correspondem, agora verificar com __eq__ para o objeto completo
                if alvo == node.data:  # Objeto exato encontrado
                    removido_flag = True
                    if not node.left or not node.right:  # Nó com 0 ou 1 filho
                        temp = node.left if node.left else node.right
                        node = None  # Libera o nó atual
                        return temp, True
                    else:  # Nó com 2 filhos
                        temp = self._min_value_node(node.right)  # Sucessor em ordem
                        node.data = temp.data  # Copia o sucessor para este nó
                        node.right, _ = self._remover(node.right, temp.data)  # Remove o sucessor
                else:  # Nomes e preços iguais, mas objeto diferente (outros atributos)
                    # Baseado na inserção, se um objeto com mesmo nome e preço mas !=,
                    # ele teria ido para a direita. Mas a busca aqui deve ser exata.
                    # Se a árvore pode ter múltiplos objetos Moto que não são __eq__ mas
                    # têm mesmo nome e preço, a lógica de busca/remoção fica mais complexa.
                    # Assumindo que __eq__ é o critério final, e nome/preço são para ordenação.
                    # Para ser robusto, se o objeto não é igual, tentamos na direita (ou esquerda,
                    # dependendo da regra de inserção para esses casos "quase-duplicados").
                    # Se a árvore SÓ armazena itens únicos por (nome, preço), então este 'else'
                    # indicaria que o item exato não está aqui, mas em uma sub-árvore.
                    # A lógica atual de inserção tenta colocar tais itens à direita.
                    node.right, removido_flag = self._remover(node.right, alvo)
                    if not removido_flag:  # Se não achou na direita, tenta na esquerda (improvável com a inserção atual)
                        node.left, removido_flag = self._remover(node.left, alvo)

        if not node:  # Se a árvore ficou vazia ou o nó foi removido
            return node, removido_flag

        if not removido_flag:  # Se não houve remoção na sub-árvore
            return node, False

        _atualizar_altura_avl(node)
        balance = self._balanceamento(node)

        # Casos de Rotação (após remoção)
        # Esquerda (Subárvore direita ficou menor)
        if balance > 1:
            if self._balanceamento(node.left) >= 0:  # Esquerda-Esquerda
                return self._rotacao_direita(node), removido_flag
            else:  # Esquerda-Direita
                node.left = self._rotacao_esquerda(node.left)
                return self._rotacao_direita(node), removido_flag

        # Direita (Subárvore esquerda ficou menor)
        if balance < -1:
            if self._balanceamento(node.right) <= 0:  # Direita-Direita
                return self._rotacao_esquerda(node), removido_flag
            else:  # Direita-Esquerda
                node.right = self._rotacao_direita(node.right)
                return self._rotacao_esquerda(node), removido_flag

        return node, removido_flag

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        """
        Busca por um objeto Moto na árvore.
        :param alvo: Objeto Moto a ser buscado.
        :return: Tupla (encontrado: bool, passos: int).
        """
        return self._buscar(self.root, alvo, 0)

    def _buscar(self, node: Optional[Node], alvo: Moto, passos: int) -> Tuple[bool, int]:
        """Método recursivo auxiliar para buscar um dado."""
        if not node:
            return False, passos

        passos += 1

        if alvo == node.data:  # Encontrou o objeto exato
            return True, passos

        # Segue a lógica de ordenação
        if alvo.nome < node.data.nome:
            return self._buscar(node.left, alvo, passos)
        elif alvo.nome > node.data.nome:
            return self._buscar(node.right, alvo, passos)
        else:  # Nomes iguais
            if alvo.preco < node.data.preco:
                return self._buscar(node.left, alvo, passos)
            elif alvo.preco > node.data.preco:
                return self._buscar(node.right, alvo, passos)
            else:
                # Nomes e preços iguais, mas o objeto Moto não é (por __eq__).
                # A inserção colocaria à direita. Então buscamos à direita.
                # Se pudesse ir para a esquerda, a busca teria que checar ambos.
                # Com a lógica de inserção atual, se nome e preço são iguais mas o objeto é
                # diferente, ele vai para a direita.
                return self._buscar(node.right, alvo, passos)

    def exibir(self) -> None:
        """Exibe todos os elementos da árvore (em ordem) no console."""
        if not self.root:
            print("Árvore AVL vazia!")
            return

        print("\n" + "=" * 70)
        print(f"=== ÁRVORE AVL (Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)
        self._displayed_count = 0  # Contador para limitar exibição
        self._em_ordem(self.root)
        if self._count > self._displayed_count:
            print(f"... e mais {self._count - self._displayed_count} motos não exibidas.")
        print("=" * 70)

    def _em_ordem(self, node: Optional[Node]) -> None:
        """Percorre a árvore em ordem para exibição."""
        if node and self._displayed_count < 50:  # Limita a exibição
            self._em_ordem(node.left)
            if self._displayed_count < 50:
                m = node.data
                print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                self._displayed_count += 1
                if self._displayed_count < 50:
                    self._em_ordem(node.right)
            else:  # Já exibiu o suficiente, não continua
                return
        elif node and self._displayed_count >= 50:  # Para a recursão se já exibiu o limite
            return

    def __len__(self) -> int:
        """Retorna o número de elementos na árvore."""
        return self._count