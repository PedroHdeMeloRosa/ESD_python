# Estruturas/radix_tree.py
from modelos.moto import Moto
from typing import Dict, List, Optional, Tuple


class RadixTree:
    """
    Implementação de uma Radix Tree (Trie Comprimida) para armazenar objetos Moto.
    A chave para a Radix Tree é o atributo `nome` do objeto Moto.
    Cada nó final pode armazenar uma lista de objetos Moto, permitindo
    múltiplas motos com o mesmo nome mas atributos diferentes.
    A operação de remoção não está implementada.
    """

    class Node:
        """Nó interno da Radix Tree."""

        def __init__(self, prefixo: str = ""):
            self.prefixo: str = prefixo
            self.filhos: Dict[str, RadixTree.Node] = {}
            self.dados: List[Moto] = []  # Lista para armazenar motos com esta chave/prefixo
            self.is_fim_de_chave: bool = False  # Indica se este nó representa o fim de uma chave válida

    def __init__(self):
        """Inicializa uma Radix Tree vazia."""
        self.raiz = self.Node()
        self._count = 0  # Para rastrear o número de motos distintas

    def inserir(self, data: Moto) -> None:
        """
        Insere um objeto Moto na Radix Tree usando seu nome como chave.
        Se uma moto idêntica (baseado em __eq__) já existir para essa chave, não é inserida novamente.
        :param data: Objeto Moto a ser inserido.
        """
        if self._inserir_recursivo(self.raiz, data.nome.lower(), data):  # Usar lower para case-insensitivity
            self._count += 1

    def _inserir_recursivo(self, node: Node, chave_restante: str, data: Moto) -> bool:
        """Método recursivo auxiliar para inserção."""
        # Caso 1: Chave restante corresponde exatamente ao prefixo do nó
        if chave_restante == node.prefixo:
            if data not in node.dados:  # Evitar duplicatas exatas
                node.dados.append(data)
                node.is_fim_de_chave = True
                return True
            return False  # Duplicata exata

        # Encontrar o ponto de divergência ou correspondência
        len_prefixo_node = len(node.prefixo)
        len_chave_restante = len(chave_restante)
        match_len = 0
        while (match_len < len_prefixo_node and
               match_len < len_chave_restante and
               node.prefixo[match_len] == chave_restante[match_len]):
            match_len += 1

        # Caso 2: Chave restante é prefixo do prefixo do nó (dividir nó atual)
        if match_len < len_prefixo_node:
            # Nó atual será dividido. Seu prefixo antigo se torna um filho.
            prefixo_filho_existente = node.prefixo[match_len:]
            novo_filho_existente = self.Node(prefixo_filho_existente)
            novo_filho_existente.filhos = node.filhos  # Move filhos do nó atual para o novo filho
            novo_filho_existente.dados = list(node.dados)  # Copia dados
            novo_filho_existente.is_fim_de_chave = node.is_fim_de_chave

            # Nó atual é atualizado
            node.prefixo = node.prefixo[:match_len]  # Encurta prefixo
            node.filhos = {prefixo_filho_existente[0]: novo_filho_existente}
            node.dados = []  # Dados são movidos para o filho ou novo nó
            node.is_fim_de_chave = False  # Não é mais fim de chave, a menos que a chave_restante termine aqui

            # Se a chave_restante termina no ponto de divisão
            if match_len == len_chave_restante:
                if data not in node.dados:  # Deveria ser node.dados aqui, se a chave termina aqui
                    node.dados.append(data)
                    node.is_fim_de_chave = True
                    return True
                return False  # Duplicata

            # Continuar inserindo o restante da chave_restante como um novo filho do nó atual
            sufixo_nova_chave = chave_restante[match_len:]
            if sufixo_nova_chave[0] not in node.filhos:
                novo_filho_para_sufixo = self.Node(sufixo_nova_chave)
                novo_filho_para_sufixo.dados.append(data)
                novo_filho_para_sufixo.is_fim_de_chave = True
                node.filhos[sufixo_nova_chave[0]] = novo_filho_para_sufixo
                return True
            else:  # Deveria ser impossível chegar aqui se o split ocorreu corretamente
                return self._inserir_recursivo(node.filhos[sufixo_nova_chave[0]], sufixo_nova_chave, data)


        # Caso 3: Prefixo do nó é prefixo da chave restante (seguir para filho ou criar novo)
        elif match_len == len_prefixo_node:
            if match_len == len_chave_restante:  # Chave é igual ao prefixo do nó
                if data not in node.dados:
                    node.dados.append(data)
                    node.is_fim_de_chave = True
                    return True
                return False  # Duplicata

            sufixo_chave = chave_restante[match_len:]
            if sufixo_chave[0] in node.filhos:
                return self._inserir_recursivo(node.filhos[sufixo_chave[0]], sufixo_chave, data)
            else:
                novo_filho = self.Node(sufixo_chave)
                novo_filho.dados.append(data)
                novo_filho.is_fim_de_chave = True
                node.filhos[sufixo_chave[0]] = novo_filho
                return True

        # Caso 4: Sem correspondência (raiz ou erro lógico) - deveria ser coberto acima
        # Para a raiz (prefixo vazio), match_len será 0.
        if node == self.raiz and match_len == 0:  # Inserindo a partir da raiz
            if chave_restante[0] in node.filhos:
                return self._inserir_recursivo(node.filhos[chave_restante[0]], chave_restante, data)
            else:
                novo_filho = self.Node(chave_restante)
                novo_filho.dados.append(data)
                novo_filho.is_fim_de_chave = True
                node.filhos[chave_restante[0]] = novo_filho
                return True

        # print(f"DEBUG: Condição de inserção não coberta para chave '{chave_restante}' no nó prefixo '{node.prefixo}'")
        return False  # Inserção falhou ou condição não coberta

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        """
        Busca por um objeto Moto específico na Radix Tree, usando seu nome como chave.
        :param alvo: Objeto Moto a ser buscado.
        :return: Tupla (encontrado: bool, passos: int). `encontrado` é True se a moto exata for achada.
        """
        passos = [0]  # Usar lista para passar por referência
        node_final = self._buscar_no_para_chave(self.raiz, alvo.nome.lower(), passos)
        if node_final and node_final.is_fim_de_chave and alvo in node_final.dados:
            return True, passos[0]
        return False, passos[0]

    def _buscar_no_para_chave(self, node: Node, chave_restante: str, passos: List[int]) -> Optional[Node]:
        """
        Busca o nó que corresponde ao final da chave_restante.
        Retorna o nó se a chave exata é encontrada, caso contrário None.
        """
        passos[0] += 1

        if not chave_restante:  # Chave vazia só corresponde se o nó é fim de chave e tem prefixo vazio
            return node if node.is_fim_de_chave and not node.prefixo else None

        len_prefixo_node = len(node.prefixo)
        len_chave_restante = len(chave_restante)

        # Se o prefixo do nó não corresponder ao início da chave_restante
        if not chave_restante.startswith(node.prefixo):
            return None

        # Se o prefixo do nó corresponde exatamente à chave_restante
        if len_prefixo_node == len_chave_restante:  # e chave_restante == node.prefixo (já verificado por startswith)
            return node if node.is_fim_de_chave else None  # Só é match se for fim de chave

        # Se o prefixo do nó é mais longo que a chave_restante (impossível se startswith passou e não são iguais)
        if len_prefixo_node > len_chave_restante:
            return None

        # Prefixo do nó é um prefixo da chave_restante. Continuar busca no filho apropriado.
        # chave_restante = "applepie", node.prefixo = "app" -> sufixo_chave = "lepie"
        sufixo_chave = chave_restante[len_prefixo_node:]
        if not sufixo_chave:  # Deveria ter sido coberto por len_prefixo_node == len_chave_restante
            return None

        primeiro_char_sufixo = sufixo_chave[0]
        if primeiro_char_sufixo in node.filhos:
            return self._buscar_no_para_chave(node.filhos[primeiro_char_sufixo], sufixo_chave, passos)

        return None

    def exibir(self) -> None:
        """Exibe todas as motos na Radix Tree no console."""
        print("\n" + "=" * 70)
        print(f"=== RADIX TREE (Motos Distintas: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)
        self._displayed_count = 0
        self._exibir_recursivo(self.raiz, "")
        if self._count > self._displayed_count:
            print(f"... e mais {self._count - self._displayed_count} motos não exibidas.")
        print("=" * 70)

    def _exibir_recursivo(self, node: Node, prefixo_acumulado: str) -> None:
        """Método recursivo auxiliar para exibir a árvore."""
        if self._displayed_count >= 50:
            return

        if node.is_fim_de_chave:
            for moto in node.dados:
                if self._displayed_count < 50:
                    print(f"{moto.marca:<15}{moto.nome:<20}{moto.preco:<12.2f}{moto.revenda:<15.2f}{moto.ano:<6}")
                    self._displayed_count += 1
                else:
                    return

        for char_inicial_filho, filho_node in sorted(node.filhos.items()):  # Ordenar para exibição consistente
            self._exibir_recursivo(filho_node, prefixo_acumulado + node.prefixo)
            if self._displayed_count >= 50:
                return

    def __len__(self) -> int:
        """Retorna o número de motos distintas na árvore."""
        return self._count
