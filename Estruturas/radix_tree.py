# Estruturas/radix_tree.py
from modelos.moto import Moto
from typing import Dict, List, Optional, Tuple


class RadixTree:
    class Node:
        def __init__(self, prefixo: str = ""):
            self.prefixo: str = prefixo
            self.filhos: Dict[str, RadixTree.Node] = {}
            self.dados: List[Moto] = []  # Armazena os objetos Moto se este nó é fim de chave
            self.is_fim_de_chave: bool = False  # Indica se este nó representa o fim de uma chave válida

    def __init__(self, max_elements: Optional[int] = None):
        self.raiz = self.Node()  # Raiz sempre tem prefixo vazio implicitamente
        self._count = 0  # Número total de OBJETOS Moto na árvore
        self.max_elements: Optional[int] = max_elements
        self._search_step_limit: Optional[int] = None

    def set_search_step_limit(self, limit: Optional[int]):
        self._search_step_limit = limit

    def inserir(self, data: Moto) -> bool:
        if self.max_elements is not None and self._count >= self.max_elements:
            return False  # Limite atingido

        # Usar o nome da moto em minúsculas como chave para consistência
        chave_str = data.nome.lower()

        foi_inserido_novo_obj = self._inserir_recursivo_v2(self.raiz, chave_str, data)
        if foi_inserido_novo_obj:
            self._count += 1
        return foi_inserido_novo_obj

    def _inserir_recursivo_v2(self, node: Node, chave_str_restante: str, data_obj: Moto) -> bool:
        """
        Lógica de inserção revisada.
        Retorna True se um novo objeto Moto foi adicionado, False caso contrário (ex: duplicata exata).
        """
        # Caso base: Se a chave restante é vazia, estamos no nó correto para inserir os dados do objeto.
        if not chave_str_restante:
            if data_obj not in node.dados:  # Evita duplicatas exatas do objeto Moto
                node.dados.append(data_obj)
                node.is_fim_de_chave = True
                return True
            return False  # Objeto Moto já existe neste nó

        # Encontrar o filho cujo prefixo COMEÇA com o primeiro caractere da chave_str_restante
        primeiro_char_chave = chave_str_restante[0]
        filho_candidato: Optional[RadixTree.Node] = None

        # Tenta encontrar um filho que compartilha um prefixo com a chave_str_restante
        for char_filho, filho_node_atual in node.filhos.items():
            if chave_str_restante.startswith(
                    filho_node_atual.prefixo):  # A chave restante começa com todo o prefixo do filho
                # A chave se encaixa perfeitamente neste filho (ou é mais longa)
                return self._inserir_recursivo_v2(filho_node_atual, chave_str_restante[len(filho_node_atual.prefixo):],
                                                  data_obj)

            elif filho_node_atual.prefixo.startswith(
                    chave_str_restante):  # O prefixo do filho começa com toda a chave restante (precisa dividir o filho)
                prefixo_comum = chave_str_restante

                # Divide o filho_node_atual
                sufixo_antigo_filho = filho_node_atual.prefixo[len(prefixo_comum):]

                # Nó que representa o que sobrou do prefixo do filho original
                novo_neto = self.Node(sufixo_antigo_filho)
                novo_neto.filhos = filho_node_atual.filhos
                novo_neto.dados = filho_node_atual.dados
                novo_neto.is_fim_de_chave = filho_node_atual.is_fim_de_chave

                # Atualiza o filho original (agora é o nó do prefixo_comum)
                filho_node_atual.prefixo = prefixo_comum
                filho_node_atual.filhos = {sufixo_antigo_filho[0]: novo_neto} if sufixo_antigo_filho else {}
                filho_node_atual.dados = []  # Dados da chave que termina aqui
                filho_node_atual.is_fim_de_chave = True  # Este nó agora representa o fim da chave_str_restante

                if data_obj not in filho_node_atual.dados:
                    filho_node_atual.dados.append(data_obj)
                    return True
                return False

            else:  # Encontrar o ponto de divergência entre chave_str_restante e prefixo do filho
                match_len = 0
                while (match_len < len(chave_str_restante) and
                       match_len < len(filho_node_atual.prefixo) and
                       chave_str_restante[match_len] == filho_node_atual.prefixo[match_len]):
                    match_len += 1

                if match_len > 0:  # Há um prefixo comum, mas não total
                    prefixo_comum_divergente = chave_str_restante[:match_len]

                    sufixo_chave_nova = chave_str_restante[match_len:]
                    sufixo_filho_existente = filho_node_atual.prefixo[match_len:]

                    # Remove o filho existente temporariamente
                    del node.filhos[char_filho]

                    # Novo nó intermediário para o prefixo comum
                    novo_no_intermediario = self.Node(prefixo_comum_divergente)
                    node.filhos[prefixo_comum_divergente[
                        0]] = novo_no_intermediario  # Assume que prefixo_comum_divergente não é vazio

                    # Reconecta o filho existente (agora neto) com seu sufixo restante
                    filho_node_atual.prefixo = sufixo_filho_existente
                    if sufixo_filho_existente:  # Só adiciona se houver sufixo
                        novo_no_intermediario.filhos[sufixo_filho_existente[0]] = filho_node_atual
                    else:  # O filho existente terminava no prefixo comum
                        novo_no_intermediario.dados.extend(filho_node_atual.dados)
                        novo_no_intermediario.is_fim_de_chave = filho_node_atual.is_fim_de_chave

                    # Cria um novo neto para o sufixo da chave que está sendo inserida
                    if sufixo_chave_nova:
                        novo_neto_para_chave_nova = self.Node(sufixo_chave_nova)
                        novo_neto_para_chave_nova.dados.append(data_obj)
                        novo_neto_para_chave_nova.is_fim_de_chave = True
                        novo_no_intermediario.filhos[sufixo_chave_nova[0]] = novo_neto_para_chave_nova
                        return True
                    else:  # A chave que está sendo inserida termina no novo nó intermediário
                        if data_obj not in novo_no_intermediario.dados:
                            novo_no_intermediario.dados.append(data_obj)
                            novo_no_intermediario.is_fim_de_chave = True
                            return True
                        return False

        # Se nenhum filho existente compartilha um prefixo, cria um novo filho para toda a chave_str_restante
        novo_filho = self.Node(chave_str_restante)
        novo_filho.dados.append(data_obj)
        novo_filho.is_fim_de_chave = True
        node.filhos[chave_str_restante[0]] = novo_filho  # Adiciona o novo filho ao nó atual
        return True

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        passos_container = [0]
        chave_str_busca = alvo.nome.lower()  # Consistência com inserção
        encontrado_obj = self._buscar_recursivo_v2(self.raiz, chave_str_busca, passos_container)

        if encontrado_obj:  # Se _buscar_recursivo_v2 retornou um nó que MARCA o fim da chave
            # Precisamos verificar se o objeto Moto específico está na lista de dados desse nó
            if alvo in encontrado_obj.dados:
                return True, passos_container[0]

        return False, passos_container[0]

    def _buscar_recursivo_v2(self, node: Node, chave_str_restante: str, passos_ref: List[int]) -> Optional[Node]:
        """
        Busca recursiva revisada.
        Retorna o NÓ onde a chave termina (se encontrada), ou None.
        A verificação do objeto Moto exato é feita no método 'buscar'.
        """
        passos_ref[0] += 1
        if self._search_step_limit is not None and passos_ref[0] > self._search_step_limit:
            return None  # Limite de passos atingido

        # Tenta encontrar um filho que corresponda ao início da chave_str_restante
        for char_filho_inicial, filho_node_atual in node.filhos.items():
            if chave_str_restante.startswith(filho_node_atual.prefixo):
                # A chave_str_restante começa com todo o prefixo do filho.
                # Continua a busca no filho com o restante da chave.
                novo_restante = chave_str_restante[len(filho_node_atual.prefixo):]
                if not novo_restante:  # Chegou exatamente ao fim do prefixo do filho
                    return filho_node_atual if filho_node_atual.is_fim_de_chave else None
                return self._buscar_recursivo_v2(filho_node_atual, novo_restante, passos_ref)

            elif filho_node_atual.prefixo.startswith(chave_str_restante):
                # O prefixo do filho é mais longo, mas começa com a chave_str_restante.
                # Isso significa que a chave_str_restante termina DENTRO do prefixo do filho.
                # Para ser um match, o nó filho precisa marcar o fim da chave_str_restante.
                # Essa situação é um pouco mais complexa para Radix Trees padrão, que comprimem.
                # Geralmente, se a chave termina no meio de um prefixo, não é um match a menos
                # que esse ponto tenha sido explicitamente marcado (o que acontece na divisão).
                # Na nossa inserção, um nó só é is_fim_de_chave se uma chave completa termina nele.
                return None  # Não encontrou um nó que termina EXATAMENTE com chave_str_restante

        # Se nenhum filho corresponde, e estamos na raiz e a raiz em si corresponde à chave_str_restante
        # (Isso é mais para tries não comprimidas ou o caso inicial).
        # Para RadixTree, se não achou filho e não é a raiz que casa perfeitamente, é falha.
        # A raiz da RadixTree tem prefixo "" implicitamente.
        if node == self.raiz and not node.filhos and not chave_str_restante and node.is_fim_de_chave:
            return node  # Raiz é o nó e a chave é vazia e é um fim de chave

        return None  # Nenhum caminho encontrado

    def exibir(self) -> None:
        print(f"\n{'=' * 70}\n=== RADIX TREE (Objetos Moto: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (₹)':<12}{'Revenda (₹)':<15}{'Ano':<6}\n{'-' * 70}")
        self._displayed_count_radix = 0
        self._exibir_recursive_radix(self.raiz, "")  # Passa prefixo acumulado vazio para raiz
        if self._count > self._displayed_count_radix:
            print(f"... e mais {self._count - self._displayed_count_radix} motos.")
        print("=" * 70)

    def _exibir_recursive_radix(self, node: Node, prefixo_atual_na_arvore: str) -> None:
        if self._displayed_count_radix >= 50: return

        # O nome completo da chave para este nó é prefixo_atual_na_arvore + node.prefixo
        # Mas só imprimimos os dados se node.is_fim_de_chave
        if node.is_fim_de_chave:
            for moto in node.dados:
                if self._displayed_count_radix < 50:
                    # Verificamos se o nome da moto realmente corresponde ao caminho da árvore
                    # (Para debug, mas a estrutura deve garantir isso)
                    # expected_full_key = prefixo_atual_na_arvore + node.prefixo
                    # if moto.nome.lower() == expected_full_key:
                    print(f"{moto.marca:<15}{moto.nome:<20}{moto.preco:<12.2f}{moto.revenda:<15.2f}{moto.ano:<6}")
                    self._displayed_count_radix += 1
                else:
                    return

        for char_inicial_filho, filho_node in sorted(node.filhos.items()):
            self._exibir_recursive_radix(filho_node,
                                         prefixo_atual_na_arvore + node.prefixo)  # Acumula o prefixo do nó atual
            if self._displayed_count_radix >= 50: return

    def __len__(self) -> int:
        return self._count

    # Remoção em RadixTree é complexa, mantendo como placeholder
    def remover(self, alvo: Moto) -> bool:  # Staticmethod removido, precisa de self
        # A remoção precisaria encontrar o nó, remover o objeto Moto da lista node.dados.
        # Se node.dados ficar vazio e o nó não tiver filhos, ele pode precisar ser removido
        # e possivelmente fundido com seu pai se o pai ficar com apenas um filho (e não for a raiz).
        # print(f"AVISO: Remoção em RadixTree não implementada completamente. {alvo.nome} pode não ser removido.")

        # Tentativa simplificada de remoção (não lida com compressão/fusão de nós)
        passos_ref = [0]
        no_alvo = self._buscar_recursivo_v2(self.raiz, alvo.nome.lower(), passos_ref)
        if no_alvo and alvo in no_alvo.dados:
            no_alvo.dados.remove(alvo)
            self._count -= 1
            if not no_alvo.dados:  # Se a lista de dados ficou vazia
                no_alvo.is_fim_de_chave = False  # Não é mais um fim de chave se não tem dados
            return True
        return False