# Estruturas/radix_tree.py
from modelos.moto import Moto
from typing import Dict, List, Optional, Tuple


class RadixTree:
    class Node:
        def __init__(self, prefixo: str = ""):
            self.prefixo: str = prefixo
            self.filhos: Dict[str, RadixTree.Node] = {}
            self.dados: List[Moto] = []
            self.is_fim_de_chave: bool = False

    def __init__(self, max_elements: Optional[int] = None):  # NOVO: max_elements
        self.raiz = self.Node()
        self._count = 0  # Número de OBJETOS Moto distintos
        self.max_elements: Optional[int] = max_elements  # NOVO ATRIBUTO

    def inserir(self, data: Moto) -> bool:  # Modificado para retornar bool
        # A RadixTree pode ter múltiplos objetos Moto sob a mesma chave de string (nome da moto)
        # O limite de max_elements aqui se refere ao número total de *objetos Moto* na árvore.
        if self.max_elements is not None and self._count >= self.max_elements:
            # print(f"AVISO (RadixTree): Capacidade máxima de {self.max_elements} elementos atingida. Não inserindo {data.nome}.")
            return False

        # A lógica de _inserir_recursivo já verifica duplicatas de objetos Moto exatos
        # dentro da lista 'dados' de um nó.
        # Retornaremos o status de _inserir_recursivo.
        inserido_novo_objeto = self._inserir_recursivo(self.raiz, data.nome.lower(), data)
        if inserido_novo_objeto:
            self._count += 1
        return inserido_novo_objeto

    def _inserir_recursivo(self, node: Node, chave_restante: str, data: Moto) -> bool:
        # ... (lógica interna de _inserir_recursivo como na sua versão funcional)
        #    Certifique-se que ele retorna True se um NOVO objeto Moto foi adicionado à lista 'dados' de um nó,
        #    e False se o objeto Moto EXATO já existia lá.
        #    A lógica anterior que você tinha para RadixTree parecia tratar isso.
        #    O importante é o `if data not in node.dados: node.dados.append(data); node.is_fim_de_chave = True; return True`
        #    e retornar False em outros casos onde não há adição.
        #
        # VOU COLAR A LÓGICA DE INSERÇÃO DA SUA VERSÃO ANTERIOR DE RADIXTREE AQUI,
        # APENAS GARANTINDO QUE O RETORNO SEJA bool PARA indicar se foi novo.
        len_prefixo_node = len(node.prefixo);
        len_chave_restante = len(chave_restante)
        match_len = 0
        while (match_len < len_prefixo_node and match_len < len_chave_restante and
               node.prefixo[match_len] == chave_restante[match_len]):
            match_len += 1
        if match_len < len_prefixo_node:  # Divide nó atual
            prefixo_filho_existente = node.prefixo[match_len:]
            novo_filho_existente = self.Node(prefixo_filho_existente)
            novo_filho_existente.filhos = node.filhos;
            novo_filho_existente.dados = list(node.dados)
            novo_filho_existente.is_fim_de_chave = node.is_fim_de_chave
            node.prefixo = node.prefixo[:match_len]
            node.filhos = {prefixo_filho_existente[0]: novo_filho_existente} if prefixo_filho_existente else {}
            node.dados = [];
            node.is_fim_de_chave = False
            if match_len == len_chave_restante:  # Chave termina no ponto de divisão
                if data not in node.dados: node.dados.append(data); node.is_fim_de_chave = True; return True
                return False  # Duplicata
            sufixo_nova_chave = chave_restante[match_len:]
            if not sufixo_nova_chave: return False  # Should not happen if len_chave_restante > match_len
            novo_filho_para_sufixo = self.Node(sufixo_nova_chave)
            novo_filho_para_sufixo.dados.append(data);
            novo_filho_para_sufixo.is_fim_de_chave = True
            node.filhos[sufixo_nova_chave[0]] = novo_filho_para_sufixo
            return True
        elif match_len == len_prefixo_node:  # Prefixo do nó é prefixo da chave
            if match_len == len_chave_restante:  # Chave igual ao prefixo
                if data not in node.dados: node.dados.append(data); node.is_fim_de_chave = True; return True
                return False  # Duplicata
            sufixo_chave = chave_restante[match_len:]
            if not sufixo_chave: return False  # Caso chave_restante seja igual ao prefixo, já tratado.
            if sufixo_chave[0] in node.filhos:
                return self._inserir_recursivo(node.filhos[sufixo_chave[0]], sufixo_chave, data)
            else:
                novo_filho = self.Node(sufixo_chave)
                novo_filho.dados.append(data);
                novo_filho.is_fim_de_chave = True
                node.filhos[sufixo_chave[0]] = novo_filho
                return True
        return False  # Caso não coberto (ex: raiz com prefixo vazio e nenhuma correspondência inicial)

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:  # ... (sem mudanças na lógica de busca em si)
        passos = [0];
        encontrado = self._buscar_recursive_radix(self.raiz, alvo.nome.lower(), alvo, passos)
        return encontrado, passos[0]

    def _buscar_recursive_radix(self, node: Node, chave_restante: str, alvo_obj: Moto, passos: List[int]) -> bool:
        # ... (lógica interna de _buscar_recursive_radix como na sua versão funcional)
        passos[0] += 1
        if not chave_restante.startswith(node.prefixo): return False
        sufixo_chave = chave_restante[len(node.prefixo):]
        if not sufixo_chave:  # Chegou ao fim da chave, no nó atual
            return node.is_fim_de_chave and alvo_obj in node.dados  # Verifica objeto exato
        if not node.filhos or sufixo_chave[0] not in node.filhos: return False
        return self._buscar_recursive_radix(node.filhos[sufixo_chave[0]], sufixo_chave, alvo_obj, passos)

    def exibir(self) -> None:  # ... (sem mudanças)
        print(f"\n{'=' * 70}\n=== RADIX TREE (Objetos Moto: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço':<12}{'Revenda':<15}{'Ano':<6}\n{'-' * 70}")
        self._displayed_count_radix = 0;
        self._exibir_recursive_radix(self.raiz)
        if self._count > self._displayed_count_radix: print(
            f"... e mais {self._count - self._displayed_count_radix} motos.")
        print("=" * 70)

    def _exibir_recursive_radix(self, node: Node) -> None:  # ... (sem mudanças)
        if self._displayed_count_radix >= 50: return
        if node.is_fim_de_chave:
            for moto in node.dados:
                if self._displayed_count_radix < 50:
                    print(f"{moto.marca:<15}{moto.nome:<20}{moto.preco:<12.2f}{moto.revenda:<15.2f}{moto.ano:<6}")
                    self._displayed_count_radix += 1
                else:
                    return
        for filho in sorted(node.filhos.values(), key=lambda n: n.prefixo):  # Ordem para consistência
            self._exibir_recursive_radix(filho)
            if self._displayed_count_radix >= 50: return

    def __len__(self) -> int:
        return self._count

    # Remoção em RadixTree é complexa, mantendo como placeholder
    def remover(self, alvo: Moto) -> bool:
        # print(f"AVISO: Remoção em RadixTree não implementada. {alvo.nome} não removido.")
        return False