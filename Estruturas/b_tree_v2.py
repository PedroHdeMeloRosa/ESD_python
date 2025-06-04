# Estruturas/b_tree_v2.py
from typing import List, Optional, Tuple, Any
from modelos.moto import Moto  # Certifique-se que Moto tem __lt__ e __eq__


class BTreeNodeV2:
    """
    Nó de uma Árvore B (Versão 2).
    Contém chaves (objetos Moto) e ponteiros para filhos.
    """

    def __init__(self, t: int, leaf: bool = False):
        self.t: int = t
        self.keys: List[Optional[Moto]] = [None] * (2 * t - 1)
        self.children: List[Optional[BTreeNodeV2]] = [None] * (2 * t)
        self.n: int = 0
        self.leaf: bool = leaf

    def __str__(self) -> str:
        key_strings = [str(k.nome if k else "None") for k in self.keys[:self.n]]
        return f"Node(Keys:{key_strings}, n:{self.n}, Leaf:{self.leaf})"

    def find_key_index(self, key_to_find: Moto) -> int:
        idx = 0
        while idx < self.n and self.keys[idx] is not None and self.keys[idx] < key_to_find:
            idx += 1
        return idx


class BTreeV2:
    """
    Implementação de uma Árvore B (Versão 2).
    Armazena objetos Moto, ordenados por sua implementação de __lt__ e __eq__.
    A remoção é placeholder e não funcionalmente completa.
    """

    def __init__(self, t: int):
        if t < 2:
            raise ValueError("Grau mínimo 't' da Árvore B deve ser pelo menos 2.")
        self.root: Optional[BTreeNodeV2] = None
        self.t: int = t
        self._count: int = 0

    def _search_recursive(self, current_node: Optional[BTreeNodeV2], key_to_find: Moto, passos: List[int]) -> Optional[
        Moto]:
        passos[0] += 1
        if current_node is None:
            return None

        idx = current_node.find_key_index(key_to_find)

        if idx < current_node.n and current_node.keys[idx] is not None and current_node.keys[idx] == key_to_find:
            return current_node.keys[idx]

        if current_node.leaf:
            return None

        return self._search_recursive(current_node.children[idx], key_to_find, passos)

    def buscar(self, key_to_find: Moto) -> Tuple[bool, int]:
        passos = [0]
        found_moto = self._search_recursive(self.root, key_to_find, passos)
        return found_moto is not None, passos[0]

    def _split_child(self, parent_node: BTreeNodeV2, child_index: int):
        full_child_node = parent_node.children[child_index]
        if full_child_node is None or full_child_node.n != (2 * self.t - 1):  # Segurança
            # print("Debug: Tentativa de dividir filho não cheio ou nulo.")
            return

        new_sibling_node = BTreeNodeV2(self.t, full_child_node.leaf)

        new_sibling_node.n = self.t - 1
        for j in range(self.t - 1):
            new_sibling_node.keys[j] = full_child_node.keys[j + self.t]
            full_child_node.keys[j + self.t] = None

        if not full_child_node.leaf:
            for j in range(self.t):
                new_sibling_node.children[j] = full_child_node.children[j + self.t]
                full_child_node.children[j + self.t] = None

        full_child_node.n = self.t - 1

        for j in range(parent_node.n, child_index, -1):
            parent_node.children[j + 1] = parent_node.children[j]

        parent_node.children[child_index + 1] = new_sibling_node

        for j in range(parent_node.n - 1, child_index - 1, -1):
            parent_node.keys[j + 1] = parent_node.keys[j]

        parent_node.keys[child_index] = full_child_node.keys[self.t - 1]
        full_child_node.keys[self.t - 1] = None

        parent_node.n += 1

    def _insert_non_full(self, node_to_insert_in: BTreeNodeV2, key_to_insert: Moto) -> bool:
        i = node_to_insert_in.n - 1

        if node_to_insert_in.leaf:
            while i >= 0 and node_to_insert_in.keys[i] is not None and key_to_insert < node_to_insert_in.keys[i]:
                node_to_insert_in.keys[i + 1] = node_to_insert_in.keys[i]
                i -= 1

            # Verifica duplicata
            if (i + 1 < node_to_insert_in.n) and \
                    node_to_insert_in.keys[i + 1] is not None and \
                    node_to_insert_in.keys[i + 1] == key_to_insert:
                return False

            node_to_insert_in.keys[i + 1] = key_to_insert
            node_to_insert_in.n += 1
            return True
        else:
            while i >= 0 and node_to_insert_in.keys[i] is not None and key_to_insert < node_to_insert_in.keys[i]:
                i -= 1
            i += 1

            child_to_descend = node_to_insert_in.children[i]
            if child_to_descend is None:  # Segurança, não deveria acontecer em árvore válida
                # print(f"Debug: Filho nulo encontrado em nó não folha durante inserção. Nó: {node_to_insert_in}, child_idx: {i}")
                # Poderia criar o filho aqui se a lógica permitisse, mas é mais um sinal de erro.
                return False

            if child_to_descend.n == (2 * self.t - 1):
                self._split_child(node_to_insert_in, i)
                if key_to_insert > node_to_insert_in.keys[i]:
                    i += 1

            return self._insert_non_full(node_to_insert_in.children[i], key_to_insert)

    def inserir(self, key_to_insert: Moto) -> None:
        if self.root is None:
            self.root = BTreeNodeV2(self.t, leaf=True)
            self.root.keys[0] = key_to_insert
            self.root.n = 1
            self._count = 1
            return

        current_root = self.root
        if current_root.n == (2 * self.t - 1):
            new_root_node = BTreeNodeV2(self.t, leaf=False)
            new_root_node.children[0] = current_root
            self.root = new_root_node
            self._split_child(new_root_node, 0)
            inserted = self._insert_non_full(new_root_node, key_to_insert)
        else:
            inserted = self._insert_non_full(current_root, key_to_insert)

        if inserted:
            self._count += 1

    @staticmethod
    def remover(key_to_remove: Moto) -> bool:
        """Placeholder para remoção. Não funcionalmente completa."""
        # A remoção completa é muito complexa para este escopo.
        # Para o benchmark, ela se comportará como se a chave não fosse encontrada
        # ou não fosse removida, dependendo do que é retornado.
        # Se _count for decrementado, a lógica de remoção real precisaria ser implementada.
        # print(f"Aviso: Remoção de {key_to_remove.nome} na BTreeV2 é placeholder.")

        # Simula uma busca para ver se existe, mas não altera a árvore
        # found, _ = self.buscar(key_to_remove)
        # if found:
        #     # Aqui iria a lógica complexa de remoção
        #     # Se fosse implementada e bem-sucedida: self._count -= 1
        #     return True # Poderia retornar True se encontrada, mesmo que não removida
        return False  # Indica que a remoção não foi (efetivamente) realizada

    def _traverse_recursive(self, node: Optional[BTreeNodeV2], result_list: List[Moto], max_items: int):
        if node is None or len(result_list) >= max_items:
            return

        i = 0
        while i < node.n:
            if not node.leaf and node.children[i] is not None:
                self._traverse_recursive(node.children[i], result_list, max_items)

            if len(result_list) < max_items and node.keys[i] is not None:
                result_list.append(node.keys[i])
            else:
                if len(result_list) >= max_items: return
            i += 1

        if not node.leaf and node.children[node.n] is not None and len(result_list) < max_items:
            self._traverse_recursive(node.children[node.n], result_list, max_items)

    def exibir(self) -> None:
        if not self.root:
            print("Árvore B (v2) vazia!")
            return

        print("\n" + "=" * 70)
        print(f"=== ÁRVORE B (v2) (Grau Mínimo t={self.t}, Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)

        motos_para_exibir: List[Moto] = []
        self._traverse_recursive(self.root, motos_para_exibir, max_items=50)

        for m in motos_para_exibir:
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")

        if self._count > len(motos_para_exibir):
            print(f"... e mais {self._count - len(motos_para_exibir)} motos não exibidas.")
        print("=" * 70)

    def __len__(self) -> int:
        return self._count