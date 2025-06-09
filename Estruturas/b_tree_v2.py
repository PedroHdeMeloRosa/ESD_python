# Estruturas/b_tree_v2.py
from typing import List, Optional, Tuple, Any
from modelos.moto import Moto


class BTreeNodeV2:
    def __init__(self, t: int, leaf: bool = False):
        self.t: int = t
        self.keys: List[Optional[Moto]] = [None] * (2 * t - 1)
        self.children: List[Optional[BTreeNodeV2]] = [None] * (2 * t)
        self.n: int = 0
        self.leaf: bool = leaf

    def find_key_index(self, key_to_find: Moto) -> int:
        idx = 0
        while idx < self.n and self.keys[idx] is not None and self.keys[idx] < key_to_find:
            idx += 1
        return idx


class BTreeV2:
    def __init__(self, t: int, max_elements: Optional[int] = None):
        if t < 2:
            raise ValueError("Grau mínimo 't' da Árvore B deve ser >= 2.")
        self.root: Optional[BTreeNodeV2] = None
        self.t: int = t
        self._count: int = 0
        self._search_step_limit: Optional[int] = None
        self.max_elements: Optional[int] = max_elements

    def set_search_step_limit(self, limit: Optional[int]):
        self._search_step_limit = limit

    def __len__(self) -> int:
        return self._count

    # --- Métodos de Busca ---
    def buscar(self, key_to_find: Moto) -> Tuple[bool, int]:
        passos_container = [0]
        found_moto = self._search_recursive(self.root, key_to_find, passos_container)
        return found_moto is not None, passos_container[0]

    def _search_recursive(self, current_node: Optional[BTreeNodeV2], key_to_find: Moto, passos_ref: List[int]) -> \
    Optional[Moto]:
        if current_node is None:
            return None

        passos_ref[0] += 1
        if self._search_step_limit is not None and passos_ref[0] > self._search_step_limit:
            return None

        idx = current_node.find_key_index(key_to_find)

        if idx < current_node.n and current_node.keys[idx] is not None and current_node.keys[idx] == key_to_find:
            return current_node.keys[idx]

        if current_node.leaf:
            return None

        return self._search_recursive(current_node.children[idx], key_to_find, passos_ref)

    # --- Métodos de Inserção ---
    def inserir(self, key_to_insert: Moto) -> bool:
        if self.max_elements is not None and self._count >= self.max_elements:
            return False

        if self.root is None:
            self.root = BTreeNodeV2(self.t, leaf=True)
            self.root.keys[0] = key_to_insert
            self.root.n = 1
            self._count = 1
            return True

        current_root = self.root
        if current_root.n == (2 * self.t - 1):
            new_root_node = BTreeNodeV2(self.t, leaf=False)
            new_root_node.children[0] = current_root
            self.root = new_root_node
            self._split_child(new_root_node, 0)
            inserted_flag = self._insert_non_full(new_root_node, key_to_insert)
        else:
            inserted_flag = self._insert_non_full(current_root, key_to_insert)

        if inserted_flag:
            self._count += 1
        return inserted_flag

    def _split_child(self, parent_node: BTreeNodeV2, child_index: int):
        full_child_node = parent_node.children[child_index]
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

    def _insert_non_full(self, node: BTreeNodeV2, key: Moto) -> bool:
        i = node.n - 1
        if node.leaf:
            while i >= 0 and node.keys[i] is not None and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1

            if (i + 1 < node.n and node.keys[i + 1] is not None and node.keys[i + 1] == key):
                return False

            node.keys[i + 1] = key
            node.n += 1
            return True
        else:
            while i >= 0 and node.keys[i] is not None and key < node.keys[i]:
                i -= 1
            i += 1

            if node.children[i].n == 2 * self.t - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            return self._insert_non_full(node.children[i], key)

    # --- Métodos de Remoção (NOVO E COMPLETO) ---
    def remover(self, key_to_remove: Moto) -> bool:
        if not self.root:
            return False

        foi_removido = self._remove_from_subtree(self.root, key_to_remove)

        if foi_removido:
            self._count -= 1
            if self.root.n == 0:
                self.root = self.root.children[0] if not self.root.leaf else None
                if self.root is None:
                    self._count = 0
        return foi_removido

    def _remove_from_subtree(self, node: BTreeNodeV2, key_to_remove: Moto) -> bool:
        idx = node.find_key_index(key_to_remove)

        if idx < node.n and node.keys[idx] == key_to_remove:
            if node.leaf:
                for i in range(idx, node.n - 1):
                    node.keys[i] = node.keys[i + 1]
                node.keys[node.n - 1] = None
                node.n -= 1
                return True
            else:
                return self._remove_from_internal_node(node, idx)
        else:
            if node.leaf:
                return False

            child_index_to_descend = idx

            if node.children[child_index_to_descend].n < self.t:
                self._fill_child(node, child_index_to_descend)

            if child_index_to_descend > node.n:
                # Se a fusão ocorreu com o filho anterior, precisamos descer para esse filho
                return self._remove_from_subtree(node.children[child_index_to_descend - 1], key_to_remove)
            else:
                return self._remove_from_subtree(node.children[child_index_to_descend], key_to_remove)

    def _remove_from_internal_node(self, node: BTreeNodeV2, key_idx: int) -> bool:
        key_to_remove = node.keys[key_idx]
        predecessor_child = node.children[key_idx]
        successor_child = node.children[key_idx + 1]

        if predecessor_child.n >= self.t:
            predecessor = self._get_predecessor(predecessor_child)
            node.keys[key_idx] = predecessor
            return self._remove_from_subtree(predecessor_child, predecessor)
        elif successor_child.n >= self.t:
            successor = self._get_successor(successor_child)
            node.keys[key_idx] = successor
            return self._remove_from_subtree(successor_child, successor)
        else:
            self._merge(node, key_idx)
            return self._remove_from_subtree(predecessor_child, key_to_remove)

    def _get_predecessor(self, node: BTreeNodeV2) -> Moto:
        current = node
        while not current.leaf:
            current = current.children[current.n]
        return current.keys[current.n - 1]

    def _get_successor(self, node: BTreeNodeV2) -> Moto:
        current = node
        while not current.leaf:
            current = current.children[0]
        return current.keys[0]

    def _fill_child(self, node: BTreeNodeV2, child_idx: int):
        if child_idx != 0 and node.children[child_idx - 1].n >= self.t:
            self._borrow_from_prev(node, child_idx)
        elif child_idx != node.n and node.children[child_idx + 1].n >= self.t:
            self._borrow_from_next(node, child_idx)
        else:
            if child_idx != node.n:
                self._merge(node, child_idx)
            else:
                self._merge(node, child_idx - 1)

    def _borrow_from_prev(self, node: BTreeNodeV2, child_idx: int):
        child = node.children[child_idx]
        sibling = node.children[child_idx - 1]

        for i in range(child.n - 1, -1, -1):
            child.keys[i + 1] = child.keys[i]

        if not child.leaf:
            for i in range(child.n, -1, -1):
                child.children[i + 1] = child.children[i]

        child.keys[0] = node.keys[child_idx - 1]

        if not child.leaf:
            child.children[0] = sibling.children[sibling.n]

        node.keys[child_idx - 1] = sibling.keys[sibling.n - 1]

        sibling.keys[sibling.n - 1] = None
        if not sibling.leaf:
            sibling.children[sibling.n] = None

        child.n += 1
        sibling.n -= 1

    def _borrow_from_next(self, node: BTreeNodeV2, child_idx: int):
        child = node.children[child_idx]
        sibling = node.children[child_idx + 1]

        child.keys[child.n] = node.keys[child_idx]

        if not child.leaf:
            child.children[child.n + 1] = sibling.children[0]

        node.keys[child_idx] = sibling.keys[0]

        for i in range(sibling.n - 1):
            sibling.keys[i] = sibling.keys[i + 1]

        if not sibling.leaf:
            for i in range(sibling.n):
                sibling.children[i] = sibling.children[i + 1]

        sibling.keys[sibling.n - 1] = None
        if not sibling.leaf:
            sibling.children[sibling.n] = None

        child.n += 1
        sibling.n -= 1

    def _merge(self, node: BTreeNodeV2, child_idx: int):
        child = node.children[child_idx]
        sibling = node.children[child_idx + 1]

        child.keys[self.t - 1] = node.keys[child_idx]

        for i in range(sibling.n):
            child.keys[i + self.t] = sibling.keys[i]

        if not child.leaf:
            for i in range(sibling.n + 1):
                child.children[i + self.t] = sibling.children[i]

        for i in range(child_idx + 1, node.n):
            node.keys[i - 1] = node.keys[i]

        for i in range(child_idx + 2, node.n + 1):
            node.children[i - 1] = node.children[i]

        node.keys[node.n - 1] = None
        node.children[node.n] = None

        node.n -= 1
        child.n += sibling.n + 1

    # --- Métodos de Exibição ---
    def exibir(self) -> None:
        if not self.root:
            print("Árvore B (v2) vazia!")
            return
        print(f"\n{'=' * 70}\n=== ÁRVORE B (v2) (t={self.t}, Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (₹)':<12}{'Revenda':<15}{'Ano':<6}\n{'-' * 70}")
        motos_ex: List[Moto] = []
        self._traverse_recursive_for_display(self.root, motos_ex, 50)
        for m in motos_ex:
            print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
        if self._count > len(motos_ex):
            print(f"... e mais {self._count - len(motos_ex)} motos.")
        print("=" * 70)

    def _traverse_recursive_for_display(self, node: Optional[BTreeNodeV2], result_list: List[Moto], max_items: int):
        if node is None or len(result_list) >= max_items:
            return
        i = 0
        while i < node.n:
            if not node.leaf:
                self._traverse_recursive_for_display(node.children[i], result_list, max_items)

            if len(result_list) < max_items and node.keys[i] is not None:
                result_list.append(node.keys[i])
            else:
                if len(result_list) >= max_items: return
            i += 1

        if not node.leaf and len(result_list) < max_items:
            self._traverse_recursive_for_display(node.children[node.n], result_list, max_items)