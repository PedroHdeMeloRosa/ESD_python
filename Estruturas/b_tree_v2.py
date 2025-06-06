# Estruturas/b_tree_v2.py
from typing import List, Optional, Tuple, Any
from modelos.moto import Moto


class BTreeNodeV2:
    def __init__(self, t: int, leaf: bool = False):
        self.t: int = t;
        self.keys: List[Optional[Moto]] = [None] * (2 * t - 1)
        self.children: List[Optional[BTreeNodeV2]] = [None] * (2 * t)
        self.n: int = 0;
        self.leaf: bool = leaf

    def find_key_index(self, key_to_find: Moto) -> int:
        idx = 0
        while idx < self.n and self.keys[idx] is not None and self.keys[idx] < key_to_find: idx += 1
        return idx


class BTreeV2:
    def __init__(self, t: int, max_elements: Optional[int] = None):  # MODIFICADO
        if t < 2: raise ValueError("Grau mínimo 't' da Árvore B deve ser >= 2.")
        self.root: Optional[BTreeNodeV2] = None;
        self.t: int = t;
        self._count: int = 0
        self._search_step_limit: Optional[int] = None  # NOVO
        self.max_elements: Optional[int] = max_elements  # NOVO

    def set_search_step_limit(self, limit: Optional[int]):  # NOVO MÉTODO
        self._search_step_limit = limit
        # if limit is not None: print(f"INFO (BTreeV2): Limite de busca -> {limit} passos.")
        # else: print("INFO (BTreeV2): Limite de busca removido.")

    def _search_recursive(self, current_node: Optional[BTreeNodeV2], key_to_find: Moto, passos_ref: List[int]) -> \
    Optional[Moto]:
        if current_node is None: return None
        passos_ref[0] += 1
        if self._search_step_limit is not None and passos_ref[0] > self._search_step_limit:  # MODIFICADO: Checa A1
            return None
        idx = current_node.find_key_index(key_to_find)
        if idx < current_node.n and current_node.keys[idx] is not None and current_node.keys[idx] == key_to_find:
            return current_node.keys[idx]
        if current_node.leaf: return None
        return self._search_recursive(current_node.children[idx], key_to_find, passos_ref)

    def buscar(self, key_to_find: Moto) -> Tuple[bool, int]:  # ... (sem mudanças)
        passos_container = [0];
        found_moto = self._search_recursive(self.root, key_to_find, passos_container)
        return found_moto is not None, passos_container[0]

    def _split_child(self, parent_node: BTreeNodeV2, child_index: int):  # ... (sem mudanças)
        full_child_node = parent_node.children[child_index];
        new_sibling_node = BTreeNodeV2(self.t, full_child_node.leaf)
        new_sibling_node.n = self.t - 1
        for j in range(self.t - 1): new_sibling_node.keys[j] = full_child_node.keys[j + self.t];full_child_node.keys[
            j + self.t] = None
        if not full_child_node.leaf:
            for j in range(self.t): new_sibling_node.children[j] = full_child_node.children[j + self.t];
            full_child_node.children[j + self.t] = None
        full_child_node.n = self.t - 1
        for j in range(parent_node.n, child_index, -1): parent_node.children[j + 1] = parent_node.children[j]
        parent_node.children[child_index + 1] = new_sibling_node
        for j in range(parent_node.n - 1, child_index - 1, -1): parent_node.keys[j + 1] = parent_node.keys[j]
        parent_node.keys[child_index] = full_child_node.keys[self.t - 1];
        full_child_node.keys[self.t - 1] = None
        parent_node.n += 1

    def _insert_non_full(self, node_to_insert_in: BTreeNodeV2, key_to_insert: Moto) -> bool:
        i = node_to_insert_in.n - 1  # Índice da última chave válida

        if node_to_insert_in.leaf:
            # Desloca chaves maiores para a direita para abrir espaço
            # Garantir que estamos comparando apenas com chaves válidas (não None)
            while i >= 0 and node_to_insert_in.keys[i] is not None and \
                    key_to_insert < node_to_insert_in.keys[i]:  # key_to_insert é Moto, keys[i] deve ser Moto
                node_to_insert_in.keys[i + 1] = node_to_insert_in.keys[i]
                i -= 1

            # Posição de inserção é i + 1
            # Verifica se já existe uma chave EXATAMENTE IGUAL nessa posição (para duplicatas)
            # Isso só acontece se keys[i+1] não for None e for igual
            if (i + 1 < node_to_insert_in.n and  # Checa se i+1 está dentro das chaves existentes
                    node_to_insert_in.keys[i + 1] is not None and
                    node_to_insert_in.keys[i + 1] == key_to_insert):
                return False  # Duplicata exata, não inseriu de novo

            node_to_insert_in.keys[i + 1] = key_to_insert
            node_to_insert_in.n += 1
            return True
        else:  # Nó não é folha
            # Encontra o filho onde a nova chave deve ser inserida
            while i >= 0 and node_to_insert_in.keys[i] is not None and \
                    key_to_insert < node_to_insert_in.keys[i]:  # key_to_insert é Moto, keys[i] deve ser Moto
                i -= 1

            # Após o loop, 'i' é o índice da chave que é <= key_to_insert (ou -1 se todas são >)
            # O filho para descer é children[i+1]
            child_descent_index = i + 1

            child_to_descend = node_to_insert_in.children[child_descent_index]
            if child_to_descend is None:  # Sanity check, não deveria ocorrer
                print(f"ERRO BTree: Filho em children[{child_descent_index}] é None durante insert_non_full.")
                return False  # Falha na inserção

            if child_to_descend.n == (2 * self.t - 1):  # Se o filho está cheio
                self._split_child(node_to_insert_in, child_descent_index)  # Divide children[child_descent_index]
                # Após o split, node_to_insert_in.keys[child_descent_index] é a nova chave mediana que subiu.
                # Precisamos reavaliar para qual filho descer:
                # Se key_to_insert é maior que a nova chave mediana, o índice do filho aumenta.

                # Adiciona verificação crucial:
                if node_to_insert_in.keys[child_descent_index] is None:
                    print(
                        f"ERRO CRÍTICO BTree: Chave em keys[{child_descent_index}] é None após split. Pai: {node_to_insert_in}")
                    # Este é um estado inconsistente, idealmente não deveria acontecer.
                    # Para evitar o crash, podemos tentar uma heurística, mas é um bug.
                    # Ex: se a chave que subiu é None, talvez a key_to_insert seja menor e vá para o filho esquerdo (original child_descent_index)
                    # ou maior e vá para o filho direito (child_descent_index + 1).
                    # Como fallback, vamos assumir que vai para o filho da esquerda se não pudermos comparar.
                elif key_to_insert > node_to_insert_in.keys[child_descent_index]:
                    child_descent_index += 1

            return self._insert_non_full(node_to_insert_in.children[child_descent_index], key_to_insert)

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
        inserted_flag = False
        if current_root.n == (2 * self.t - 1):  # Raiz está cheia
            new_root_node = BTreeNodeV2(self.t, leaf=False)
            new_root_node.children[0] = current_root
            self.root = new_root_node  # Nova raiz
            self._split_child(new_root_node, 0)  # Divide a raiz antiga (filho 0 da nova)
            # Agora new_root_node tem 1 chave e 2 filhos.
            # A chave que subiu está em new_root_node.keys[0]
            # Decide em qual dos dois filhos (children[0] ou children[1]) da nova raiz inserir.
            # Esta lógica já está implícita em _insert_non_full
            inserted_flag = self._insert_non_full(new_root_node, key_to_insert)
        else:  # Raiz não está cheia
            inserted_flag = self._insert_non_full(current_root, key_to_insert)

        if inserted_flag:
            self._count += 1
        return inserted_flag

    def remover(self, key_to_remove: Moto) -> bool:
        return False  # Placeholder

    def _traverse_recursive(self, node: Optional[BTreeNodeV2], result_list: List[Moto],
                            max_items: int):  # ... (sem mudanças)
        if node is None or len(result_list) >= max_items: return
        i = 0
        while i < node.n:
            if not node.leaf and node.children[i] is not None: self._traverse_recursive(node.children[i], result_list,
                                                                                        max_items)
            if len(result_list) < max_items and node.keys[i] is not None:
                result_list.append(node.keys[i])
            else:
                if len(result_list) >= max_items: return
            i += 1
        if not node.leaf and node.children[node.n] is not None and len(result_list) < max_items:
            self._traverse_recursive(node.children[node.n], result_list, max_items)

    def exibir(self) -> None:  # ... (sem mudanças)
        if not self.root: print("Árvore B (v2) vazia!"); return
        print(f"\n{'=' * 70}\n=== ÁRVORE B (v2) (t={self.t}, Elementos: {self._count}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço':<12}{'Revenda':<15}{'Ano':<6}\n{'-' * 70}")
        motos_ex: List[Moto] = [];
        self._traverse_recursive(self.root, motos_ex, 50)
        for m in motos_ex: print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
        if self._count > len(motos_ex): print(f"... e mais {self._count - len(motos_ex)} motos.")
        print("=" * 70)

    def __len__(self) -> int:
        return self._count