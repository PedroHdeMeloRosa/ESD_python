# Estruturas/hash_table.py
from modelos.moto import Moto
from typing import List, Tuple


class HashTable:
    """
    Implementação de uma Tabela Hash com tratamento de colisão por encadeamento.
    Utiliza o hash embutido do Python para objetos Moto (requer __hash__ e __eq__ em Moto).
    Realiza redimensionamento quando o fator de carga máximo é atingido.
    """

    def __init__(self, capacidade: int = 101, fator_carga_max: float = 0.7):
        """
        Inicializa a Tabela Hash.
        :param capacidade: Capacidade inicial da tabela.
        :param fator_carga_max: Fator de carga máximo antes do redimensionamento.
        """
        if capacidade <= 0:
            raise ValueError("Capacidade deve ser um inteiro positivo.")
        if not (0 < fator_carga_max <= 1):
            raise ValueError("Fator de carga máximo deve estar entre 0 (exclusivo) e 1 (inclusivo).")

        self.capacidade = capacidade
        self.fator_carga_max = fator_carga_max
        self.elementos = 0
        self.table: List[List[Moto]] = [[] for _ in range(capacidade)]

    def _hash(self, chave_moto: Moto) -> int:
        """Calcula o índice hash para uma Moto."""
        return hash(chave_moto) % self.capacidade

    def _redimensionar(self) -> None:
        """Dobra a capacidade da tabela e redistribui os elementos."""
        nova_capacidade = self.capacidade * 2 + 1  # Garante que seja ímpar, pode ajudar na distribuição
        print(f"Redimensionando tabela hash de {self.capacidade} para {nova_capacidade}...")
        nova_tabela: List[List[Moto]] = [[] for _ in range(nova_capacidade)]

        old_table = self.table
        self.table = nova_tabela
        self.capacidade = nova_capacidade
        self.elementos = 0  # Será recontado durante a reinserção

        for bucket in old_table:
            for item in bucket:
                # Precisamos chamar inserir para que o contador de elementos e o hash sejam recalculados corretamente
                # com a nova capacidade.
                self.inserir(item)  # Chamada recursiva na ideia, mas com a nova tabela

    def inserir(self, data: Moto) -> None:
        """
        Insere um objeto Moto na tabela.
        Se a moto já existir (de acordo com __eq__), ela não é inserida novamente para evitar duplicatas no mesmo bucket.
        :param data: Objeto Moto a ser inserido.
        """
        # Verifica se precisa redimensionar ANTES de calcular o índice e inserir
        if (self.elementos + 1) / self.capacidade > self.fator_carga_max:
            # Armazena a tabela atual e a capacidade para a operação de reinserção
            # que acontecerá dentro de _redimensionar se chamarmos inserir recursivamente.
            # Esta lógica de redimensionamento foi simplificada acima.
            # Aqui, o _redimensionar já vai mover os itens.
            # A verificação de duplicatas será feita abaixo.
            old_table = self.table
            old_capacidade = self.capacidade

            nova_capacidade_temp = self.capacidade * 2 + 1
            nova_tabela_temp: List[List[Moto]] = [[] for _ in range(nova_capacidade_temp)]

            self.table = nova_tabela_temp
            self.capacidade = nova_capacidade_temp
            self.elementos = 0

            for bucket in old_table:
                for item_moto in bucket:
                    idx_temp = hash(item_moto) % self.capacidade  # Novo hash com nova capacidade
                    self.table[idx_temp].append(item_moto)  # Adiciona à nova tabela
                    self.elementos += 1  # Incrementa elementos

        idx = self._hash(data)

        # Evitar duplicatas no mesmo bucket
        for item_existente in self.table[idx]:
            if item_existente == data:
                return  # Item já existe, não insere novamente

        self.table[idx].append(data)
        self.elementos += 1

    def remover(self, alvo: Moto) -> bool:
        """
        Remove um objeto Moto da tabela.
        :param alvo: Objeto Moto a ser removido.
        :return: True se o item foi removido, False caso contrário.
        """
        idx = self._hash(alvo)
        bucket = self.table[idx]

        for i, item in enumerate(bucket):
            if item == alvo:
                del bucket[i]
                self.elementos -= 1
                return True
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        """
        Busca por um objeto Moto na tabela.
        :param alvo: Objeto Moto a ser buscado.
        :return: Tupla (encontrado: bool, passos: int).
        """
        idx = self._hash(alvo)
        bucket = self.table[idx]
        passos = 0

        for item in bucket:
            passos += 1  # Conta cada comparação dentro do bucket
            if item == alvo:
                return True, passos
        return False, passos + (1 if not bucket else 0)  # +1 se o bucket estava vazio (acesso ao bucket)

    def exibir(self) -> None:
        """Exibe o conteúdo da tabela hash no console."""
        print(f"\n{'=' * 70}")
        print(
            f"=== TABELA HASH (Elementos: {self.elementos}, Capacidade: {self.capacidade}, Fator de Carga: {self.elementos / self.capacidade:.2f}) ===")
        print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print("-" * 70)

        count_displayed = 0
        for i, bucket in enumerate(self.table):
            if bucket:  # Só imprime buckets não vazios
                # print(f"Bucket {i}:") # Opcional: para ver a distribuição
                for item in bucket:
                    if count_displayed < 50:  # Limita a exibição
                        m = item
                        print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                        count_displayed += 1
                    else:
                        break
            if count_displayed >= 50:
                break

        if self.elementos > count_displayed:
            print(f"... e mais {self.elementos - count_displayed} motos não exibidas.")
        print("=" * 70)

    def __len__(self) -> int:
        """Retorna o número de elementos na tabela hash."""
        return self.elementos