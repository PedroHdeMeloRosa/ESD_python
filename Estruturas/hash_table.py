from modelos.moto import Moto
from typing import List, Tuple, Dict, Any, Optional


class HashTable:
    def __init__(self, capacidade: int = 101,
                 fator_carga_max: float = 0.7,
                 max_elements: Optional[int] = None):
        if capacidade <= 0:
            raise ValueError("Capacidade deve ser um inteiro positivo.")
        if not (0 < fator_carga_max <= 1):
            raise ValueError("Fator de carga máximo deve estar entre 0 (exclusivo) e 1 (inclusivo).")

        self.capacidade = capacidade
        self.fator_carga_max = fator_carga_max
        self.elementos = 0
        self.table: List[List[Moto]] = [[] for _ in range(capacidade)]
        self.max_elements: Optional[int] = max_elements

        # --- NOVOS ATRIBUTOS PARA CONTAR COLISÕES ---
        self.total_colisoes_insercao = 0
        self.total_insercoes_sucesso = 0

    def _hash(self, chave_moto: Moto) -> int:
        return hash(chave_moto) % self.capacidade

    def _redimensionar(self) -> None:
        nova_capacidade = self.capacidade * 2 + 1
        old_table = self.table

        # Resetando a instância para a nova capacidade
        self.capacidade = nova_capacidade
        self.table = [[] for _ in range(nova_capacidade)]
        self.elementos = 0

        # Resetando os contadores de colisão, pois estamos re-inserindo tudo
        self.total_colisoes_insercao = 0
        self.total_insercoes_sucesso = 0

        # Re-insere todos os elementos antigos na nova tabela
        for bucket in old_table:
            for item_moto in bucket:
                self.inserir(item_moto)  # Chamar inserir para re-contar colisões corretamente

    def inserir(self, data: Moto) -> bool:
        if self.max_elements is not None and self.elementos >= self.max_elements:
            return False

        # Redimensiona ANTES de calcular o índice para garantir que o hash seja para a tabela correta
        if (self.elementos + 1) / self.capacidade > self.fator_carga_max:
            self._redimensionar()

        idx = self._hash(data)
        bucket = self.table[idx]

        # Verifica se o item já existe (duplicata)
        for item_existente in bucket:
            if item_existente == data:
                return False  # Duplicata, não inseriu

        # --- LÓGICA DE CONTAGEM DE COLISÃO ---
        # Se o bucket já tem um ou mais itens, esta inserção é uma colisão.
        if len(bucket) > 0:
            self.total_colisoes_insercao += 1

        # Insere o novo dado
        bucket.append(data)
        self.elementos += 1
        self.total_insercoes_sucesso += 1  # Conta apenas inserções bem-sucedidas (não duplicatas)
        return True

    def remover(self, alvo: Moto) -> bool:
        idx = self._hash(alvo)
        bucket = self.table[idx]
        for i, item in enumerate(bucket):
            if item == alvo:
                del bucket[i]
                self.elementos -= 1
                # Opcional: ajustar contadores de colisão na remoção é complexo
                # e geralmente não é feito. Vamos focar nas colisões de inserção.
                return True
        return False

    def buscar(self, alvo: Moto) -> Tuple[bool, int]:
        idx = self._hash(alvo)
        bucket = self.table[idx]
        passos = 1  # 1 passo para o cálculo do hash
        for item in bucket:
            passos += 1  # +1 passo para cada comparação dentro do bucket
            if item == alvo:
                return True, passos
        return False, passos

    def exibir(self) -> None:
        stats = self.obter_estatisticas_colisao()

        print(f"\n{'=' * 80}")
        print(f"=== TABELA HASH (Elementos: {self.elementos}, Capacidade: {self.capacidade}) ===")
        print(f"Fator de Carga: {stats['fator_carga_real']:.3f} | Fator Máx Configurado: {self.fator_carga_max}")
        print(
            f"Total Inserções c/ Colisão: {stats['total_colisoes_insercao']} ({stats['percentual_insercoes_com_colisao']:.2f}%)")
        print(
            f"Buckets Ocupados c/ Colisão: {stats['num_buckets_com_colisao']}/{stats['num_buckets_ocupados']} ({stats['percent_buckets_com_colisao_de_ocupados']:.2f}%) | Max Bucket: {stats['max_comprimento_bucket']}")
        print(f"{'-' * 80}")
        print(f"{'Marca':<15}{'Modelo':<25}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}")
        print(f"{'-' * 80}")

        count_displayed = 0
        for bucket in self.table:
            for item in bucket:
                if count_displayed < 50:
                    m = item
                    print(f"{m.marca:<15}{m.nome:<25}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}")
                    count_displayed += 1
                else:
                    break
            if count_displayed >= 50:
                break

        if self.elementos > count_displayed:
            print(f"... e mais {self.elementos - count_displayed} motos não exibidas.")
        print("=" * 80)

    def __len__(self) -> int:
        return self.elementos

    def obter_estatisticas_colisao(self) -> Dict[str, Any]:
        if self.capacidade == 0:
            return {
                'fator_carga_real': 0, 'num_buckets_vazios': 0, 'num_buckets_ocupados': 0,
                'num_buckets_com_colisao': 0, 'max_comprimento_bucket': 0,
                'avg_comprimento_bucket_ocupado': 0.0,
                'percent_buckets_com_colisao_de_total': 0.0,
                'percent_buckets_com_colisao_de_ocupados': 0.0,
                'total_insercoes_sucesso': self.total_insercoes_sucesso,
                'total_colisoes_insercao': self.total_colisoes_insercao,
                'percentual_insercoes_com_colisao': 0.0
            }

        n_vazios, n_colisao, n_ocupados, max_len, soma_len_ocupados = 0, 0, 0, 0, 0
        for bucket in self.table:
            tam = len(bucket)
            if tam == 0:
                n_vazios += 1
            else:
                n_ocupados += 1
                soma_len_ocupados += tam
                if tam > max_len:
                    max_len = tam
                if tam > 1:
                    n_colisao += 1

        # --- NOVAS ESTATÍSTICAS ---
        percent_colisao_insercao = (self.total_colisoes_insercao / self.total_insercoes_sucesso * 100) \
            if self.total_insercoes_sucesso > 0 else 0.0

        avg_len_ocup = (soma_len_ocupados / n_ocupados) if n_ocupados > 0 else 0.0
        perc_col_total = (n_colisao / self.capacidade * 100) if self.capacidade > 0 else 0.0
        perc_col_ocup = (n_colisao / n_ocupados * 100) if n_ocupados > 0 else 0.0

        return {
            'fator_carga_real': self.elementos / self.capacidade if self.capacidade > 0 else 0,
            'num_buckets_vazios': n_vazios,
            'num_buckets_ocupados': n_ocupados,
            'num_buckets_com_colisao': n_colisao,
            'max_comprimento_bucket': max_len,
            'avg_comprimento_bucket_ocupado': avg_len_ocup,
            'percent_buckets_com_colisao_de_total': perc_col_total,
            'percent_buckets_com_colisao_de_ocupados': perc_col_ocup,
            'total_insercoes_sucesso': self.total_insercoes_sucesso,
            'total_colisoes_insercao': self.total_colisoes_insercao,
            'percentual_insercoes_com_colisao': percent_colisao_insercao
        }

    def analisar_distribuicao_hash(self, num_bins_histograma=10) -> None:
        """
        Gera um histograma da distribuição do comprimento dos buckets para analisar a
        qualidade da função de hash.
        """
        if self.elementos == 0:
            print("Tabela Hash vazia. Nenhuma análise de distribuição para fazer.")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        comprimentos_buckets = [len(bucket) for bucket in self.table]

        # Chi-quadrado para teste de qualidade da distribuição (avançado, mas impressionante)
        # H0: A distribuição dos elementos é uniforme.
        observed_counts, _ = np.histogram(comprimentos_buckets, bins=range(max(comprimentos_buckets) + 2))
        total_elementos = self.elementos
        total_buckets = self.capacidade
        expected_count_per_bucket = total_elementos / total_buckets

        # A estatística Chi-quadrado é mais complexa de aplicar diretamente aqui,
        # pois a expectativa não é que o *comprimento* dos buckets seja uniforme,
        # mas que cada *elemento* tenha a mesma probabilidade de cair em qualquer bucket.
        # Vamos focar na visualização e em estatísticas descritivas que são mais diretas.

        variancia_comprimentos = np.var(comprimentos_buckets)

        print("\n--- Análise de Distribuição da Tabela Hash ---")
        print(f"Variância do comprimento dos buckets: {variancia_comprimentos:.4f}")
        print("(Idealmente, uma variância próxima de λ (fator de carga) indica uma boa distribuição de Poisson)")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(comprimentos_buckets, bins=range(max(comprimentos_buckets) + 2),
                    align='left', rwidth=0.8, color='cornflowerblue', edgecolor='black')
            ax.set_title('Distribuição do Comprimento dos Buckets (Hash Quality)')
            ax.set_xlabel('Número de Itens no Bucket (Comprimento)')
            ax.set_ylabel('Número de Buckets com esse Comprimento')
            ax.set_xticks(range(max(comprimentos_buckets) + 2))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Não foi possível gerar o gráfico de distribuição: {e}")