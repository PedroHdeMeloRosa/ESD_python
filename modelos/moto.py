# modelos/moto.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


# import seaborn as sns # Seaborn pode melhorar a estética, mas adiciona dependência
# from matplotlib.pyplot import hlines # hlines está em plt

@dataclass(eq=True, frozen=False)  # eq=True é padrão, frozen=False é padrão. frozen=True faria hashável automaticamente
class Moto:
    """
    Representa uma motocicleta com seus atributos.
    A implementação de __eq__ e __hash__ é crucial para uso em conjuntos e tabelas hash.
    A implementação de __lt__ define uma ordem natural (usada por AVLTree, sorted).
    """
    marca: str
    nome: str
    preco: float
    revenda: float
    ano: int

    # _hash_cache: int = field(init=False, repr=False, default=None) # Para otimizar hash

    def __lt__(self, other: Any) -> bool:
        """Define a ordenação: primariamente por nome, secundariamente por preço."""
        if not isinstance(other, Moto):
            return NotImplemented
        if self.nome != other.nome:
            return self.nome < other.nome
        return self.preco < other.preco

    def __eq__(self, other: Any) -> bool:
        """Verifica igualdade baseada em todos os atributos."""
        if not isinstance(other, Moto):
            return False
        return (self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano) == \
            (other.marca.lower(), other.nome.lower(), other.preco, other.revenda, other.ano)

    def __hash__(self) -> int:
        """Gera um hash baseado em todos os atributos."""
        # if self._hash_cache is None: # Cache de hash
        #     self._hash_cache = hash((self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano))
        # return self._hash_cache
        return hash((self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano))


class MotoEstatisticas:
    """Classe utilitária para calcular estatísticas e gerar gráficos sobre uma lista de Motos."""

    @staticmethod
    def calcular_estatisticas(motos: List[Moto]) -> Dict[str, Dict[str, float]]:
        """
        Calcula diversas estatísticas descritivas de uma lista de motos.
        :param motos: Lista de objetos Moto.
        :return: Dicionário com as estatísticas.
        """
        if not motos:
            # Retorna um dicionário com valores padrão ou vazios se a lista de motos estiver vazia
            stats_template = {'media': 0.0, 'mediana': 0.0, 'desvio_padrao': 0.0, 'variancia': 0.0, 'moda': 0}
            return {
                'preco': stats_template.copy(),
                'revenda': stats_template.copy(),
                'ano': {key: stats_template[key] for key in ['moda', 'media', 'mediana']},
                'depreciacao': {'media': 0.0, 'mediana': 0.0},
                'taxa_depreciacao': {'media': 0.0, 'mediana': 0.0}
            }

        precos = [m.preco for m in motos]
        revendas = [m.revenda for m in motos]
        anos = [m.ano for m in motos]
        depreciacoes = [(m.preco - m.revenda) for m in motos]
        # Evitar divisão por zero para taxa de depreciação
        taxas_depreciacao = [((m.preco - m.revenda) / m.preco * 100) for m in motos if m.preco > 0]

        estatisticas_calculadas: Dict[str, Dict[str, float]] = {
            'preco': {
                'media': statistics.mean(precos) if precos else 0.0,
                'mediana': statistics.median(precos) if precos else 0.0,
                'desvio_padrao': statistics.stdev(precos) if len(precos) > 1 else 0.0,
                'variancia': statistics.variance(precos) if len(precos) > 1 else 0.0
            },
            'revenda': {
                'media': statistics.mean(revendas) if revendas else 0.0,
                'mediana': statistics.median(revendas) if revendas else 0.0,
                'desvio_padrao': statistics.stdev(revendas) if len(revendas) > 1 else 0.0,
                'variancia': statistics.variance(revendas) if len(revendas) > 1 else 0.0
            },
            'ano': {
                'moda': statistics.mode(anos) if anos else 0,
                # mode pode dar erro se não houver moda única ou lista vazia
                'media': statistics.mean(anos) if anos else 0.0,
                'mediana': statistics.median(anos) if anos else 0.0
            },
            'depreciacao': {
                'media': statistics.mean(depreciacoes) if depreciacoes else 0.0,
                'mediana': statistics.median(depreciacoes) if depreciacoes else 0.0
            },
            'taxa_depreciacao': {
                'media': statistics.mean(taxas_depreciacao) if taxas_depreciacao else 0.0,
                'mediana': statistics.median(taxas_depreciacao) if taxas_depreciacao else 0.0
            }
        }
        # Tratar caso de moda para anos
        try:
            estatisticas_calculadas['ano']['moda'] = statistics.mode(anos) if anos else 0
        except statistics.StatisticsError:  # Múltiplas modas ou lista vazia
            estatisticas_calculadas['ano']['moda'] = Counter(anos).most_common(1)[0][0] if anos else 0

        return estatisticas_calculadas

    @staticmethod
    def gerar_graficos(motos: List[Moto]) -> None:
        """
        Gera e exibe um conjunto de gráficos estatísticos sobre a lista de motos.
        :param motos: Lista de objetos Moto.
        """
        if not motos:
            print("Não há dados de motos para gerar gráficos.")
            return

        try:
            plt.style.use('seaborn-v0_8-whitegrid')  # Estilo mais moderno se seaborn estiver disponível
        except:
            plt.style.use('default')  # Fallback

        plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 90})

        # --- PRIMEIRA FIGURA (gráficos consolidados) ---
        fig1 = plt.figure(figsize=(18, 15))
        fig1.suptitle("Análise Estatística Geral de Motocicletas", fontsize=16, y=0.98)

        # Grid 3x3
        ax1 = fig1.add_subplot(3, 3, 1)  # Histograma Preços
        ax2 = fig1.add_subplot(3, 3, 2)  # Boxplot Preços/Revendas
        ax3 = fig1.add_subplot(3, 3, 3)  # Distribuição Anos
        ax4 = fig1.add_subplot(3, 1, 2)  # Dispersão Preço vs Revenda (ocupa uma linha)
        ax5 = fig1.add_subplot(3, 3, 7)  # Top Marcas
        ax6 = fig1.add_subplot(3, 3, (8, 9))  # Depreciação Média por Ano

        precos = np.array([m.preco for m in motos])
        revendas = np.array([m.revenda for m in motos])
        anos = np.array([m.ano for m in motos])
        marcas = [m.marca for m in motos]
        anos_unicos = sorted(list(set(anos)))

        # 1. Histograma de Preços
        ax1.hist(precos, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Distribuição de Preços', fontsize=12)
        ax1.set_xlabel('Preço (R$)', fontsize=10)
        ax1.set_ylabel('Frequência', fontsize=10)
        ax1.tick_params(axis='x', rotation=30)

        # 2. Boxplot Comparativo Preços e Revendas
        ax2.boxplot([precos, revendas], labels=['Preços', 'Revendas'], patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='black'),
                    medianprops=dict(color='red'))
        ax2.set_title('Comparativo Preços vs Revendas', fontsize=12)
        ax2.set_ylabel('Valores (R$)', fontsize=10)

        # 3. Distribuição por Ano
        contador_anos = Counter(anos)
        ax3.bar(contador_anos.keys(), contador_anos.values(), color='salmon', edgecolor='black', alpha=0.7)
        ax3.set_title('Distribuição por Ano de Fabricação', fontsize=12)
        ax3.set_xlabel('Ano', fontsize=10)
        ax3.set_ylabel('Quantidade', fontsize=10)
        ax3.set_xticks(sorted(list(contador_anos.keys()))[::max(1, len(contador_anos) // 5)])  # Ajustar ticks do eixo X

        # 4. Dispersão Preço vs Revenda (colorido por ano)
        scatter = ax4.scatter(precos, revendas, c=anos, cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
        ax4.set_title('Relação Preço Original vs Valor de Revenda', fontsize=14)
        ax4.set_xlabel('Preço Original (R$)', fontsize=12)
        ax4.set_ylabel('Valor de Revenda (R$)', fontsize=12)
        legend1 = ax4.legend(*scatter.legend_elements(num=min(5, len(anos_unicos))), title="Anos", loc="upper right",
                             bbox_to_anchor=(1.15, 1))
        ax4.add_artist(legend1)
        ax4.plot([min(precos), max(precos)], [min(precos), max(precos)], 'r--', alpha=0.5,
                 label="Preço = Revenda")  # Linha de referência
        ax4.legend(loc='lower right')

        # 5. Top Marcas
        top_marcas = Counter(marcas).most_common(10)
        ax5.barh([m[0] for m in top_marcas][::-1], [m[1] for m in top_marcas][::-1], color='cornflowerblue',
                 edgecolor='black', alpha=0.7)
        ax5.set_title('Top 10 Marcas Mais Frequentes', fontsize=12)
        ax5.set_xlabel('Quantidade', fontsize=10)

        # 6. Depreciação Média por Ano
        depreciacao_media_ano: Dict[int, float] = {}
        for ano_u in anos_unicos:
            taxas_ano = [((m.preco - m.revenda) / m.preco * 100)
                         for m in motos if m.ano == ano_u and m.preco > 0]
            if taxas_ano:
                depreciacao_media_ano[ano_u] = statistics.mean(taxas_ano)

        if depreciacao_media_ano:
            anos_plot = sorted(depreciacao_media_ano.keys())
            deprec_plot = [depreciacao_media_ano[a] for a in anos_plot]
            ax6.plot(anos_plot, deprec_plot, marker='o', linestyle='-', color='darkred', mfc='lightcoral')
            ax6.set_title('Taxa de Depreciação Média por Ano', fontsize=12)
            ax6.set_xlabel('Ano de Fabricação', fontsize=10)
            ax6.set_ylabel('Depreciação Média (%)', fontsize=10)
            ax6.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar para o supertítulo

        try:
            plt.show()
        except Exception as e_show:
            print(f"Não foi possível exibir os gráficos interativamente ({e_show}). Salvando em arquivo...")
            try:
                fig1.savefig('estatisticas_motos_consolidadas.png')
                print("Gráficos consolidados salvos como 'estatisticas_motos_consolidadas.png'")
            except Exception as e_save:
                print(f"Erro ao salvar gráficos consolidados: {e_save}")
        finally:
            plt.close(fig1)

        # --- SEGUNDA FIGURA (subplots por ano - opcional ou simplificado) ---
        # Esta parte pode gerar muitas figuras. Considere se é essencial ou simplificar.
        # Se mantida, adicione tratamento de erro similar.
        # Para este exemplo, vamos omitir a segunda figura detalhada para brevidade,
        # mas a lógica original pode ser mantida com os devidos cuidados.

    @staticmethod
    def prever_tendencias(motos: List[Moto], anos_futuros: int = 5) -> None:
        """
        Tenta prever tendências de preços e revendas para os próximos anos usando regressão linear.
        :param motos: Lista de objetos Moto.
        :param anos_futuros: Número de anos no futuro para prever.
        """
        if not motos:
            print("Não há dados para prever tendências.")
            return
        if anos_futuros <= 0:
            print("Número de anos futuros deve ser positivo.")
            return

        dados_por_ano: Dict[int, Dict[str, List[float]]] = {}
        for moto in motos:
            if moto.ano not in dados_por_ano:
                dados_por_ano[moto.ano] = {'precos': [], 'revendas': []}
            dados_por_ano[moto.ano]['precos'].append(moto.preco)
            dados_por_ano[moto.ano]['revendas'].append(moto.revenda)

        anos_hist = sorted(dados_por_ano.keys())
        if len(anos_hist) < 2:  # Precisa de pelo menos 2 pontos para regressão
            print("Dados insuficientes (menos de 2 anos distintos) para previsão de tendências.")
            return

        medias_precos = [statistics.mean(dados_por_ano[ano]['precos']) for ano in anos_hist]
        medias_revendas = [statistics.mean(dados_por_ano[ano]['revendas']) for ano in anos_hist]

        def _realizar_previsao(eixo_x: List[int], eixo_y: List[float], num_anos_futuros: int) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if len(eixo_x) < 2: return None, None, None

            coef = np.polyfit(eixo_x, eixo_y, 1)  # Regressão linear (grau 1)
            modelo = np.poly1d(coef)

            ultimo_ano_hist = max(eixo_x)
            anos_para_prever = np.array(list(range(min(eixo_x), ultimo_ano_hist + num_anos_futuros + 1)))
            previsao_valores = modelo(anos_para_prever)
            return anos_para_prever, previsao_valores, coef

        anos_prev_preco, val_prev_preco, coef_preco = _realizar_previsao(anos_hist, medias_precos, anos_futuros)
        anos_prev_revenda, val_prev_revenda, coef_revenda = _realizar_previsao(anos_hist, medias_revendas, anos_futuros)

        plt.figure(figsize=(14, 7))
        plt.suptitle("Previsão de Tendências de Preços e Revendas (Regressão Linear)", fontsize=16)

        if anos_prev_preco is not None and val_prev_preco is not None and coef_preco is not None:
            plt.subplot(1, 2, 1)
            plt.scatter(anos_hist, medias_precos, label='Média Histórica Preços', color='blue', alpha=0.7)
            plt.plot(anos_prev_preco, val_prev_preco, 'r--',
                     label=f'Tendência Preços\ny={coef_preco[0]:.2f}x + {coef_preco[1]:.0f}')
            plt.title('Preços Médios', fontsize=14)
            plt.xlabel('Ano', fontsize=12)
            plt.ylabel('Preço Médio (R$)', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.7)

        if anos_prev_revenda is not None and val_prev_revenda is not None and coef_revenda is not None:
            plt.subplot(1, 2, 2)
            plt.scatter(anos_hist, medias_revendas, label='Média Histórica Revendas', color='green', alpha=0.7)
            plt.plot(anos_prev_revenda, val_prev_revenda, 'm--',
                     label=f'Tendência Revendas\ny={coef_revenda[0]:.2f}x + {coef_revenda[1]:.0f}')
            plt.title('Valores Médios de Revenda', fontsize=14)
            plt.xlabel('Ano', fontsize=12)
            plt.ylabel('Revenda Média (R$)', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        try:
            plt.show()
        except Exception as e_show:
            print(f"Não foi possível exibir o gráfico de tendências ({e_show}). Salvando em arquivo...")
            try:
                plt.savefig('previsao_tendencias_motos.png')
                print("Gráfico de tendências salvo como 'previsao_tendencias_motos.png'")
            except Exception as e_save:
                print(f"Erro ao salvar gráfico de tendências: {e_save}")
        finally:
            plt.close()