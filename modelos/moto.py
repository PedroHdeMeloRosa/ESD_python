from dataclasses import dataclass
from typing import List, Dict
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from matplotlib.pyplot import hlines
from setuptools.command.rotate import rotate


@dataclass
class Moto:
    marca: str
    nome: str
    preco: float
    revenda: float
    ano: int

    def __lt__(self, other):
        if not isinstance(other, Moto):
            return NotImplemented
        return self.preco < other.preco

    def __eq__(self, other):
        if not isinstance(other, Moto):
            return False
        return (self.marca, self.nome, self.preco, self.revenda, self.ano) == \
            (other.marca, other.nome, other.preco, other.revenda, other.ano)

    def __hash__(self):
        return hash((self.marca, self.nome, self.preco, self.revenda, self.ano))


class MotoEstatisticas:
    @staticmethod
    def calcular_estatisticas(motos: List[Moto]) -> Dict[str, Dict[str, float]]:
        precos = [m.preco for m in motos]
        revendas = [m.revenda for m in motos]
        anos = [m.ano for m in motos]
        depreciacoes = [(m.preco - m.revenda) for m in motos]
        taxas_depreciacao = [((m.preco - m.revenda) / m.preco * 100) for m in motos if m.preco > 0]

        return {
            'preco': {
                'media': statistics.mean(precos),
                'mediana': statistics.median(precos),
                'desvio_padrao': statistics.stdev(precos) if len(precos) > 1 else 0.0,
                'variancia': statistics.variance(precos) if len(precos) > 1 else 0.0
            },
            'revenda': {
                'media': statistics.mean(revendas),
                'mediana': statistics.median(revendas),
                'desvio_padrao': statistics.stdev(revendas) if len(revendas) > 1 else 0.0,
                'variancia': statistics.variance(revendas) if len(revendas) > 1 else 0.0
            },
            'ano': {
                'moda': statistics.mode(anos),
                'media': statistics.mean(anos),
                'mediana': statistics.median(anos)
            },
            'depreciacao': {
                'media': statistics.mean(depreciacoes),
                'mediana': statistics.median(depreciacoes)
            },
            'taxa_depreciacao': {
                'media': statistics.mean(taxas_depreciacao) if taxas_depreciacao else 0.0,
                'mediana': statistics.median(taxas_depreciacao) if taxas_depreciacao else 0.0
            }
        }

    def gerar_graficos(motos: List[Moto]) -> None:
        """Gera gráficos sem dependência de layout engine avançado"""
        try:
            # Configuração básica segura
            plt.style.use('default')
            plt.rcParams.update({'figure.autolayout': True})

            # PRIMEIRA FIGURA (gráficos consolidados)
            fig1 = plt.figure(figsize=(18, 15))

            # Grid 3x3 com ajustes manuais de posição
            ax1 = plt.subplot2grid((3, 3), (0, 0))  # Histograma
            ax2 = plt.subplot2grid((3, 3), (0, 1))  # Boxplot
            ax3 = plt.subplot2grid((3, 3), (0, 2))  # Distribuição anos
            ax4 = plt.subplot2grid((3, 3), (1, 0), colspan=3)  # Dispersão (linha completa)
            ax5 = plt.subplot2grid((3, 3), (2, 0))  # Top marcas
            ax6 = plt.subplot2grid((3, 3), (2, 1), colspan=2)  # Depreciação

            # Dados para os gráficos
            precos = [m.preco for m in motos]
            revendas = [m.revenda for m in motos]
            anos = [m.ano for m in motos]
            marcas = [m.marca for m in motos]
            anos_unicos = sorted(set(anos))

            # 1. Histograma de Preços
            ax1.hist(precos, bins=20, color='#1f77b4', edgecolor='white')
            ax1.set_title('Distribuição de Preços', pad=10)
            ax1.set(xlabel='Preço (R$)', ylabel='Frequência')
            ax1.set_xticklabels(ax1.get_xticks(), rotation=45, ha='right')

            # 2. Boxplot Comparativo
            box = ax2.boxplot([precos, revendas],
                              labels=['Preços', 'Revendas'],
                              patch_artist=True)
            for patch in box['boxes']:
                patch.set_facecolor('#2ca02c')
            ax2.set_title('Comparação de Valores', pad=10)
            ax2.set_ylabel('Valores (R$)')

            # 3. Distribuição por Ano
            contador_anos = Counter(anos)
            ax3.bar(contador_anos.keys(), contador_anos.values(), color='#d62728')
            ax3.set_title('Distribuição por Ano', pad=10)
            ax3.set(xlabel='Ano', ylabel='Quantidade')

            # 4. Dispersão Preço vs Revenda
            for ano in anos_unicos:
                idx = [i for i, m in enumerate(motos) if m.ano == ano]
                ax4.scatter([precos[i] for i in idx],
                            [revendas[i] for i in idx],
                            label=str(ano), alpha=0.6)
            ax4.set_title('Relação Preço vs Revenda (Todos os Anos)', pad=10)
            ax4.set(xlabel='Preço (R$)', ylabel='Revenda (R$)')
            ax4.legend(title='Ano', bbox_to_anchor=(1.05, 1), loc='upper left')

            # 5. Top Marcas
            top_marcas = Counter(marcas).most_common(10)
            ax5.barh([m[0] for m in top_marcas],
                     [m[1] for m in top_marcas],
                     color='#9467bd')
            ax5.set_title('Top 10 Marcas', pad=10)
            ax5.set_xlabel('Quantidade')

            # 6. Depreciação por Ano
            depreciacao = {ano: [(m.preco - m.revenda) / m.preco * 100
                                 for m in motos if m.ano == ano and m.preco > 0]
                           for ano in anos_unicos}
            ax6.plot(sorted(depreciacao.keys()),
                     [np.mean(depreciacao[a]) for a in sorted(depreciacao.keys())],
                     'o-', color='#8c564b')
            ax6.set_title('Depreciação Média por Ano', pad=10)
            ax6.set(xlabel='Ano', ylabel='Depreciação (%)')
            ax6.set_yticks(np.arange(39, 42, 0.5))
            ax6.grid(axis="y", linestyle='--', color="blue", alpha=0.5)

            # Ajuste manual do layout
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                                hspace=0.4, wspace=0.3)

            # Exibir/salvar primeira figura
            try:
                plt.show()
            except:
                plt.savefig('estatisticas_motos_consolidadas.png')
                print("Gráficos consolidados salvos como backup")
            finally:
                plt.close(fig1)

            # SEGUNDA FIGURA (subplots por ano)
            n_anos = len(anos_unicos)
            n_cols = 3  # 3 gráficos por linha
            n_rows = (n_anos + n_cols - 1) // n_cols

            fig2, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig2.suptitle('Relação Preço-Revenda por Ano', y=1.02)

            for i, ano in enumerate(anos_unicos):
                if n_rows > 1:
                    ax = axs[i // n_cols, i % n_cols]
                else:
                    ax = axs[i] if n_cols > 1 else axs

                dados_ano = [(m.preco / 1000, m.revenda / 1000) for m in motos if m.ano == ano]  # Dividindo por 1000

                if dados_ano:
                    precos_ano, revendas_ano = zip(*dados_ano)
                    ax.scatter(precos_ano, revendas_ano, alpha=0.7, label=f'Ano {ano}')
                    ax.set_title(f'Ano {ano}')

                    # Configuração dos eixos com valores em milhares
                    ax.set_xlabel('Preço (mil R$)', labelpad=10)
                    ax.set_ylabel('Revenda (mil R$)', labelpad=10)

                    # Formatação dos rótulos
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

                    # Rotação e alinhamento
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                    ax.grid(True, alpha=0.3)
                    ax.legend()

            # Ocultar eixos vazios
            for j in range(i + 1, n_rows * n_cols):
                if n_rows > 1:
                    fig2.delaxes(axs.flatten()[j])
                else:
                    fig2.delaxes(axs[j])

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                                hspace=1, wspace=0.3)

            # Exibir/salvar segunda figura
            try:
                plt.show()
            except:
                plt.savefig('estatisticas_motos_por_ano.png')
                print("Gráficos por ano salvos como backup")
            finally:
                plt.close(fig2)

        except Exception as e:
            print(f"Erro durante geração de gráficos: {str(e)}")
            plt.close('all')

    @staticmethod
    def prever_tendencias(motos: List[Moto], anos_futuros: int = 5) -> None:
        # Agrupar por ano e calcular médias
        dados_por_ano = {}
        for moto in motos:
            if moto.ano not in dados_por_ano:
                dados_por_ano[moto.ano] = {'precos': [], 'revendas': []}
            dados_por_ano[moto.ano]['precos'].append(moto.preco)
            dados_por_ano[moto.ano]['revendas'].append(moto.revenda)

        # Calcular médias por ano
        anos = sorted(dados_por_ano.keys())
        medias_precos = [statistics.mean(dados_por_ano[ano]['precos']) for ano in anos]
        medias_revendas = [statistics.mean(dados_por_ano[ano]['revendas']) for ano in anos]

        # Prever usando regressão linear simples
        def prever(dados_x, dados_y, anos_futuro):
            if len(dados_x) < 2:
                return dados_x, dados_y, []

            coef = np.polyfit(dados_x, dados_y, 1)
            modelo = np.poly1d(coef)

            anos_futuro_range = list(range(min(anos), max(anos) + anos_futuro + 1))
            previsao = modelo(anos_futuro_range)

            return anos_futuro_range, previsao, coef

        # Previsões
        anos_range_preco, previsao_preco, coef_preco = prever(anos, medias_precos, anos_futuros)
        anos_range_revenda, previsao_revenda, coef_revenda = prever(anos, medias_revendas, anos_futuros)

        # Plotar resultados
        plt.figure(figsize=(12, 10))

        # Preços
        plt.subplot(1, 2, 1)
        plt.scatter(anos, medias_precos, label='Dados Históricos')
        plt.plot(anos_range_preco, previsao_preco, 'r--', label='Tendência')
        plt.title(f'Preços: y = {coef_preco[0]:.2f}x + {coef_preco[1]:.2f}')
        plt.xlabel('Ano')
        plt.ylabel('Preço Médio (R$)')
        plt.legend()
        plt.grid(True)

        # Revendas
        plt.subplot(1, 2, 2)
        plt.scatter(anos, medias_revendas, label='Dados Históricos')
        plt.plot(anos_range_revenda, previsao_revenda, 'r--', label='Tendência')
        plt.title(f'Revendas: y = {coef_revenda[0]:.2f}x + {coef_revenda[1]:.2f}')
        plt.xlabel('Ano')
        plt.ylabel('Revenda Média (R$)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()