# modelos/moto.py
from dataclasses import dataclass, field  # field não está sendo usado, pode remover se não for adicionar _hash_cache
from typing import List, Dict, Any, Tuple, Optional
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


@dataclass(eq=True, frozen=False)
class Moto:
    marca: str
    nome: str
    preco: float
    revenda: float
    ano: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Moto): return NotImplemented
        if self.nome != other.nome: return self.nome < other.nome
        return self.preco < other.preco

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Moto): return False
        return (self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano) == \
            (other.marca.lower(), other.nome.lower(), other.preco, other.revenda, other.ano)

    def __hash__(self) -> int:
        return hash((self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano))


class MotoEstatisticas:
    @staticmethod
    def calcular_estatisticas(motos: List[Moto]) -> Dict[str, Dict[str, Any]]:
        if not motos:
            stats_template: Dict[str, Any] = {'media': 0.0, 'mediana': 0.0, 'desvio_padrao': 0.0, 'variancia': 0.0,
                                              'moda': 0}
            return {'preco': stats_template.copy(), 'revenda': stats_template.copy(),
                    'ano': {key: stats_template[key] for key in ['moda', 'media', 'mediana']},
                    'depreciacao': {'media': 0.0, 'mediana': 0.0}, 'taxa_depreciacao': {'media': 0.0, 'mediana': 0.0}}

        precos = [m.preco for m in motos if isinstance(m.preco, (int, float)) and m.preco is not None]
        revendas = [m.revenda for m in motos if isinstance(m.revenda, (int, float)) and m.revenda is not None]
        anos = [m.ano for m in motos if isinstance(m.ano, int) and m.ano is not None]

        # depreciacoes e taxas_dep usam m.preco e m.revenda, então precisam do mesmo tipo de filtro
        depreciacoes = [(m.preco - m.revenda) for m in motos
                        if isinstance(m.preco, (int, float)) and m.preco is not None and \
                        isinstance(m.revenda, (int, float)) and m.revenda is not None]

        taxas_dep = [((m.preco - m.revenda) / m.preco * 100) for m in motos
                     if isinstance(m.preco, (int, float)) and m.preco is not None and m.preco > 0 and \
                     isinstance(m.revenda, (int, float)) and m.revenda is not None]

        estats: Dict[str, Dict[str, Any]] = {
            'preco': {'media': statistics.mean(precos) if precos else 0.0,
                      'mediana': statistics.median(precos) if precos else 0.0,
                      'desvio_padrao': statistics.stdev(precos) if len(precos) > 1 else 0.0,
                      'variancia': statistics.variance(precos) if len(precos) > 1 else 0.0},
            'revenda': {'media': statistics.mean(revendas) if revendas else 0.0,
                        'mediana': statistics.median(revendas) if revendas else 0.0,
                        'desvio_padrao': statistics.stdev(revendas) if len(revendas) > 1 else 0.0,
                        'variancia': statistics.variance(revendas) if len(revendas) > 1 else 0.0},
            'ano': {'moda': 0,
                    'media': statistics.mean(anos) if anos else 0.0,
                    'mediana': statistics.median(anos) if anos else 0.0},
            'depreciacao': {'media': statistics.mean(depreciacoes) if depreciacoes else 0.0,
                            'mediana': statistics.median(depreciacoes) if depreciacoes else 0.0},
            'taxa_depreciacao': {'media': statistics.mean(taxas_dep) if taxas_dep else 0.0,
                                 'mediana': statistics.median(taxas_dep) if taxas_dep else 0.0}
        }
        if anos:
            try:
                estats['ano']['moda'] = statistics.mode(anos)
            except statistics.StatisticsError:
                ca = Counter(anos)
                if ca:
                    max_freq = ca.most_common(1)[0][1]
                    modas = [a for a, f in ca.items() if f == max_freq]
                    estats['ano']['moda'] = modas if len(modas) > 1 else modas[0]
        return estats

    @staticmethod
    def gerar_graficos(motos: List[Moto]) -> None:
        if not motos:  # Correção da sintaxe aqui
            print("Não há dados de motos para gerar gráficos.")
            return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        plt.rcParams.update({'figure.autolayout': False, 'figure.dpi': 90, 'font.size': 9})

        fig1 = None
        try:
            fig1 = plt.figure(figsize=(16, 14))
            fig1.suptitle("Análise Estatística Geral de Motocicletas", fontsize=16, y=0.99)
            gs = fig1.add_gridspec(3, 2, hspace=0.45, wspace=0.3)  # Aumentado hspace

            ax1 = fig1.add_subplot(gs[0, 0])
            ax2 = fig1.add_subplot(gs[0, 1])
            ax3 = fig1.add_subplot(gs[1, 0])
            ax5 = fig1.add_subplot(gs[1, 1])
            ax6 = fig1.add_subplot(gs[2, 0])
            ax4 = fig1.add_subplot(gs[2, 1])

            # Preparar dados filtrando None e tipos incorretos
            precos_plot = np.array(
                [m.preco for m in motos if isinstance(m.preco, (int, float)) and m.preco is not None])
            revendas_plot = np.array(
                [m.revenda for m in motos if isinstance(m.revenda, (int, float)) and m.revenda is not None])
            anos_plot_raw = [m.ano for m in motos if isinstance(m.ano, int) and m.ano is not None]
            anos_plot = np.array(anos_plot_raw)

            marcas_plot_raw = [m.marca for m in motos if isinstance(m.marca, str) and m.marca]  # Filtra strings vazias
            anos_unicos = sorted(list(set(anos_plot_raw)))

            # 1. Histograma de Preços
            if len(precos_plot) > 0:
                ax1.hist(precos_plot, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax1.set_title('Distribuição de Preços', fontsize=11);
                ax1.set_xlabel('Preço (₹)', fontsize=9)
                ax1.set_ylabel('Frequência', fontsize=9);
                ax1.tick_params(axis='x', rotation=30, labelsize=8)
                ax1.tick_params(axis='y', labelsize=8)
            else:
                ax1.text(0.5, 0.5, "Sem dados de preço", ha='center', va='center', fontsize=10)

            # 2. Boxplot Preços/Revendas
            if len(precos_plot) > 0 or len(revendas_plot) > 0:  # Mostra mesmo que um esteja vazio
                data_for_boxplot = []
                labels_for_boxplot = []
                if len(precos_plot) > 0: data_for_boxplot.append(precos_plot); labels_for_boxplot.append('Preços Orig.')
                if len(revendas_plot) > 0: data_for_boxplot.append(revendas_plot); labels_for_boxplot.append(
                    'Val. Revenda')
                if data_for_boxplot:
                    ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True,
                                boxprops=dict(facecolor='lightgreen', color='black'), medianprops=dict(color='red'))
                ax2.set_title('Preços Originais vs. Valores de Revenda', fontsize=11);
                ax2.set_ylabel('Valores (₹)', fontsize=9)
                ax2.tick_params(axis='x', labelsize=9);
                ax2.tick_params(axis='y', labelsize=8)
            else:
                ax2.text(0.5, 0.5, "Sem dados de preço/revenda", ha='center', va='center', fontsize=10)

            # 3. Distribuição por Ano
            if len(anos_plot) > 0:
                ca = Counter(anos_plot);
                ax3.bar(ca.keys(), ca.values(), color='salmon', edgecolor='black', alpha=0.7)
                ax3.set_title('Distribuição por Ano de Fabricação', fontsize=11);
                ax3.set_xlabel('Ano', fontsize=9)
                ax3.set_ylabel('Quantidade', fontsize=9)
                ax3.set_xticks(sorted(list(ca.keys()))[::max(1, len(ca) // 6)]);
                ax3.tick_params(axis='x', labelsize=8)
                ax3.tick_params(axis='y', labelsize=8)
            else:
                ax3.text(0.5, 0.5, "Sem dados de ano", ha='center', va='center', fontsize=10)

            # 5. Top Marcas
            if marcas_plot_raw:
                tm = Counter(marcas_plot_raw).most_common(10)
                ax5.barh([m[0] for m in tm][::-1], [m[1] for m in tm][::-1], color='cornflowerblue', edgecolor='black',
                         alpha=0.7)
                ax5.set_title('Top 10 Marcas Mais Frequentes', fontsize=11);
                ax5.set_xlabel('Quantidade', fontsize=9)
                ax5.tick_params(axis='y', labelsize=8);
                ax5.tick_params(axis='x', labelsize=8)
            else:
                ax5.text(0.5, 0.5, "Sem dados de marca", ha='center', va='center', fontsize=10)

            # 6. Taxa de Depreciação Média por Ano
            depreciacao_media_ano_plot: Dict[int, float] = {}
            if anos_unicos:
                for ano_u in anos_unicos:
                    taxas_ano_u = []
                    for m in motos:
                        if m.ano == ano_u and isinstance(m.preco,
                                                         (int, float)) and m.preco is not None and m.preco > 0 and \
                                isinstance(m.revenda, (int, float)) and m.revenda is not None:
                            taxa = ((m.preco - m.revenda) / m.preco) * 100
                            if -50 <= taxa <= 100:  # Filtro de outlier
                                taxas_ano_u.append(taxa)
                    if taxas_ano_u: depreciacao_media_ano_plot[ano_u] = statistics.mean(taxas_ano_u)

            if depreciacao_media_ano_plot:
                anos_dep = sorted(depreciacao_media_ano_plot.keys())
                deprec_vals = [depreciacao_media_ano_plot[a] for a in anos_dep]
                ax6.plot(anos_dep, deprec_vals, marker='o', linestyle='-', color='darkred', mfc='lightcoral')
                ax6.set_title('Taxa de Depreciação Média por Ano', fontsize=11);
                ax6.set_xlabel('Ano Fabricação', fontsize=9)
                ax6.set_ylabel('Depreciação Média (%)', fontsize=9);
                ax6.grid(True, ls=':', alpha=0.7)
                ax6.tick_params(axis='both', labelsize=8)
            else:
                ax6.text(0.5, 0.5, "Sem dados válidos para\ntaxa de depreciação", ha='center', va='center', fontsize=10)

            # 4. Dispersão Preço vs. Revenda
            # Preparar dados alinhados para scatter plot
            scatter_data = [(m.preco, m.revenda, m.ano) for m in motos if
                            isinstance(m.preco, (int, float)) and m.preco is not None and
                            isinstance(m.revenda, (int, float)) and m.revenda is not None and
                            isinstance(m.ano, int) and m.ano is not None]
            if scatter_data:
                precos_s, revendas_s, anos_s = zip(*scatter_data)
                precos_s_np = np.array(precos_s)
                revendas_s_np = np.array(revendas_s)
                anos_s_np = np.array(anos_s)

                scatter = ax4.scatter(precos_s_np, revendas_s_np, c=anos_s_np, cmap='viridis', alpha=0.5,
                                      edgecolors='w', linewidth=0.3, s=30)
                ax4.set_title('Preço Original vs. Valor de Revenda', fontsize=11);
                ax4.set_xlabel('Preço Original (₹)', fontsize=9)
                ax4.set_ylabel('Valor de Revenda (₹)', fontsize=9)
                if len(precos_s_np) > 0:  # Só adiciona linha de referência se houver dados
                    ax4.plot([precos_s_np.min(), precos_s_np.max()], [precos_s_np.min(), precos_s_np.max()], 'r--',
                             alpha=0.5, label="Preço=Revenda")
                    ax4.legend(loc='lower right', fontsize=8)  # Legenda para a linha y=x

                anos_unicos_scatter = sorted(list(set(anos_s_np)))
                if len(anos_unicos_scatter) > 0:  # Tenta criar legenda de cores se houver variedade de anos
                    try:
                        legend_elements = scatter.legend_elements(num=min(5, len(anos_unicos_scatter)))
                        legend1 = ax4.legend(legend_elements[0], legend_elements[1], title="Anos", loc="upper left",
                                             bbox_to_anchor=(1.01, 1), fontsize=8)
                        ax4.add_artist(legend1)  # Adiciona a legenda de cores
                    except (AttributeError, ValueError, IndexError) as e_leg:
                        # print(f"Aviso: Não foi possível criar legenda de cores para anos no gráfico de dispersão: {e_leg}")
                        pass
                ax4.tick_params(axis='both', labelsize=8)
            else:
                ax4.text(0.5, 0.5, "Sem dados para dispersão", ha='center', va='center', fontsize=10)

            plt.tight_layout(
                rect=[0, 0.02, 1, 0.96])  # Ajuste para y=0.02 por causa dos xlabels, e x=0 por causa dos ylabels
            print("\nTentando exibir gráficos consolidados...")
            plt.show()  # Tenta exibir
        except Exception as e_show:
            print(
                f"AVISO: Não foi possível exibir os gráficos interativamente ({e_show}). Tentando salvar em arquivo...")
            try:
                if fig1: fig1.savefig('estatisticas_motos_consolidadas.png')
                print("INFO: Gráficos consolidados salvos como 'estatisticas_motos_consolidadas.png'")
            except Exception as e_save:
                print(f"ERRO: Não foi possível salvar os gráficos consolidados: {e_save}")
        finally:
            if fig1:
                plt.close(fig1)

    @staticmethod
    def prever_tendencias(motos: List[Moto], anos_futuros: int = 5) -> None:
        if not motos: print("Não há dados para prever tendências."); return
        if anos_futuros <= 0: print("Anos futuros deve ser positivo."); return

        dados_por_ano: Dict[int, Dict[str, List[float]]] = {}
        for moto in motos:
            if not (isinstance(moto.ano, int) and isinstance(moto.preco, (int, float)) and isinstance(moto.revenda,
                                                                                                      (int, float)) and \
                    moto.ano is not None and moto.preco is not None and moto.revenda is not None): continue  # Pula dados inválidos

            if moto.ano not in dados_por_ano: dados_por_ano[moto.ano] = {'precos': [], 'revendas': []}
            dados_por_ano[moto.ano]['precos'].append(moto.preco)
            dados_por_ano[moto.ano]['revendas'].append(moto.revenda)

        anos_hist_precos = sorted([ano for ano, data in dados_por_ano.items() if data['precos']])
        medias_precos = [statistics.mean(dados_por_ano[ano]['precos']) for ano in anos_hist_precos if
                         dados_por_ano[ano]['precos']]

        anos_hist_revendas = sorted([ano for ano, data in dados_por_ano.items() if data['revendas']])
        medias_revendas = [statistics.mean(dados_por_ano[ano]['revendas']) for ano in anos_hist_revendas if
                           dados_por_ano[ano]['revendas']]

        # Garante que as listas de anos e médias tenham o mesmo tamanho para polyfit
        anos_hist_precos = [ano for ano, data in dados_por_ano.items() if
                            data['precos'] and statistics.mean(data['precos']) in medias_precos]
        anos_hist_revendas = [ano for ano, data in dados_por_ano.items() if
                              data['revendas'] and statistics.mean(data['revendas']) in medias_revendas]

        if len(anos_hist_precos) < 2 and len(anos_hist_revendas) < 2:
            print("Dados insuficientes (< 2 anos distintos com dados válidos) para previsão.");
            return

        def _reg_lin(x_in: List[int], y_in: List[float], n_fut: int) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if len(x_in) < 2 or len(y_in) < 2 or len(x_in) != len(y_in): return None, None, None
            try:
                coef = np.polyfit(x_in, y_in, 1);
                modelo = np.poly1d(coef)
                if not x_in: return None, None, None  # Segurança
                last_x = max(x_in);
                min_x = min(x_in)
                anos_prev = np.array(list(range(min_x, last_x + n_fut + 1)))
                val_prev = modelo(anos_prev);
                return anos_prev, val_prev, coef
            except Exception as e_fit:
                print(f"Erro na regressão linear: {e_fit}"); return None, None, None

        fig_prev = None
        try:
            fig_prev = plt.figure(figsize=(14, 7));
            fig_prev.suptitle("Previsão de Tendências (Regressão Linear)", fontsize=16)
            plot_count = 0

            if len(anos_hist_precos) >= 2:
                a_p_preco, v_p_preco, c_preco = _reg_lin(anos_hist_precos, medias_precos, anos_futuros)
                if a_p_preco is not None:
                    plot_count += 1
                    ax_pr = fig_prev.add_subplot(1, 2, plot_count) if len(
                        anos_hist_revendas) >= 2 else fig_prev.add_subplot(1, 1, 1)
                    ax_pr.scatter(anos_hist_precos, medias_precos, label='Média Hist. Preços', c='blue', alpha=0.7)
                    ax_pr.plot(a_p_preco, v_p_preco, 'r--',
                               label=f'Tendência Preços\ny={c_preco[0]:.2f}x+{c_preco[1]:.0f}')
                    ax_pr.set_title('Preços Médios', fontsize=14);
                    ax_pr.set_xlabel('Ano', fontsize=12);
                    ax_pr.set_ylabel('Preço Médio (₹)', fontsize=12)
                    ax_pr.legend();
                    ax_pr.grid(True, ls=':', alpha=0.7)

            if len(anos_hist_revendas) >= 2:
                a_p_rev, v_p_rev, c_rev = _reg_lin(anos_hist_revendas, medias_revendas, anos_futuros)
                if a_p_rev is not None:
                    plot_count += 1
                    ax_rev = fig_prev.add_subplot(1, 2, plot_count) if len(
                        anos_hist_precos) >= 2 else fig_prev.add_subplot(1, 1, 1)
                    ax_rev.scatter(anos_hist_revendas, medias_revendas, label='Média Hist. Revendas', c='green',
                                   alpha=0.7)
                    ax_rev.plot(a_p_rev, v_p_rev, 'm--', label=f'Tendência Revendas\ny={c_rev[0]:.2f}x+{c_rev[1]:.0f}')
                    ax_rev.set_title('Valores Médios Revenda', fontsize=14);
                    ax_rev.set_xlabel('Ano', fontsize=12);
                    ax_rev.set_ylabel('Revenda Média (₹)', fontsize=12)
                    ax_rev.legend();
                    ax_rev.grid(True, ls=':', alpha=0.7)

            if plot_count == 0:
                print("Nenhum dado suficiente para plotar tendências de preço ou revenda.")
                if fig_prev: plt.close(fig_prev)
                return

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            print("\nExibindo gráfico de previsão de tendências... (Feche a janela para continuar)")
            plt.show()
        except Exception as e_show:
            print(f"Tendências: erro ao preparar ou mostrar gráficos ({e_show}).")
        finally:
            if fig_prev: plt.close(fig_prev)