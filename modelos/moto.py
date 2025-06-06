# modelos/moto.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


@dataclass(eq=True, frozen=False)
class Moto:
    marca: str;
    nome: str;
    preco: float;
    revenda: float;
    ano: int

    def __lt__(self, other: Any) -> bool:  # ... (como antes)
        if not isinstance(other, Moto): return NotImplemented
        if self.nome != other.nome: return self.nome < other.nome
        return self.preco < other.preco

    def __eq__(self, other: Any) -> bool:  # ... (como antes)
        if not isinstance(other, Moto): return False
        return (self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano) == \
            (other.marca.lower(), other.nome.lower(), other.preco, other.revenda, other.ano)

    def __hash__(self) -> int:  # ... (como antes)
        return hash((self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano))


class MotoEstatisticas:
    @staticmethod
    def calcular_estatisticas(motos: List[Moto]) -> Dict[str, Dict[str, Any]]:
        # ... (Implementação robusta como na sua última versão) ...
        if not motos:
            stats_template: Dict[str, Any] = {'media': 0.0, 'mediana': 0.0, 'desvio_padrao': 0.0, 'variancia': 0.0,
                                              'moda': 0}
            return {'preco': stats_template.copy(), 'revenda': stats_template.copy(),
                    'ano': {key: stats_template[key] for key in ['moda', 'media', 'mediana']},
                    'depreciacao': {'media': 0.0, 'mediana': 0.0}, 'taxa_depreciacao': {'media': 0.0, 'mediana': 0.0}}
        precos = [m.preco for m in motos if isinstance(m.preco, (int, float))]
        revendas = [m.revenda for m in motos if isinstance(m.revenda, (int, float))]
        anos = [m.ano for m in motos if isinstance(m.ano, int)]
        precos_v = [p for p in precos if p is not None];
        revendas_v = [r for r in revendas if r is not None];
        anos_v = [a for a in anos if a is not None]
        depreciacoes = [(m.preco - m.revenda) for m in motos if
                        isinstance(m.preco, (int, float)) and isinstance(m.revenda, (int, float))]
        taxas_dep = [((m.preco - m.revenda) / m.preco * 100) for m in motos if
                     isinstance(m.preco, (int, float)) and m.preco > 0 and isinstance(m.revenda, (int, float))]

        estats: Dict[str, Dict[str, Any]] = {
            'preco': {'media': statistics.mean(precos_v) if precos_v else 0.0,
                      'mediana': statistics.median(precos_v) if precos_v else 0.0,
                      'desvio_padrao': statistics.stdev(precos_v) if len(precos_v) > 1 else 0.0,
                      'variancia': statistics.variance(precos_v) if len(precos_v) > 1 else 0.0},
            'revenda': {'media': statistics.mean(revendas_v) if revendas_v else 0.0,
                        'mediana': statistics.median(revendas_v) if revendas_v else 0.0,
                        'desvio_padrao': statistics.stdev(revendas_v) if len(revendas_v) > 1 else 0.0,
                        'variancia': statistics.variance(revendas_v) if len(revendas_v) > 1 else 0.0},
            'ano': {'moda': 0, 'media': statistics.mean(anos_v) if anos_v else 0.0,
                    'mediana': statistics.median(anos_v) if anos_v else 0.0},
            'depreciacao': {'media': statistics.mean(depreciacoes) if depreciacoes else 0.0,
                            'mediana': statistics.median(depreciacoes) if depreciacoes else 0.0},
            'taxa_depreciacao': {'media': statistics.mean(taxas_dep) if taxas_dep else 0.0,
                                 'mediana': statistics.median(taxas_dep) if taxas_dep else 0.0}
        }
        if anos_v:
            try:
                estats['ano']['moda'] = statistics.mode(anos_v)
            except statistics.StatisticsError:
                ca = Counter(anos_v)
                if ca:
                    max_freq = ca.most_common(1)[0][1]; modas = [a for a, f in ca.items() if f == max_freq];
                    estats['ano']['moda'] = modas if len(modas) > 1 else modas[0]
                else:
                    estats['ano']['moda'] = 0
        else:
            estats['ano']['moda'] = 0
        return estats

    @staticmethod
    def gerar_graficos(motos: List[Moto], fig_to_use: Optional[plt.Figure] = None) -> Optional[plt.Figure]:
        if not fig_to_use:
            print("Não há dados de motos para gerar gráficos.")
            return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        plt.rcParams.update({'figure.autolayout': False, 'figure.dpi': 90,
                             'font.size': 9})  # autolayout pode conflitar com tight_layout

        fig1 = None
        try:
            fig1 = plt.figure(figsize=(16, 14))  # Ajustado figsize
            fig1.suptitle("Análise Estatística Geral de Motocicletas", fontsize=16, y=0.99)

            # Grid 3x2 para 6 gráficos principais
            gs = fig1.add_gridspec(3, 2, hspace=0.4, wspace=0.3)  # Usando GridSpec para melhor controle
            ax1 = fig1.add_subplot(gs[0, 0])  # Histograma Preços
            ax2 = fig1.add_subplot(gs[0, 1])  # Boxplot Preços/Revendas
            ax3 = fig1.add_subplot(gs[1, 0])  # Distribuição Anos
            ax5 = fig1.add_subplot(gs[1, 1])  # Top Marcas
            ax6 = fig1.add_subplot(gs[2, 0])  # Depreciação Média por Ano
            ax4 = fig1.add_subplot(gs[2, 1])  # Dispersão Preço vs Revenda (coloquei aqui para preencher)

            precos = np.array([m.preco for m in motos if isinstance(m.preco, (int, float)) and m.preco is not None])
            revendas = np.array(
                [m.revenda for m in motos if isinstance(m.revenda, (int, float)) and m.revenda is not None])
            anos = np.array([m.ano for m in motos if isinstance(m.ano, int) and m.ano is not None])
            marcas = [m.marca for m in motos]  # Marcas podem ser strings vazias se não validadas antes
            anos_unicos = sorted(list(set(a for a in anos if a is not None)))

            if len(precos) > 0:
                ax1.hist(precos, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax1.set_title('Distribuição de Preços', fontsize=11);
                ax1.set_xlabel('Preço (₹)', fontsize=9);
                ax1.set_ylabel('Frequência', fontsize=9)
                ax1.tick_params(axis='x', rotation=30, labelsize=8);
                ax1.tick_params(axis='y', labelsize=8)
            else:
                ax1.text(0.5, 0.5, "Sem dados de preço", ha='center', va='center')

            if len(precos) > 0 and len(revendas) > 0:
                ax2.boxplot([precos, revendas], labels=['Preços Orig.', 'Val. Revenda'], patch_artist=True,
                            boxprops=dict(facecolor='lightgreen', color='black'), medianprops=dict(color='red'))
                ax2.set_title('Preços Originais vs. Valores de Revenda', fontsize=11);
                ax2.set_ylabel('Valores (₹)', fontsize=9)
                ax2.tick_params(axis='x', labelsize=9);
                ax2.tick_params(axis='y', labelsize=8)
            else:
                ax2.text(0.5, 0.5, "Sem dados de preço/revenda", ha='center', va='center')

            if len(anos) > 0:
                ca = Counter(anos);
                ax3.bar(ca.keys(), ca.values(), color='salmon', edgecolor='black', alpha=0.7)
                ax3.set_title('Distribuição por Ano de Fabricação', fontsize=11);
                ax3.set_xlabel('Ano', fontsize=9);
                ax3.set_ylabel('Quantidade', fontsize=9)
                ax3.set_xticks(sorted(list(ca.keys()))[::max(1, len(ca) // 6)]);
                ax3.tick_params(axis='x', labelsize=8);
                ax3.tick_params(axis='y', labelsize=8)
            else:
                ax3.text(0.5, 0.5, "Sem dados de ano", ha='center', va='center')

            if marcas:
                tm = Counter(m for m in marcas if m).most_common(10)  # Filtra marcas vazias
                if tm:
                    ax5.barh([m[0] for m in tm][::-1], [m[1] for m in tm][::-1], color='cornflowerblue',
                             edgecolor='black', alpha=0.7)
                    ax5.set_title('Top 10 Marcas Mais Frequentes', fontsize=11);
                    ax5.set_xlabel('Quantidade', fontsize=9)
                    ax5.tick_params(axis='y', labelsize=8);
                    ax5.tick_params(axis='x', labelsize=8)
                else:
                    ax5.text(0.5, 0.5, "Sem dados de marca", ha='center', va='center')
            else:
                ax5.text(0.5, 0.5, "Sem dados de marca", ha='center', va='center')

            # --- GRÁFICO DE TAXA DE DEPRECIAÇÃO - AJUSTES ---
            depreciacao_por_ano_valida: Dict[int, List[float]] = {}
            if anos_unicos:
                for ano_u in anos_unicos:
                    # Calcula taxas de depreciação apenas para motos com preço > 0
                    taxas_validas_para_este_ano = []
                    for m in motos:
                        if m.ano == ano_u and isinstance(m.preco, (int, float)) and m.preco > 0 and \
                                isinstance(m.revenda, (int, float)) and m.revenda is not None and m.preco is not None:
                            taxa = ((m.preco - m.revenda) / m.preco) * 100
                            # Considerar apenas taxas "realistas" para a média, e.g., entre -50% e 100%
                            if -50 <= taxa <= 100:  # Filtro de outlier para taxa de depreciação
                                taxas_validas_para_este_ano.append(taxa)

                    if taxas_validas_para_este_ano:
                        depreciacao_por_ano_valida[ano_u] = taxas_validas_para_este_ano

            if depreciacao_por_ano_valida:
                anos_plot_dep = sorted(depreciacao_por_ano_valida.keys())
                # Calcula média de taxas válidas para cada ano
                deprec_plot_avg = [statistics.mean(depreciacao_por_ano_valida[a]) for a in anos_plot_dep]

                ax6.plot(anos_plot_dep, deprec_plot_avg, marker='o', linestyle='-', color='darkred', mfc='lightcoral')
                ax6.set_title('Taxa de Depreciação Média por Ano', fontsize=11)
                ax6.set_xlabel('Ano de Fabricação', fontsize=9);
                ax6.set_ylabel('Depreciação Média (%)', fontsize=9)
                ax6.grid(True, linestyle=':', alpha=0.7);
                ax6.tick_params(axis='both', labelsize=8)

            else:
                ax6.text(0.5, 0.5, "Sem dados válidos para\ntaxa de depreciação", ha='center', va='center')

            if len(precos) > 0 and len(revendas) > 0 and len(anos) == len(precos) and len(anos_unicos) > 0:
                # Gráfico de Dispersão Preço vs. Revenda
                # Certifique-se de que 'anos' usado para colorir corresponda aos 'precos' e 'revendas'
                anos_para_scatter = np.array([m.ano for m in motos if
                                              isinstance(m.preco, (int, float)) and isinstance(m.revenda, (
                                              int, float)) and isinstance(m.ano, int) and m.ano is not None])
                if len(precos) == len(anos_para_scatter) and len(revendas) == len(anos_para_scatter):
                    scatter = ax4.scatter(precos, revendas, c=anos_para_scatter, cmap='viridis', alpha=0.5,
                                          edgecolors='w', linewidth=0.3, s=30)
                    ax4.set_title('Preço Original vs. Valor de Revenda', fontsize=11)
                    ax4.set_xlabel('Preço Original (₹)', fontsize=9);
                    ax4.set_ylabel('Valor de Revenda (₹)', fontsize=9)
                    try:  # Legenda de cores para os anos
                        legend1 = ax4.legend(*scatter.legend_elements(num=min(5, len(anos_unicos))),
                                             title="Anos", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
                        ax4.add_artist(legend1)
                    except Exception:
                        pass  # Ignora erro na legenda se não puder criar
                    ax4.plot([min(precos), max(precos)], [min(precos), max(precos)], 'r--', alpha=0.5,
                             label="Preço=Revenda")
                    ax4.legend(loc='lower right', fontsize=8)
                    ax4.tick_params(axis='both', labelsize=8)
                else:
                    ax4.text(0.5, 0.5, "Inconsistência de dados para dispersão", ha='center', va='center')
            else:
                ax4.text(0.5, 0.5, "Sem dados para dispersão", ha='center', va='center')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta para o supertítulo e títulos dos eixos

            print("\nTentando exibir gráficos consolidados...")
            return fig_to_use  # Tenta exibir
        except Exception as e_show:
            print(
                f"AVISO: Não foi possível exibir os gráficos interativamente ({e_show}). Tentando salvar em arquivo...")
            try:
                if fig1: fig1.savefig('estatisticas_motos_consolidadas.png')
                print("INFO: Gráficos consolidados salvos como 'estatisticas_motos_consolidadas.png'")
            except Exception as e_save:
                print(f"ERRO: Não foi possível salvar os gráficos consolidados: {e_save}")
        finally:
            if fig1:  # Garante que a figura é fechada se foi criada
                plt.close(fig1)

        # A segunda figura com subplots por ano foi omitida para simplificar, mas a lógica original
        # poderia ser adicionada aqui com tratamento de erro similar, se desejado.

    @staticmethod
    def prever_tendencias(motos: List[Moto], anos_futuros: int = 5) -> None:
        # (Implementação de prever_tendencias como na sua última versão funcional completa,
        # com tratamento para listas vazias antes da regressão e try-except para plotagem)
        if not motos: print("Não há dados para prever tendências."); return
        if anos_futuros <= 0: print("Anos futuros deve ser positivo."); return
        dados_por_ano: Dict[int, Dict[str, List[float]]] = {}
        for moto in motos:
            if not isinstance(moto.ano, int) or not isinstance(moto.preco, (int, float)) or not isinstance(moto.revenda,
                                                                                                           (int,
                                                                                                            float)): continue
            if moto.ano not in dados_por_ano: dados_por_ano[moto.ano] = {'precos': [], 'revendas': []}
            dados_por_ano[moto.ano]['precos'].append(moto.preco);
            dados_por_ano[moto.ano]['revendas'].append(moto.revenda)

        anos_h_precos = sorted([ano for ano, data in dados_por_ano.items() if data['precos']])
        med_precos = [statistics.mean(dados_por_ano[ano]['precos']) for ano in anos_h_precos]
        anos_h_revendas = sorted([ano for ano, data in dados_por_ano.items() if data['revendas']])
        med_revendas = [statistics.mean(dados_por_ano[ano]['revendas']) for ano in anos_h_revendas]

        if len(anos_h_precos) < 2 and len(anos_h_revendas) < 2:
            print("Dados insuficientes (< 2 anos distintos com dados) para previsão.");
            return

        def _reg_lin(x_in: List[int], y_in: List[float], n_fut: int) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if len(x_in) < 2 or len(y_in) < 2 or len(x_in) != len(y_in): return None, None, None
            try:
                coef = np.polyfit(x_in, y_in, 1);
                modelo = np.poly1d(coef)
                last_x = max(x_in);
                anos_prev = np.array(list(range(min(x_in), last_x + n_fut + 1)))
                val_prev = modelo(anos_prev);
                return anos_prev, val_prev, coef
            except Exception as e_fit:
                print(f"Erro na regressão linear: {e_fit}");
                return None, None, None

        fig_prev = None
        try:
            fig_prev = plt.figure(figsize=(14, 7));
            fig_prev.suptitle("Previsão de Tendências (Regressão Linear)", fontsize=16)
            a_p_preco, v_p_preco, c_preco = _reg_lin(anos_h_precos, med_precos, anos_futuros)
            if a_p_preco is not None and v_p_preco is not None and c_preco is not None:
                ax_pr = fig_prev.add_subplot(1, 2, 1);
                ax_pr.scatter(anos_h_precos, med_precos, label='Média Hist. Preços', color='blue', alpha=0.7)
                ax_pr.plot(a_p_preco, v_p_preco, 'r--',
                           label=f'Tendência Preços\ny={c_preco[0]:.2f}x + {c_preco[1]:.0f}')
                ax_pr.set_title('Preços Médios', fontsize=14);
                ax_pr.set_xlabel('Ano', fontsize=12);
                ax_pr.set_ylabel('Preço Médio (₹)', fontsize=12)
                ax_pr.legend();
                ax_pr.grid(True, ls=':', alpha=0.7)

            a_p_rev, v_p_rev, c_rev = _reg_lin(anos_h_revendas, med_revendas, anos_futuros)
            if a_p_rev is not None and v_p_rev is not None and c_rev is not None:
                ax_rev = fig_prev.add_subplot(1, 2, 2);
                ax_rev.scatter(anos_h_revendas, med_revendas, label='Média Hist. Revendas', color='green', alpha=0.7)
                ax_rev.plot(a_p_rev, v_p_rev, 'm--', label=f'Tendência Revendas\ny={c_rev[0]:.2f}x + {c_rev[1]:.0f}')
                ax_rev.set_title('Valores Médios Revenda', fontsize=14);
                ax_rev.set_xlabel('Ano', fontsize=12);
                ax_rev.set_ylabel('Revenda Média (₹)', fontsize=12)
                ax_rev.legend();
                ax_rev.grid(True, ls=':', alpha=0.7)

            if hasattr(fig_prev, 'axes') and not fig_prev.axes:  # Se nenhum subplot foi adicionado
                print("Nenhum dado válido para plotar tendências.")
                if fig_prev: plt.close(fig_prev)
                return

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            print("\nExibindo gráfico de previsão de tendências... (Feche a janela para continuar)")
            plt.show()
        except Exception as e_show:
            print(f"Tendências: erro ao mostrar ({e_show}).");
        finally:
            if fig_prev: plt.close(fig_prev)  # Fecha mesmo se não salvou, para limpar.