# modelos/moto.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec


@dataclass(eq=True, frozen=False)
class Moto:
    # ... (definição da classe Moto como antes) ...
    marca: str;
    nome: str;
    preco: float;
    revenda: float;
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
        # ... (método calcular_estatisticas como na última correção, sem novas alterações) ...
        if not motos:
            stats_template_float = {'media': 0.0, 'mediana': 0.0, 'desvio_padrao': 0.0, 'variancia': 0.0}
            stats_template_int_moda = {'moda': [0], 'media': 0.0, 'mediana': 0.0}
            return {'preco': stats_template_float.copy(), 'revenda': stats_template_float.copy(),
                    'ano': stats_template_int_moda.copy(), 'depreciacao': {'media': 0.0, 'mediana': 0.0},
                    'taxa_depreciacao': {'media': 0.0, 'mediana': 0.0}}
        precos = [m.preco for m in motos];
        revendas = [m.revenda for m in motos];
        anos = [int(m.ano) for m in motos]
        valores_depreciacao_abs = [(m.preco - m.revenda) for m in motos]
        percentuais_taxa_depreciacao = [((m.preco - m.revenda) / m.preco * 100) for m in motos if m.preco > 0]
        estats: Dict[str, Dict[str, Any]] = {
            'preco': {'media': statistics.mean(precos) if precos else 0.0,
                      'mediana': statistics.median(precos) if precos else 0.0,
                      'desvio_padrao': statistics.stdev(precos) if len(precos) > 1 else 0.0,
                      'variancia': statistics.variance(precos) if len(precos) > 1 else 0.0},
            'revenda': {'media': statistics.mean(revendas) if revendas else 0.0,
                        'mediana': statistics.median(revendas) if revendas else 0.0,
                        'desvio_padrao': statistics.stdev(revendas) if len(revendas) > 1 else 0.0,
                        'variancia': statistics.variance(revendas) if len(revendas) > 1 else 0.0},
            'ano': {'media': statistics.mean(anos) if anos else 0.0,
                    'mediana': statistics.median(anos) if anos else 0.0},
            'depreciacao': {'media': statistics.mean(valores_depreciacao_abs) if valores_depreciacao_abs else 0.0,
                            'mediana': statistics.median(valores_depreciacao_abs) if valores_depreciacao_abs else 0.0},
            'taxa_depreciacao': {
                'media': statistics.mean(percentuais_taxa_depreciacao) if percentuais_taxa_depreciacao else 0.0,
                'mediana': statistics.median(percentuais_taxa_depreciacao) if percentuais_taxa_depreciacao else 0.0}}
        try:
            estats['ano']['moda'] = statistics.multimode(anos) if anos else [0]
        except statistics.StatisticsError:
            counter_anos = Counter(anos);
            estats['ano']['moda'] = [counter_anos.most_common(1)[0][0]] if counter_anos else [0]
        return estats

    @staticmethod
    def gerar_graficos(motos: List[Moto]) -> None:
        if not motos:
            print("Não há dados de motos para gerar gráficos.")
            return

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')

        plt.rcParams.update({'figure.dpi': 100, 'font.size': 10})

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle("Análise Estatística Geral e Depreciação de Motocicletas", fontsize=20, y=0.98)

        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1], wspace=0.25, hspace=0.45)  # Ajustado hspace

        ax1_hist_preco = fig.add_subplot(gs[0, 0])
        ax2_top_marcas = fig.add_subplot(gs[0, 1])
        ax3_preco_revenda = fig.add_subplot(gs[1, :])

        # --- MODIFICAÇÃO AQUI ---
        ax4_boxplot_preco_revenda = fig.add_subplot(gs[2, 0])  # Boxplot Preço vs Revenda
        ax5_deprec_taxa = fig.add_subplot(gs[2, 1])  # Taxa Média de Depreciação Anual (mantido)

        precos_np = np.array([m.preco for m in motos])
        revendas_np = np.array([m.revenda for m in motos])
        anos_np = np.array([m.ano for m in motos])
        marcas = [m.marca for m in motos]
        anos_unicos_int = sorted(list(set(int(a) for a in anos_np)))

        # 1. Histograma de Preços (ax1_hist_preco)
        ax1_hist_preco.hist(precos_np, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1_hist_preco.set_title('Distribuição de Preços', fontsize=15)
        ax1_hist_preco.set_xlabel('Preço (₹)', fontsize=12)
        ax1_hist_preco.set_ylabel('Frequência', fontsize=12)
        ax1_hist_preco.tick_params(axis='x', rotation=20, labelsize=10)
        ax1_hist_preco.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

        # 2. Top Marcas (ax2_top_marcas)
        top_marcas = Counter(marcas).most_common(10)
        ax2_top_marcas.barh([m[0] for m in top_marcas][::-1], [m[1] for m in top_marcas][::-1], color='cornflowerblue',
                            edgecolor='black', alpha=0.7)
        ax2_top_marcas.set_title('Top 10 Marcas por Quantidade', fontsize=15)
        ax2_top_marcas.set_xlabel('Quantidade de Motos', fontsize=12)
        ax2_top_marcas.tick_params(axis='y', labelsize=10)
        ax2_top_marcas.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

        # 3. Dispersão Preço vs Revenda (ax3_preco_revenda)
        scatter = ax3_preco_revenda.scatter(precos_np, revendas_np, c=anos_np, cmap='viridis', alpha=0.6,
                                            edgecolors='w', linewidth=0.5, s=50)
        ax3_preco_revenda.set_title('Relação Preço Original vs. Valor de Revenda', fontsize=16)
        ax3_preco_revenda.set_xlabel('Preço Original (₹)', fontsize=13)
        ax3_preco_revenda.set_ylabel('Valor de Revenda (₹)', fontsize=13)
        ax3_preco_revenda.tick_params(labelsize=11)
        min_val_disp = 0;
        max_val_disp = 100000
        if len(precos_np) > 0 and len(revendas_np) > 0:
            min_val_disp = min(precos_np.min(), revendas_np.min()) * 0.9
            max_val_disp = max(precos_np.max(), revendas_np.max()) * 1.05
        ax3_preco_revenda.plot([min_val_disp, max_val_disp], [min_val_disp, max_val_disp], 'r--', alpha=0.7,
                               label="Preço = Revenda")
        ax3_preco_revenda.legend(fontsize=11)
        if len(motos) > 0:
            cbar = fig.colorbar(scatter, ax=ax3_preco_revenda, orientation='vertical', fraction=0.03, pad=0.02)
            cbar.set_label('Ano de Fabricação', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
        ax3_preco_revenda.set_xlim(left=max(0, min_val_disp * 0.95));
        ax3_preco_revenda.set_ylim(
            bottom=max(0, min_val_disp * 0.95))  # Ajuste para não começar necessariamente em 0 se os dados forem altos
        ax3_preco_revenda.grid(True, linestyle=':', alpha=0.5)

        # 4. Boxplot Comparativo Preços de Venda e Valores de Revenda (ax4_boxplot_preco_revenda) --- MODIFICADO
        if len(precos_np) > 0 and len(revendas_np) > 0:
            box_data = [precos_np, revendas_np]
            box_labels = ['Preços Originais', 'Valores de Revenda']
            bp = ax4_boxplot_preco_revenda.boxplot(box_data, labels=box_labels, patch_artist=True,
                                                   medianprops={'color': 'red', 'linewidth': 1.5},
                                                   showfliers=True)  # Mostra outliers
            colors_boxplot = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors_boxplot):
                patch.set_facecolor(color)
            ax4_boxplot_preco_revenda.set_title('Distribuição: Preços vs. Revendas', fontsize=15)
            ax4_boxplot_preco_revenda.set_ylabel('Valor (₹)', fontsize=12)
            ax4_boxplot_preco_revenda.tick_params(axis='y', labelsize=10)
            ax4_boxplot_preco_revenda.tick_params(axis='x', labelsize=11)
            ax4_boxplot_preco_revenda.grid(True, axis='y', linestyle=':', alpha=0.7)
        else:
            ax4_boxplot_preco_revenda.text(0.5, 0.5, 'Dados insuficientes para Boxplot', ha='center', va='center',
                                           fontsize=12, color='gray')
            ax4_boxplot_preco_revenda.set_xticks([]);
            ax4_boxplot_preco_revenda.set_yticks([])

        # 5. Depreciação TAXA Média por Ano de Fabricação (ax5_deprec_taxa) - MANTIDO
        depreciacao_taxa_media_ano: Dict[int, float] = {}
        for ano_u in anos_unicos_int:
            taxas_dep_ano = [((m.preco - m.revenda) / m.preco * 100)
                             for m in motos if int(m.ano) == ano_u and m.preco > 0]
            if taxas_dep_ano:
                depreciacao_taxa_media_ano[ano_u] = statistics.mean(taxas_dep_ano)

        if depreciacao_taxa_media_ano:
            anos_plot_taxa = sorted(depreciacao_taxa_media_ano.keys())
            deprec_plot_taxa = [depreciacao_taxa_media_ano[a] for a in anos_plot_taxa]
            ax5_deprec_taxa.plot(anos_plot_taxa, deprec_plot_taxa, marker='x', linestyle='--', color='forestgreen',
                                 mec='darkgreen', markersize=7, linewidth=1.5)
            ax5_deprec_taxa.set_title('Taxa Média de Depreciação Anual', fontsize=15)
            ax5_deprec_taxa.set_xlabel('Ano de Fabricação', fontsize=12)
            ax5_deprec_taxa.set_ylabel('Depreciação Média (%)', fontsize=12)
            ax5_deprec_taxa.grid(True, linestyle=':', alpha=0.7)
            ax5_deprec_taxa.xaxis.set_major_locator(
                mticker.MaxNLocator(integer=True, nbins=min(len(anos_plot_taxa), 8)))
            ax5_deprec_taxa.tick_params(axis='x', rotation=25, labelsize=10)
            ax5_deprec_taxa.tick_params(axis='y', labelsize=10)
        else:
            ax5_deprec_taxa.text(0.5, 0.5, 'Dados insuficientes para Taxa de Depreciação', ha='center', va='center',
                                 fontsize=12, color='gray')
            ax5_deprec_taxa.set_xticks([]);
            ax5_deprec_taxa.set_yticks([])

        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.95)  # Ajuste final

        try:
            print("\nINFO: Gráfico de estatísticas gerais gerado. Feche a janela para continuar.")
            plt.show()
        except Exception as e_show:
            print(f"AVISO: Não foi possível exibir os gráficos interativamente ({e_show}).")
        finally:
            if 'fig' in locals() and fig is not None: plt.close(fig)

    @staticmethod
    def prever_tendencias(motos: List[Moto], anos_futuros: int = 5) -> None:
        # ... (código de prever_tendencias como na última versão estável) ...
        if not motos: print("Não há dados para prever tendências."); return
        if anos_futuros <= 0: print("Número de anos futuros deve ser positivo."); return
        dados_por_ano: Dict[int, Dict[str, List[float]]] = {}
        for moto in motos:
            ano_int = int(moto.ano);
            if ano_int not in dados_por_ano: dados_por_ano[ano_int] = {'precos': [], 'revendas': []}
            dados_por_ano[ano_int]['precos'].append(moto.preco);
            dados_por_ano[ano_int]['revendas'].append(moto.revenda)
        anos_hist = sorted(dados_por_ano.keys())
        if len(anos_hist) < 2: print("Dados insuficientes (<2 anos distintos) para previsão."); return
        medias_precos = [statistics.mean(dados_por_ano[ano]['precos']) for ano in anos_hist]
        medias_revendas = [statistics.mean(dados_por_ano[ano]['revendas']) for ano in anos_hist]

        def _realizar_previsao(x: List[int], y: List[float], n_fut: int) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if len(x) < 2: return None, None, None
            coef = np.polyfit(x, y, 1);
            modelo = np.poly1d(coef)
            ult_ano = max(x);
            anos_p = np.array(list(range(min(x), ult_ano + n_fut + 1)))
            prev_v = modelo(anos_p);
            return anos_p, prev_v, coef

        ap_p, vp_p, c_p = _realizar_previsao(anos_hist, medias_precos, anos_futuros)
        ap_r, vp_r, c_r = _realizar_previsao(anos_hist, medias_revendas, anos_futuros)
        fig, (axp, axr) = plt.subplots(1, 2, figsize=(15, 7));
        fig.suptitle("Previsão de Tendências (Regressão Linear)", fontsize=16, y=0.98)
        if ap_p is not None and c_p is not None:  # Adicionado check para c_p
            axp.scatter(anos_hist, medias_precos, label='Média Hist. Preços', color='blue', alpha=0.7)
            axp.plot(ap_p, vp_p, 'r--', label=f'Tendência Preços\ny={c_p[0]:.2f}x + {c_p[1]:.0f}')
            axp.set_title('Preços Médios', fontsize=14);
            axp.set_xlabel('Ano', fontsize=12);
            axp.set_ylabel('Preço (₹)', fontsize=12)
            axp.legend();
            axp.grid(True, ls=':', alpha=0.7);
            axp.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if ap_r is not None and c_r is not None:  # Adicionado check para c_r
            axr.scatter(anos_hist, medias_revendas, label='Média Hist. Revendas', color='green', alpha=0.7)
            axr.plot(ap_r, vp_r, 'm--', label=f'Tendência Revendas\ny={c_r[0]:.2f}x + {c_r[1]:.0f}')
            axr.set_title('Valores Revenda', fontsize=14);
            axr.set_xlabel('Ano', fontsize=12);
            axr.set_ylabel('Revenda (₹)', fontsize=12)
            axr.legend();
            axr.grid(True, ls=':', alpha=0.7);
            axr.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout(rect=[0, 0.03, 1, 0.94]);
        try:
            print("\nINFO: Gráfico de previsão gerado. Feche para continuar."); plt.show()
        except Exception as e:
            print(f"AVISO: Erro ao exibir gráfico de tendências: {e}")
        finally:
            if 'fig' in locals() and fig is not None: plt.close(fig)