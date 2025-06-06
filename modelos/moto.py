# modelos/moto.py
from dataclasses import dataclass, field
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
        if not isinstance(other, Moto):
            return NotImplemented
        if self.nome != other.nome:
            return self.nome < other.nome
        return self.preco < other.preco

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Moto):
            return False
        return (self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano) == \
            (other.marca.lower(), other.nome.lower(), other.preco, other.revenda, other.ano)

    def __hash__(self) -> int:
        return hash((self.marca.lower(), self.nome.lower(), self.preco, self.revenda, self.ano))


class MotoEstatisticas:
    @staticmethod
    def calcular_estatisticas(motos: List[Moto]) -> Dict[str, Dict[str, Any]]:  # Modificado para Any para moda
        if not motos:
            stats_template: Dict[str, Any] = {'media': 0.0, 'mediana': 0.0, 'desvio_padrao': 0.0, 'variancia': 0.0,
                                              'moda': 0}  # Moda pode ser lista
            return {
                'preco': stats_template.copy(),
                'revenda': stats_template.copy(),
                'ano': {key: stats_template[key] for key in ['moda', 'media', 'mediana']},  # Moda para ano
                'depreciacao': {'media': 0.0, 'mediana': 0.0},
                'taxa_depreciacao': {'media': 0.0, 'mediana': 0.0}
            }

        precos = [m.preco for m in motos if isinstance(m.preco, (int, float))]  # Garante numérico
        revendas = [m.revenda for m in motos if isinstance(m.revenda, (int, float))]
        anos = [m.ano for m in motos if isinstance(m.ano, int)]  # Garante int

        # Filtra para evitar erros em estatísticas se listas estiverem vazias após filtragem
        precos_validos = [p for p in precos if p is not None]
        revendas_validas = [r for r in revendas if r is not None]
        anos_validos = [a for a in anos if a is not None]

        depreciacoes = [(m.preco - m.revenda) for m in motos if
                        isinstance(m.preco, (int, float)) and isinstance(m.revenda, (int, float))]
        taxas_depreciacao = [((m.preco - m.revenda) / m.preco * 100) for m in motos if
                             isinstance(m.preco, (int, float)) and m.preco > 0 and isinstance(m.revenda, (int, float))]

        estatisticas_calculadas: Dict[str, Dict[str, Any]] = {  # Any para moda
            'preco': {
                'media': statistics.mean(precos_validos) if precos_validos else 0.0,
                'mediana': statistics.median(precos_validos) if precos_validos else 0.0,
                'desvio_padrao': statistics.stdev(precos_validos) if len(precos_validos) > 1 else 0.0,
                'variancia': statistics.variance(precos_validos) if len(precos_validos) > 1 else 0.0
            },
            'revenda': {
                'media': statistics.mean(revendas_validas) if revendas_validas else 0.0,
                'mediana': statistics.median(revendas_validas) if revendas_validas else 0.0,
                'desvio_padrao': statistics.stdev(revendas_validas) if len(revendas_validas) > 1 else 0.0,
                'variancia': statistics.variance(revendas_validas) if len(revendas_validas) > 1 else 0.0
            },
            'ano': {
                'moda': 0,  # Será preenchido abaixo
                'media': statistics.mean(anos_validos) if anos_validos else 0.0,
                'mediana': statistics.median(anos_validos) if anos_validos else 0.0
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

        if anos_validos:
            try:
                estatisticas_calculadas['ano']['moda'] = statistics.mode(anos_validos)
            except statistics.StatisticsError:  # Múltiplas modas
                # Counter retorna lista de tuplas (valor, contagem), pegamos os valores com a maior contagem
                contagem_anos = Counter(anos_validos)
                if contagem_anos:  # Verifica se não está vazio
                    max_freq = contagem_anos.most_common(1)[0][1]
                    modas = [ano for ano, freq in contagem_anos.items() if freq == max_freq]
                    estatisticas_calculadas['ano']['moda'] = modas if len(modas) > 1 else modas[
                        0]  # Lista se múltiplas, senão valor único
                else:
                    estatisticas_calculadas['ano'][
                        'moda'] = 0  # Fallback se Counter vazio (improvável se anos_validos não é)
        else:
            estatisticas_calculadas['ano']['moda'] = 0

        return estatisticas_calculadas

    @staticmethod
    def gerar_graficos(motos: List[Moto]) -> None:
        # (Implementação de gerar_graficos como na última versão funcional completa que você me forneceu)
        # Vou colar aqui para garantir.
        if not motos:
            print("Não há dados de motos para gerar gráficos.")
            return
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 90})
        fig1 = plt.figure(figsize=(15, 12));
        fig1.suptitle("Análise Estatística Geral de Motocicletas", fontsize=16, y=0.98)
        ax1 = fig1.add_subplot(3, 3, 1);
        ax2 = fig1.add_subplot(3, 3, 2);
        ax3 = fig1.add_subplot(3, 3, 3)
        ax4 = fig1.add_subplot(3, 1, 2);
        ax5 = fig1.add_subplot(3, 3, 7);
        ax6 = fig1.add_subplot(3, 3, (8, 9))
        precos = np.array([m.preco for m in motos if isinstance(m.preco, (int, float))]);
        revendas = np.array([m.revenda for m in motos if isinstance(m.revenda, (int, float))])
        anos = np.array([m.ano for m in motos if isinstance(m.ano, int)]);
        marcas = [m.marca for m in motos];
        anos_unicos = sorted(list(set(a for a in anos if a is not None)))
        if len(precos) > 0: ax1.hist(precos, bins=20, color='skyblue', edgecolor='black', alpha=0.7); ax1.set_title(
            'Distribuição Preços', fontsize=12); ax1.set_xlabel('Preço (₹)', fontsize=10); ax1.set_ylabel('Frequência',
                                                                                                          fontsize=10); ax1.tick_params(
            axis='x', rotation=30)
        if len(precos) > 0 and len(revendas) > 0: ax2.boxplot([precos, revendas], labels=['Preços', 'Revendas'],
                                                              patch_artist=True,
                                                              boxprops=dict(facecolor='lightgreen', color='black'),
                                                              medianprops=dict(color='red')); ax2.set_title(
            'Preços vs Revendas', fontsize=12); ax2.set_ylabel('Valores (₹)', fontsize=10)
        if len(anos) > 0: ca = Counter(anos); ax3.bar(ca.keys(), ca.values(), color='salmon', edgecolor='black',
                                                      alpha=0.7); ax3.set_title('Distribuição Anos',
                                                                                fontsize=12); ax3.set_xlabel('Ano',
                                                                                                             fontsize=10); ax3.set_ylabel(
            'Qtd', fontsize=10); ax3.set_xticks(sorted(list(ca.keys()))[::max(1, len(ca) // 5)])
        if len(precos) > 0 and len(revendas) > 0 and len(anos) == len(precos): sc = ax4.scatter(precos, revendas,
                                                                                                c=anos, cmap='viridis',
                                                                                                alpha=0.6,
                                                                                                edgecolors='w',
                                                                                                lw=0.5);ax4.set_title(
            'Preço vs Revenda', fontsize=14); ax4.set_xlabel('Preço (₹)', fontsize=12); ax4.set_ylabel('Revenda (₹)',
                                                                                                       fontsize=12); leg1 = ax4.legend(
            *sc.legend_elements(num=min(5, len(anos_unicos))), title="Anos", loc="upper right",
            bbox_to_anchor=(1.15, 1));ax4.add_artist(leg1);ax4.plot([min(precos), max(precos)],
                                                                    [min(precos), max(precos)], 'r--', alpha=0.5,
                                                                    label="Preço=Revenda");ax4.legend(loc='lower right')
        if marcas: tm = Counter(marcas).most_common(10);ax5.barh([m[0] for m in tm][::-1], [m[1] for m in tm][::-1],
                                                                 color='cornflowerblue', edgecolor='black',
                                                                 alpha=0.7);ax5.set_title('Top 10 Marcas',
                                                                                          fontsize=12);ax5.set_xlabel(
            'Qtd', fontsize=10)
        dep_media_ano: Dict[int, float] = {};
        if anos_unicos:
            for au in anos_unicos: taxas_a = [((m.preco - m.revenda) / m.preco * 100) for m in motos if
                                              m.ano == au and isinstance(m.preco,
                                                                         (int, float)) and m.preco > 0 and isinstance(
                                                  m.revenda, (int, float))];
            if taxas_a: dep_media_ano[au] = statistics.mean(taxas_a)
        if dep_media_ano: ap = sorted(dep_media_ano.keys());dp = [dep_media_ano[a] for a in ap];ax6.plot(ap, dp,
                                                                                                         marker='o',
                                                                                                         ls='-',
                                                                                                         color='darkred',
                                                                                                         mfc='lightcoral');ax6.set_title(
            'Depreciação Média/Ano', fontsize=12);ax6.set_xlabel('Ano Fab.', fontsize=10);ax6.set_ylabel(
            'Depreciação Média (%)', fontsize=10);ax6.grid(True, ls=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 1, 0.96]);
        try:
            plt.show()
        except Exception as e_show:
            print(f"Gráficos: erro ao mostrar ({e_show}). Salvando...");
        try:
            fig1.savefig('estatisticas_motos_consolidadas.png'); print(
                "Gráficos salvos em 'estatisticas_motos_consolidadas.png'")
        except Exception as e_save:
            print(f"Gráficos: erro ao salvar: {e_save}")
        finally:
            plt.close(fig1)

    @staticmethod
    def prever_tendencias(motos: List[Moto], anos_futuros: int = 5) -> None:
        # (Implementação de prever_tendencias como na última versão funcional completa que você me forneceu)
        if not motos: print("Não há dados para prever tendências."); return
        if anos_futuros <= 0: print("Anos futuros deve ser positivo."); return
        dados_por_ano: Dict[int, Dict[str, List[float]]] = {}
        for moto in motos:
            if not isinstance(moto.ano, int) or not isinstance(moto.preco, (int, float)) or not isinstance(moto.revenda,
                                                                                                           (int,
                                                                                                            float)): continue  # Pula dados inválidos
            if moto.ano not in dados_por_ano: dados_por_ano[moto.ano] = {'precos': [], 'revendas': []}
            dados_por_ano[moto.ano]['precos'].append(moto.preco);
            dados_por_ano[moto.ano]['revendas'].append(moto.revenda)
        anos_h = sorted(dados_por_ano.keys())
        if len(anos_h) < 2: print("Dados insuficientes (< 2 anos distintos) para previsão."); return
        med_precos = [statistics.mean(dados_por_ano[ano]['precos']) for ano in anos_h if dados_por_ano[ano]['precos']]
        med_revendas = [statistics.mean(dados_por_ano[ano]['revendas']) for ano in anos_h if
                        dados_por_ano[ano]['revendas']]
        # Garante que anos_h corresponda aos dados médios válidos
        anos_h_precos = [ano for ano in anos_h if dados_por_ano[ano]['precos']]
        anos_h_revendas = [ano for ano in anos_h if dados_por_ano[ano]['revendas']]

        def _reg_lin(x_in: List[int], y_in: List[float], n_fut: int) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            if len(x_in) < 2 or len(y_in) < 2 or len(x_in) != len(y_in): return None, None, None
            coef = np.polyfit(x_in, y_in, 1);
            modelo = np.poly1d(coef)
            last_x = max(x_in);
            anos_prev = np.array(list(range(min(x_in), last_x + n_fut + 1)))
            val_prev = modelo(anos_prev);
            return anos_prev, val_prev, coef

        fig_prev = None
        try:
            fig_prev = plt.figure(figsize=(14, 7));
            fig_prev.suptitle("Previsão de Tendências (Regressão Linear)", fontsize=16)
            a_p_preco, v_p_preco, c_preco = _reg_lin(anos_h_precos, med_precos, anos_futuros)
            if a_p_preco is not None:
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
            if a_p_rev is not None:
                ax_rev = fig_prev.add_subplot(1, 2, 2);
                ax_rev.scatter(anos_h_revendas, med_revendas, label='Média Hist. Revendas', color='green', alpha=0.7)
                ax_rev.plot(a_p_rev, v_p_rev, 'm--', label=f'Tendência Revendas\ny={c_rev[0]:.2f}x + {c_rev[1]:.0f}')
                ax_rev.set_title('Valores Médios Revenda', fontsize=14);
                ax_rev.set_xlabel('Ano', fontsize=12);
                ax_rev.set_ylabel('Revenda Média (₹)', fontsize=12)
                ax_rev.legend();
                ax_rev.grid(True, ls=':', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 1, 0.95]);
            plt.show()
        except Exception as e_show:
            print(f"Tendências: erro ao mostrar ({e_show}). Salvando...");
        try:
            if fig_prev: fig_prev.savefig('previsao_tendencias_motos.png'); print(
                "Tendências salvas em 'previsao_tendencias_motos.png'")
        except Exception as e_save:
            print(f"Tendências: erro ao salvar: {e_save}")
        finally:
            if fig_prev: plt.close(fig_prev)