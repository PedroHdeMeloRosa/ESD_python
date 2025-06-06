# modelos/data_handler.py
import csv
import os
from modelos.moto import Moto
from typing import List
import datetime  # Para o ano máximo dinâmico


class DataHandler:
    """Classe para manipulação de dados, como leitura de datasets de motocicletas."""

    @staticmethod
    def ler_dataset(caminho: str) -> List[Moto]:
        """
        Lê um dataset de motos de um arquivo CSV.
        Espera-se que o CSV tenha as colunas: 'brand', 'nome', 'preco', 'revenda', 'ano'.
        Valida os dados e reporta o número de linhas ignoradas.

        :param caminho: Caminho para o arquivo CSV.
        :return: Lista de objetos Moto.
        """
        motos: List[Moto] = []
        linhas_ignoradas_count = 0  # NOVO: Contador
        linhas_lidas_total = 0  # NOVO: Contador

        ano_atual = datetime.date.today().year
        ano_maximo_permitido = ano_atual + 2  # Permite motos até 2 anos no futuro (ajustável)

        try:
            with open(caminho, 'r', encoding='utf-8') as arquivo:
                leitor_csv = csv.DictReader(arquivo)
                linhas_lidas_total = 0  # Reset para este arquivo

                if not leitor_csv.fieldnames:
                    print(f"Erro: O arquivo CSV '{caminho}' parece estar vazio ou não tem cabeçalho.")
                    return motos

                colunas_esperadas = ['brand', 'nome', 'preco', 'revenda', 'ano']
                colunas_presentes_no_csv = [col.strip().lower() for col in leitor_csv.fieldnames]
                colunas_esperadas_lower = [col.lower() for col in colunas_esperadas]
                colunas_faltantes = [c_esp for c_esp in colunas_esperadas_lower if
                                     c_esp not in colunas_presentes_no_csv]

                if colunas_faltantes:
                    print(f"Erro: Colunas faltando no CSV '{caminho}': {', '.join(colunas_faltantes)}.")
                    print(f"       Esperadas: {', '.join(colunas_esperadas_lower)}")
                    print(f"       Encontradas: {', '.join(colunas_presentes_no_csv)}")
                    return motos

                mapa_colunas = {norm: orig for orig, norm in zip(leitor_csv.fieldnames, colunas_presentes_no_csv)}

                for i, linha in enumerate(leitor_csv):
                    linhas_lidas_total += 1
                    num_linha_arquivo = i + 2  # +1 para 1-based index, +1 para pular cabeçalho

                    try:
                        marca_str = linha.get(mapa_colunas.get('brand', 'brand_fallback'), "").strip()
                        nome_str = linha.get(mapa_colunas.get('nome', 'nome_fallback'), "").strip()
                        preco_str = linha.get(mapa_colunas.get('preco', 'preco_fallback'), "0").strip()
                        revenda_str = linha.get(mapa_colunas.get('revenda', 'revenda_fallback'), "0").strip()
                        ano_str = linha.get(mapa_colunas.get('ano', 'ano_fallback'), "0").strip()

                        if not marca_str:
                            # print(f"Aviso [Linha {num_linha_arquivo}]: Marca vazia. Linha ignorada: {linha}") # Muito verboso
                            linhas_ignoradas_count += 1;
                            continue
                        if not nome_str:
                            # print(f"Aviso [Linha {num_linha_arquivo}]: Nome vazio. Linha ignorada: {linha}")
                            linhas_ignoradas_count += 1;
                            continue

                        try:
                            preco_val = float(preco_str) if preco_str else 0.0
                            if preco_val < 0: preco_val = 0.0
                        except ValueError:
                            # print(f"Aviso [Linha {num_linha_arquivo}]: Preço inválido ('{preco_str}'). Usando 0.0.")
                            preco_val = 0.0
                            # Se um preço inválido deve pular a linha:
                            # linhas_ignoradas_count += 1; continue

                        try:
                            revenda_val = float(revenda_str) if revenda_str else 0.0
                            if revenda_val < 0: revenda_val = 0.0
                        except ValueError:
                            # print(f"Aviso [Linha {num_linha_arquivo}]: Revenda inválida ('{revenda_str}'). Usando 0.0.")
                            revenda_val = 0.0
                            # Se uma revenda inválida deve pular a linha:
                            # linhas_ignoradas_count += 1; continue

                        try:
                            ano_val = int(ano_str) if ano_str else 0
                            if not (1900 <= ano_val <= ano_maximo_permitido):  # MODIFICADO: ano_maximo_permitido
                                # print(f"Aviso [Linha {num_linha_arquivo}]: Ano inválido ({ano_val}). Linha ignorada: {linha}")
                                linhas_ignoradas_count += 1;
                                continue
                        except ValueError:
                            # print(f"Aviso [Linha {num_linha_arquivo}]: Valor de ano inválido ('{ano_str}'). Linha ignorada.")
                            linhas_ignoradas_count += 1;
                            continue

                        # Se revenda for maior que preço após conversões (e preço > 0), ajustar revenda.
                        if preco_val > 0 and revenda_val > preco_val:
                            # print(f"Aviso [Linha {num_linha_arquivo}]: Revenda ({revenda_val}) > Preço ({preco_val}). Ajustando revenda para 90% do preço.")
                            revenda_val = preco_val * 0.9

                        motos.append(
                            Moto(marca=marca_str, nome=nome_str, preco=preco_val, revenda=revenda_val, ano=ano_val))

                    except KeyError as e:
                        # print(f"Aviso [Linha {num_linha_arquivo}]: Coluna '{e}' não encontrada. Linha ignorada: {linha}")
                        linhas_ignoradas_count += 1
                    except Exception as e_linha:
                        # print(f"Aviso [Linha {num_linha_arquivo}]: Erro inesperado: '{e_linha}'. Linha ignorada: {linha}")
                        linhas_ignoradas_count += 1

        except FileNotFoundError:
            print(f"Erro Crítico: Arquivo de dataset não encontrado em '{os.path.abspath(caminho)}'")
            return []
        except Exception as e_geral:
            print(f"Erro Crítico: Erro inesperado ao tentar ler o arquivo '{caminho}': {e_geral}")
            return []

        if linhas_ignoradas_count > 0:
            print(
                f"AVISO: {linhas_ignoradas_count} de {linhas_lidas_total} linhas foram ignoradas devido a dados faltantes ou inválidos.")

        if not motos and linhas_lidas_total > 0:  # Se leu linhas mas nenhuma moto foi criada
            print(
                f"AVISO: Nenhuma moto válida foi carregada do arquivo '{caminho}' após processar {linhas_lidas_total} linhas.")
        elif not motos:  # Se não leu nenhuma linha (ex: arquivo existe mas está vazio)
            print(f"AVISO: Nenhum dado de moto foi carregado do arquivo '{caminho}'. O arquivo pode estar vazio.")
        else:
            print(f"✅ {len(motos)} motos carregadas com sucesso de '{caminho}'.")

        return motos