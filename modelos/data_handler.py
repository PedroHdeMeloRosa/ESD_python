# modelos/data_handler.py
import csv
import os
from modelos.moto import Moto  # Certifique-se que a classe Moto está definida em modelos/moto.py
from typing import List


class DataHandler:
    """Classe para manipulação de dados, como leitura de datasets de motocicletas."""

    @staticmethod
    def ler_dataset(caminho: str) -> List[Moto]:
        """
        Lê um dataset de motos de um arquivo CSV.
        Espera-se que o CSV tenha as colunas: 'brand', 'nome', 'preco', 'revenda', 'ano'.

        :param caminho: Caminho para o arquivo CSV.
        :return: Lista de objetos Moto.
        """
        motos: List[Moto] = []

        try:
            with open(caminho, 'r', encoding='utf-8') as arquivo:
                leitor_csv = csv.DictReader(arquivo)

                # Validação do cabeçalho
                if not leitor_csv.fieldnames:
                    print(f"Erro: O arquivo CSV '{caminho}' parece estar vazio ou não tem cabeçalho.")
                    return motos

                colunas_esperadas = ['brand', 'nome', 'preco', 'revenda', 'ano']
                colunas_presentes_no_csv = [col.strip().lower() for col in
                                            leitor_csv.fieldnames]  # Normaliza para minúsculas e remove espaços

                # Ajusta os nomes das colunas esperadas para minúsculas para comparação case-insensitive
                colunas_esperadas_lower = [col.lower() for col in colunas_esperadas]

                colunas_faltantes = [
                    col_esperada for col_esperada in colunas_esperadas_lower
                    if col_esperada not in colunas_presentes_no_csv
                ]

                if colunas_faltantes:
                    print(f"Erro: Colunas obrigatórias faltando no CSV '{caminho}': {', '.join(colunas_faltantes)}.")
                    print(f"       Colunas esperadas (case-insensitive): {', '.join(colunas_esperadas_lower)}")
                    print(
                        f"       Colunas encontradas no arquivo (normalizadas): {', '.join(colunas_presentes_no_csv)}")
                    return motos

                # Mapeamento de nomes de coluna no CSV para os nomes esperados (caso haja variação de case)
                # Isso torna a leitura mais robusta a pequenas variações no cabeçalho.
                mapa_colunas = {col_csv_norm: col_csv_original for col_csv_original, col_csv_norm in
                                zip(leitor_csv.fieldnames, colunas_presentes_no_csv)}

                for i, linha in enumerate(leitor_csv):
                    try:
                        # Obtém os valores usando os nomes originais das colunas do CSV,
                        # mas acessando pelo nome normalizado (minúsculo) que esperamos.
                        marca_str = linha.get(mapa_colunas['brand'], "").strip()
                        nome_str = linha.get(mapa_colunas['nome'], "").strip()
                        preco_str = linha.get(mapa_colunas['preco'], "0").strip()
                        revenda_str = linha.get(mapa_colunas['revenda'], "0").strip()
                        ano_str = linha.get(mapa_colunas['ano'], "0").strip()

                        # Validação básica dos dados lidos
                        if not marca_str:
                            print(f"Aviso: Marca vazia na linha {i + 2}. Linha ignorada: {linha}")
                            continue
                        if not nome_str:
                            print(f"Aviso: Nome vazio na linha {i + 2}. Linha ignorada: {linha}")
                            continue

                        # Conversão para os tipos corretos
                        try:
                            preco_val = float(preco_str) if preco_str else 0.0
                            if preco_val < 0:
                                print(
                                    f"Aviso: Preço negativo ({preco_val}) na linha {i + 2}. Usando 0.0. Linha: {linha}")
                                preco_val = 0.0
                        except ValueError:
                            print(
                                f"Aviso: Valor de preço inválido ('{preco_str}') na linha {i + 2}. Usando 0.0. Linha: {linha}")
                            preco_val = 0.0

                        try:
                            revenda_val = float(revenda_str) if revenda_str else 0.0
                            if revenda_val < 0:
                                print(
                                    f"Aviso: Revenda negativa ({revenda_val}) na linha {i + 2}. Usando 0.0. Linha: {linha}")
                                revenda_val = 0.0
                        except ValueError:
                            print(
                                f"Aviso: Valor de revenda inválido ('{revenda_str}') na linha {i + 2}. Usando 0.0. Linha: {linha}")
                            revenda_val = 0.0

                        try:
                            ano_val = int(ano_str) if ano_str else 0
                            # Uma validação simples para o ano
                            if not (1900 <= ano_val <= 2050):  # Ajuste o range conforme necessário
                                print(
                                    f"Aviso: Ano inválido ({ano_val}) na linha {i + 2}. Usando 0 ou ignorando. Linha: {linha}")
                                # Decida se quer ignorar a linha ou usar um valor padrão para o ano.
                                # Por ora, vamos ignorar a linha se o ano for muito discrepante.
                                if ano_val == 0 and not ano_str:  # Se era vazio e virou 0
                                    print(
                                        f"Aviso: Ano vazio na linha {i + 2}, tratando como inválido. Linha ignorada. {linha}")
                                continue
                        except ValueError:
                            print(
                                f"Aviso: Valor de ano inválido ('{ano_str}') na linha {i + 2}. Linha ignorada. {linha}")
                            continue

                        moto = Moto(
                            marca=marca_str,
                            nome=nome_str,
                            preco=preco_val,
                            revenda=revenda_val,
                            ano=ano_val
                        )
                        motos.append(moto)

                    except KeyError as e:
                        # Este erro não deveria ocorrer se a validação do cabeçalho passou,
                        # mas é uma salvaguarda.
                        print(f"Aviso: Coluna esperada '{e}' não encontrada na linha {i + 2}. Linha ignorada: {linha}")
                    except Exception as e_linha:
                        # Captura qualquer outra exceção durante o processamento da linha
                        print(
                            f"Aviso: Erro inesperado ao processar linha {i + 2}: '{e_linha}'. Linha ignorada: {linha}")

        except FileNotFoundError:
            print(f"Erro Crítico: Arquivo de dataset não encontrado em '{os.path.abspath(caminho)}'")
            # Em um aplicativo real, você poderia levantar a exceção para ser tratada mais acima
            # raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
            return []  # Retorna lista vazia para o programa tentar continuar ou informar o usuário
        except Exception as e_geral:
            print(f"Erro Crítico: Erro inesperado ao tentar ler o arquivo '{caminho}': {e_geral}")
            # raise # Re-levanta a exceção se quiser que o programa pare
            return []

        if not motos:
            print(
                f"Aviso: Nenhum dado de moto foi carregado do arquivo '{caminho}' ou todas as linhas continham erros.")
        else:
            print(f"✅ {len(motos)} motos carregadas com sucesso de '{caminho}'.")
        return motos