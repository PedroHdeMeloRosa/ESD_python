import csv
import os
from modelos.moto import Moto
from typing import List

class DataHandler:
    @staticmethod
    def ler_dataset(caminho: str) -> List[Moto]:
        motos = []
        try:
            with open(caminho, 'r', encoding='utf-8') as arquivo:
                leitor = csv.DictReader(arquivo)
                for linha in leitor:
                    try:
                        # Adapte os nomes das colunas conforme seu CSV
                        moto = Moto(
                            marca=linha['brand'],
                            nome=linha['nome'],
                            preco=float(linha['preco']),
                            revenda=float(linha['revenda']),
                            ano=int(linha['ano'])
                        )
                        motos.append(moto)
                    except (ValueError, KeyError) as e:
                        print(f"Erro ao processar linha: {linha} - {e}")
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado - {caminho}")
            raise
        except Exception as e:
            print(f"Erro inesperado: {e}")
            raise
        return motos