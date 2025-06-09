# ui/helpers.py
import datetime
from typing import Optional

from modelos.moto import Moto  # Assegure que Moto está corretamente definida


def validar_entrada_generica(prompt: str, tipo_esperado: type, erro_msg: str, condicao_extra=None,
                             msg_condicao_extra=""):
    """
    Função genérica para validar entrada do usuário.
    Pede input repetidamente até que um valor válido do tipo esperado
    seja fornecido e opcionalmente atenda a uma condição extra.
    """
    while True:
        try:
            valor_str = input(prompt).strip()
            # Permite -1 para campos numéricos em busca se prompt indicar
            if tipo_esperado == float and valor_str == "-1" and "(-1 para qualquer)" in prompt:
                return -1.0
            if tipo_esperado == int and valor_str == "-1" and "(-1 para qualquer)" in prompt:
                return -1

            valor = tipo_esperado(valor_str)
            if condicao_extra and not condicao_extra(valor):
                print(msg_condicao_extra or "Erro: Valor inválido! Tente novamente.")
                continue
            return valor
        except ValueError:
            print(erro_msg)


def validar_float(prompt: str, permitir_negativo_um: bool = False, min_val: Optional[float] = None,
                  max_val: Optional[float] = None) -> float:
    """
    Valida entrada float do usuário.
    Pode permitir -1 (para qualquer valor em buscas) e aplicar limites mínimo/máximo.
    """
    while True:
        try:
            entrada_str = input(prompt).strip()
            if permitir_negativo_um and entrada_str == "-1":
                return -1.0

            valor = float(entrada_str)

            if min_val is not None and valor < min_val and not (permitir_negativo_um and valor == -1):
                print(f"Erro: O valor deve ser maior ou igual a {min_val}.")
                continue
            if max_val is not None and valor > max_val and not (permitir_negativo_um and valor == -1):
                print(f"Erro: O valor deve ser menor ou igual a {max_val}.")
                continue
            return valor
        except ValueError:
            print("Erro: Valor inválido! Deve ser um número (ex: 123.45). Tente novamente.")


def validar_int(prompt: str, permitir_negativo_um: bool = False, min_val: Optional[int] = None,
                max_val: Optional[int] = None) -> int:
    """
    Valida entrada int do usuário.
    Pode permitir -1 (para qualquer valor em buscas) e aplicar limites mínimo/máximo.
    """
    while True:
        try:
            entrada_str = input(prompt).strip()
            if permitir_negativo_um and entrada_str == "-1":
                return -1

            valor = int(entrada_str)

            if min_val is not None and valor < min_val and not (permitir_negativo_um and valor == -1):
                print(f"Erro: O valor deve ser maior ou igual a {min_val}.")
                continue
            if max_val is not None and valor > max_val and not (permitir_negativo_um and valor == -1):
                print(f"Erro: O valor deve ser menor ou igual a {max_val}.")
                continue
            return valor
        except ValueError:
            print("Erro: Valor inválido! Deve ser um número inteiro (ex: 123). Tente novamente.")


def obter_dados_moto(para_busca: bool = False) -> Moto:
    """
    Coleta os dados de uma moto do usuário via entrada de console.
    :param para_busca: Se True, a lógica permite que campos numéricos recebam -1.
    :return: Um objeto Moto com os dados inseridos.
    """
    print("\n--- Inserir Dados da Moto ---")
    marca = input("Marca: ").strip().upper()
    while not marca:
        print("Erro: A marca não pode ser vazia.")
        marca = input("Marca: ").strip().upper()

    nome = input("Modelo: ").strip()
    while not nome:
        print("Erro: O modelo não pode ser vazio.")
        nome = input("Modelo: ").strip()

    if not para_busca:
        # Usando min_val e max_val para validação mais robusta
        preco = validar_float("Preço (₹): ", min_val=0.0)
        revenda = validar_float("Valor de Revenda (₹): ", min_val=0.0)
        ano = validar_int("Ano de Fabricação: ", min_val=1900, max_val=datetime.date.today().year + 1)
    else:
        print("Digite os valores exatos para busca nos respectivos campos.")
        preco = validar_float("Preço (₹): ", permitir_negativo_um=True)
        revenda = validar_float("Valor de Revenda (₹): ", permitir_negativo_um=True)
        # Definindo um range seguro para anos na busca, mesmo com -1
        ano = validar_int("Ano de Fabricação: ", permitir_negativo_um=True)

    # Criando a moto com todos os parâmetros obrigatórios
    return Moto(marca=marca, nome=nome, preco=preco, revenda=revenda, ano=ano)