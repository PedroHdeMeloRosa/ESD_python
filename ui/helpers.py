from modelos.moto import Moto

def validar_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Erro: Valor inválido! Tente novamente.")

def validar_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Erro: Valor inválido! Tente novamente.")

def obter_dados_moto(busca: bool = False) -> Moto:
    marca = input("Marca: ").strip()
    nome = input("Modelo: ").strip()

    if not busca:
        preco = validar_float("Preço: ")
        revenda = validar_float("Revenda: ")
        ano = validar_int("Ano: ")
    else:
        preco = validar_float("Preço (-1 para qualquer): ")
        revenda = validar_float("Revenda (-1 para qualquer): ")
        ano = validar_int("Ano (-1 para qualquer): ")

    # Criando a moto com todos os parâmetros obrigatórios
    moto = Moto(marca=marca, nome=nome, preco=preco, revenda=revenda, ano=ano)
    return moto