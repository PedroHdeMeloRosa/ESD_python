import os
from typing import Any, List
from modelos.moto import Moto, MotoEstatisticas
from ui.helpers import validar_float, validar_int, obter_dados_moto


def menu_estrutura(estrutura: Any, nome_estrutura: str, dataset: List[Moto]) -> None:
    while True:
        print(f"\n=== MENU {nome_estrutura} ===")
        print("1. Inserir Moto")
        print("2. Remover Moto")
        print("3. Buscar Moto")
        print("4. Exibir Todas")
        print("5. Estatísticas e Gráficos")
        print("6. Simular Tendências")
        print("7. Filtrar e Ordenar")
        print("0. Voltar")
        escolha = input("Escolha: ").strip()

        if escolha == '1':
            nova_moto = obter_dados_moto()

            if nome_estrutura == "RADIX TREE":
                encontrado, _ = estrutura.buscar(nova_moto)
                if encontrado:
                    print("\nErro: Moto já existe!")
                else:
                    estrutura.inserir(nova_moto)
                    dataset.append(nova_moto)  # Adiciona ao dataset também
                    print("\nInserção bem sucedida!")
            else:
                estrutura.inserir(nova_moto)
                dataset.append(nova_moto)  # Adiciona ao dataset também
                print("\nOperação realizada!")

        elif escolha == '2':
            if nome_estrutura in ["BLOOM FILTER", "RADIX TREE"]:
                print("\nErro: Operação não suportada!")
            else:
                moto = obter_dados_moto(True)
                if estrutura.remover(moto):
                    # Tenta remover do dataset também
                    for i, m in enumerate(dataset):
                        if m == moto:
                            del dataset[i]
                            break
                    print("\nRemoção realizada! (Um item removido)")
                else:
                    print("\nNenhuma moto encontrada com esses critérios!")

        elif escolha == '3':
            moto = obter_dados_moto(True)

            if nome_estrutura == "BLOOM FILTER":
                encontrado = estrutura.buscar(moto)
                print(f"\nResultado da busca: {'Encontrado' if encontrado else 'Não encontrado'}")
            else:
                encontrado, passos = estrutura.buscar(moto)
                status = "Encontrado" if encontrado else "Não encontrado"
                print(f"\nResultado da busca: {status} em {passos} passos")

        elif escolha == '4':
            estrutura.exibir()

        elif escolha == '5':
            estatisticas = MotoEstatisticas.calcular_estatisticas(dataset)
            print("\n=== ESTATÍSTICAS DETALHADAS ===")
            print(f"Preços:")
            print(f"  Média: R${estatisticas['preco']['media']:.2f}")
            print(f"  Mediana: R${estatisticas['preco']['mediana']:.2f}")
            print(f"  Desvio Padrão: R${estatisticas['preco']['desvio_padrao']:.2f}")
            print(f"  Variância: R${estatisticas['preco']['variancia']:.2f}")

            print(f"\nRevendas:")
            print(f"  Média: R${estatisticas['revenda']['media']:.2f}")
            print(f"  Mediana: R${estatisticas['revenda']['mediana']:.2f}")
            print(f"  Desvio Padrão: R${estatisticas['revenda']['desvio_padrao']:.2f}")
            print(f"  Variância: R${estatisticas['revenda']['variancia']:.2f}")

            print(f"\nDepreciação:")
            print(f"  Média: R${estatisticas['depreciacao']['media']:.2f}")
            print(f"  Mediana: R${estatisticas['depreciacao']['mediana']:.2f}")
            print(f"  Taxa Média de Depreciação: {estatisticas['taxa_depreciacao']['media']:.2f}%")
            print(f"  Taxa Mediana de Depreciação: {estatisticas['taxa_depreciacao']['mediana']:.2f}%")

            print(f"\nAnos:")
            print(f"  Moda: {estatisticas['ano']['moda']}")
            print(f"  Média: {estatisticas['ano']['media']:.1f}")
            print(f"  Mediana: {estatisticas['ano']['mediana']}")

            MotoEstatisticas.gerar_graficos(dataset)

        elif escolha == '6':
            try:
                anos_futuros = int(input("Quantos anos no futuro para prever? "))
                if anos_futuros < 0:
                    print("Erro: O número de anos deve ser não negativo.")
                else:
                    MotoEstatisticas.prever_tendencias(dataset, anos_futuros)
            except ValueError:
                print("Erro: Entrada inválida. Digite um número inteiro.")

        elif escolha == '7':
            print("\n=== FILTRAR E ORDENAR ===")
            print("1. Filtrar por marca")
            print("2. Filtrar por faixa de preço")
            print("3. Ordenar por preço (crescente)")
            print("4. Ordenar por ano (decrescente)")
            print("5. Filtrar por taxa de depreciação máxima")
            opcao = input("Escolha: ").strip()

            filtradas = dataset[:]  # Inicia com todos os dados

            if opcao == '1':
                marca = input("Digite a marca: ").strip().lower()
                filtradas = [m for m in filtradas if m.marca.lower() == marca]

            elif opcao == '2':
                min_preco = validar_float("Preço mínimo: ")
                max_preco = validar_float("Preço máximo: ")
                if min_preco > max_preco:
                    min_preco, max_preco = max_preco, min_preco
                filtradas = [m for m in filtradas if min_preco <= m.preco <= max_preco]

            elif opcao == '3':
                filtradas = sorted(filtradas, key=lambda m: m.preco)

            elif opcao == '4':
                filtradas = sorted(filtradas, key=lambda m: m.ano, reverse=True)

            elif opcao == '5':
                max_taxa = validar_float("Taxa máxima de depreciação (%): ")
                filtradas = [
                    m for m in filtradas
                    if m.preco > 0 and ((m.preco - m.revenda) / m.preco * 100) <= max_taxa
                ]

            else:
                print("Opção inválida. Voltando ao menu.")
                continue

            # Exibir resultados
            print("\n" + "=" * 70)
            print(f"{'Marca':<15}{'Modelo':<20}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}{'Deprec.%':<8}")
            print("-" * 70)
            for m in filtradas[:50]:  # Limita a exibir 50 registros
                deprec = ((m.preco - m.revenda) / m.preco * 100) if m.preco > 0 else 0.0
                print(f"{m.marca:<15}{m.nome:<20}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}{deprec:<8.1f}")
            print(f"\nTotal de motos: {len(filtradas)} (exibindo até 50)")
            print("=" * 70)

        elif escolha == '0':
            break

        else:
            print("\nOpção inválida!")