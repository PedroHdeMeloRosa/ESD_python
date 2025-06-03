# ui/menu.py
import os
from typing import Any, List, Optional  # Adicionado Optional
from modelos.moto import Moto, MotoEstatisticas
from ui.helpers import validar_float, validar_int, obter_dados_moto
import datetime  # Para helpers


# Variável global para armazenar o dataset, se necessário ser acessado por outras partes do UI
# No entanto, é melhor passar como parâmetro. Mas para simplificar o exemplo do menu,
# se menu_estrutura for chamado múltiplas vezes com o mesmo dataset, ele é passado.
# GLOBAL_DATASET_UI: List[Moto] = []


def menu_estrutura(estrutura: Any, nome_estrutura: str, dataset_principal: List[Moto]) -> None:
    """
    Exibe o menu de operações para uma estrutura de dados específica.

    :param estrutura: A instância da estrutura de dados (ex: LinkedList, AVLTree).
    :param nome_estrutura: Nome amigável da estrutura para exibição.
    :param dataset_principal: A lista principal de motos, usada para algumas operações
                              como estatísticas e como fonte para popular/remover da estrutura.
                              Alterações (inserção/remoção) na estrutura também devem refletir aqui.
    """
    # global GLOBAL_DATASET_UI
    # GLOBAL_DATASET_UI = dataset_principal # Se precisar acessar o dataset de outras funcs do UI

    while True:
        print(f"\n{'=' * 15} MENU: {nome_estrutura.upper()} {'=' * 15}")
        print(f"Total de motos na estrutura: {len(estrutura) if hasattr(estrutura, '__len__') else 'N/A'}")
        print("1. Inserir Moto")
        if nome_estrutura not in [
            "BLOOM FILTER"]:  # RadixTree pode ter remoção complexa, mas não explicitamente proibida
            print("2. Remover Moto")
        else:
            print("2. Remover Moto (Não suportado para Bloom Filter)")
        print("3. Buscar Moto")
        print("4. Exibir Todas as Motos na Estrutura")
        print("5. Estatísticas Detalhadas do Dataset Completo")
        print("6. Simular Tendências Futuras (Dataset Completo)")
        print("7. Filtrar e Ordenar (Dataset Completo)")
        print("0. Voltar ao Menu Principal")

        escolha = input("Escolha uma opção: ").strip()

        if escolha == '1':
            print("\n--- Inserir Nova Moto ---")
            nova_moto = obter_dados_moto(para_busca=False)

            # Lógica específica para RadixTree (baseada em nome como chave primária)
            # A RadixTree em si já lida com não adicionar Moto idênticas se a chave (nome) for a mesma.
            # A busca antes da inserção aqui seria mais para feedback ao usuário.
            if nome_estrutura == "RADIX TREE":
                # A RadixTree implementada armazena uma lista de motos por chave (nome).
                # A inserção na RadixTree já verifica se a *moto exata* está na lista do nó.
                # Não é preciso buscar antes aqui no menu, a menos que queira um feedback específico.
                pass  # A inserção abaixo cuidará disso.

            try:
                # Tenta inserir na estrutura
                if hasattr(estrutura, 'inserir'):
                    # Para estruturas que retornam algo na inserção (ex: AVL pode retornar flag)
                    # ou para medir desempenho individualmente (embora centralizado no main.py)
                    resultado_insercao = estrutura.inserir(nova_moto)

                    # Adiciona ao dataset_principal SE AINDA NÃO EXISTIR para evitar duplicatas na lista Python
                    # A estrutura interna pode ter sua própria lógica de duplicatas.
                    if nova_moto not in dataset_principal:
                        dataset_principal.append(nova_moto)
                        print(f"\n✅ Moto '{nova_moto.nome}' inserida com sucesso na estrutura e no dataset principal!")
                    elif nova_moto in dataset_principal and hasattr(estrutura, 'buscar') and \
                            estrutura.buscar(nova_moto)[0]:
                        print(f"\nℹ️ Moto '{nova_moto.nome}' já existe no dataset e na estrutura.")
                    else:  # Existe no dataset, mas talvez não na estrutura (ex: após limpar estrutura)
                        dataset_principal.append(nova_moto)  # Garante que está no dataset
                        print(f"\n✅ Moto '{nova_moto.nome}' inserida com sucesso na estrutura (já existia no dataset).")
                else:
                    print(
                        f"\n❌ Erro: A estrutura {nome_estrutura} não suporta inserção direta ou o método não foi encontrado.")

            except Exception as e:
                print(f"\n❌ Erro ao inserir na estrutura {nome_estrutura}: {e}")


        elif escolha == '2':
            if nome_estrutura == "BLOOM FILTER":
                print("\n❌ Operação de remoção não é suportada diretamente por Bloom Filters.")
                continue
            # RadixTree: remoção é complexa, assumindo que não está implementada a menos que `remover` exista.
            if not hasattr(estrutura, 'remover'):
                print(f"\n❌ Operação de remoção não implementada para {nome_estrutura}.")
                continue

            print("\n--- Remover Moto ---")
            moto_para_remover = obter_dados_moto(para_busca=True)  # Permite busca genérica

            try:
                removido_da_estrutura = estrutura.remover(moto_para_remover)
                if removido_da_estrutura:
                    print(f"\n✅ Moto '{moto_para_remover.nome}' removida da estrutura.")
                    # Tenta remover do dataset_principal também
                    # Cuidado: se houver múltiplas motos "iguais" no dataset, isso removerá a primeira.
                    try:
                        dataset_principal.remove(moto_para_remover)
                        print("✅ Moto também removida do dataset principal.")
                    except ValueError:
                        print("ℹ️ Moto não encontrada no dataset principal (ou já removida).")
                else:
                    print(
                        f"\n🟡 Moto '{moto_para_remover.nome}' não encontrada na estrutura {nome_estrutura} com os critérios fornecidos.")
            except Exception as e:
                print(f"\n❌ Erro ao remover da estrutura {nome_estrutura}: {e}")

        elif escolha == '3':
            if not hasattr(estrutura, 'buscar'):
                print(f"\n❌ Operação de busca não implementada para {nome_estrutura}.")
                continue

            print("\n--- Buscar Moto ---")
            moto_para_buscar = obter_dados_moto(para_busca=True)

            try:
                if nome_estrutura == "BLOOM FILTER":
                    encontrado_bf = estrutura.buscar(moto_para_buscar)
                    print(
                        f"\nResultado da busca no Bloom Filter: {'PROVAVELMENTE ENCONTRADO' if encontrado_bf else 'DEFINITIVAMENTE NÃO ENCONTRADO'}")
                    if encontrado_bf:
                        print("  (Lembre-se que Bloom Filters podem ter falsos positivos)")
                else:
                    encontrado, passos = estrutura.buscar(moto_para_buscar)
                    status = "✅ Encontrado" if encontrado else "🟡 Não encontrado"
                    print(f"\nResultado da busca: {status} em {passos} passos/comparações.")
            except Exception as e:
                print(f"\n❌ Erro ao buscar na estrutura {nome_estrutura}: {e}")


        elif escolha == '4':
            if hasattr(estrutura, 'exibir'):
                print(f"\n--- Exibindo Motos em {nome_estrutura} ---")
                estrutura.exibir()
            else:
                print(f"\n❌ Operação de exibição não implementada para {nome_estrutura}.")

        elif escolha == '5':
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio. Não há estatísticas para calcular.")
                continue
            print("\n--- Estatísticas Detalhadas do Dataset Completo ---")
            estatisticas = MotoEstatisticas.calcular_estatisticas(dataset_principal)
            print(f"\nPreços (Total: {len(dataset_principal)} motos):")
            print(f"  Média: R${estatisticas['preco']['media']:.2f}")
            print(f"  Mediana: R${estatisticas['preco']['mediana']:.2f}")
            print(f"  Desvio Padrão: R${estatisticas['preco']['desvio_padrao']:.2f}")
            # ... (restante da exibição de estatísticas como antes) ...
            print(f"\nRevendas:")
            print(f"  Média: R${estatisticas['revenda']['media']:.2f}")
            # ...
            print(f"\nAnos:")
            print(f"  Moda: {estatisticas['ano']['moda']}")  # Pode ser uma lista se houver múltiplas modas
            # ...
            print("\nGerando gráficos estatísticos do dataset completo...")
            MotoEstatisticas.gerar_graficos(dataset_principal)


        elif escolha == '6':
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio. Não é possível simular tendências.")
                continue
            try:
                anos_futuros = validar_int("Quantos anos no futuro para prever a tendência (ex: 5)? ", min_val=1,
                                           max_val=50)
                print("\n--- Simulando Tendências Futuras (Dataset Completo) ---")
                MotoEstatisticas.prever_tendencias(dataset_principal, anos_futuros)
            except ValueError:  # Já tratado em validar_int, mas como segurança
                print("Erro: Entrada inválida para anos futuros.")

        elif escolha == '7':
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio. Não é possível filtrar ou ordenar.")
                continue
            submenu_filtrar_ordenar(dataset_principal)


        elif escolha == '0':
            print(f"\nVoltando ao Menu Principal...")
            break

        else:
            print("\n❌ Opção inválida! Tente novamente.")

        input("\nPressione Enter para continuar...")  # Pausa para o usuário ler a saída


def submenu_filtrar_ordenar(dataset: List[Moto]):
    """Permite filtrar e ordenar o dataset principal (uma cópia dele)."""
    print("\n--- Filtrar e Ordenar Dataset Completo ---")
    print("1. Filtrar por marca")
    print("2. Filtrar por faixa de preço original")
    print("3. Ordenar por preço original (crescente)")
    print("4. Ordenar por ano de fabricação (mais novo primeiro)")
    print("5. Filtrar por taxa de depreciação máxima (ex: até 30%)")
    print("0. Voltar")
    opcao = input("Escolha uma opção de filtro/ordenação: ").strip()

    # Trabalhar com uma cópia para não alterar o dataset original do menu
    dados_para_exibir = list(dataset)  # Cópia superficial

    if opcao == '1':
        marca_filtro = input("Digite a marca para filtrar: ").strip().upper()
        dados_para_exibir = [m for m in dados_para_exibir if m.marca.upper() == marca_filtro]
    elif opcao == '2':
        min_preco = validar_float("Preço mínimo original: ", min_val=0)
        max_preco = validar_float("Preço máximo original: ", min_val=min_preco)
        dados_para_exibir = [m for m in dados_para_exibir if min_preco <= m.preco <= max_preco]
    elif opcao == '3':
        dados_para_exibir.sort(key=lambda m: m.preco)  # Ordena in-place a cópia
    elif opcao == '4':
        dados_para_exibir.sort(key=lambda m: m.ano, reverse=True)  # Ordena in-place
    elif opcao == '5':
        max_taxa_deprec = validar_float("Taxa máxima de depreciação permitida (ex: 30 para 30%): ", min_val=0,
                                        max_val=100)
        dados_para_exibir = [
            m for m in dados_para_exibir
            if m.preco > 0 and ((m.preco - m.revenda) / m.preco * 100) <= max_taxa_deprec
        ]
    elif opcao == '0':
        return
    else:
        print("Opção inválida.")
        return

    if not dados_para_exibir:
        print("\nNenhuma moto encontrada com os critérios especificados.")
    else:
        print("\n" + "=" * 80)
        print(f"{'Marca':<15}{'Modelo':<25}{'Preço (R$)':<12}{'Revenda (R$)':<15}{'Ano':<6}{'Deprec.%':<10}")
        print("-" * 80)
        for i, m in enumerate(dados_para_exibir):
            if i >= 50:  # Limitar exibição
                print(f"... e mais {len(dados_para_exibir) - 50} motos.")
                break
            deprec_percent = ((m.preco - m.revenda) / m.preco * 100) if m.preco > 0 else 0.0
            print(f"{m.marca:<15}{m.nome:<25}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}{deprec_percent:<10.1f}")
        print(f"\nTotal de motos filtradas/ordenadas: {len(dados_para_exibir)} (exibindo até 50)")
        print("=" * 80)