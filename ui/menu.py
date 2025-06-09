# ui/menu.py
import os
from typing import Any, List, Optional, Dict, Tuple

# As dependências externas a esta pasta são importadas.
from modelos.moto import Moto, MotoEstatisticas
from ui.helpers import validar_float, validar_int, obter_dados_moto


def exibir_estatisticas_texto(dataset: List[Moto]):
    """Função auxiliar para exibir estatísticas textuais e evitar repetição de código."""
    if not dataset:
        print("\n🟡 Dataset vazio. Não há estatísticas para calcular.")
        return

    print("\n--- Estatísticas Detalhadas do Dataset ---")
    estatisticas = MotoEstatisticas.calcular_estatisticas(dataset)
    print(f"Total de Motos Analisadas: {len(dataset)}")

    print("\nPreços (₹):")
    print(f"  Média: {estatisticas['preco']['media']:.2f}")
    print(f"  Mediana: {estatisticas['preco']['mediana']:.2f}")
    print(f"  Desvio Padrão: {estatisticas['preco']['desvio_padrao']:.2f}")

    print("\nValores de Revenda (₹):")
    print(f"  Média: {estatisticas['revenda']['media']:.2f}")
    print(f"  Mediana: {estatisticas['revenda']['mediana']:.2f}")

    print("\nAno de Fabricação:")
    moda_anos = estatisticas['ano']['moda']
    if isinstance(moda_anos, list):
        print(f"  Moda (ano mais comum): {', '.join(map(str, moda_anos))}")
    else:
        print(f"  Moda (ano mais comum): {moda_anos}")
    print(f"  Média (ano): {estatisticas['ano']['media']:.1f}")

    print("\nTaxa de Depreciação (%):")
    print(f"  Média: {estatisticas['taxa_depreciacao']['media']:.2f}%")
    print(f"  Mediana: {estatisticas['taxa_depreciacao']['mediana']:.2f}%")


def menu_estrutura(estrutura: Any, nome_estrutura: str, dataset_principal: List[Moto]) -> None:
    """
    Exibe o menu de operações para uma estrutura de dados específica,
    permitindo interagir com ela individualmente.
    """
    while True:
        print(f"\n{'=' * 15} MENU: {nome_estrutura.upper()} {'=' * 15}")
        struct_len = -1
        if hasattr(estrutura, '__len__'):
            try:
                struct_len = len(estrutura)
            except TypeError:
                pass

        print(f"Total de motos na estrutura: {struct_len if struct_len != -1 else 'N/A'}")
        print(f"Total de motos no dataset principal (Original): {len(dataset_principal)}")

        print("1. Inserir Moto")
        print("2. Remover Moto" + (" (Não suportado)" if nome_estrutura == "BLOOM FILTER" else ""))
        print("3. Buscar Moto")
        print("4. Exibir Todas as Motos na Estrutura")
        print("5. Estatísticas e Gráficos do Dataset Completo")
        print("6. Simular Tendências Futuras (Dataset Completo)")
        print("7. Filtrar e Ordenar (Dataset Completo)")

        if nome_estrutura == "TABELA HASH":
            print("8. Analisar Distribuição do Hash (Gráfico)")

        print("0. Voltar ao Menu Principal")

        escolha = input("Escolha uma opção: ").strip()

        if escolha == '1':
            print("\n--- Inserir Nova Moto ---")
            nova_moto = obter_dados_moto(para_busca=False)
            try:
                if hasattr(estrutura, 'inserir'):
                    inseriu_status = estrutura.inserir(nova_moto)
                    if inseriu_status is not False:
                        print(f"\n✅ Moto '{nova_moto.nome}' inserida na estrutura!")
                        if nova_moto not in dataset_principal:
                            dataset_principal.append(nova_moto)
                    else:
                        print(
                            f"\n🟡 Inserção falhou na estrutura {nome_estrutura} (capacidade máxima atingida ou item duplicado).")
                else:
                    print(f"\n❌ Erro: {nome_estrutura} não suporta inserção.")
            except Exception as e:
                print(f"\n❌ Erro ao inserir na estrutura {nome_estrutura}: {e}")

        elif escolha == '2':
            if nome_estrutura == "BLOOM FILTER" or not hasattr(estrutura, 'remover'):
                print("\n❌ Operação de remoção não suportada/implementada para esta estrutura.")
            else:
                print("\n--- Remover Moto ---")
                moto_para_remover = obter_dados_moto(para_busca=True)
                try:
                    removido = estrutura.remover(moto_para_remover)
                    if removido:
                        print(f"\n✅ Moto '{moto_para_remover.nome}' removida da estrutura.")
                        try:
                            dataset_principal.remove(moto_para_remover)
                        except ValueError:
                            pass
                    else:
                        print(f"\n🟡 Moto '{moto_para_remover.nome}' não encontrada/removida da {nome_estrutura}.")
                except Exception as e:
                    print(f"\n❌ Erro ao remover da estrutura {nome_estrutura}: {e}")

        elif escolha == '3':
            if not hasattr(estrutura, 'buscar'):
                print(f"\n❌ Operação de busca não implementada para {nome_estrutura}.")
            else:
                print("\n--- Buscar Moto ---")
                moto_para_buscar = obter_dados_moto(para_busca=True)
                try:
                    if nome_estrutura == "BLOOM FILTER":
                        encontrado_bf = estrutura.buscar(moto_para_buscar)
                        print(
                            f"\nResultado da busca: {'PROVAVELMENTE ENCONTRADO' if encontrado_bf else 'DEFINITIVAMENTE NÃO ENCONTRADO'}")
                        if encontrado_bf: print("  (Lembre-se que Bloom Filters podem ter falsos positivos)")
                    else:
                        encontrado, passos = estrutura.buscar(moto_para_buscar)
                        status = "✅ Encontrado" if encontrado else "🟡 Não encontrado"
                        if hasattr(estrutura,
                                   '_search_step_limit') and estrutura._search_step_limit is not None and passos > estrutura._search_step_limit and not encontrado:
                            status += f" (Limite de {estrutura._search_step_limit} passos atingido)"
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
            # Chama a função de texto e depois a de gráficos
            exibir_estatisticas_texto(dataset_principal)
            MotoEstatisticas.gerar_graficos(dataset_principal)

        elif escolha == '6':
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio.")
            else:
                anos_futuros = validar_int("Quantos anos no futuro para prever (ex: 5)? ", min_val=1, max_val=50)
                MotoEstatisticas.prever_tendencias(dataset_principal, anos_futuros)

        elif escolha == '7':
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio.")
            else:
                submenu_filtrar_ordenar(dataset_principal)

        elif nome_estrutura == "TABELA HASH" and escolha == '8':
            if hasattr(estrutura, 'analisar_distribuicao_hash'):
                estrutura.analisar_distribuicao_hash()
            else:
                print("\n❌ Erro: Método de análise de distribuição não encontrado na estrutura.")

        elif escolha == '0':
            print(f"\nVoltando ao Menu Principal...")
            break
        else:
            print("\n❌ Opção inválida! Tente novamente.")

        input("\nPressione Enter para continuar...")


def submenu_filtrar_ordenar(dataset: List[Moto]):
    """Permite filtrar e ordenar uma cópia do dataset principal."""
    print("\n--- Filtrar e Ordenar Dataset Completo ---")
    print("1. Filtrar por marca")
    print("2. Filtrar por faixa de preço original")
    print("3. Ordenar por preço original (crescente)")
    print("4. Ordenar por ano de fabricação (mais novo primeiro)")
    print("5. Filtrar por taxa de depreciação máxima (ex: até 30%)")
    print("0. Voltar")

    opcao = input("Escolha uma opção: ").strip()
    dados_para_exibir = list(dataset)

    if opcao == '1':
        marca_filtro = input("Digite a marca para filtrar: ").strip().upper()
        if marca_filtro: dados_para_exibir = [m for m in dados_para_exibir if m.marca.upper() == marca_filtro]
    elif opcao == '2':
        min_preco = validar_float("Preço mínimo: ", min_val=0)
        max_preco = validar_float("Preço máximo: ", min_val=min_preco)
        dados_para_exibir = [m for m in dados_para_exibir if min_preco <= m.preco <= max_preco]
    elif opcao == '3':
        dados_para_exibir.sort(key=lambda m: m.preco)
    elif opcao == '4':
        dados_para_exibir.sort(key=lambda m: m.ano, reverse=True)
    elif opcao == '5':
        max_taxa_deprec = validar_float("Taxa máxima de depreciação (%): ", min_val=0, max_val=100)
        dados_para_exibir = [m for m in dados_para_exibir if
                             m.preco > 0 and ((m.preco - m.revenda) / m.preco * 100) <= max_taxa_deprec]
    elif opcao == '0':
        return
    else:
        print("Opção inválida.")
        return

    if not dados_para_exibir:
        print("\nNenhuma moto encontrada com os critérios especificados.")
    else:
        print("\n" + "=" * 80)
        print(f"{'Marca':<15}{'Modelo':<25}{'Preço (₹)':<12}{'Revenda (₹)':<15}{'Ano':<6}{'Deprec.%':<10}")
        print("-" * 80)
        for i, m in enumerate(dados_para_exibir):
            if i >= 50:
                print(f"... e mais {len(dados_para_exibir) - 50} motos.")
                break
            deprec_percent = ((m.preco - m.revenda) / m.preco * 100) if m.preco > 0 else 0.0
            print(f"{m.marca:<15}{m.nome:<25}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}{deprec_percent:<10.1f}")
        print(f"\nTotal: {len(dados_para_exibir)} (exibindo até 50)")
        print("=" * 80)


def submenu_testes_restricao(analyzer: Any, configuracoes_restricoes: Dict[str, Dict[str, Any]]):
    """
    Exibe um submenu para escolher e executar testes com condições restritivas.
    """
    while True:
        print("\n" + "=" * 20 + " TESTES COM CONDIÇÕES RESTRITIVAS " + "=" * 20)
        if not configuracoes_restricoes:
            print("Nenhuma configuração de teste de restrição definida.")
            return

        categorias: Dict[str, List[Tuple[str, str]]] = {}
        for id_teste, config in configuracoes_restricoes.items():
            cat = config.get("categoria", "Outros")
            if cat not in categorias:
                categorias[cat] = []
            categorias[cat].append((id_teste, config.get("nome", id_teste)))

        idx_global = 1
        opcoes_menu_mapeamento: Dict[str, str] = {}
        for cat_nome, testes_na_categoria in sorted(categorias.items()):
            print(f"\n--- {cat_nome.upper()} ---")
            for id_teste, nome_teste in sorted(testes_na_categoria, key=lambda x: x[1]):
                print(f"{idx_global}. {nome_teste}")
                opcoes_menu_mapeamento[str(idx_global)] = id_teste
                idx_global += 1

        print("\n0. Voltar ao Menu Principal")
        escolha_teste_num = input("Escolha um teste de restrição para executar: ").strip()

        if escolha_teste_num == '0':
            break

        id_teste_escolhido = opcoes_menu_mapeamento.get(escolha_teste_num)
        if not id_teste_escolhido:
            print("\nOpção inválida!")
            continue

        config_escolhida = configuracoes_restricoes[id_teste_escolhido]
        print(f"\nIniciando teste: {config_escolhida['nome']}...")

        try:
            default_init_size_restr = 1000
            init_s_str_restr = input(f"Tamanho da amostra (Padrão {default_init_size_restr}): ").strip()
            init_sample_restr = int(init_s_str_restr) if init_s_str_restr else default_init_size_restr

            bench_ops_s_restr = input("Número de operações para benchmarks (padrão 100): ").strip()
            bench_ops_restr = int(bench_ops_s_restr) if bench_ops_s_restr else 100

            run_scal_restr_input = input("Rodar testes de escalabilidade também? (s/n, padrão n): ").strip().lower()
            run_scal_restr = run_scal_restr_input == 's'

            scal_sizes_restr: Optional[List[int]] = None
            if run_scal_restr:
                sizes_str_restr = input("Tamanhos N para escalar (ex: 100,500). VAZIO para padrão: ").strip()
                if sizes_str_restr:
                    scal_sizes_restr = [int(s.strip()) for s in sizes_str_restr.split(',') if s.strip().isdigit()]

            analyzer.run_suite_with_restriction(
                restriction_config=config_escolhida,
                init_sample_size=init_sample_restr,
                benchmark_ops_count=bench_ops_restr,
                run_scalability_flag=run_scal_restr,
                scalability_sizes=scal_sizes_restr
            )
        except ValueError:
            print("ERRO: Entrada inválida. Use números inteiros para amostras e operações.")
        except Exception as e:
            import traceback
            print(f"ERRO inesperado durante o teste: {e}")
            traceback.print_exc()