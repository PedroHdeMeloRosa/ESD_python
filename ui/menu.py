# ui/menu.py
import os
from typing import Any, List,Optional,Dict
from modelos.moto import Moto, MotoEstatisticas
from ui.helpers import validar_float, validar_int, obter_dados_moto


# import datetime # helpers.py usa datetime, não diretamente aqui

def menu_estrutura(estrutura: Any, nome_estrutura: str, dataset_principal: List[Moto]) -> None:
    """
    Exibe o menu de operações para uma estrutura de dados específica.
    (Docstring completo como na sua versão)
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
        print(f"Total de motos no dataset principal: {len(dataset_principal)}")

        print("1. Inserir Moto")
        # Verifica se a remoção é suportada/implementada
        suporta_remocao = hasattr(estrutura, 'remover')
        # Particularidades para Bloom Filter e BTree (com remoção placeholder)
        if nome_estrutura == "BLOOM FILTER":
            print("2. Remover Moto (Não suportado para Bloom Filter)")
        elif nome_estrutura == "ÁRVORE B" and suporta_remocao:
            print("2. Remover Moto (Aviso: Implementação placeholder na B-Tree)")
        elif suporta_remocao:
            print("2. Remover Moto")
        else:  # Se não tem 'remover' ou é uma das exceções já tratadas
            print(f"2. Remover Moto (Não implementado/suportado para {nome_estrutura})")

        print("3. Buscar Moto")
        print("4. Exibir Todas as Motos na Estrutura")
        print("5. Estatísticas Detalhadas do Dataset Completo")
        print("6. Simular Tendências Futuras (Dataset Completo)")
        print("7. Filtrar e Ordenar (Dataset Completo)")
        print("0. Voltar ao Menu Principal")

        escolha = input("Escolha uma opção: ").strip()

        if escolha == '1':  # Inserir
            print("\n--- Inserir Nova Moto ---")
            nova_moto = obter_dados_moto(para_busca=False)
            try:
                if hasattr(estrutura, 'inserir'):
                    estrutura.inserir(nova_moto)
                    if nova_moto not in dataset_principal:
                        dataset_principal.append(nova_moto)
                        print(f"\n✅ Moto '{nova_moto.nome}' inserida na estrutura e adicionada ao dataset principal!")
                    else:
                        print(
                            f"\n✅ Moto '{nova_moto.nome}' inserida na estrutura (já existia no dataset principal ou foi inserida com sucesso).")
                else:
                    print(f"\n❌ Erro: {nome_estrutura} não suporta inserção ou método não encontrado.")
            except Exception as e:
                print(f"\n❌ Erro ao inserir na estrutura {nome_estrutura}: {e}")

        elif escolha == '2':  # Remover
            if nome_estrutura == "BLOOM FILTER":
                print("\n❌ Operação de remoção não é suportada por Bloom Filters.")
            elif not hasattr(estrutura, 'remover'):  # Checagem genérica
                print(f"\n❌ Operação de remoção não implementada para {nome_estrutura}.")
            else:
                print("\n--- Remover Moto ---")
                moto_para_remover = obter_dados_moto(para_busca=True)
                try:
                    removido_da_estrutura = estrutura.remover(moto_para_remover)
                    if removido_da_estrutura:
                        print(f"\n✅ Moto '{moto_para_remover.nome}' removida da estrutura.")
                        try:
                            dataset_principal.remove(moto_para_remover)
                            print("✅ Moto também removida do dataset principal.")
                        except ValueError:
                            print("ℹ️ Moto não encontrada no dataset principal (ou já removida).")
                    elif nome_estrutura == "ÁRVORE B":
                        print(
                            f"\n🟡 Remoção na B-Tree é placeholder ou moto '{moto_para_remover.nome}' não foi efetivamente removida/encontrada.")
                    else:
                        print(f"\n🟡 Moto '{moto_para_remover.nome}' não encontrada/removida da {nome_estrutura}.")
                except Exception as e:
                    print(f"\n❌ Erro ao remover da estrutura {nome_estrutura}: {e}")

        elif escolha == '3':  # Buscar
            if not hasattr(estrutura, 'buscar'):
                print(f"\n❌ Operação de busca não implementada para {nome_estrutura}.")
            else:
                # ... (código da sua versão que funcionava) ...
                print("\n--- Buscar Moto ---")
                moto_para_buscar = obter_dados_moto(para_busca=True)
                try:
                    if nome_estrutura == "BLOOM FILTER":
                        encontrado_bf = estrutura.buscar(moto_para_buscar)
                        print(
                            f"\nResultado da busca no Bloom Filter: {'PROVAVELMENTE ENCONTRADO' if encontrado_bf else 'DEFINITIVAMENTE NÃO ENCONTRADO'}")
                        if encontrado_bf: print("  (Lembre-se que Bloom Filters podem ter falsos positivos)")
                    else:
                        encontrado, passos = estrutura.buscar(moto_para_buscar)
                        status = "✅ Encontrado" if encontrado else "🟡 Não encontrado"
                        print(f"\nResultado da busca: {status} em {passos} passos/comparações.")
                except Exception as e:
                    print(f"\n❌ Erro ao buscar na estrutura {nome_estrutura}: {e}")


        elif escolha == '4':  # Exibir
            if hasattr(estrutura, 'exibir'):
                # ... (código da sua versão que funcionava) ...
                print(f"\n--- Exibindo Motos em {nome_estrutura} ---")
                estrutura.exibir()
            else:
                print(f"\n❌ Operação de exibição não implementada para {nome_estrutura}.")

        elif escolha == '5':  # Estatísticas do Dataset Completo
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio. Não há estatísticas para calcular.")
            else:
                # ... (CÓDIGO COMPLETO DE EXIBIÇÃO DE ESTATÍSTICAS E CHAMADA DE GRÁFICOS - como corrigido anteriormente) ...
                print("\n--- Estatísticas Detalhadas do Dataset Completo ---")
                estatisticas = MotoEstatisticas.calcular_estatisticas(dataset_principal)
                print(f"\nPreços (Total: {len(dataset_principal)} motos):")
                print(f"  Média: R${estatisticas['preco']['media']:.2f}")
                print(f"  Mediana: R${estatisticas['preco']['mediana']:.2f}")
                print(f"  Desvio Padrão: R${estatisticas['preco']['desvio_padrao']:.2f}")
                print(f"  Variância: R${estatisticas['preco']['variancia']:.2f}")
                print(f"\nRevendas:")
                print(f"  Média: R${estatisticas['revenda']['media']:.2f}")
                print(f"  Mediana: R${estatisticas['revenda']['mediana']:.2f}")
                print(f"  Desvio Padrão: R${estatisticas['revenda']['desvio_padrao']:.2f}")
                print(f"  Variância: R${estatisticas['revenda']['variancia']:.2f}")
                print(f"\nAnos:")
                moda_anos = estatisticas['ano']['moda']
                if isinstance(moda_anos, list):
                    print(f"  Moda(s): {', '.join(map(str, moda_anos))}")
                else:
                    print(f"  Moda: {moda_anos}")
                print(f"  Média: {estatisticas['ano']['media']:.1f}")
                print(f"  Mediana: {estatisticas['ano']['mediana']}")
                print(f"\nDepreciação (Valor Absoluto):")
                print(f"  Média: R${estatisticas['depreciacao']['media']:.2f}")
                print(f"  Mediana: R${estatisticas['depreciacao']['mediana']:.2f}")
                print(f"\nTaxa de Depreciação (%):")
                print(f"  Média: {estatisticas['taxa_depreciacao']['media']:.2f}%")
                print(f"  Mediana: {estatisticas['taxa_depreciacao']['mediana']:.2f}%")
                print("\nGerando gráficos estatísticos do dataset completo...")
                MotoEstatisticas.gerar_graficos(dataset_principal)


        elif escolha == '6':  # Simular Tendências
            # ... (código da sua versão que funcionava) ...
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio.")
            else:
                try:
                    anos_futuros = validar_int("Quantos anos no futuro para prever (ex: 5)? ", min_val=1, max_val=50)
                    MotoEstatisticas.prever_tendencias(dataset_principal, anos_futuros)
                except ValueError:
                    print("Erro: Entrada inválida para anos futuros.")

        elif escolha == '7':  # Filtrar e Ordenar
            # ... (código da sua versão que funcionava, chamando submenu_filtrar_ordenar) ...
            if not dataset_principal:
                print("\n🟡 Dataset principal está vazio.")
            else:
                submenu_filtrar_ordenar(dataset_principal)

        elif escolha == '0':
            print(f"\nVoltando ao Menu Principal...")
            break
        else:
            print("\n❌ Opção inválida! Tente novamente.")

        input("\nPressione Enter para continuar...")


def submenu_filtrar_ordenar(dataset: List[Moto]):
    # ... (CÓDIGO COMPLETO DA SUA VERSÃO QUE FUNCIONAVA - SEM MUDANÇAS NECESSÁRIAS AQUI) ...
    print("\n--- Filtrar e Ordenar Dataset Completo ---")
    print("1. Filtrar por marca")
    print("2. Filtrar por faixa de preço original")
    print("3. Ordenar por preço original (crescente)")
    print("4. Ordenar por ano de fabricação (mais novo primeiro)")
    print("5. Filtrar por taxa de depreciação máxima (ex: até 30%)")
    print("0. Voltar")
    opcao = input("Escolha uma opção de filtro/ordenação: ").strip()
    dados_para_exibir = list(dataset)
    if opcao == '1':
        marca_filtro = input("Digite a marca para filtrar: ").strip().upper()
        if marca_filtro:
            dados_para_exibir = [m for m in dados_para_exibir if m.marca.upper() == marca_filtro]
        else:
            print("Marca vazia, nenhum filtro aplicado.")
    elif opcao == '2':
        min_preco = validar_float("Preço mínimo original: ", min_val=0)
        max_preco = validar_float("Preço máximo original: ", min_val=min_preco)
        dados_para_exibir = [m for m in dados_para_exibir if min_preco <= m.preco <= max_preco]
    elif opcao == '3':
        dados_para_exibir.sort(key=lambda m: m.preco)
    elif opcao == '4':
        dados_para_exibir.sort(key=lambda m: m.ano, reverse=True)
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
            if i >= 50:
                print(f"... e mais {len(dados_para_exibir) - 50} motos.")
                break
            deprec_percent = ((m.preco - m.revenda) / m.preco * 100) if m.preco > 0 else 0.0
            print(f"{m.marca:<15}{m.nome:<25}{m.preco:<12.2f}{m.revenda:<15.2f}{m.ano:<6}{deprec_percent:<10.1f}")
        print(f"\nTotal de motos filtradas/ordenadas: {len(dados_para_exibir)} (exibindo até 50)")
        print("=" * 80)


def submenu_testes_restricao(analyzer: Any, configuracoes_restricoes: Dict[str, Dict[str, Any]]):
    """
    Exibe um submenu para o usuário escolher e executar testes com condições restritivas.
    :param analyzer: A instância do StructureAnalyzer.
    :param configuracoes_restricoes: Dicionário com as configurações dos testes de restrição.
    """
    while True:
        print("\n" + "=" * 20 + " TESTES COM CONDIÇÕES RESTRITIVAS " + "=" * 20)
        if not configuracoes_restricoes:
            print("Nenhuma configuração de teste de restrição definida.")
            return

        categorias = {}  # Dicionário para agrupar testes por categoria
        for id_teste, config in configuracoes_restricoes.items():
            cat = config.get("categoria", "Outros")
            if cat not in categorias:
                categorias[cat] = []
            categorias[cat].append((id_teste, config.get("nome", id_teste)))

        idx_global = 1
        opcoes_menu = {}  # Mapeia número da opção para id_teste

        for cat_nome, testes_na_categoria in sorted(categorias.items()):
            print(f"\n--- {cat_nome.upper()} ---")
            for id_teste, nome_teste in sorted(testes_na_categoria, key=lambda x: x[1]):
                print(f"{idx_global}. {nome_teste}")
                opcoes_menu[str(idx_global)] = id_teste
                idx_global += 1

        print("0. Voltar ao Menu Principal")
        escolha_teste_num = input("Escolha um teste de restrição para executar: ").strip()

        if escolha_teste_num == '0':
            break

        id_teste_escolhido = opcoes_menu.get(escolha_teste_num)
        if not id_teste_escolhido:
            print("Opção inválida.")
            continue

        config_escolhida = configuracoes_restricoes[id_teste_escolhido]
        print(f"\nExecutando teste: {config_escolhida['nome']}...")

        try:
            # Configurar parâmetros para a suíte de análise sob restrição
            default_init_size_restr = analyzer.last_init_sample_size if analyzer.last_init_sample_size is not None else len(
                analyzer.motorcycles_full_dataset_original) // 2 or 1000
            init_s_str_restr = input(
                f"Tamanho da amostra para este teste (Padrão {default_init_size_restr}. VAZIO para padrão): ").strip()
            init_sample_restr = int(init_s_str_restr) if init_s_str_restr else default_init_size_restr
            if init_sample_restr <= 0: init_sample_restr = default_init_size_restr

            bench_ops_s_restr = input(f"Número de operações de benchmark para este teste (padrão 100): ").strip()
            bench_ops_restr = int(bench_ops_s_restr) if bench_ops_s_restr else 100
            if bench_ops_restr < 0: bench_ops_restr = 100

            # Perguntar se quer rodar escalabilidade também sob esta restrição
            run_scal_restr_input = input(
                "Rodar também testes de escalabilidade sob esta restrição? (s/n, padrão n): ").strip().lower()
            run_scal_restr = True if run_scal_restr_input == 's' else False
            scal_sizes_restr = None
            if run_scal_restr:
                sizes_str_restr = input(
                    "Tamanhos N para escalabilidade (ex: 100,500,1000). VAZIO para padrão: ").strip()
                if sizes_str_restr:
                    scal_sizes_restr = [int(s.strip()) for s in sizes_str_restr.split(',')]
                    if any(s <= 0 for s in scal_sizes_restr):
                        print("AVISO: Tamanhos N de escalabilidade devem ser positivos. Usando padrão.")
                        scal_sizes_restr = None

            analyzer.run_suite_with_restriction(
                restriction_config=config_escolhida,
                init_sample_size=init_sample_restr,
                benchmark_ops_count=bench_ops_restr,
                run_scalability_flag=run_scal_restr,
                scalability_sizes=scal_sizes_restr
            )
        except ValueError:
            print("ERRO: Entrada inválida para parâmetros do teste.")
        except Exception as e:
            print(f"ERRO ao executar teste com restrição '{config_escolhida['nome']}': {e}")

        input("\nPressione Enter para continuar...")