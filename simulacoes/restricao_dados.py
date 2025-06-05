# simulacoes/restricao_dados.py
import random
from typing import List
from modelos.moto import Moto
import copy

def corromper_precos_aleatoriamente(dataset: List[Moto], percentual_corrompido: float = 0.1, fator_outlier: float = 5.0) -> List[Moto]:
    dataset_corrompido = [copy.deepcopy(m) for m in dataset]
    num_para_corromper = int(len(dataset_corrompido) * percentual_corrompido)
    if num_para_corromper == 0 and len(dataset_corrompido) > 0 and percentual_corrompido > 0:
        num_para_corromper = 1
    indices_corrompidos = random.sample(range(len(dataset_corrompido)), k=min(num_para_corromper, len(dataset_corrompido)))
    for idx in indices_corrompidos:
        moto = dataset_corrompido[idx]
        if random.random() < 0.5:
            preco_original = moto.preco; modificador = 1 + random.uniform(-fator_outlier, fator_outlier)
            while 0.8 < modificador < 1.2 and fator_outlier > 0.2: modificador = 1 + random.uniform(-fator_outlier, fator_outlier)
            moto.preco = preco_original * modificador
            if moto.preco < 0: moto.preco = random.uniform(1000, 5000)
        else:
            if random.random() < 0.3: moto.preco = 0.0
            else: moto.preco /= random.uniform(5, 20)
        if moto.revenda > moto.preco : moto.revenda = moto.preco * random.uniform(0.5, 0.9)
        if moto.revenda < 0 and moto.preco >= 0 : moto.revenda = 0
        elif moto.revenda < 0 and moto.preco < 0 : moto.revenda = moto.preco
    print(f"INFO: {len(indices_corrompidos)} registros tiveram seus preços corrompidos.")
    return dataset_corrompido

def introduzir_anos_anomalos(dataset: List[Moto], percentual_anomalo: float = 0.05, ano_min_valido:int=1950, ano_max_valido:int=2030) -> List[Moto]:
    dataset_modificado = [copy.deepcopy(m) for m in dataset]
    num_para_modificar = int(len(dataset_modificado) * percentual_anomalo)
    if num_para_modificar == 0 and len(dataset_modificado) > 0 and percentual_anomalo > 0: num_para_modificar = 1
    indices_modificados = random.sample(range(len(dataset_modificado)), k=min(num_para_modificar, len(dataset_modificado)))
    for idx in indices_modificados:
        if random.random() < 0.5: dataset_modificado[idx].ano = random.randint(1880, ano_min_valido - 1)
        else: dataset_modificado[idx].ano = random.randint(ano_max_valido + 1, ano_max_valido + 50)
    print(f"INFO: {len(indices_modificados)} registros tiveram seus anos modificados para valores anômalos.")
    return dataset_modificado

def reduzir_resolucao_precos(dataset: List[Moto]) -> List[Moto]: # Não está nos 10 testes atuais, mas pode ser adicionado
    dataset_modificado = [copy.deepcopy(m) for m in dataset]
    count = 0
    for moto in dataset_modificado:
        try:
            moto.preco = float(int(moto.preco)); moto.revenda = float(int(moto.revenda))
            count +=1
        except ValueError: pass
    print(f"INFO: Preços e revendas de {count} motos convertidos para inteiros.")
    return dataset_modificado