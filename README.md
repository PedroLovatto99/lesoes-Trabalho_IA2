# Clusterização de Lesões no Futebol

Script em Python que agrupa jogadores de futebol por perfil de lesão (frequência, intervalo entre lesões, reincidência e tempo de recuperação) usando K-Means.

## Pré-requisitos

- Python 3.10 ou superior instalado
- O arquivo de dados `full_dataset_thesis - 1.csv` (baixado do Kaggle) na mesma pasta do script

## 1. Criar e ativar o ambiente virtual


```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

> Se o PowerShell bloquear a execução do script de ativação, rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

No `cmd.exe`, use `.\env\Scripts\activate.bat` em vez do comando acima.

## 2. Instalar as dependências

Com o ambiente virtual ativado (o prompt deve mostrar `(env)`):

```powershell
pip install -r requirements.txt
```

## 3. Rodar o script

```powershell
python injuries_football.py
```

O script vai:

1. Carregar e limpar o dataset de lesões;
2. Categorizar cada lesão (Muscular, Articular/ligamentar, Óssea, Outras) e mostrar a distribuição em um gráfico de barras;
3. Agregar as lesões por jogador e calcular as features usadas no modelo;
4. Rodar o K-Means para diferentes valores de `k` (método do cotovelo e silhouette score) e treinar o modelo final (`k=4`);
5. Exibir no terminal as estatísticas, distribuições e o perfil de cada cluster.

Os gráficos abrem em janelas separadas enquanto o script continua rodando; o terminal mostra os resultados finais sem esperar você fechá-las. Ao final, o script aguarda você fechar as janelas dos gráficos para encerrar.

## Saídas geradas

- `img/barras_categoria_lesao.png` — distribuição das lesões por categoria;
- `img/cotovelo_silhouette.png` — gráficos do método do cotovelo e do silhouette score usados para escolher o `k`.

A pasta `img/` é criada automaticamente pelo script, caso não exista.

## Desativar o ambiente virtual

```powershell
deactivate
```
