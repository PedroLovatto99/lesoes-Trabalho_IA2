# Clusterização de Lesões no Futebol

Script em Python que agrupa jogadores de futebol por perfil de lesão (frequência, intervalo entre lesões, reincidência e tempo de recuperação) usando K-Means.

## Pré-requisitos

- Python 3.10 ou superior instalado

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
2. Categorizar cada lesão (Muscular, Articular/ligamentar, Óssea, Outras);
3. Agregar as lesões por jogador e calcular as features usadas no modelo;
4. Rodar o K-Means para diferentes valores de `k` e treinar o modelo final (`k=4`);
5. Exibir no terminal as estatísticas, distribuições e o perfil de cada cluster.

## Saídas geradas

- `img/cotovelo_silhouette.png` — gráficos do método do cotovelo e do silhouette score usados para escolher o `k`;
- `img/tamanho_clusters.png` — gráfico com o tamanho de cada cluster final;
- `perfis_jogadores_clusters.csv` — tabela com o perfil e o cluster de cada jogador.

A pasta `img/` é criada automaticamente pelo script, caso não exista.

## Desativar o ambiente virtual

```powershell
deactivate
```
