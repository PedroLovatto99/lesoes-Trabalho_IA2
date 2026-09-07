
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =============================================================================
# 1. Configuracao e carregamento dos dados
# =============================================================================

# ajuste o caminho/nome do arquivo conforme baixado do Kaggle
CAMINHO_CSV = 'full_dataset_thesis - 1.csv'

# diretorio onde as imagens geradas serao salvas
DIR_IMG = 'img'
os.makedirs(DIR_IMG, exist_ok=True)

df = pd.read_csv(CAMINHO_CSV)
print(f"Total de registros: {len(df)}")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df.head().T)

# =============================================================================
# 2. Limpeza dos dados
# =============================================================================

df['days_num'] = df['Days'].str.replace(' days', '', regex=False).str.replace(' day', '', regex=False).astype(int)

df['injury_from_parsed'] = pd.to_datetime(df['injury_from_parsed'])
df['injury_until_parsed'] = pd.to_datetime(df['injury_until_parsed'])

df['injury_lower'] = df['Injury'].str.lower().str.strip()

print("\nValores nulos por coluna:")
print(df.isnull().sum())

nao_lesao = ['corona virus', 'ill', 'fitness', 'flu', 'rest', 'quarantine', 'cold',
             'fever', 'stomach flu', 'influenza', 'infection', 'virus', 'tonsillitis',
             'stomach problems']
antes = len(df)
df = df[~df['injury_lower'].isin(nao_lesao)].copy()
print(f"\nRegistros removidos (doenca/afastamento nao relacionado a lesao): {antes - len(df)}")
print(f"Registros restantes: {len(df)}")


# =============================================================================
# 3. Categorizacao do tipo de lesao
# =============================================================================

def categoria_lesao(texto):
    muscular = ['muscle', 'hamstring', 'calf', 'thigh', 'adductor', 'groin', 'strain',
                'fiber', 'bundle', 'dead leg', 'pubalgia', 'lumbago']
    articular = ['knee', 'ankle', 'ligament', 'cruciate', 'meniscus', 'achilles',
                 'shoulder', 'hip', 'capsular', 'joint']
    ossea = ['fracture', 'broken', 'bone']
    for kw in muscular:
        if kw in texto:
            return 'Muscular'
    for kw in articular:
        if kw in texto:
            return 'Articular/ligamentar'
    for kw in ossea:
        if kw in texto:
            return 'Óssea'
    return 'Outras'


df['categoria_lesao'] = df['injury_lower'].apply(categoria_lesao)
print("\nDistribuição por categoria de lesão:")
print(df['categoria_lesao'].value_counts())


# =============================================================================
# 4. Construcao das features por jogador (agregacao)
# =============================================================================

df = df.sort_values(['player_name', 'injury_from_parsed'])

registros = []
for jogador, grupo in df.groupby('player_name'):
    if len(grupo) < 2:
        continue 
    grupo = grupo.sort_values('injury_from_parsed')

    gaps = []
    mesmos_tipos = 0
    total_pares = 0
    for i in range(len(grupo) - 1):
        gap = (grupo.iloc[i + 1]['injury_from_parsed'] - grupo.iloc[i]['injury_until_parsed']).days
        gaps.append(max(gap, 0))  # trata sobreposição/erro de data como intervalo 0
        total_pares += 1
        if grupo.iloc[i]['categoria_lesao'] == grupo.iloc[i + 1]['categoria_lesao']:
            mesmos_tipos += 1

    registros.append({
        'player_name': jogador,
        'n_lesoes': len(grupo),
        'idade': grupo.iloc[-1]['player_age'],
        'intervalo_medio': np.mean(gaps),
        'pct_reincidencia': mesmos_tipos / total_pares,
        'media_dias_parado': grupo['days_num'].mean(),
    })

jogadores_df = pd.DataFrame(registros)
print(f"\nTotal de jogadores com 2+ lesões: {len(jogadores_df)}")
print(jogadores_df.describe().round(1))


# =============================================================================
# 5. Transformacao (log) e preparacao para o K-Means
# =============================================================================

jogadores_df['n_lesoes_log'] = np.log1p(jogadores_df['n_lesoes'])
jogadores_df['intervalo_log'] = np.log1p(jogadores_df['intervalo_medio'])
jogadores_df['dias_log'] = np.log1p(jogadores_df['media_dias_parado'])

features = ['n_lesoes_log', 'idade', 'intervalo_log', 'pct_reincidencia', 'dias_log']
X = StandardScaler().fit_transform(jogadores_df[features])
print(f"\nMatriz final: {X.shape[0]} jogadores x {X.shape[1]} features")


# =============================================================================
# 6. Escolha do numero de clusters (k)
# =============================================================================

inercias = []
silhouettes = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(k_range, inercias, marker='o')
ax[0].set_title('Método do cotovelo')
ax[0].set_xlabel('k')
ax[0].set_ylabel('Inércia')

ax[1].plot(k_range, silhouettes, marker='o', color='darkorange')
ax[1].set_title('Silhouette score por k')
ax[1].set_xlabel('k')
ax[1].set_ylabel('Silhouette')
plt.tight_layout()
plt.savefig(os.path.join(DIR_IMG, 'cotovelo_silhouette.png'), dpi=150)
plt.show()

print("\nSilhouette score por k:")
for k, s in zip(k_range, silhouettes):
    print(f"k={k} | silhouette={s:.3f}")


# =============================================================================
# 7. Treinamento do modelo final
# =============================================================================

K_ESCOLHIDO = 4

kmeans_final = KMeans(n_clusters=K_ESCOLHIDO, random_state=42, n_init=10)
jogadores_df['cluster'] = kmeans_final.fit_predict(X)

sil_final = silhouette_score(X, jogadores_df['cluster'])
print(f"\nSilhouette score final (k={K_ESCOLHIDO}): {sil_final:.3f}")


# =============================================================================
# 8. Interpretacao dos clusters
# =============================================================================

colunas_perfil = ['n_lesoes', 'idade', 'intervalo_medio', 'pct_reincidencia', 'media_dias_parado']

print("\n=== Média geral (linha de base, todos os jogadores) ===")
print(jogadores_df[colunas_perfil].mean().round(1))

print("\n=== Perfil de cada cluster ===")
perfil = jogadores_df.groupby('cluster')[colunas_perfil].mean().round(1)
perfil['n_jogadores'] = jogadores_df.groupby('cluster').size()
print(perfil)

fig, ax = plt.subplots(figsize=(7, 4))
tamanhos = jogadores_df['cluster'].value_counts().sort_index()
ax.bar(tamanhos.index.astype(str), tamanhos.values, color='steelblue')
ax.set_xlabel('Cluster')
ax.set_ylabel('Número de jogadores')
ax.set_title(f'Tamanho de cada cluster (k={K_ESCOLHIDO})')
plt.savefig(os.path.join(DIR_IMG, 'tamanho_clusters.png'), dpi=150)
plt.show()
