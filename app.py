import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances

@st.cache_resource
def carregar_modelo_e_dados():
    modelo = joblib.load('models/modelo_kmeans.joblib')
    scaler = joblib.load('models/scaler.joblib')
    df_identificadores = pd.read_csv('data/identificadores_com_clusters.csv')
    dados_treino_padronizados = np.load('data/dados_treino_padronizados.npy')
    df_modelo_final = pd.read_excel('data/df_modelo_final.xlsx')
    return modelo, scaler, df_identificadores, dados_treino_padronizados, df_modelo_final

modelo_kmeans, scaler, df_identificadores, dados_treino_padronizados, df_modelo_final = carregar_modelo_e_dados()

def encontrar_similares(novo_jogador_stats, modelo_kmeans, scaler, dados_originais_padronizados, df_identificadores):
    """
    Encontra jogadores similares a um novo jogador usando um modelo K-Means treinado.

    Parâmetros:
    - novo_jogador_stats (pd.DataFrame): DataFrame com 1 linha contendo as stats do novo jogador.
    - modelo_kmeans (KMeans): O objeto do modelo K-Means já treinado.
    - scaler (StandardScaler): O objeto do scaler já treinado nos dados originais.
    - dados_originais_padronizados (np.array): O array numpy com todos os dados que foram usados para treinar o modelo, já padronizados.
    - df_identificadores (pd.DataFrame): DataFrame com os nomes e clusters dos jogadores originais.

    Retorna:
    - pd.DataFrame: Um DataFrame com os jogadores mais similares rankeados pela distância.
    """

    print("--- Iniciando busca por jogadores similares ---")

    novo_jogador_scaled = scaler.transform(novo_jogador_stats)
    print("Stats do novo jogador padronizadas com sucesso.")

    cluster_previsto = modelo_kmeans.predict(novo_jogador_scaled)[0]
    print(f"Cluster previsto para o novo jogador: {cluster_previsto}")

    indices_posicionais = np.where(modelo_kmeans.labels_ == cluster_previsto)[0]
    
    if len(indices_posicionais) == 0:
        print(f"Nenhum jogador encontrado no Cluster {cluster_previsto}.")
        return None

    #df_cluster_previsto = df_identificadores[df_identificadores['cluster'] == cluster_previsto].copy()

    #if df_cluster_previsto.empty:
        #print(f"Nenhum jogador encontrado no Cluster {cluster_previsto}.")
        #return None

    #print(f"Encontrados {len(df_cluster_previsto)} jogadores no mesmo cluster.")

    #indices_cluster = df_cluster_previsto.index
    dados_cluster = dados_originais_padronizados[indices_posicionais]
    df_cluster_previsto = df_identificadores.iloc[indices_posicionais].copy()

    distancias = euclidean_distances(novo_jogador_scaled, dados_cluster)

    df_cluster_previsto['distancia'] = distancias[0]
    ranking_final = df_cluster_previsto.sort_values(by='distancia')

    print("--- Ranking de similaridade gerado com sucesso! ---")
    
    return ranking_final, cluster_previsto



st.title('Players Clustering App - MD')
st.write("Encontre jogadores com perfis estatísticos parecidos no nosso dataset.")

lista_jogadores = df_identificadores['player'].unique()

jogador_selecionado = st.selectbox('Selecione um jogador para encontrar similares:', options=lista_jogadores)

if st.button('Encontrar Jogadores Similares'):
    if jogador_selecionado:
        try:

            indice_jogador = df_identificadores[df_identificadores['player'] == jogador_selecionado].index[0]

            stats_jogador_selecionado = df_modelo_final.loc[[indice_jogador]]

            ranking, cluster = encontrar_similares(
                novo_jogador_stats=stats_jogador_selecionado,
                modelo_kmeans=modelo_kmeans,
                scaler=scaler,
                dados_originais_padronizados=dados_treino_padronizados,
                df_identificadores=df_identificadores
            )

            st.success(f"Análise completa para: {jogador_selecionado}")
            st.write(f"Este jogador pertence ao **Perfil (Cluster) {cluster}**.")

            st.subheader(f"Top 10 Jogadores com Estilo Similar:")
            
            st.dataframe(ranking[ranking['player'] != jogador_selecionado].head(10))

        except IndexError:
            st.error(f"Não foi possível encontrar as estatísticas para o jogador {jogador_selecionado}. Verifique os dados.")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {e}")

    else:
        st.error("Por favor, selecione um jogador.")