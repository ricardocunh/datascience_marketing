# -*- coding: utf-8 -*-
"""
Base de Dados analisado - Kaggle

https://www.kaggle.com/arjunbhasin2013/ccdata

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster  import KMeans
from sklearn.decomposition import PCA

from google.colab import drive
drive.mount('/content/drive')

creditcards_df = pd.read_csv('Marketing_data.csv')

creditcards_df.shape

creditcards_df.head()

# Observando o tipo de dados
creditcards_df.info()

# Visualizar  estatistica dos dados
creditcards_df.describe()

# Encontrar dados sobre um cliente especifico (cliente que maior fez compra a vista)
creditcards_df[creditcards_df['ONEOFF_PURCHASES'] == 40761.250000]

# Buscando o cliente que fez uma maior saque do limite do cartão de crédito
creditcards_df['CASH_ADVANCE'].max()

creditcards_df[creditcards_df['CASH_ADVANCE'] == 47137.21176]

"""## VISUALIZAÇÃO E EXPLORAÇÃO DOS DADOS"""

# Verificando se há registros nulos na tabela
sns.heatmap(creditcards_df.isnull()); # As barrinhas são os dados nulos encontrados

# Fazendo a contagem dos elementos nulos existentes
creditcards_df.isnull().sum()

creditcards_df['MINIMUM_PAYMENTS'].mean()

# Realizando o tratamentos nos dados para não gerar erros no algoritmo
# Preenchendo os dados nulos e trabalando com a média dos dados

creditcards_df.loc[(creditcards_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcards_df['MINIMUM_PAYMENTS'].mean()

creditcards_df['CREDIT_LIMIT'].mean()

creditcards_df.loc[(creditcards_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcards_df['CREDIT_LIMIT'].mean()

creditcards_df.isnull().sum()

sns.heatmap(creditcards_df.isnull());

# Verificando se existe dados duplicados
creditcards_df.duplicated().sum()

# Apagando o identificado do cleinte pois não é relevante para o agrupamento
creditcards_df.drop('CUST_ID', axis = 1, inplace = True)

creditcards_df.head()

creditcards_df.columns

plt.figure(figsize=(10,50))
for i in range(len(creditcards_df.columns)):
  plt.subplot(17, 1, i + 1)  # Sub gráfico, gerando 17 gráfico, ou seja, um grafico por linha, 17 linhas e uma coluna juntamento com o ID do gráfico
  sns.distplot(creditcards_df[creditcards_df.columns[i]], kde = True) #gerando o histograma, kde para geração da linha
  plt.title(creditcards_df.columns[i]) # definindo um titulo para o gráfico
plt.tight_layout(); # isso fará que as palavras não fiquem em cima uma as outras

# Gerar a matriz de correlações
correlations = creditcards_df.corr()

# gerando o mapa de calor
f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations, annot=True);

"""## K-MEANS => PARA ATUAR COM AGRUPAMENTO DE DADOS
Algoritmos não supervisionado (clustering - agrupamento)

### QUAL A TÉCNICA PARA SABER QUANTOS GRUPOS DE CLUSTER UMA BASE DE DADOS PRECISA SER AGRUPADO OU QUE PODE SER AGRUPADO?

UTILIZANDO O ELBOW METHOD (MÉTODO DO COTOVELO)
"""

# Descobrindo o número ideal de clusters nessa base de dados

# visualizando os valores minimos e máximos
min(creditcards_df['BALANCE']), max(creditcards_df['BALANCE'])

# Colocar os dados em escala
scaler = StandardScaler()
creditcards_df_scaled = scaler.fit_transform(creditcards_df)

# Substituido o nome da coluna BALANCE para o  número do indice do dataset que no caso está no "0"
min(creditcards_df_scaled[0]), max(creditcards_df_scaled[0])

creditcards_df_scaled

# nesse modelo trabalharemos até 20 clusters
wcss_1 = []
range_values = range(1, 20)
for i in range_values:
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(creditcards_df_scaled)
  wcss_1.append(kmeans.inertia_)

print(wcss_1)

# gerando o gráfico do wcss_1 contendo os números de clusters
plt.plot(wcss_1, 'bx-')
plt.xlabel('clusters')
plt.ylabel('WCSS');

# Provavel que seja utilizado de 7 a 8 clusters no dataset

"""# Agrupamento com K-means"""

kmeans = KMeans(n_clusters=8) # testaremos com 8 clusters
kmeans.fit(creditcards_df_scaled) # fit => ira fara todo o treinamento do algoritmo kmeans de encontrar o centroide, calcular as médias até que ele consiga colocar cada um dos registros
labels = kmeans.labels_

# visualizando que grupo cada cliente pertence e o todos de clientes
labels, len(labels)

# Checando quantos clientes existem em cada grupo
np.unique(labels, return_counts=True)

"""Centroide nada mais é do que a média de cada grupo

Cada grupo tem seu centroide
"""

# Verificando a média de cada clusters
kmeans.cluster_centers_

# Criando um dataframe com os dados
cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcards_df.columns])
cluster_centers

"""## 4 Grupos com potências interesses para o Banco

* Grupo 0: Clientes que pagam pouco juros para o banco e são cuidadosos com seu dinheiro. Possui menos dinheiro na conta corrente (105) e não sacam muito dinheiro do limite do cartão (301). 23% de pagamento da fatura completa do cartão de crédito
* Grupo 3 (que apresenta mais risco ao banco): usam o cartão de crédito como "emprestimos" (setor mais lucrativo para o banco), possuem muito dinheiro  na conta corrente (5041) e sacam muito dinheiro do cartão de crédito (5168), compram pouco (0.29) e usam bastante o limite do cartão para saques (0.51). Pagam muito pouco a fatura completa (0.03)
* Grupo 6 (VIP/Prime): Limite do cartão alto (16043) e o mais alto percentual de pagamento da fatura completa (0.51). Aumentar o limite do cartão e o hábito de compras
* Grupo 7 (clientes novos): clientes mais novos (7.23) e que mantém pouco dinheiro na conta corrente (867)
"""

# Criando o caminho inverso, ira retornar os valores originais
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcards_df.columns])
cluster_centers

# Verificado o total de labels
labels, len(labels)

# Adicionando cada um dos grupos para cada um dos clientes
# Criando um novo dataframe para realizarmos a concatenação com o formato de dicionário, com o atributo 'cluster' e passamos o label concatenando com as colunas "axis=1"
creditcards_df_cluster = pd.concat([creditcards_df, pd.DataFrame({'cluster': labels})], axis=1)
creditcards_df_cluster.head()

from IPython.core.pylabtools import figsize
# Gerando um histograma utilizando os grupos para facilitar novas analises
for i in creditcards_df.columns:  # Percorrendo inicialmente neste for cada uma das colunas
  plt.figure(figsize=(35,5))
  for j in range(8): # Percorrendo 8x cada um dos atribusots dos clusters separados pelos grupos
    plt.subplot(1, 8, j + 1)  # Tera uma linha e 8 colunas, cada linha terá 8 gráficos
    cluster = creditcards_df_cluster[creditcards_df_cluster['cluster'] == j]  # a variavel "j" varia de 0 até 7 para coletarmos os registros específicos de cada grupo
    cluster[i].hist(bins = 20)  # número de divisões igual a 20 (bins = 20)
    plt.title('{} \nCluster {}'.format(i, j))
  plt.show()

# Criando uma codificação adicional simulado como se fosse ser enviados essa base de dados para o pessoal do marketing para serem feitas as campanhas
credit_ordered = creditcards_df_cluster.sort_values(by = 'cluster')
credit_ordered.head()

# Checando os 5 ultimos valores
credit_ordered.tail()

# Salvando o arquivo em um CSV
credit_ordered.to_csv('cluster.csv')
