import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import matplotlib.pyplot as plt


# Carregar o dataframe
df_train = pd.read_csv('dados_trab2/conjunto_de_treinamento.csv')

# Pré-processamento
# 1. Retirada de ID
df_train = df_train.drop(columns=['Id'])
# Codificar atributos não numéricos
label_encoders = {}
for column in df_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column])
    label_encoders[column] = le

# Remover outliers
quantile_low = df_train.quantile(0.001)
quantile_high = df_train.quantile(0.98)

for column in df_train.columns:
    df_train = df_train[(df_train[column] >= quantile_low[column]) & (df_train[column] <= quantile_high[column])]

# Separar features e target
X = df_train.drop(columns=['preco'])
y = df_train['preco']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=2)  # Mantém 95% da variância
X_pca = pca.fit_transform(X_scaled)
# Criar um DataFrame para X_pca
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_pca, y, test_size=0.2, random_state=42)


# # Fixar número de componentes e clusters
# n_components = 2
# n_clusters = 2

# # Aplicar PCA com 2 componentes
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X_scaled)

# # Aplicar KMeans com 2 clusters
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(X_pca)

# # Adicionar os rótulos de cluster ao DataFrame PCA
# df_pca['cluster'] = cluster_labels

# # Plotar gráfico de Componente 1 vs Componente 2 com hue = 'cluster'
# plt.figure(figsize=(10, 7))
# plt.scatter(df_pca['PC1'], df_pca['PC2'], c=y, cmap='viridis', alpha=0.5)
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.title('Gráfico de Componente 1 vs Componente 2 com hue = cluster')
# plt.colorbar(label='Cluster')
# plt.show()


best_score = -1
best_n_components = 0
best_n_clusters = 0

# Fixar número de componentes
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_train)

# Utilizar KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_pca, y_train)

# Prever e calcular o erro quadrático médio
y_pred = knn_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# # Plotar gráfico de Componente 1 vs Componente 2 com hue = 'cluster'
# plt.figure(figsize=(10, 7))
# plt.scatter(df_pca['PC1'], df_pca['PC2'], c=y, cmap='viridis', alpha=0.5)
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.title('Gráfico de Componente 1 vs Componente 2 com hue = Y')
# plt.colorbar(label='Cluster')
# plt.show()

# Calcular o erro (residuals)
residuals = y_test - y_pred

# Juntar os dois gráficos em uma figura
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# Gráfico de Componente 1 vs Componente 2 com hue = Y
scatter1 = axes[0].scatter(df_pca['PC1'], df_pca['PC2'], c=y, cmap='viridis', alpha=0.5)
axes[0].set_xlabel('Componente Principal 1')
axes[0].set_ylabel('Componente Principal 2')
axes[0].set_title('Gráfico de Componente 1 vs Componente 2 com hue = Y')
fig.colorbar(scatter1, ax=axes[0], label='Cluster')

# Gráfico de Componente 1 vs Componente 2 com hue = Erro (y_test - y_pred)
scatter2 = axes[1].scatter(X_test['PC1'], X_test['PC2'], c=residuals, cmap='coolwarm', alpha=0.5)
axes[1].set_xlabel('Componente Principal 1')
axes[1].set_ylabel('Componente Principal 2')
axes[1].set_title('Gráfico de Componente 1 vs Componente 2 com hue = Erro (y_test - y_pred)')
fig.colorbar(scatter2, ax=axes[1], label='Erro')

plt.tight_layout()
plt.show()

# best_score = mse
# best_n_components = n_components
# best_n_clusters = 5  # Número de vizinhos utilizado no KNeighborsRegressor

# print(f"Best score: {best_score}")
# print(f"Best number of components: {best_n_components}")
# print(f"Best number of clusters: {best_n_clusters}")

# # Plotar gráficos PCi vs PCj numa grade
# num_components = df_pca.shape[1]
# fig, axes = plt.subplots(num_components, num_components, figsize=(15, 15))

# for i in range(num_components):
#     for j in range(num_components):
#         if i != j:
#             axes[i, j].scatter(df_pca[f'PC{i+1}'], df_pca[f'PC{j+1}'], alpha=0.5)
#             axes[i, j].set_xlabel(f'PC{i+1}')
#             axes[i, j].set_ylabel(f'PC{j+1}')
#         else:
#             axes[i, j].text(0.5, 0.5, f'PC{i+1}', fontsize=12, ha='center')
#         axes[i, j].set_xticks([])
#         axes[i, j].set_yticks([])

# plt.tight_layout()
# plt.show()