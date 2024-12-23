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

from sklearn.ensemble import ExtraTreesRegressor


# Carregar o dataframe
df_train = pd.read_csv('dados_trab2/conjunto_de_treinamento.csv')
df_test = pd.read_csv('dados_trab2/conjunto_de_teste.csv')

# ----------------------------  ------------------------------------------------------------
# ----------------------------  ------------------------------------------------------------


# Pré-processamento

# 1. Retirada de ID
df_train.index = df_train['Id']
df_train = df_train.drop(columns=['Id'])
df_test.index = df_test['Id']
df_test = df_test.drop(columns=['Id'])


# ----------------------------  ------------------------------------------------------------



# Codificar atributos não numéricos
label_encoders = {}
for column in df_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column])
    label_encoders[column] = le

for column in df_test.select_dtypes(include=['object']).columns:
    le = label_encoders[column]
    df_test[column] = df_test[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.fit(np.append(le.classes_, x)).transform([x])[0])

# ----------------------------  ------------------------------------------------------------

# Remover outliers
quantile_low = df_train.quantile(0.00001)
quantile_high = df_train.quantile(0.97)


print(f'Low quantil: {quantile_low} ; High quantil: {quantile_high}')

for column in ['preco']: 
    df_train = df_train[(df_train[column] >= quantile_low[column]) & (df_train[column] <= quantile_high[column])]

# ----------------------------  ------------------------------------------------------------

# Normalizar os dados 
scaler = StandardScaler()
# scaler_test = StandardScaler()
predictors = df_train.drop(columns=['preco']).columns
df_train[predictors] = scaler.fit_transform(df_train[predictors])
df_test[predictors] = scaler.transform(df_test[predictors])

# ----------------------------  ------------------------------------------------------------

## Encontrar variáveis de maior correlação

corr_scaled = df_train.corr()

# Identificar colunas com correlação maior que k
k = 0.5
high_correlation_columns = corr_scaled.columns[corr_scaled['preco'] > k].drop('preco').tolist()

# Criar um novo DataFrame apenas com essas colunas
X_scaled = df_train[high_correlation_columns].to_numpy()
X_test_scaled = df_test[high_correlation_columns].to_numpy()

print(len(df_train),len(X_scaled))

print(f'Colunas com alta correlação: {high_correlation_columns}')


# ----------------------------  ------------------------------------------------------------

# # Aplicar PCA

# Dividir os dados em treino e validatee
X_train, X_validate, y_train, y_validate = train_test_split(X_scaled, df_train['preco'], test_size=0.1, random_state=42)

# Fixar número de componentes
n_components = 0.95
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_train)

X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

X_validate_pca = pca.transform(X_validate)
X_validate_pca_df = pd.DataFrame(X_validate_pca, columns=[f'PC{i+1}' for i in range(X_validate_pca.shape[1])])

X_test_pca = pca.transform(X_test_scaled)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

print(f"Number of components: {X_pca_df.shape[1]}")

# ----------------------------  ------------------------------------------------------------
# ----------------------------  ------------------------------------------------------------

# Treinamento e Predição

# Treinar modelo

# model = RandomForestRegressor(random_state=42,  n_estimators = 200, max_depth = 30, min_samples_split = 2,min_samples_leaf = 1)
model = ExtraTreesRegressor(random_state=42 ,n_estimators =  500, min_samples_split = 10, min_samples_leaf =  2, max_depth =  20)


model.fit(X_pca_df, y_train)

# Prever e calcular o erro quadrático médio
y_pred = model.predict(X_validate_pca_df)
mse = mean_squared_error(y_validate, y_pred)

print(f"Mean Squared Error: {mse}")


# ----------------------------  ------------------------------------------------------------

# Validacao cruzada
cv = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = cross_val_score(model, X_pca_df, y_train, cv=cv, scoring='neg_mean_squared_error')
mean_cv_score = -cross_val_scores.mean()

print(f"Cross-validated Mean Squared Error: {mean_cv_score}")


# ----------------------------  ------------------------------------------------------------

# Treinar com 100% dos dados de treino, treinar o modelo e fazer previsões nos dados de teste.

# Fit the model again using 100% of the training data

model.fit(pca.transform(X_scaled), df_train['preco'])

y_test_pred = model.predict(X_test_pca)

## save predictions

predictions_df = pd.DataFrame({ 'preco': y_test_pred , 'Id': df_test.index})  
predictions_df.to_csv('predictions/ExtraTrees_tuned_predicted_prices_pca_outliers1e-5_97e-2_allIds.csv', index=False)


# ----------------------------  ------------------------------------------------------------

var1 = 'PC1'
var2 = 'PC3'
# Calcular o erro (residuals)
residuals = y_validate - y_pred

# Juntar os dois gráficos em uma figura
fig, axes = plt.subplots(2, 2, figsize=(20, 7))
# Determinar os limites dos eixos
x_lim = (min(X_pca_df[var1].min(), X_validate_pca_df[var1].min(), X_test_pca_df[var1].min()), 
         max(X_pca_df[var1].max(), X_validate_pca_df[var1].max(), X_test_pca_df[var1].max()))
y_lim = (min(X_pca_df[var2].min(), X_validate_pca_df[var2].min(), X_test_pca_df[var2].min()), 
         max(X_pca_df[var2].max(), X_validate_pca_df[var2].max(), X_test_pca_df[var2].max()))

# Aplicar os limites aos gráficos
for ax in axes.flatten():
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

# Gráfico de Componente 1 vs Componente 2 com hue = Y
scatter1 = axes[0][0].scatter(X_pca_df[var1], X_pca_df[var2], c=y_train, cmap='viridis', alpha=0.5)
axes[0][0].set_xlabel(var1)
axes[0][0].set_ylabel(var2)
axes[0][0].set_title('Gráfico de ' + var1 + ' vs ' + var2 + ' com hue = Y')
fig.colorbar(scatter1, ax=axes[0][0], label='Preço')

# Gráfico de var1 vs var2 com hue = Erro (y_validate - y_pred)
scatter2 = axes[0][1].scatter(X_validate_pca_df[var1], X_validate_pca_df[var2], c=residuals, cmap='coolwarm', alpha=0.5)
axes[0][1].set_xlabel(var1)
axes[0][1].set_ylabel(var2)
axes[0][1].set_title('Gráfico de ' + var1 + ' vs ' + var2 + ' com hue = Erro (y_validate - y_pred)')
fig.colorbar(scatter2, ax=axes[0][1], label='Erro')

# Gráfico de var1 vs var2 para X_test_pca_df
scatter3 = axes[1][0].scatter(X_test_pca_df[var1], X_test_pca_df[var2], c=y_test_pred, cmap='viridis', alpha=0.5)
axes[1][0].set_xlabel(var1)
axes[1][0].set_ylabel(var2)
axes[1][0].set_title('Gráfico de ' + var1 + ' vs ' + var2 + ' para X_test_pca_df')
fig.colorbar(scatter3, ax=axes[1][0], label='Preço')



plt.tight_layout()
plt.show()