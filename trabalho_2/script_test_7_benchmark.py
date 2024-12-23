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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt



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

# Benchmark
# Definir o modelo
model = ExtraTreesRegressor(random_state=42 ,n_estimators =  500, min_samples_split = 10, min_samples_leaf =  2, max_depth =  20)

# Definir o grid de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 500 , 800],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # 'min_samples_split': [2, 5, 10, 15, 20],
    # 'min_samples_leaf': [1, 2, 4, 6, 8],
    # 'bootstrap': [False, True]
}

# Configurar o GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Realizar o fit do GridSearchCV
grid_search.fit(X_pca_df, y_train)

# Obter os melhores parâmetros e o melhor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Avaliar o modelo no conjunto de validação
y_pred = best_model.predict(X_validate_pca_df)
mse = mean_squared_error(y_validate, y_pred)

# Criar um DataFrame com os resultados
results_df = pd.DataFrame([best_params])
results_df['mse'] = mse
# Imprimir os resultados
print(results_df)

# Escrever os resultados em um arquivo txt
results_df.to_csv('benchmark_results_ExtraTree.txt', sep='\t', index=False)
