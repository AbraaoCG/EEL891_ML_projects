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
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, PassiveAggressiveRegressor, Ridge, Lasso, LassoLars, Lars, ElasticNet, OrthogonalMatchingPursuit, ARDRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor

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


# Definir os grids de parâmetros para múltiplos modelos de regressão
param_grids = {
    'SVR': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    },
    'GradientBoostingRegressor': {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    },
    'AdaBoostRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 1.0]
    },
    'MLPRegressor': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    'TheilSenRegressor': {
        'max_subpopulation': [1e3, 1e4, 1e5],
        'n_subsamples': [None, 1e3, 1e4],
        'fit_intercept': [True, False]
    },
    'RANSACRegressor': {
        'min_samples': [0.1, 0.5, 0.9],
        'residual_threshold': [None, 1.0, 5.0],
        'is_data_valid': [None],
        'is_model_valid': [None]
    },
    'PassiveAggressiveRegressor': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000],
        'tol': [1e-3, 1e-4]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    },
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000]
    },
    'LassoLars': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False]
    },
    'Lars': {
        'fit_intercept': [True, False],
        'n_nonzero_coefs': [500, 1000, 2000]
    },
    'ElasticNet': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True, False]
    },
    'RandomForestRegressor': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'DecisionTreeRegressor': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBRegressor': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10]
    },

    'GradientBoostingRegressor': {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    },
    'KNeighborsRegressor': {
        'n_neighbors': [3, 5, 10, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'ARDRegression': {
        'tol': [1e-3, 1e-4, 1e-5],
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    },
    'ExtraTreesRegressor': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'BayesianRidge': {
        'tol': [1e-3, 1e-4, 1e-5],
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4]
    },
    'AdaBoostRegressor': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 1.0]
    }
}

# Inicializar os modelos
models = {
    'SVR': SVR(),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
    'MLPRegressor': MLPRegressor(random_state=42),
    'TheilSenRegressor': TheilSenRegressor(),
    'RANSACRegressor': RANSACRegressor(random_state=42),
    'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=42),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'LassoLars': LassoLars(),
    'Lars': Lars(),
    'ElasticNet': ElasticNet(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'XGBRegressor': XGBRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'ARDRegression': ARDRegression(),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
    'BayesianRidge': BayesianRidge()
}

# DataFrame para armazenar os resultados
results_df = pd.DataFrame(columns=['Model', 'Best_Params', 'MSE', 'Cross_Val_MSE'])

# Realizar RandomizedSearchCV para cada modelo
for model_name in models:
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_pca_df, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    y_pred = best_model.predict(X_validate_pca_df)
    mse = mean_squared_error(y_validate, y_pred)
    cross_val_mse = -cross_val_score(best_model, X_pca_df, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    print(f"Mean Squared Error: {mse}")
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': model_name,
        'Best_Params': best_params,
        'MSE': mse,
        'Cross_Val_MSE': cross_val_mse
    }])], ignore_index=True)

# Imprimir os resultados
print(results_df)

# Escrever os resultados em um arquivo txt
results_df.to_csv('benchmark_results.txt', sep='\t', index=False)
