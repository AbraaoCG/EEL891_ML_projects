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
quantile_low = df_train.quantile(0.05)
quantile_high = df_train.quantile(0.95)

for column in df_train.columns:
    df_train = df_train[(df_train[column] >= quantile_low[column]) & (df_train[column] <= quantile_high[column])]

# Separar features e target
X = df_train.drop(columns=['preco'])
y = df_train['preco']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA(n_components=0.95)  # Mantém 95% da variância
X_pca = pca.fit_transform(X_scaled)
# Criar um DataFrame para X_pca
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_pca, y, test_size=0.2, random_state=42)

# print(df_pca)

# Modelo LASSO

# # Definir o modelo
# model = Lasso()

# # Definir a grade de hiperparâmetros
# param_distributions = {
#     'alpha': np.logspace(-4, 0, 50),
#     'max_iter': [1000, 2000, 3000, 4000],
#     'tol': [1e-4, 1e-3, 1e-2]
# }

# # Configurar a busca aleatória
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, 
#                                    n_iter=100, cv=5, random_state=42, n_jobs=-1)

# # Executar a busca aleatória
# random_search.fit(X_train, y_train)

# # Melhor modelo encontrado
# best_params = random_search.best_params_
# best_score = -random_search.best_score_

# print(f"Best Score: {best_score:.2f}")
# print(f"Best Parameters: {best_params}")



# Criar o melhor modelo encontrado
best_model = Lasso(tol=0.01, max_iter=3000, alpha=np.float64(1.0))

# Validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = cross_val_score(best_model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)

# Calcular a média e o desvio padrão do MSE
mean_mse = -np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"Cross-Validation Mean MSE: {mean_mse:.2f}")
print(f"Cross-Validation MSE Std Dev: {std_mse:.2f}")

# Treinar o modelo
best_model.fit(X_train, y_train)

# Previsões no conjunto de teste
y_test_pred = best_model.predict(X_test)

# Imprimir 10 valores (y_test ; y_pred)
for actual, predicted in zip(y_test[:10], y_test_pred[:10]):
    print(f'Actual: {actual}, Predicted: {predicted}')

# Calcular o resíduo/erro
residuals = y_test - y_test_pred

# # Mostrar gráfico com o resíduo/erro
# plt.scatter(X_test['PC1'], y_test)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('Predicted Values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs Predicted Values')
# plt.show()
# # Criar um conjunto de plots com os componentes principais e os valores reais de y_test

i = 1
plt.figure(figsize=(8, 6))
plt.scatter(X_test[f'PC{i}'], y_test, alpha=0.5)
plt.xlabel(f'Principal Component ')
plt.ylabel('Preço')
plt.title(f'Principal Component {i} vs Preço')
plt.show()
