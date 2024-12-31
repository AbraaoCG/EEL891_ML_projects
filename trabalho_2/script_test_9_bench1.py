import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
import umap

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

initHigh = 0.9999999 
initLow = 0.00000001

columns_to_check = ['preco', 'diferenciais', 'area_util']# df_train.columns.to_list()

quantile_ranges = {col: [initLow,initHigh] for col in columns_to_check}




quantile_ranges['preco'] = [0.0001, 0.96]
quantile_ranges['diferenciais'] = [0.001, 0.999]
#quantile_ranges['area_util'] = [0.00001, 0.98]


# quantile_low = df_train[columns_to_check].quantile(0.00001)
# quantile_high = df_train[columns_to_check].quantile(0.97)

print('Análise de Quantil -- Primeira redução de outliers')
for column in columns_to_check: 
    Q1 = df_train[column].quantile(quantile_ranges[column][0])
    Q3 = df_train[column].quantile(quantile_ranges[column][1])
    print(f'{column} - Low quantil 1: {Q1} ; High quantil 1: {Q3}')
    df_train = df_train[(df_train[column] >= Q1) & (df_train[column] <= Q3)]


# initialIdsCount = len(df_test)
# # Filtrar df_test pelos quantis de outliers de treino
# for column in ['area_util']:
#     Q1 = df_train[column].quantile(quantile_ranges[column][0])
#     Q3 = df_train[column].quantile(quantile_ranges[column][1])
#     df_test = df_test[(df_test[column] >= Q1) & (df_test[column] <= Q3)]

# finalidsCount = len(df_test)

# print(f'Droped {initialIdsCount - finalidsCount} ids from test set')

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
# Aplicar PCA

# Fixar número de componentes
n_components = 0.95
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index = df_train.index)
pca_df['preco'] = df_train['preco']

# ----------------------------  ------------------------------------------------------------

# Segunda camada de remoção de outliers (Dentro dos componentes principais)
lowQ = 0.00001
HighQ = 0.9999

quantile_low = pca_df.quantile(lowQ)
quantile_high = pca_df.quantile(HighQ)

# print('Análise de Quantil -- Segunda redução de outliers')
# print(f'Low quantil 2: \n {quantile_low} ; High quantil 2: \n {quantile_high}')


if(lowQ == 0 and HighQ == 1):
    pca_df = pca_df
else:
    for column in pca_df.columns.drop('preco'): 
        pca_df = pca_df[(pca_df[column] >= quantile_low[column]) & (pca_df[column] <= quantile_high[column])]

X_pca_df = pca_df.drop(columns=['preco']).to_numpy()
Y_pca_series = pca_df['preco']

# ----------------------------  ------------------------------------------------------------

# Dividir os dados em treino e validate
X_train_pca, X_validate_pca, y_train, y_validate = train_test_split(X_pca_df, Y_pca_series, test_size=0.1, random_state=42)

X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])

X_validate_pca_df = pd.DataFrame(X_validate_pca, columns=[f'PC{i+1}' for i in range(X_validate_pca.shape[1])])

X_test_pca = pca.transform(X_test_scaled)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

print(f"Number of components: {X_pca_df.shape[1]}")

# ----------------------------  ------------------------------------------------------------


# ----------------------------  ------------------------------------------------------------
# ----------------------------  ------------------------------------------------------------

# Treinamento e Predição

# Treinar modelo

# model = RandomForestRegressor(random_state=42,  n_estimators = 200, max_depth = 30, min_samples_split = 2,min_samples_leaf = 1)
# model = ExtraTreesRegressor(random_state=42 ,n_estimators =  100, min_samples_split = 4, min_samples_leaf =  5, max_depth =  40)
model = GradientBoostingRegressor(random_state=42, n_estimators = 500, max_depth = 20, min_samples_split = 20, min_samples_leaf = 32)

model.fit(X_train_pca, y_train)
# Prever e calcular o erro quadrático médio
y_pred = model.predict(X_validate_pca_df)
mse = mean_squared_error(y_validate, y_pred)
rmse = np.sqrt(mse)

# Calcular o erro percentual quadrático médio da raiz (RMSPE)
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

rmspe_value = rmspe(y_validate, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Root Mean Squared Percentage Error: {rmspe_value}")

# # Print some predictions
# for i in range(0,50):
#     # idx = np.random.randint(0, len(y_validate))
#     print(f'Preço real: {y_validate.values[i]}, Preço previsto: {y_pred[i]}')



# ----------------------------  ------------------------------------------------------------

# Validacao cruzada
cv = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = cross_val_score(model, X_train_pca, y_train, cv=cv, scoring='neg_mean_squared_error')
mean_cv_score = -cross_val_scores.mean()

print(f"Cross-validated Mean Squared Error: {mean_cv_score}")


# ----------------------------  ------------------------------------------------------------

# Treinar com 100% dos dados de treino, treinar o modelo e fazer previsões nos dados de teste.

# Fit the model again using 100% of the training data

model.fit(X_pca_df, Y_pca_series)

y_test_pred = model.predict(X_test_pca)

## save predictions

predictions_df = pd.DataFrame({ 'preco': y_test_pred , 'Id': df_test.index})  
# predictions_df.to_csv('predictions/GBR_tuned2_predicted_prices_pca2_outliers1_custom2_outliers2_1e-5_9999e-4_allIds.csv', index=False)


# ----------------------------  ------------------------------------------------------------

var1 = 'PC1'
var2 = 'PC3'
# Calcular o erro (residuals)
residuals = y_validate - y_pred

# Juntar os dois gráficos em uma figura
fig, axes = plt.subplots(2, 2, figsize=(20, 7))

# Determinar os limites dos eixos
x_lim = (min(X_train_pca_df[var1].min(), X_validate_pca_df[var1].min(), X_test_pca_df[var1].min()), 
         max(X_train_pca_df[var1].max(), X_validate_pca_df[var1].max(), X_test_pca_df[var1].max()))
y_lim = (min(X_train_pca_df[var2].min(), X_validate_pca_df[var2].min(), X_test_pca_df[var2].min()), 
         max(X_train_pca_df[var2].max(), X_validate_pca_df[var2].max(), X_test_pca_df[var2].max()))

# Aplicar os limites aos gráficos
for ax in axes.flatten()[:-1]:  # Exclude the last subplot (axes[1][1])
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

# Gráfico de Componente 1 vs Componente 2 com hue = Y
scatter1 = axes[0][0].scatter(X_train_pca_df[var1], X_train_pca_df[var2], c=y_train, cmap='viridis', alpha=0.5)
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


# Plotar o erro para cada aquisição no teste
axes[1][1].scatter(X_validate_pca_df[var1], residuals, alpha=0.5)
axes[1][1].axhline(y=0, color='r', linestyle='--')
axes[1][1].set_title('Erro para cada aquisição no teste (ExtraTrees)')
axes[1][1].set_xlabel('Índice')
axes[1][1].set_ylabel('Erro (Real - Previsto)')



plt.tight_layout()
plt.show()
