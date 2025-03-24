import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBClassifier
import seaborn as sns

def get_confusion_matrix(y_validate,y_pred):
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_validate, y_pred)



    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greys', cbar=False)
    plt.title('Confusion Matrix for XGBoostClassifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    from sklearn.metrics import classification_report

    classification_rep = classification_report(y_validate, y_pred)
    print('\nClassification Report for XGBoostClassifier:')
    print(classification_rep)



# Carregar o dataframe
df_train = pd.read_csv('dados_trabalho1/conjunto_de_treinamento.csv')
df_test = pd.read_csv('dados_trabalho1/conjunto_de_teste.csv')

# ----------------------------  ------------------------------------------------------------
# ----------------------------  ------------------------------------------------------------


# Pré-processamento

# 1. Retirada de ID
df_train.index = df_train['id_solicitante']
df_train = df_train.drop(columns=['id_solicitante'])
df_test.index = df_test['id_solicitante']
df_test = df_test.drop(columns=['id_solicitante'])

alvo = 'inadimplente'
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

### Verificação de colunas e linhas com null / NaN

# alterar DataFrame --> sem linhas com valores nulos (excluindo duas colunas com muitos valores nulos para evitar perda excessiva de dados --> mais de 12 mil)
null_columns_to_drop = ['profissao_companheiro','grau_instrucao_companheiro']
df_train = df_train.drop(columns=null_columns_to_drop).dropna()

df_test = df_test.drop(columns=null_columns_to_drop)


## É preciso lidar com valores nulos nos atributos dos dados de teste --> Usarei a mediana dos valores para preencher.

# Verificar quais colunas têm valores nulos
null_columns_test = df_test.isnull().sum()
null_columns_test = null_columns_test[null_columns_test > 0]

if (not null_columns_test.empty):
    # Preencher valores nulos com a mediana das colunas
    df_test.fillna(df_test.median(), inplace=True)


# ----------------------------  ------------------------------------------------------------
# A partir de uma avaliação visual dos gráficos de densidade, seleciona-se algumas variáveis que apresentam uma separação um pouco mais clara entre os adimplentes e inadimplentes.

# predictors = ['idade', 'valor_patrimonio_pessoal', 'renda_extra', 'produto_solicitado', 'possui_telefone_residencial', 'possui_outros_cartoes', 'mastercard', 'possui_carro', 'meses_no_trabalho', 'estado_civil', 'dia_de_vencimento', 'codigo_area_telefone_residencial', 'ocupacao']

# ----------------------------  ------------------------------------------------------------

# Normalizar os dados 
scaler = StandardScaler()
# scaler_test = StandardScaler()
predictors = df_train.drop(columns=[alvo]).columns
df_train[predictors] = scaler.fit_transform(df_train[predictors])
df_test[predictors] = scaler.transform(df_test[predictors])
X_scaled = df_train[predictors].to_numpy()
X_test_scaled = df_test[predictors].to_numpy()

# ----------------------------  ------------------------------------------------------------

# Aplicar PCA

# Fixar número de componentes
n_components = 0.95
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index = df_train.index)
pca_df[alvo] = df_train[alvo]

# ----------------------------  ------------------------------------------------------------

# Segunda camada de remoção de outliers (Dentro dos componentes principais)
lowQ = 0# 0.00000001 # 
HighQ = 1# 0.9999999 # 

quantile_low = pca_df.quantile(lowQ)
quantile_high = pca_df.quantile(HighQ)

# print('Análise de Quantil -- Segunda redução de outliers')
# print(f'Low quantil 2: \n {quantile_low} ; High quantil 2: \n {quantile_high}')


if(lowQ == 0 and HighQ == 1):
    pca_df = pca_df
else:
    for column in pca_df.columns.drop(alvo): 
        pca_df = pca_df[(pca_df[column] >= quantile_low[column]) & (pca_df[column] <= quantile_high[column])]

X_pca_df = pca_df.drop(columns=[alvo]).to_numpy()
Y_pca_series = pca_df[alvo]

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

# model = XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.01, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=4, max_leaves=None,
#               min_child_weight=None, monotone_constraints=None,
#               multi_strategy=None, n_estimators=850, n_jobs=None,
#               num_parallel_tree=None, random_state=None)



model = XGBClassifier(
    colsample_bytree=0.6480869299533999,
    gamma=0.49887024252447093,
    learning_rate=0.09003430428258549,
    max_depth=4,
    min_child_weight=2,
    n_estimators=319,
    subsample=0.5514787512499894,
    random_state=42
)


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

predictions_df = pd.DataFrame({ alvo: y_test_pred , 'id_solicitante': df_test.index})  
#predictions_df.to_csv('predictions5.csv', index=False)


# ----------------------------  ------------------------------------------------------------

# var1 = 'PC1'
# var2 = 'PC2'
# # Calcular o erro (residuals)
# residuals = y_validate - y_pred

# # Juntar os dois gráficos em uma figura
# fig, axes = plt.subplots(2, 2, figsize=(20, 7))

# # Determinar os limites dos eixos
# x_lim = (min(X_train_pca_df[var1].min(), X_validate_pca_df[var1].min(), X_test_pca_df[var1].min()), 
#          max(X_train_pca_df[var1].max(), X_validate_pca_df[var1].max(), X_test_pca_df[var1].max()))
# y_lim = (min(X_train_pca_df[var2].min(), X_validate_pca_df[var2].min(), X_test_pca_df[var2].min()), 
#          max(X_train_pca_df[var2].max(), X_validate_pca_df[var2].max(), X_test_pca_df[var2].max()))

# # Aplicar os limites aos gráficos
# for ax in axes.flatten()[:-1]:  # Exclude the last subplot (axes[1][1])
#     ax.set_xlim(x_lim)
#     ax.set_ylim(y_lim)

# # Gráfico de Componente 1 vs Componente 2 com hue = Y
# scatter1 = axes[0][0].scatter(X_train_pca_df[var1], X_train_pca_df[var2], c=y_train, cmap='viridis', alpha=0.5)
# axes[0][0].set_xlabel(var1)
# axes[0][0].set_ylabel(var2)
# axes[0][0].set_title('Gráfico de ' + var1 + ' vs ' + var2 + ' com hue = inadimplente')
# fig.colorbar(scatter1, ax=axes[0][0], label='inadimplente')

# # Gráfico de var1 vs var2 com hue = Erro (y_validate - y_pred)
# scatter2 = axes[0][1].scatter(X_validate_pca_df[var1], X_validate_pca_df[var2], c=residuals, cmap='coolwarm', alpha=0.5)
# axes[0][1].set_xlabel(var1)
# axes[0][1].set_ylabel(var2)
# axes[0][1].set_title('Gráfico de ' + var1 + ' vs ' + var2 + ' com hue = Erro (y_validate - y_pred)')
# fig.colorbar(scatter2, ax=axes[0][1], label='Erro')

# # # Gráfico de var1 vs var2 com hue = Erro (y_validate - y_pred)
# # scatter2 = axes[0][1].scatter(y_validate, y_pred, c=residuals, cmap='coolwarm', alpha=0.5)
# # axes[0][1].set_xlabel(var1)
# # axes[0][1].set_ylabel(var2)
# # axes[0][1].set_title('Gráfico de y_validate vs y_pred com hue = Erro (y_validate - y_pred)')
# # fig.colorbar(scatter2, ax=axes[0][1], label='Erro')

# # Gráfico de var1 vs var2 para X_test_pca_df
# scatter3 = axes[1][0].scatter(X_test_pca_df[var1], X_test_pca_df[var2], c=y_test_pred, cmap='viridis', alpha=0.5)
# axes[1][0].set_xlabel(var1)
# axes[1][0].set_ylabel(var2)
# axes[1][0].set_title('Gráfico de ' + var1 + ' vs ' + var2 + ' para X_test_pca_df')
# fig.colorbar(scatter3, ax=axes[1][0], label='Preço')


# # Plotar o erro para cada aquisição no teste
# axes[1][1].scatter(X_validate_pca_df[var1], residuals, alpha=0.5)
# axes[1][1].axhline(y=0, color='r', linestyle='--')
# axes[1][1].set_title('Erro para cada aquisição no teste')
# axes[1][1].set_xlabel('Índice')
# axes[1][1].set_ylabel('Erro (Real - Previsto)')



# plt.tight_layout()
# plt.show()



get_confusion_matrix(y_validate,y_pred)