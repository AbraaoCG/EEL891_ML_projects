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
from sklearn.model_selection import RandomizedSearchCV

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

# # Remover outliers

# initHigh = 0.99999
# initLow = 0.000001

# columns_to_check = df_train.columns.to_list()

# quantile_ranges = {col: [initLow,initHigh] for col in columns_to_check}



# print('Análise de Quantil -- Primeira redução de outliers')
# for column in columns_to_check: 
#     Q1 = df_train[column].quantile(quantile_ranges[column][0])
#     Q3 = df_train[column].quantile(quantile_ranges[column][1])
#     print(f'{column} - Low quantil 1: {Q1} ; High quantil 1: {Q3}')
#     df_train = df_train[(df_train[column] >= Q1) & (df_train[column] <= Q3)]



# Normalizar os dados 
scaler = StandardScaler()
# scaler_test = StandardScaler()
predictors = df_train.drop(columns=[alvo]).columns
df_train[predictors] = scaler.fit_transform(df_train[predictors])
df_test[predictors] = scaler.transform(df_test[predictors])
X_scaled = df_train[predictors].to_numpy()
X_test_scaled = df_test[predictors].to_numpy()

# ----------------------------  ------------------------------------------------------------

## Encontrar variáveis de maior correlação

# corr_scaled = df_train.corr()

# # Identificar colunas com correlação maior que k
# k = 0.5
# high_correlation_columns = corr_scaled.columns[corr_scaled[alvo] > k].drop(alvo).tolist()

# # Criar um novo DataFrame apenas com essas colunas
# X_scaled = df_train[high_correlation_columns].to_numpy()
# X_test_scaled = df_test[high_correlation_columns].to_numpy()

# print(len(df_train),len(X_scaled))

# print(f'Colunas com alta correlação: {high_correlation_columns}')


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
lowQ = 0.00001 # 0# 
HighQ = 0.9999 # 1# 

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

# Definir a grade de parâmetros para o RandomizedSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'min_child_weight': [1, 2, 3, 4, 5],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# Inicializar o modelo
model = XGBClassifier()

# Configurar o RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100, scoring='neg_mean_squared_error', cv=3, verbose=1, random_state=42, n_jobs=-1)

# Ajustar o modelo
random_search.fit(X_train_pca, y_train)

# Melhor estimador
best_model = random_search.best_estimator_

# Treinar o melhor modelo
best_model.fit(X_train_pca, y_train)

# Prever e calcular o erro quadrático médio
y_pred = best_model.predict(X_validate_pca_df)
mse = mean_squared_error(y_validate, y_pred)
rmse = np.sqrt(mse)

# Calcular o erro percentual quadrático médio da raiz (RMSPE)
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

rmspe_value = rmspe(y_validate, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Root Mean Squared Percentage Error: {rmspe_value}")

# Validacao cruzada
cv = KFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = cross_val_score(best_model, X_train_pca, y_train, cv=cv, scoring='neg_mean_squared_error')
mean_cv_score = -cross_val_scores.mean()

print(f"Cross-validated Mean Squared Error: {mean_cv_score}")
