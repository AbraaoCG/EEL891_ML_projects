import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor, RANSACRegressor

# Função para calcular RMSPE
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


# Read the CSV file
df_train = pd.read_csv('dados_trab2/conjunto_de_treinamento.csv')
df_test = pd.read_csv('dados_trab2/conjunto_de_teste.csv')


excluded_columns = ['Id' ]
df_train = df_train.drop(excluded_columns, axis=1)

test_ids = df_test['Id']
df_test = df_test.drop(excluded_columns, axis=1)




# Criar uma cópia dos dataframes originais
df_train_encoded = df_train.copy()
df_test_encoded = df_test.copy()

# Identificar colunas do tipo String
string_columns = df_train.select_dtypes(include=['object']).columns

# Inicializar o LabelEncoder
le = LabelEncoder()

# Codificar colunas do tipo String no dataframe de treino
for col in string_columns:
    df_train_encoded[col] = le.fit_transform(df_train_encoded[col])

# Codificar colunas do tipo String no dataframe de teste
for col in string_columns:
    df_test_encoded[col] = le.fit_transform(df_test_encoded[col])


usingData = df_train_encoded
usingTestData = df_test_encoded

## TESTAR O MODELO COM DIFERENTES VALORES de delimitação de outliers;

initHigh = 0.9
initLow = 0.0
# Calcular o IQR para as colunas especificadas
columns_to_check = usingData.columns.drop('preco').to_list()
quantile_ranges = {col: [initLow,initHigh] for col in columns_to_check}


usingData2_no_outlier = usingData.copy()
for col in columns_to_check:# [selectColumnVisualize]:
    qLow, qHigh = quantile_ranges[col]
    Q1 = usingData[col].quantile(qLow)
    Q3 = usingData[col].quantile(qHigh)
    IQR = Q3 - Q1

    fac = 1.5
    usingData2_no_outlier = usingData2_no_outlier[~((usingData2_no_outlier[col] > (Q3 + fac * IQR)) )]


# Verificar colunas com apenas um valor único
unique_value_columns = usingData2_no_outlier.columns[usingData2_no_outlier.nunique() == 1]

# Remover essas colunas
usingData2_no_outlier = usingData2_no_outlier.drop(unique_value_columns, axis=1)

# print(f'Colunas removidas: {unique_value_columns.tolist()}')
# print(f'Colunas restantes: {usingData2_no_outlier.columns.tolist()}')

if (0):
    usingData = usingData2_no_outlier.copy()


X = usingData.drop('preco', axis = 1)
y = usingData['preco']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Criar um scorer para RMSPE
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

# Inicializar o modelo de Regressão Linear
model = RANSACRegressor()

# Realizar a validação cruzada com 5 folds
scores = cross_val_score(model, X_train, y_train, cv=5, scoring=rmspe_scorer)

# Calcular a média e o desvio padrão do RMSPE
mean_rmspe = -scores.mean()
std_rmspe = scores.std()

print(f'Mean RMSPE: {mean_rmspe}')
print(f'Standard Deviation of RMSPE: {std_rmspe}')

# Treinar o modelo nos dados de treino completos
model.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test)

# Calcular o RMSPE nos dados de teste
test_rmspe = rmspe(y_test, y_pred)
print(f'Test RMSPE: {test_rmspe}')

print(y_pred[:10])
    
print(y_test[:10])



if (0):
    # Selecionar apenas as colunas do usingData final
    usingTestData_final = usingTestData[usingData.columns.drop('preco')]

    # Fazer previsões nos dados de teste
    y_test_pred = model.predict(usingTestData_final)

    # Criar um DataFrame com os IDs e os preços previstos
    df_predictions = pd.DataFrame({'Id': test_ids, 'preco': y_test_pred})

    # Salvar o DataFrame em um arquivo CSV
    df_predictions.to_csv('MLP_predicted_prices5.csv', index=False)


