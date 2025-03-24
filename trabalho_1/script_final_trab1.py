## Importação e Pré-Processamento
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

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




### Importação
# Read the CSV file
df_train = pd.read_csv('dados_trabalho1/conjunto_de_treinamento.csv')
# Read the test CSV file
df_test = pd.read_csv('dados_trabalho1/conjunto_de_teste.csv')

# df_train.describe()

### Verificação manual de atributos + Codificação de atributos não numéricos


testIDs = df_test['id_solicitante' ]

excluded_columns = ['id_solicitante' ]

df_train = df_train.drop(excluded_columns, axis=1)
df_test = df_test.drop(excluded_columns, axis=1)


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


### Verificação de colunas e linhas com null / NaN

# Criar um DataFrame sem linhas com valores nulos (excluindo duas colunas com muitos valores nulos para evitar perda excessiva de dados --> mais de 12 mil)
null_columns_to_drop = ['profissao_companheiro','grau_instrucao_companheiro']
df_no_null_rows = df_train_encoded.drop(columns=null_columns_to_drop).dropna()

df_test_no_null_rows = df_test_encoded.drop(columns=null_columns_to_drop)
# df_test_no_null_rows = df_test_encoded.drop(columns=null_columns_to_drop).dropna()

## É preciso lidar com valores nulos nos atributos dos dados de teste --> Usarei a mediana dos valores para preencher.

# Verificar quais colunas têm valores nulos
null_columns_test = df_test_encoded.isnull().sum()
null_columns_test = null_columns_test[null_columns_test > 0]

if (not null_columns_test.empty):
    # Preencher valores nulos com a mediana das colunas
    df_test_no_null_rows.fillna(df_test_encoded.median(), inplace=True)

### Selecionar dataSet
usingData = df_no_null_rows
usingTestData = df_test_no_null_rows

### Discretizar valores float.
for colname in usingData.select_dtypes("float"):
    usingData[colname] = usingData[colname].astype(int)
    usingTestData[colname] = usingTestData[colname].astype(int)    

### Dividindo e Normalizando
from sklearn.model_selection import train_test_split

X = usingData.drop('inadimplente', axis = 1)
y = usingData['inadimplente']

# Divisão em treino e validação
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


X_train_scaled= scaler.fit_transform(X_train)
X_validate_scaled = scaler.transform(X_validate)

# print("Normalized input data(X):\n", X_train_scaled)

## Treino e predicao usando:
#### Using SMOTE with XGBoost Classifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

X_blc, y_blc = SMOTE().fit_resample(X_train_scaled, y_train)
# xgb = XGBClassifier()

bestXgbModel = XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.01, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=4, max_leaves=None,
              min_child_weight=None, monotone_constraints=None,
              multi_strategy=None, n_estimators=850, n_jobs=None,
              num_parallel_tree=None, random_state=None)


bestXgbModel.fit(X_blc, y_blc)

y_pred = bestXgbModel.predict(X_validate_scaled)

get_confusion_matrix(y_validate,y_pred)

# Perform cross-validation
cv_scores = cross_val_score(bestXgbModel, X_blc, y_blc, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: ", np.mean(cv_scores))



# test data

# Refit the model using the entire dataset
X_full = usingData.drop('inadimplente', axis=1)
y_full = usingData['inadimplente']

# Scale the full dataset
X_full_scaled = scaler.fit_transform(X_full)

# Balance the full dataset using SMOTE
X_full_blc, y_full_blc = SMOTE().fit_resample(X_full_scaled, y_full)

# Refit the model
bestXgbModel.fit(X_full_blc, y_full_blc)

X_test = usingTestData

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Predict using the trained model
y_test_pred = bestXgbModel.predict(X_test_scaled)

# Add predictions to the test dataframe with the original 'id_solicitante' column
df_test['inadimplente_pred'] = y_test_pred
df_test['id_solicitante'] = testIDs

# Create a new dataframe with 'id_solicitante' and 'inadimplente_pred'
df_predictions = df_test[['id_solicitante', 'inadimplente_pred']]

# Rename the columns to match the required header
df_predictions.columns = ['id_solicitante', 'inadimplente']

# # Save the predictions to a CSV file
# df_predictions.to_csv('predictions7.csv', index=False)


