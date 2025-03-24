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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV

X_blc, y_blc = SMOTE().fit_resample(X_train_scaled, y_train)
xgb = XGBClassifier()

# Definir os grids de parâmetros para múltiplos modelos de classificação
param_grids = {
    'XGBClassifier': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10]
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'LogisticRegression': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['lbfgs', 'saga']
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10]
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'GaussianNB': {}
}

# Inicializar os modelos
models = {
    'XGBClassifier': XGBClassifier(random_state=42),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    # 'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    # 'SVC': SVC(random_state=42),
    # 'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    # 'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
    # 'KNeighborsClassifier': KNeighborsClassifier(),
    # 'GaussianNB': GaussianNB()
}

# DataFrame para armazenar os resultados
results_df = pd.DataFrame(columns=['Model', 'Best_Params', 'Accuracy', 'Cross_Val_Accuracy'])

# Realizar GridSearchCV para cada modelo
for model_name in models:
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    grid_search = RandomizedSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1, n_iter=50, random_state=42)
    grid_search.fit(X_blc, y_blc)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_validate_scaled)
    accuracy = (y_pred == y_validate).mean()
    cross_val_accuracy = cross_val_score(best_model, X_blc, y_blc, scoring='accuracy', cv=5).mean()
    
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Model': model_name,
        'Best_Params': best_params,
        'Accuracy': accuracy,
        'Cross_Val_Accuracy': cross_val_accuracy
    }])], ignore_index=True)

print(results_df)