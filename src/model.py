import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_model():
    df = pd.read_csv('data/processed/accidents_clean.csv')

    # Selecionar colunas relevantes
    features = ['uf', 'tipo_pista', 'fase_dia', 'condicao_metereologica', 'tipo_acidente']
    target = 'classificacao_acidente'

    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    # Separar treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pré-processamento (OneHotEncoding)
    preprocessor = ColumnTransformer([
        ('encoder', OneHotEncoder(handle_unknown='ignore'), features)
    ])

    # Modelo base
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Treinamento
    pipeline.fit(X_train, y_train)

    # Avaliação
    y_pred = pipeline.predict(X_test)
    print('Relatório de Classificação:')
    print(classification_report(y_test, y_pred))
    print('Matriz de Confusão:')
    print(confusion_matrix(y_test, y_pred))

    # Exportar modelo
    joblib.dump(pipeline, 'src/model.joblib')
    print('✅ Modelo salvo em src/model.joblib')

if __name__ == '__main__':
    train_model()
