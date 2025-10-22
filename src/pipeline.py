import pandas as pd
from pathlib import Path

RAW = Path('data/raw/accidents_brazil.csv')
CLEAN = Path('data/processed/accidents_clean.csv')

def clean_data():
    df = pd.read_csv(RAW)
    print('Antes:', df.shape)

    df.columns = df.columns.str.lower().str.strip()

    # Ajuste fino: inclui 'uf' e 'municipio'
    col_estado = next((c for c in df.columns if 'estado' in c or 'uf' == c or 'state' in c), None)
    col_cidade = next((c for c in df.columns if 'municipio' in c or 'munic' in c or 'city' in c), None)
    col_data = next((c for c in df.columns if 'data' in c or 'date' in c), None)

    if not col_estado or not col_cidade:
        print('Erro: não encontrei colunas de estado (uf) ou cidade (municipio).')
        print('Colunas disponíveis:', df.columns.tolist())
        return

    df.dropna(subset=[col_estado, col_cidade], inplace=True)

    if col_data:
        df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
        df = df[df[col_data].notna()]

    df[col_estado] = df[col_estado].astype(str).str.upper().str.strip()
    df[col_cidade] = df[col_cidade].astype(str).str.title().str.strip()
    df.drop_duplicates(inplace=True)

    df.to_csv(CLEAN, index=False)
    print('Depois:', df.shape)
    print('Arquivo limpo salvo em:', CLEAN)

if __name__ == '__main__':
    clean_data()
