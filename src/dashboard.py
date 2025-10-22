import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Acidentes de Trânsito no Brasil', layout='wide')

DATA_PATH = 'data/processed/accidents_clean.csv'
MODEL_PATH = 'src/model.joblib'

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    # datas
    if 'data_inversa' in df.columns:
        df['data_inversa'] = pd.to_datetime(df['data_inversa'], errors='coerce')
        df['ano'] = df['data_inversa'].dt.year
        df['mes'] = df['data_inversa'].dt.to_period('M').astype(str)
    # texto padronizado
    if 'uf' in df.columns:
        df['uf'] = df['uf'].astype(str).str.upper().str.strip()
    for c in ['tipo_pista', 'fase_dia', 'condicao_metereologica', 'tipo_acidente', 'classificacao_acidente', 'municipio']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# ---------- Sidebar filtros globais ----------
st.sidebar.header('Filtros')
ufs = sorted(df['uf'].dropna().unique()) if 'uf' in df.columns else []
uf_sel = st.sidebar.multiselect('Estado (UF)', ufs, default=ufs[:5] if ufs else [])

tipo_pista_vals = sorted(df['tipo_pista'].dropna().unique()) if 'tipo_pista' in df.columns else []
tp_sel = st.sidebar.multiselect('Tipo de Pista', tipo_pista_vals, default=tipo_pista_vals[:3] if tipo_pista_vals else [])

fase_vals = sorted(df['fase_dia'].dropna().unique()) if 'fase_dia' in df.columns else []
fase_sel = st.sidebar.multiselect('Fase do Dia', fase_vals, default=fase_vals[:3] if fase_vals else [])

cond_vals = sorted(df['condicao_metereologica'].dropna().unique()) if 'condicao_metereologica' in df.columns else []
cond_sel = st.sidebar.multiselect('Condição Meteorológica', cond_vals, default=cond_vals[:3] if cond_vals else [])

# aplica filtros base
df_f = df.copy()
if uf_sel:
    df_f = df_f[df_f['uf'].isin(uf_sel)]
if tp_sel:
    df_f = df_f[df_f['tipo_pista'].isin(tp_sel)]
if fase_sel:
    df_f = df_f[df_f['fase_dia'].isin(fase_sel)]
if cond_sel:
    df_f = df_f[df_f['condicao_metereologica'].isin(cond_sel)]

st.title('Predição de Acidentes de Trânsito no Brasil')
st.caption('Explore, visualize e teste o modelo preditivo treinado em dados nacionais.')

tabs = st.tabs(['Visão Geral', 'Mapa', 'Séries Temporais', 'Predição', 'Sobre'])

# ---------- Visão Geral ----------
with tabs[0]:
    st.subheader('Resumo dos Dados (após filtros)')
    c1, c2, c3, c4 = st.columns(4)
    total = len(df_f)
    ano_min = int(df_f['ano'].min()) if 'ano' in df_f.columns and total else None
    ano_max = int(df_f['ano'].max()) if 'ano' in df_f.columns and total else None
    c1.metric('Registros', f'{total:,}'.replace(',', '.'))
    c2.metric('Estados', df_f['uf'].nunique() if 'uf' in df_f.columns else 0)
    c3.metric('Municípios', df_f['municipio'].nunique() if 'municipio' in df_f.columns else 0)
    c4.metric('Período', f'{ano_min}–{ano_max}' if ano_min and ano_max else 'n/d')

    st.divider()
    cA, cB = st.columns([2, 3])

    with cA:
        if 'classificacao_acidente' in df_f.columns:
            vc = df_f['classificacao_acidente'].value_counts().reset_index()
            vc.columns = ['classificacao_acidente', 'contagem']
            fig = px.bar(vc, x='classificacao_acidente', y='contagem', title='Distribuição por Classificação')
            st.plotly_chart(fig, use_container_width=True)
        if 'tipo_acidente' in df_f.columns:
            top = df_f['tipo_acidente'].value_counts().head(10).reset_index()
            top.columns = ['tipo_acidente', 'contagem']
            fig2 = px.bar(top, x='tipo_acidente', y='contagem', title='Top 10 Tipos de Acidente')
            st.plotly_chart(fig2, use_container_width=True)

    with cB:
        if 'uf' in df_f.columns:
            uf_rank = df_f['uf'].value_counts().reset_index()
            uf_rank.columns = ['uf', 'contagem']
            fig3 = px.bar(uf_rank, x='uf', y='contagem', title='Ocorrências por UF')
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander('Amostra do DataFrame'):
        st.dataframe(df_f.head(50), use_container_width=True)

# ---------- Mapa ----------
with tabs[1]:
    st.subheader('Mapa de Acidentes')
    if 'latitude' in df_f.columns and 'longitude' in df_f.columns:
        max_pts = st.slider('Quantidade de pontos no mapa (amostra aleatória)', 1000, 20000, 5000, step=1000)
        df_map = df_f[['latitude', 'longitude', 'uf', 'tipo_acidente', 'classificacao_acidente']].dropna()
        if len(df_map) > max_pts:
            df_map = df_map.sample(max_pts, random_state=42)
        figm = px.scatter_mapbox(
            df_map,
            lat='latitude',
            lon='longitude',
            color='classificacao_acidente' if 'classificacao_acidente' in df_map.columns else None,
            hover_data=['uf', 'tipo_acidente'],
            zoom=3,
            height=650
        )
        figm.update_layout(mapbox_style='open-street-map', margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(figm, use_container_width=True)
    else:
        st.info('Colunas latitude/longitude não encontradas no dataset filtrado.')

# ---------- Séries Temporais ----------
with tabs[2]:
    st.subheader('Evolução Temporal')
    if 'mes' in df_f.columns:
        serie = df_f.groupby('mes').size().reset_index(name='ocorrencias').sort_values('mes')
        figt = px.line(serie, x='mes', y='ocorrencias', markers=True, title='Acidentes por Mês')
        st.plotly_chart(figt, use_container_width=True)
    elif 'ano' in df_f.columns:
        serie = df_f.groupby('ano').size().reset_index(name='ocorrencias').sort_values('ano')
        figt = px.line(serie, x='ano', y='ocorrencias', markers=True, title='Acidentes por Ano')
        st.plotly_chart(figt, use_container_width=True)
    else:
        st.info('Coluna de data não disponível para série temporal.')

# ---------- Predição ----------
with tabs[3]:
    st.subheader('Simular Predição de Gravidade')
    if model is None:
        st.warning('Modelo não encontrado ou incompatível. Treine e salve em src/model.joblib.')
    else:
        c1, c2, c3 = st.columns(3)
        c4, c5 = st.columns(2)

        def pick(col):
            vals = sorted(df[col].dropna().unique()) if col in df.columns else []
            return vals[0] if vals else None, vals

        _, uf_vals = pick('uf')
        _, tp_vals = pick('tipo_pista')
        _, fase_vals = pick('fase_dia')
        _, cond_vals = pick('condicao_metereologica')
        _, tipo_vals = pick('tipo_acidente')

        uf_in = c1.selectbox('UF', uf_vals)
        tp_in = c2.selectbox('Tipo de Pista', tp_vals)
        fase_in = c3.selectbox('Fase do Dia', fase_vals)
        cond_in = c4.selectbox('Condição Meteorológica', cond_vals)
        tipo_in = c5.selectbox('Tipo de Acidente', tipo_vals)

        if st.button('Prever'):
            entrada = pd.DataFrame([{
                'uf': uf_in,
                'tipo_pista': tp_in,
                'fase_dia': fase_in,
                'condicao_metereologica': cond_in,
                'tipo_acidente': tipo_in
            }])
            try:
                pred = model.predict(entrada)[0]
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(entrada)
                    prob_dict = {cls: float(proba[0][i]) for i, cls in enumerate(model.classes_)}
                else:
                    prob_dict = {}
                cL, cR = st.columns([1, 2])
                cL.metric('Classificação Prevista', str(pred))
                if prob_dict:
                    prob_df = pd.DataFrame({
                        'classe': list(prob_dict.keys()),
                        'probabilidade': [round(v, 4) for v in prob_dict.values()]
                    }).sort_values('probabilidade', ascending=False)
                    cR.write(prob_df)
            except Exception as e:
                st.error(f'Falha na predição: {e}')

# ---------- Sobre ----------
with tabs[4]:
    st.subheader('Sobre o Projeto')
    st.markdown(
        '''
        Projeto de análise e predição de acidentes de trânsito no Brasil (2017–2023).
        Pipeline: coleta, limpeza, EDA, modelo de classificação e dashboard interativo.
        Principais colunas usadas: uf, tipo_pista, fase_dia, condicao_metereologica, tipo_acidente, classificacao_acidente, latitude, longitude, data_inversa.
        '''
    )
    st.markdown('Como executar:')
    st.code('streamlit run src/dashboard.py', language='bash')
