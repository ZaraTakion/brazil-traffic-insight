# 🌌 Brazil Traffic Insight  
**⚙️ Data Science Saga: Brasil sob o olhar dos dados (2017–2023)**  

```
██████╗ ██████╗  █████╗ ███████╗██╗██╗     ███████╗    ████████╗██████╗  █████╗ ███████╗███████╗██╗ ██████╗██╗  ██╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝██║██║     ██╔════╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██║██╔════╝██║ ██╔╝
██████╔╝██████╔╝███████║███████╗██║██║     █████╗         ██║   ██████╔╝███████║███████╗███████╗██║██║     █████╔╝ 
██╔══██╗██╔══██╗██╔══██║╚════██║██║██║     ██╔══╝         ██║   ██╔══██╗██╔══██║╚════██║╚════██║██║██║     ██╔═██╗ 
██████╔╝██║  ██║██║  ██║███████║██║███████╗███████╗       ██║   ██║  ██║██║  ██║███████║███████║██║╚██████╗██║  ██╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝╚═╝  ╚═╝
```

> _“Entre números e caos, a verdade sobre o trânsito brasileiro emerge.”_

---

## 🎌 Sinopse
Um projeto em estilo **mangá‑científico**, onde cada capítulo revela uma camada da história escondida nos dados.  
De **milhares de acidentes** surgem padrões, previsões e visualizações que transformam estatísticas em narrativa visual.

---

## 📖 Capítulos da Obra
1. **Capítulo I – O Despertar dos Dados**: limpeza e padronização do dataset.  
2. **Capítulo II – Olhos da Análise**: exploração visual com gráficos e correlações.  
3. **Capítulo III – A Árvore das Decisões**: modelagem preditiva (Random Forest).  
4. **Capítulo IV – O Portal Streamlit**: dashboard interativo e responsivo.  
5. **Capítulo V – O Futuro**: roadmap e evolução do projeto.

---

## ⚙️ Tecnologias que dão poder ao herói

| Tipo | Ferramentas |
|------|--------------|
| Linguagem | Python 3.10+ |
| Dados | Pandas, NumPy |
| Visualização | Plotly, Seaborn, Matplotlib |
| Machine Learning | Scikit‑learn, Joblib |
| Interface | Streamlit |
| Geodados | Folium, Plotly Mapbox |
| Dataset | [Kaggle – Car Accidents in Brazil (2017‑2023)](https://www.kaggle.com/datasets/mlippo/car-accidents-in-brazil-2017-2023) |

---

## 🗂 Estrutura do Arco Principal

```
brazil-traffic-insight/
│
├── data/
│   ├── raw/               # Dados brutos
│   └── processed/         # Dados tratados
│
├── notebooks/
│   ├── 01_exploracao_inicial.ipynb
│   └── 02_modelagem.ipynb
│
├── src/
│   ├── pipeline.py        # Purificação dos dados
│   ├── model.py           # Treinamento do modelo
│   └── dashboard.py       # Interface interativa
│
├── assets/                # Prints, mapas e ícones
├── requirements.txt
└── README.md
```

---

## ⚔️ Como invocar o projeto

```bash
git clone https://github.com/ZaraTakion/brazil-traffic-insight.git
cd brazil-traffic-insight
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/dashboard.py
```

Abra o portal: [http://localhost:8501](http://localhost:8501)

---

## 🧩 Habilidades desbloqueadas

🧠 **Predição de Gravidade** – insira variáveis e veja o modelo responder.  
🌦 **Filtros Dinâmicos** – UF, tipo de pista, fase do dia e condição climática.  
🗺 **Mapa Interativo** – visualize a dispersão dos acidentes.  
📈 **Séries Temporais** – observe o fluxo dos anos.  
🎛 **Interface Responsiva** – controle total, leveza total.

---

## 🧮 A Árvore das Decisões

| Item | Valor |
|------|-------|
| Algoritmo | Random Forest Classifier |
| Entradas | UF, tipo_pista, fase_dia, condição_metereológica, tipo_acidente |
| Saída | classificação_acidente |
| Métricas | Acurácia / F1‑Score |

> _Os arquivos `accidents_clean.csv` e `model.joblib` foram banidos do reino por serem pesados demais (> 100 MB).  
> Podem ser regenerados localmente via `pipeline.py` e `model.py`._

---

## 🌠 Roadmap dos Próximos Episódios
- [ ] Deploy no Streamlit Cloud / Hugging Face Spaces  
- [ ] Integração com APIs climáticas  
- [ ] Modo noturno com tema cyberpunk  
- [ ] Comparativo entre modelos (XGBoost, CatBoost)  

---

## 🎭 Autor

**Rodrigo A. Maciel Pinheiro (Zara Takion)**  
Estudante de Sistemas para Internet | Ciência de Dados & Web Design  
[GitHub](https://github.com/ZaraTakion) • [LinkedIn](https://www.linkedin.com)

> _“Entre o código e o caos, há sempre um padrão esperando ser revelado.”_
