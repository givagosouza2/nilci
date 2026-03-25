import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu
from statsmodels.stats.anova import anova_lm


st.set_page_config(page_title="Análise do efeito da idade", layout="wide")

st.title("Análise do efeito da idade sobre um parâmetro de desempenho")

st.markdown(
    '''
Carregue um arquivo **CSV com cabeçalho**, contendo:
- uma coluna com **idade**
- uma coluna com o **parâmetro de desempenho**
'''
)

uploaded_file = st.file_uploader("Carregue o arquivo CSV", type=["csv"])


def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)


def format_p(p):
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Não foi possível ler o arquivo: {e}")
        st.stop()

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df_raw.head())

    if df_raw.shape[1] < 2:
        st.error("O arquivo deve conter pelo menos duas colunas com cabeçalho.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        coluna_idade = st.selectbox("Selecione a coluna de idade", df_raw.columns)

    with col2:
        opcoes_desempenho = [c for c in df_raw.columns if c != coluna_idade]
        coluna_desempenho = st.selectbox(
            "Selecione a coluna do parâmetro de desempenho",
            opcoes_desempenho
        )

    nome_parametro = st.text_input(
        "Nome para exibição do parâmetro",
        value=coluna_desempenho
    )

    df = df_raw[[coluna_idade, coluna_desempenho]].copy()
    df.columns = ["idade", "desempenho"]

    df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    df["desempenho"] = pd.to_numeric(df["desempenho"], errors="coerce")
    df = df.dropna(subset=["idade", "desempenho"]).copy()

    st.success(f"Foram carregadas {len(df)} observações válidas.")

    st.header("Estatísticas descritivas")

    q1 = df["desempenho"].quantile(0.25)
    q3 = df["desempenho"].quantile(0.75)
    iqr_val = q3 - q1

    st.write(f"Mediana: {df['desempenho'].median():.4f}")
    st.write(f"IQR: {iqr_val:.4f}")

    pearson_r, pearson_p = pearsonr(df["idade"], df["desempenho"])
    spearman_rho, spearman_p = spearmanr(df["idade"], df["desempenho"])

    st.header("Correlação")
    st.write(f"Pearson r = {pearson_r:.4f}, p = {format_p(pearson_p)}")
    st.write(f"Spearman rho = {spearman_rho:.4f}, p = {format_p(spearman_p)}")

    X = sm.add_constant(df["idade"])
    model = sm.OLS(df["desempenho"], X).fit()

    st.header("Regressão linear")
    st.write(model.summary())

    fig, ax = plt.subplots()
    ax.scatter(df["idade"], df["desempenho"])
    ax.set_xlabel("Idade")
    ax.set_ylabel(nome_parametro)
    st.pyplot(fig)

else:
    st.info("Aguardando arquivo CSV")
