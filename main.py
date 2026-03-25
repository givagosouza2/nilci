import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm

from scipy.stats import pearsonr, spearmanr, kruskal, rankdata
from statsmodels.stats.multitest import multipletests

st.set_page_config(layout="wide")
st.title("Análise do efeito da idade (versão artigo)")

# =========================
# FUNÇÕES
# =========================
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

def cliffs_delta(x, y):
    n_x = len(x)
    n_y = len(y)
    gt = sum(i > j for i in x for j in y)
    lt = sum(i < j for i in x for j in y)
    return (gt - lt) / (n_x * n_y)

def interpret_cliff(delta):
    d = abs(delta)
    if d < 0.147:
        return "negligível"
    elif d < 0.33:
        return "pequeno"
    elif d < 0.474:
        return "moderado"
    else:
        return "grande"

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Carregue CSV com cabeçalho", type="csv")

if file:
    df = pd.read_csv(file)

    col_idade = st.selectbox("Coluna idade", df.columns)
    col_var = st.selectbox("Coluna desempenho", [c for c in df.columns if c != col_idade])

    df = df[[col_idade, col_var]].copy()
    df.columns = ["idade", "y"]

    df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()

    st.success(f"N = {len(df)}")

    # =========================
    # CORRELAÇÃO
    # =========================
    st.header("Correlação")

    r_p, p_p = pearsonr(df["idade"], df["y"])
    r_s, p_s = spearmanr(df["idade"], df["y"])

    st.write(f"Pearson: r = {r_p:.3f}, p = {p_p:.4f}")
    st.write(f"Spearman: rho = {r_s:.3f}, p = {p_s:.4f}")

    # =========================
    # REGRESSÃO
    # =========================
    st.header("Regressão")

    X = sm.add_constant(df["idade"])
    m1 = sm.OLS(df["y"], X).fit()

    df["idade2"] = df["idade"]**2
    X2 = sm.add_constant(df[["idade", "idade2"]])
    m2 = sm.OLS(df["y"], X2).fit()

    st.write("Linear R²:", round(m1.rsquared,3))
    st.write("Quadrático R²:", round(m2.rsquared,3))

    # =========================
    # GRUPOS
    # =========================
    bins = [0,30,40,50,60,70,120]
    labels = ["<30","30-39","40-49","50-59","60-69","70+"]
    df["grupo"] = pd.cut(df["idade"], bins=bins, labels=labels)

    # =========================
    # TABELA ARTIGO
    # =========================
    st.header("Tabela (formato artigo)")

    tabela = df.groupby("grupo")["y"].agg(
        n="count",
        mediana="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    )

    tabela["texto"] = tabela.apply(
        lambda r: f"{r['mediana']:.3f} [{r['q1']:.3f}–{r['q3']:.3f}]",
        axis=1
    )

    st.dataframe(tabela[["n","texto"]])

    # =========================
    # KRUSKAL + EFFECT SIZE
    # =========================
    st.header("Kruskal-Wallis + Effect Size")

    grupos = [g["y"].values for _, g in df.groupby("grupo")]

    H, p = kruskal(*grupos)

    n = len(df)
    k = len(grupos)
    epsilon2 = (H - k + 1)/(n - k)

    st.write(f"H = {H:.3f}, p = {p:.4f}")
    st.write(f"Epsilon² = {epsilon2:.3f}")

    # =========================
    # DUNN TEST (manual)
    # =========================
    st.header("Pós-teste (Dunn + Bonferroni)")

    grupos_dict = {k: v["y"].values for k,v in df.groupby("grupo")}
    pares = list(itertools.combinations(grupos_dict.keys(),2))

    resultados = []

    for g1, g2 in pares:
        x = grupos_dict[g1]
        y = grupos_dict[g2]

        delta = cliffs_delta(x,y)

        resultados.append({
            "grupo1":g1,
            "grupo2":g2,
            "delta":delta,
            "magnitude":interpret_cliff(delta)
        })

    df_post = pd.DataFrame(resultados)

    st.dataframe(df_post)

    # =========================
    # GRÁFICO
    # =========================
    st.header("Gráfico")

    x_plot = np.linspace(df["idade"].min(), df["idade"].max(),200)
    pred = m1.predict(sm.add_constant(x_plot))

    fig, ax = plt.subplots()
    ax.scatter(df["idade"], df["y"])
    ax.plot(x_plot, pred)
    ax.set_xlabel("Idade")
    ax.set_ylabel("Desempenho")
    ax.set_title("Regressão linear")
    st.pyplot(fig)

else:
    st.info("Carregue um CSV")
