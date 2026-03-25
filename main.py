import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

from scipy.stats import kruskal, norm, pearsonr, rankdata, spearmanr
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


st.set_page_config(page_title="Análise do efeito da idade", layout="wide")
st.title("Análise do efeito da idade sobre um parâmetro de desempenho — versão premium")


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================
def iqr(x: pd.Series) -> float:
    return float(x.quantile(0.75) - x.quantile(0.25))


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


def interpret_cliffs_delta(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligível"
    if d < 0.33:
        return "pequeno"
    if d < 0.474:
        return "moderado"
    return "grande"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)

    return (gt - lt) / (len(x) * len(y))


def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    denom = n - k
    if denom <= 0:
        return np.nan
    return (H - k + 1) / denom


def interpret_epsilon_squared(eps: float) -> str:
    if pd.isna(eps):
        return ""
    if eps < 0.01:
        return "trivial"
    if eps < 0.08:
        return "pequeno"
    if eps < 0.26:
        return "moderado"
    return "grande"


def dunn_posthoc(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    p_adjust: str = "holm"
) -> pd.DataFrame:
    """
    Pós-teste de Dunn manual, com correção múltipla.
    Usa ranks globais e correção para empates.
    """
    data = df[[group_col, value_col]].dropna().copy()
    data[group_col] = data[group_col].astype(str)

    groups = list(data[group_col].unique())
    n_total = len(data)

    # ranks globais
    values = data[value_col].to_numpy(dtype=float)
    ranks = rankdata(values, method="average")
    data["_rank"] = ranks

    # estatísticas por grupo
    stats_by_group = (
        data.groupby(group_col)
        .agg(
            n=(value_col, "count"),
            rank_mean=("_rank", "mean"),
        )
        .reset_index()
    )

    group_info = {
        row[group_col]: {"n": int(row["n"]), "rank_mean": float(row["rank_mean"])}
        for _, row in stats_by_group.iterrows()
    }

    # correção de empates
    counts = pd.Series(values).value_counts()
    tie_sum = np.sum(counts**3 - counts)
    tie_correction = 1.0 - tie_sum / (n_total**3 - n_total) if n_total > 1 else 1.0

    const = (n_total * (n_total + 1) / 12.0) * tie_correction

    rows = []
    raw_pvals = []

    for g1, g2 in itertools.combinations(groups, 2):
        n1 = group_info[g1]["n"]
        n2 = group_info[g2]["n"]
        r1 = group_info[g1]["rank_mean"]
        r2 = group_info[g2]["rank_mean"]

        se = math.sqrt(const * (1 / n1 + 1 / n2))
        if se == 0:
            z = np.nan
            p_raw = np.nan
        else:
            z = (r1 - r2) / se
            p_raw = 2 * (1 - norm.cdf(abs(z)))

        raw_pvals.append(p_raw)
        rows.append({
            "grupo_1": g1,
            "grupo_2": g2,
            "n_1": n1,
            "n_2": n2,
            "mean_rank_1": r1,
            "mean_rank_2": r2,
            "z": z,
            "p_bruto": p_raw
        })

    results = pd.DataFrame(rows)

    if len(results) > 0:
        valid_mask = results["p_bruto"].notna()
        pvals_valid = results.loc[valid_mask, "p_bruto"].to_numpy()

        if len(pvals_valid) > 0:
            reject_holm, p_holm, _, _ = multipletests(pvals_valid, alpha=0.05, method="holm")
            reject_bonf, p_bonf, _, _ = multipletests(pvals_valid, alpha=0.05, method="bonferroni")

            results.loc[valid_mask, "p_holm"] = p_holm
            results.loc[valid_mask, "sig_holm"] = reject_holm
            results.loc[valid_mask, "p_bonferroni"] = p_bonf
            results.loc[valid_mask, "sig_bonferroni"] = reject_bonf

    return results


def article_summary_table(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    table = (
        df.groupby(group_col, observed=False)[value_col]
        .agg(
            n="count",
            media="mean",
            dp="std",
            mediana="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
            minimo="min",
            maximo="max",
        )
        .reset_index()
    )
    table["mediana [Q1–Q3]"] = table.apply(
        lambda r: f"{r['mediana']:.4f} [{r['q1']:.4f}–{r['q3']:.4f}]"
        if pd.notna(r["mediana"]) else "",
        axis=1
    )
    return table


# =========================================================
# INTERFACE
# =========================================================
st.markdown(
    """
Carregue um arquivo **CSV com cabeçalho** contendo:
- uma coluna de **idade**
- uma coluna do **parâmetro de desempenho**
"""
)

uploaded_file = st.file_uploader("Carregue o arquivo CSV", type=["csv"])

if uploaded_file is None:
    st.info("Aguardando o carregamento de um arquivo CSV.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Não foi possível ler o arquivo: {e}")
    st.stop()

if df_raw.shape[1] < 2:
    st.error("O arquivo deve conter pelo menos duas colunas com cabeçalho.")
    st.stop()

st.subheader("Pré-visualização")
st.dataframe(df_raw.head())

col1, col2 = st.columns(2)
with col1:
    coluna_idade = st.selectbox("Selecione a coluna de idade", df_raw.columns)
with col2:
    coluna_desempenho = st.selectbox(
        "Selecione a coluna do parâmetro de desempenho",
        [c for c in df_raw.columns if c != coluna_idade]
    )

nome_parametro = st.text_input("Nome para exibição do parâmetro", value=coluna_desempenho)

c1, c2, c3 = st.columns(3)
with c1:
    interpretar_maior_como_pior = st.checkbox(
        "Valores maiores indicam pior desempenho",
        value=False
    )
with c2:
    mostrar_tabela_limpa = st.checkbox("Mostrar tabela após limpeza", value=False)
with c3:
    correcao_multipla_preferida = st.selectbox(
        "Correção múltipla preferida",
        ["holm", "bonferroni"],
        index=0
    )

df = df_raw[[coluna_idade, coluna_desempenho]].copy()
df.columns = ["idade", "desempenho"]

df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
df["desempenho"] = pd.to_numeric(df["desempenho"], errors="coerce")
df = df.dropna(subset=["idade", "desempenho"]).copy()

if len(df) < 5:
    st.error("Após a limpeza, sobraram poucos dados para análise.")
    st.stop()

if mostrar_tabela_limpa:
    st.subheader("Dados após limpeza")
    st.dataframe(df)

st.success(f"Foram carregadas {len(df)} observações válidas.")


# =========================================================
# 1. ESTATÍSTICAS DESCRITIVAS
# =========================================================
st.header("1. Estatísticas descritivas")

descritivas = df[["idade", "desempenho"]].describe().T
q1_geral = df["desempenho"].quantile(0.25)
q3_geral = df["desempenho"].quantile(0.75)
iqr_geral = q3_geral - q1_geral
mediana_geral = df["desempenho"].median()

a1, a2 = st.columns(2)
with a1:
    st.dataframe(descritivas)
with a2:
    robusto = pd.DataFrame({
        "Métrica": ["Mediana", "Q1", "Q3", "IQR"],
        "Valor": [mediana_geral, q1_geral, q3_geral, iqr_geral]
    })
    st.dataframe(robusto.style.format({"Valor": "{:.4f}"}))


# =========================================================
# 2. CORRELAÇÕES
# =========================================================
st.header("2. Correlações")

pearson_r, pearson_p = pearsonr(df["idade"], df["desempenho"])
spearman_rho, spearman_p = spearmanr(df["idade"], df["desempenho"])

correl_df = pd.DataFrame({
    "Teste": ["Pearson", "Spearman"],
    "Coeficiente": [pearson_r, spearman_rho],
    "p": [pearson_p, spearman_p]
})

st.dataframe(correl_df.style.format({
    "Coeficiente": "{:.4f}",
    "p": "{:.6f}"
}))

if interpretar_maior_como_pior:
    if spearman_rho > 0:
        st.info("Correlação positiva sugere piora com a idade.")
    else:
        st.info("Correlação negativa sugere melhora com a idade.")
else:
    if spearman_rho < 0:
        st.info("Correlação negativa sugere redução do parâmetro com a idade.")
    else:
        st.info("Correlação positiva sugere aumento do parâmetro com a idade.")


# =========================================================
# 3. REGRESSÕES
# =========================================================
st.header("3. Regressões")

X_linear = sm.add_constant(df["idade"])
modelo_linear = sm.OLS(df["desempenho"], X_linear).fit()

df["idade2"] = df["idade"] ** 2
X_quad = sm.add_constant(df[["idade", "idade2"]])
modelo_quad = sm.OLS(df["desempenho"], X_quad).fit()

comparacao_modelos = pd.DataFrame({
    "Modelo": ["Linear", "Quadrático"],
    "R²": [modelo_linear.rsquared, modelo_quad.rsquared],
    "AIC": [modelo_linear.aic, modelo_quad.aic],
    "BIC": [modelo_linear.bic, modelo_quad.bic]
})
st.dataframe(comparacao_modelos.style.format({
    "R²": "{:.4f}",
    "AIC": "{:.2f}",
    "BIC": "{:.2f}"
}))

anova_result = anova_lm(modelo_linear, modelo_quad)
st.subheader("Comparação entre modelo linear e quadrático")
st.dataframe(anova_result)

coef_linear = pd.DataFrame({
    "Parâmetro": modelo_linear.params.index,
    "Coeficiente": modelo_linear.params.values,
    "p": modelo_linear.pvalues.values
})
coef_quad = pd.DataFrame({
    "Parâmetro": modelo_quad.params.index,
    "Coeficiente": modelo_quad.params.values,
    "p": modelo_quad.pvalues.values
})

b1, b2 = st.columns(2)
with b1:
    st.subheader("Coeficientes do modelo linear")
    st.dataframe(coef_linear.style.format({
        "Coeficiente": "{:.6f}",
        "p": "{:.6f}"
    }))
with b2:
    st.subheader("Coeficientes do modelo quadrático")
    st.dataframe(coef_quad.style.format({
        "Coeficiente": "{:.6f}",
        "p": "{:.6f}"
    }))


# =========================================================
# 4. GRÁFICOS
# =========================================================
st.header("4. Gráficos")

idades_plot = np.linspace(df["idade"].min(), df["idade"].max(), 300)

X_plot_linear = sm.add_constant(pd.DataFrame({"idade": idades_plot}))
pred_linear = modelo_linear.predict(X_plot_linear)

X_plot_quad = sm.add_constant(pd.DataFrame({
    "idade": idades_plot,
    "idade2": idades_plot ** 2
}))
pred_quad = modelo_quad.predict(X_plot_quad)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(df["idade"], df["desempenho"], alpha=0.7, label="Dados")
ax1.plot(idades_plot, pred_linear, linewidth=2, label="Regressão linear")
ax1.plot(idades_plot, pred_quad, linewidth=2, label="Regressão quadrática")
ax1.set_xlabel("Idade (anos)")
ax1.set_ylabel(nome_parametro)
ax1.set_title("Efeito da idade sobre o parâmetro")
ax1.grid(True, alpha=0.3)
ax1.legend()
st.pyplot(fig1)

residuos = modelo_linear.resid
ajustados = modelo_linear.fittedvalues

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.scatter(ajustados, residuos, alpha=0.7)
ax2.axhline(0, linestyle="--")
ax2.set_xlabel("Valores ajustados")
ax2.set_ylabel("Resíduos")
ax2.set_title("Resíduos da regressão linear")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)


# =========================================================
# 5. FAIXAS ETÁRIAS
# =========================================================
st.header("5. Análise por faixas etárias")

bins = [0, 29, 39, 49, 59, 69, 120]
labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
df["faixa_etaria"] = pd.cut(df["idade"], bins=bins, labels=labels)

resumo_faixa = article_summary_table(df, "faixa_etaria", "desempenho")
st.dataframe(
    resumo_faixa.style.format({
        "media": "{:.4f}",
        "dp": "{:.4f}",
        "mediana": "{:.4f}",
        "q1": "{:.4f}",
        "q3": "{:.4f}",
        "iqr": "{:.4f}",
        "minimo": "{:.4f}",
        "maximo": "{:.4f}"
    })
)

fig3, ax3 = plt.subplots(figsize=(10, 6))
df.boxplot(column="desempenho", by="faixa_etaria", grid=False, ax=ax3)
ax3.set_title(f"{nome_parametro} por faixa etária")
ax3.set_xlabel("Faixa etária")
ax3.set_ylabel(nome_parametro)
plt.suptitle("")
st.pyplot(fig3)


# =========================================================
# 6. KRUSKAL-WALLIS + EFFECT SIZE
# =========================================================
st.header("6. Kruskal–Wallis + tamanho de efeito")

df_kw = df.dropna(subset=["faixa_etaria", "desempenho"]).copy()
ordem_faixas = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]

grupos_validos = []
faixas_validas = []
for faixa in ordem_faixas:
    vals = df_kw.loc[df_kw["faixa_etaria"] == faixa, "desempenho"].values
    if len(vals) > 0:
        grupos_validos.append(vals)
        faixas_validas.append(faixa)

if len(grupos_validos) >= 2:
    H, p_kw = kruskal(*grupos_validos)
    eps2 = epsilon_squared_kruskal(H, len(df_kw), len(grupos_validos))

    kw_df = pd.DataFrame({
        "Estatística H": [H],
        "p": [p_kw],
        "Epsilon²": [eps2],
        "Magnitude": [interpret_epsilon_squared(eps2)]
    })

    st.dataframe(kw_df.style.format({
        "Estatística H": "{:.4f}",
        "p": "{:.6f}",
        "Epsilon²": "{:.4f}"
    }))
else:
    st.warning("Não há grupos suficientes para executar o Kruskal–Wallis.")
    st.stop()


# =========================================================
# 7. PÓS-TESTE DE DUNN COM P-VALUES
# =========================================================
st.header("7. Pós-teste de Dunn com comparações múltiplas")

df_dunn = df_kw.loc[df_kw["faixa_etaria"].isin(faixas_validas)].copy()
df_dunn["faixa_etaria"] = df_dunn["faixa_etaria"].astype(str)

posthoc = dunn_posthoc(df_dunn, "faixa_etaria", "desempenho")

if len(posthoc) > 0:
    # Cliff's delta par a par
    grupos_dict = {
        faixa: df_dunn.loc[df_dunn["faixa_etaria"] == faixa, "desempenho"].to_numpy(dtype=float)
        for faixa in faixas_validas
    }

    deltas = []
    mags = []
    for _, row in posthoc.iterrows():
        g1 = row["grupo_1"]
        g2 = row["grupo_2"]
        delta = cliffs_delta(grupos_dict[g1], grupos_dict[g2])
        deltas.append(delta)
        mags.append(interpret_cliffs_delta(delta))

    posthoc["cliffs_delta"] = deltas
    posthoc["magnitude_delta"] = mags

    colunas_mostrar = [
        "grupo_1", "grupo_2",
        "n_1", "n_2",
        "mean_rank_1", "mean_rank_2",
        "z", "p_bruto",
        "p_holm", "sig_holm",
        "p_bonferroni", "sig_bonferroni",
        "cliffs_delta", "magnitude_delta"
    ]

    st.dataframe(
        posthoc[colunas_mostrar].style.format({
            "mean_rank_1": "{:.3f}",
            "mean_rank_2": "{:.3f}",
            "z": "{:.4f}",
            "p_bruto": "{:.6f}",
            "p_holm": "{:.6f}",
            "p_bonferroni": "{:.6f}",
            "cliffs_delta": "{:.4f}",
        })
    )

    if correcao_multipla_preferida == "holm":
        sigs = posthoc.loc[posthoc["sig_holm"] == True, ["grupo_1", "grupo_2", "p_holm"]].copy()
        sigs.rename(columns={"p_holm": "p_corrigido"}, inplace=True)
    else:
        sigs = posthoc.loc[posthoc["sig_bonferroni"] == True, ["grupo_1", "grupo_2", "p_bonferroni"]].copy()
        sigs.rename(columns={"p_bonferroni": "p_corrigido"}, inplace=True)

    st.subheader("Comparações significativas")
    if len(sigs) == 0:
        st.info("Nenhuma comparação múltipla permaneceu significativa após correção.")
    else:
        st.dataframe(sigs.style.format({"p_corrigido": "{:.6f}"}))


# =========================================================
# 8. RESUMO POR IDADE EXATA
# =========================================================
st.header("8. Resumo por idade exata")

resumo_idade = (
    df.groupby("idade")["desempenho"]
    .agg(
        n="count",
        media="mean",
        dp="std",
        mediana="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
        minimo="min",
        maximo="max"
    )
    .reset_index()
)

st.dataframe(
    resumo_idade.style.format({
        "idade": "{:.0f}",
        "media": "{:.4f}",
        "dp": "{:.4f}",
        "mediana": "{:.4f}",
        "q1": "{:.4f}",
        "q3": "{:.4f}",
        "iqr": "{:.4f}",
        "minimo": "{:.4f}",
        "maximo": "{:.4f}"
    })
)

fig4, ax4 = plt.subplots(figsize=(11, 6))
ax4.plot(resumo_idade["idade"], resumo_idade["mediana"], marker="o")
ax4.set_xlabel("Idade (anos)")
ax4.set_ylabel(f"Mediana de {nome_parametro}")
ax4.set_title("Mediana do parâmetro por idade")
ax4.grid(True, alpha=0.3)
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(11, 6))
ax5.plot(resumo_idade["idade"], resumo_idade["iqr"], marker="o")
ax5.set_xlabel("Idade (anos)")
ax5.set_ylabel("Amplitude interquartil (IQR)")
ax5.set_title("Variabilidade do parâmetro por idade")
ax5.grid(True, alpha=0.3)
st.pyplot(fig5)


# =========================================================
# 9. RESUMO AUTOMÁTICO
# =========================================================
st.header("9. Resumo automático")

direcao = "positiva" if spearman_rho > 0 else "negativa"

texto_kw = (
    f"Kruskal–Wallis: H = {H:.3f}, p = {format_p(p_kw)}, "
    f"epsilon² = {eps2:.3f} ({interpret_epsilon_squared(eps2)})."
)

st.write(
    f"""
- O coeficiente de **Spearman** foi **{spearman_rho:.4f}** com **p = {format_p(spearman_p)}**.
- O coeficiente de **Pearson** foi **{pearson_r:.4f}** com **p = {format_p(pearson_p)}**.
- O modelo **linear** apresentou **R² = {modelo_linear.rsquared:.4f}**.
- O modelo **quadrático** apresentou **R² = {modelo_quad.rsquared:.4f}**.
- A associação monotônica entre idade e **{nome_parametro}** foi **{direcao}**.
- **{texto_kw}**
"""
)
