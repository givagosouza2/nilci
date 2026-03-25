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
    coluna_desempenho = st.selectbox(
        "Selecione a coluna do parâmetro de desempenho",
        [c for c in df_raw.columns if c != coluna_idade]
    )

nome_parametro = st.text_input(
    "Nome para exibição do parâmetro",
    value=coluna_desempenho
)

st.subheader("Configurações adicionais")

c1, c2 = st.columns(2)

with c1:
    interpretar_maior_como_pior = st.checkbox(
        "Interpretar valores maiores como pior desempenho",
        value=False
    )

with c2:
    mostrar_tabela_completa = st.checkbox(
        "Mostrar tabela completa após limpeza",
        value=False
    )

df = df_raw[[coluna_idade, coluna_desempenho]].copy()
df.columns = ["idade", "desempenho"]

df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
df["desempenho"] = pd.to_numeric(df["desempenho"], errors="coerce")
df = df.dropna(subset=["idade", "desempenho"]).copy()

if len(df) < 5:
    st.error("Após a limpeza, sobraram poucos dados para análise.")
    st.stop()

if mostrar_tabela_completa:
    st.subheader("Dados após limpeza")
    st.dataframe(df)

st.success(f"Foram carregadas {len(df)} observações válidas.")

# =========================
# Estatísticas descritivas
# =========================
st.header("1. Estatísticas descritivas")

descritivas = df[["idade", "desempenho"]].describe().T
q1_geral = df["desempenho"].quantile(0.25)
q3_geral = df["desempenho"].quantile(0.75)
iqr_geral = q3_geral - q1_geral
mediana_geral = df["desempenho"].median()

c1, c2 = st.columns(2)

with c1:
    st.dataframe(descritivas)

with c2:
    resumo_robusto = pd.DataFrame({
        "Métrica": ["Mediana", "Q1", "Q3", "IQR"],
        "Valor": [mediana_geral, q1_geral, q3_geral, iqr_geral]
    })
    st.dataframe(resumo_robusto)

# =========================
# Correlações
# =========================
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
        st.info("Como valores maiores foram definidos como pior desempenho, uma correlação positiva sugere piora com a idade.")
    else:
        st.info("Como valores maiores foram definidos como pior desempenho, uma correlação negativa sugere melhora com a idade.")
else:
    if spearman_rho < 0:
        st.info("Como valores maiores não foram definidos como pior desempenho, uma correlação negativa sugere redução do parâmetro com a idade.")
    else:
        st.info("Como valores maiores não foram definidos como pior desempenho, uma correlação positiva sugere aumento do parâmetro com a idade.")

# =========================
# Regressões
# =========================
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
st.subheader("Comparação formal entre modelo linear e quadrático")
st.dataframe(anova_result)

st.subheader("Coeficientes do modelo linear")
coef_linear = pd.DataFrame({
    "Parâmetro": modelo_linear.params.index,
    "Coeficiente": modelo_linear.params.values,
    "p": modelo_linear.pvalues.values
})
st.dataframe(coef_linear.style.format({
    "Coeficiente": "{:.6f}",
    "p": "{:.6f}"
}))

st.subheader("Coeficientes do modelo quadrático")
coef_quad = pd.DataFrame({
    "Parâmetro": modelo_quad.params.index,
    "Coeficiente": modelo_quad.params.values,
    "p": modelo_quad.pvalues.values
})
st.dataframe(coef_quad.style.format({
    "Coeficiente": "{:.6f}",
    "p": "{:.6f}"
}))

# =========================
# Gráfico scatter + regressões
# =========================
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

# =========================
# Faixas etárias
# =========================
st.header("5. Análise por faixas etárias")

bins = [0, 29, 39, 49, 59, 69, 120]
labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
df["faixa_etaria"] = pd.cut(df["idade"], bins=bins, labels=labels)

resumo_faixa = df.groupby("faixa_etaria", observed=False)["desempenho"].agg(
    n="count",
    media="mean",
    dp="std",
    mediana="median",
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75),
    iqr=iqr,
    minimo="min",
    maximo="max"
)

st.dataframe(resumo_faixa.style.format("{:.4f}"))

fig3, ax3 = plt.subplots(figsize=(10, 6))
df.boxplot(column="desempenho", by="faixa_etaria", grid=False, ax=ax3)
ax3.set_title(f"{nome_parametro} por faixa etária")
ax3.set_xlabel("Faixa etária")
ax3.set_ylabel(nome_parametro)
plt.suptitle("")
st.pyplot(fig3)

# =========================
# Kruskal-Wallis
# =========================
st.header("6. Kruskal-Wallis")

df_kw = df.dropna(subset=["faixa_etaria", "desempenho"]).copy()
ordem_faixas = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]

grupos = [
    df_kw.loc[df_kw["faixa_etaria"] == faixa, "desempenho"].values
    for faixa in ordem_faixas
    if len(df_kw.loc[df_kw["faixa_etaria"] == faixa, "desempenho"].values) > 0
]

faixas_validas = [
    faixa for faixa in ordem_faixas
    if len(df_kw.loc[df_kw["faixa_etaria"] == faixa, "desempenho"].values) > 0
]

if len(grupos) >= 2:
    H, p_kw = kruskal(*grupos)
    n = len(df_kw)
    k = len(grupos)
    epsilon_squared = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan

    kw_df = pd.DataFrame({
        "Estatística H": [H],
        "p": [p_kw],
        "Epsilon²": [epsilon_squared]
    })

    st.dataframe(kw_df.style.format({
        "Estatística H": "{:.4f}",
        "p": "{:.6f}",
        "Epsilon²": "{:.4f}"
    }))

    # Pós-teste
    st.subheader("Pós-teste: Mann-Whitney com correção de Bonferroni")

    grupos_dict = {
        faixa: df_kw.loc[df_kw["faixa_etaria"] == faixa, "desempenho"].values
        for faixa in faixas_validas
    }

    pares = list(itertools.combinations(faixas_validas, 2))
    alpha = 0.05
    alpha_bonf = alpha / len(pares) if len(pares) > 0 else np.nan

    resultados_posthoc = []

    for g1, g2 in pares:
        x = grupos_dict[g1]
        y = grupos_dict[g2]
        stat, p = mannwhitneyu(x, y, alternative="two-sided")

        resultados_posthoc.append({
            "grupo_1": g1,
            "grupo_2": g2,
            "U": stat,
            "p": p,
            "limiar_bonferroni": alpha_bonf,
            "significativo": p < alpha_bonf
        })

    if resultados_posthoc:
        df_posthoc = pd.DataFrame(resultados_posthoc)
        st.dataframe(df_posthoc.style.format({
            "U": "{:.2f}",
            "p": "{:.6f}",
            "limiar_bonferroni": "{:.6f}"
        }))
else:
    st.warning("Não há grupos suficientes para executar o Kruskal-Wallis.")

# =========================
# Resumo por idade exata
# =========================
st.header("7. Resumo por idade exata")

resumo_idade = df.groupby("idade")["desempenho"].agg(
    n="count",
    media="mean",
    dp="std",
    mediana="median",
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75),
    iqr=iqr,
    minimo="min",
    maximo="max"
).reset_index()

st.dataframe(resumo_idade.style.format({
    "idade": "{:.0f}",
    "media": "{:.4f}",
    "dp": "{:.4f}",
    "mediana": "{:.4f}",
    "q1": "{:.4f}",
    "q3": "{:.4f}",
    "iqr": "{:.4f}",
    "minimo": "{:.4f}",
    "maximo": "{:.4f}"
}))

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

# =========================
# Texto resumo
# =========================
st.header("8. Resumo automático")

direcao = "positiva" if spearman_rho > 0 else "negativa"
st.write(
    f"""
"""
)

else:
st.info("Aguardando o carregamento de um arquivo CSV com cabeçalho.")
