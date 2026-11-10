# App Streamlit — Gráficos 5.1 e 5.2 de Fiani (Teoria dos Jogos) com cara ou coroa
# Rodar:  streamlit run app_cara_coroa.py   (cara_coroa.py na mesma pasta)
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cara_coroa as cc

st.set_page_config(page_title="Cara ou coroa: estratégias mistas", layout="wide")
st.title("Cara ou coroa: recompensas esperadas e melhores respostas")
st.caption("Adaptado do jogo de prevenção de ataque de Fiani, Teoria dos Jogos, cap. 5 "
           "(Gráficos 5.1 e 5.2). Jogador 1 = Azul (linhas), Jogador 2 = Vermelho (colunas), "
           "Cara = porto Sul, Coroa = porto Norte.")

A0, B0 = cc.matriz_padrao()
with st.sidebar:
    st.header("Estratégia mista do Jogador 1")
    p = st.slider("p = Pr(Jogador 1 jogar Cara)", 0.0, 1.0, 0.30, 0.01)
    with st.expander("Avançado", expanded=False):
        q = st.slider("q = Pr(Jogador 2 jogar Cara)", 0.0, 1.0, 0.50, 0.01)
        ver_j2 = st.toggle("Mostrar Jogador 1 e equilíbrio", value=False)
        st.caption("Matriz de recompensas (padrão = jogo do livro)")
        A, B = np.zeros((2, 2)), np.zeros((2, 2))
        for i, li in enumerate(cc.EST):
            for j, cj in enumerate(cc.EST):
                c1, c2 = st.columns(2)
                A[i, j] = c1.number_input(f"J1 {li} × J2 {cj}: J1", value=float(A0[i, j]), step=1.0, key=f"a{i}{j}")
                B[i, j] = c2.number_input(f"J1 {li} × J2 {cj}: J2", value=float(B0[i, j]), step=1.0, key=f"b{i}{j}")

st.dataframe({"(J1, J2)": ["J1: Cara", "J1: Coroa"],
              "J2: Cara": [f"({A[0,0]:g}, {B[0,0]:g})", f"({A[1,0]:g}, {B[1,0]:g})"],
              "J2: Coroa": [f"({A[0,1]:g}, {B[0,1]:g})", f"({A[1,1]:g}, {B[1,1]:g})"]}, hide_index=True)

c1, c2 = st.columns([1.45, 1])
with c1:
    fig = cc.plot_grafico_5_1(B, p); st.pyplot(fig); plt.close(fig)
with c2:
    fig = cc.plot_grafico_5_2(B, p); st.pyplot(fig); plt.close(fig)

if ver_j2:
    c3, c4 = st.columns([1.45, 1])
    with c3:
        fig = cc.plot_recompensas_j1(A, q); st.pyplot(fig); plt.close(fig)
    with c4:
        fig = cc.plot_equilibrio(A, B, p, q); st.pyplot(fig); plt.close(fig)

st.code(cc.resumo(A, B, p, q), language=None)
with st.expander("Fórmulas (para copiar)"):
    st.code(cc.FORMULAS, language=None)
