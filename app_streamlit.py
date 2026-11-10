# =====================================================================
#  App Streamlit — Batalha do Mar de Bismarck (Haywood, 1954)
#  Rodar:  streamlit run app_streamlit.py
#  (bismarck_nucleo.py deve estar na mesma pasta)
# =====================================================================
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import bismarck_nucleo as bn

st.set_page_config(page_title="Batalha do Mar de Bismarck", layout="wide")
st.title("Batalha do Mar de Bismarck (1943) — um jogo de soma zero")
st.caption("Haywood, O. G. (1954). Military Decision and Game Theory. "
           "Journal of the Operations Research Society of America, 2(4), 365–385.")

# ---------------- barra lateral: decisões e parâmetros ----------------
with st.sidebar:
    st.header("Decisões")
    recon = st.radio("Aliados (aviões) concentram o reconhecimento no:", bn.ESTRATEGIAS, horizontal=True)
    rota = st.radio("Comboio japonês (navios) navega pela rota:", bn.ESTRATEGIAS, horizontal=True)
    if st.button("Zarpar de Rabaul / nova simulação", type="primary", use_container_width=True):
        st.session_state.seed = np.random.randint(1_000_000_000)
    st.session_state.setdefault("seed", 42)

    P = bn.parametros_padrao()
    with st.expander("Avançado", expanded=False):
        st.caption("Padrão: modelo estocástico com bombardeio rasante.")
        modo_lbl = st.radio("Modelo", ["Estocástico (clima + detecção)", "Matriz do artigo (determinístico)"])
        modo = "artigo" if modo_lbl.startswith("Matriz") else "estocastico"
        rasante = st.toggle("Bombardeio rasante (skip bombing)", value=True,
                            help="A surpresa tática citada por Haywood: aviões aliados modificados para ataque a baixa altitude.")
        st.markdown("**Prob. de avistar por dia**")
        P["p_conc_chuva"] = st.slider("Concentrado + chuva", 0.0, 1.0, P["p_conc_chuva"], 0.05)
        P["p_conc_limpo"] = st.slider("Concentrado + limpo", 0.0, 1.0, P["p_conc_limpo"], 0.05)
        P["p_esp_chuva"] = st.slider("Esparso + chuva", 0.0, 1.0, P["p_esp_chuva"], 0.05)
        P["p_esp_limpo"] = st.slider("Esparso + limpo", 0.0, 1.0, P["p_esp_limpo"], 0.05)
        P["p_lae"] = st.slider("Dia 3 perto de Lae (limpo)", 0.0, 1.0, P["p_lae"], 0.05)
        st.markdown("**Prob. de afundar um navio por dia de bombardeio**")
        P["q_transp_rasante"] = st.slider("Transporte — rasante", 0.0, 1.0, P["q_transp_rasante"], 0.05)
        P["q_destr_rasante"] = st.slider("Destróier — rasante", 0.0, 1.0, P["q_destr_rasante"], 0.05)
        P["q_transp_conv"] = st.slider("Transporte — convencional", 0.0, 1.0, P["q_transp_conv"], 0.05)
        P["q_destr_conv"] = st.slider("Destróier — convencional", 0.0, 1.0, P["q_destr_conv"], 0.05)

rng = np.random.default_rng(st.session_state.seed)
res = bn.simular(recon, rota, P, modo=modo, rasante=rasante, rng=rng)

# ---------------- indicadores ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dias de bombardeio", res["B"])
c2.metric("Transportes afundados", f"{res['T']:.1f} / {P['n_transp']}")
c3.metric("Destróieres afundados", f"{res['D']:.1f} / {P['n_destr']}")
c4.metric("Tropas perdidas", f"{res['tropas_perdidas']:,.0f}".replace(",", "."))

aba1, aba2, aba3, aba4 = st.tabs(["Mapa da travessia", "Matriz do jogo", "Monte Carlo (avançado)", "Teoria e fórmulas"])

with aba1:
    fig = bn.plot_mapa(res, P); st.pyplot(fig); plt.close(fig)
    st.code(bn.resumo_texto(res, P), language=None)

with aba2:
    col_a, col_b = st.columns(2)
    for col, m, tit in [(col_a, "artigo", "Matriz do artigo (dias de bombardeio)"),
                        (col_b, "estocastico", "Matriz do modelo: E[dias de bombardeio]")]:
        A = bn.matriz_payoff(P, m)
        fig = bn.plot_matriz(A, recon, rota, tit); col.pyplot(fig); plt.close(fig)
        sol = bn.resolver_jogo(A)
        if sol["tipo"] == "sela":
            col.success(f"Ponto de sela em (Aliados {bn.ESTRATEGIAS[sol['i']]}, Japão "
                        f"{bn.ESTRATEGIAS[sol['j']]}), valor = {sol['valor']:.3f}")
        else:
            col.warning(f"Sem ponto de sela. Estratégias mistas: Aliados Norte com prob. "
                        f"{sol['p']:.3f}; Japão Norte com prob. {sol['q']:.3f}; valor = {sol['valor']:.3f}")
    st.info("A célula com borda azul é a combinação escolhida na barra lateral. "
            "Mova os sliders de detecção para ver o ponto de sela desaparecer.")

with aba3:
    n = st.select_slider("Número de travessias", [1_000, 5_000, 10_000, 50_000], value=10_000)
    mc = bn.monte_carlo(recon, rota, P, rasante=rasante, n=n, rng=np.random.default_rng(st.session_state.seed))
    fig = bn.plot_monte_carlo(mc, P); st.pyplot(fig); plt.close(fig)
    st.write(f"P(comboio chegar intacto) = {(mc['T'] == 0).mean():.3f} · "
             f"P(perder todos os transportes) = {(mc['T'] == P['n_transp']).mean():.3f} · "
             f"tropas perdidas médias = {mc['tropas'].mean():,.0f}".replace(",", "."))

with aba4:
    from pathlib import Path
    arq = Path(__file__).parent / "TEORIA.md"
    st.markdown(arq.read_text(encoding="utf-8") if arq.exists()
                else "Coloque TEORIA.md na mesma pasta do app para exibir as fórmulas.")
