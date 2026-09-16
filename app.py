import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Simulação HHI", layout="wide")

st.title("📊 Simulação Interativa: O Paradoxo da Concentração Gradual")
st.markdown("""
Esta aplicação simula o mercado hipotético com **200 empresas**, cada uma com **0,5% de participação de mercado**. 
O agente adquire as empresas **uma a uma** de forma sequencial. O objetivo é demonstrar como aquisições fracionadas mantêm a variação do HHI ($\Delta HHI$) abaixo do limiar regulatório de **100 pontos**, permitindo a formação de um monopólio sem gatilhos automáticos de alerta.
""")

# Inicializar o estado da sessão para o número de empresas adquiridas (k)
if 'num_adquiridas' not in st.session_state:
    st.session_state.num_adquiridas = 1

# Layout em colunas
col_controles, col_grafico = st.columns([1.2, 1.8], gap="large")

with col_controles:
    st.subheader("Painel de Controle")
    
    # Botões de Decremento e Incremento
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Desfazer (-1)", use_container_width=True):
            if st.session_state.num_adquiridas > 1:
                st.session_state.num_adquiridas -= 1
    with col_btn2:
        if st.button("Adicionar (+1) ➡️", use_container_width=True):
            if st.session_state.num_adquiridas < 199:
                st.session_state.num_adquiridas += 1

    # Slider para navegação direta
    st.session_state.num_adquiridas = st.slider(
        "Número de empresas absorvidas ($k$):", 
        min_value=1, 
        max_value=199, 
        value=st.session_state.num_adquiridas,
        step=1
    )

    k = st.session_state.num_adquiridas
    
    # Cálculos matemáticos baseados nas notas
    # Market share da adquirente antes da transação atual
    market_share_anterior = k * 0.5
    market_share_alvo = 0.5
    market_share_novo = (k + 1) * 0.5
    
    # Delta HHI = 2 * S1 * S2
    delta_hhi = 2 * market_share_anterior * market_share_alvo
    
    # HHI Total atual do mercado
    num_restantes = 200 - (k + 1)
    hhi_total = (market_share_novo ** 2) + (num_restantes * (0.5 ** 2))

    st.markdown("---")
    st.markdown(f"### Status da Transação #{k}")
    st.metric(label="Market Share da Adquirente (Antes)", value=f"{market_share_anterior:.1f}%")
    st.metric(label="Variação do HHI ($\Delta HHI = 2 \cdot S_1 \cdot S_2$)", value=f"{delta_hhi:.2f} pontos")
    st.metric(label="HHI Total do Mercado", value=f"{hhi_total:.1f} pontos")
    
    if delta_hhi < 100:
        st.success("✅ **Abaixo de 100 pontos:** Transação passaria sem alerta automático do CADE.")
    else:
        st.error("⚠️ **Acima de 100 pontos:** Acima do limiar de notificação mandatória.")

with col_grafico:
    st.subheader("Trajetória do $\Delta HHI$ ao Longo das Aquisições")
    
    # Gerar dados para o gráfico de todas as 199 etapas
    etapas = list(range(1, 200))
    deltas = [2 * (i * 0.5) * 0.5 for i in etapas]
    
    df_plot = pd.DataFrame({
        'Etapa': etapas,
        'Delta_HHI': deltas
    })
    
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df_plot['Etapa'], df_plot['Delta_HHI'], color='#1f77b4', lw=2.2, label=r'$\Delta \text{HHI}$')
    ax.axhline(y=100, color='crimson', linestyle='--', linewidth=1.5, label='Limiar CADE (100 pts)')
    
    # Destacar o ponto atual selecionado
    ax.scatter([k], [delta_hhi], color='darkorange', s=120, zorder=5, label=f'Passo Atual ({k})')
    
    ax.set_title("Evolução da Variação do HHI por Aquisição Fracionada", fontsize=11, fontweight='bold')
    ax.set_xlabel("Número de empresas absorvidas", fontsize=10)
    ax.set_ylabel(r"$\Delta \text{HHI}$ (pontos)", fontsize=10)
    ax.legend(frameon=True, facecolor='#f8f9fa', edgecolor='none')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig)

st.info("""
**Insight Econômico:** 
Como $\Delta HHI = 2 \times S_1 \times 0{,}5 = S_1$, a variação do HHI em cada aquisição de $0{,}5\%$ é numericamente igual ao market share que o adquirente já acumulou. 
Mesmo na **última aquisição** (quando o agente já detém $99{,}5\%$ e compra o último $0{,}5\%$), a variação é de exatos **$99{,}5$ pontos**, ficando abaixo da linha de corte de $100$ pontos do CADE. Isso prova a necessidade de flexibilização e análise comportamental em mercados fragmentados.
""")

