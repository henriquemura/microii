import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página do Streamlit
st.set_page_config(page_title="Simulador Cournot", layout="wide")
st.title("Simulador do Modelo de Cournot")

# ==========================================
# 1. ENTRADA DE VARIÁVEIS (SIDEBAR)
# ==========================================
st.sidebar.header("Modifique as Variáveis Aqui")
d0 = st.sidebar.number_input("Intercepto da Demanda (d0 - P máximo)", value=3300.0, step=100.0)
d1 = st.sidebar.number_input("Inclinação da Demanda (d1)", value=0.0005, format="%.5f", step=0.0001)
s0 = st.sidebar.number_input("Custo Marginal (s0 - MC)", value=1000.0, step=100.0)
n = st.sidebar.number_input("Número de firmas a competir (n)", value=3, min_value=1, step=1)

# ==========================================
# 2. CÁLCULOS DO MODELO DE COURNOT
# ==========================================
if s0 >= d0:
    st.error("Erro: O Custo Marginal (s0) deve ser menor que o intercepto da demanda (d0).")
else:
    # Quantidade individual por firma
    q = (d0 - s0) / (d1 * (n + 1))
    
    # Quantidade total do mercado
    Q = n * q
    
    # Preço de equilíbrio
    P = d0 - d1 * Q

    # Quantidade em concorrência perfeita (para calcular a perda de peso morto)
    Q_pc = (d0 - s0) / d1

    # ==========================================
    # 3. CÁLCULOS DE BEM-ESTAR (WELFARE)
    # ==========================================
    # Excedente do Consumidor (CS)
    CS = 0.5 * (d0 - P) * Q
    
    # Excedente do Produtor individual e Total (PS)
    PS_firm = (P - s0) * q
    PS_total = PS_firm * n
    
    # Perda de Peso Morto (DWL)
    DWL = 0.5 * (P - s0) * (Q_pc - Q)

    # ==========================================
    # 4. EXIBIÇÃO DOS RESULTADOS (TEXTO)
    # ==========================================
    st.subheader("--- RESULTADOS DE COURNOT ---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Preço (P)", f" {P:,.2f}")
    col2.metric("Quantidade por Firma (q)", f"{q/1000:,.0f}k")
    col3.metric("Quantidade Total (Q)", f"{Q/1000:,.0f}k")

    st.subheader("--- ANÁLISE DE BEM-ESTAR ---")
    col4, col5, col6 = st.columns(3)
    col4.metric("Excedente do Consumidor (CS)", f" {CS/1000000:,.2f}M")
    col5.metric("Excedente do Produtor Total (PS)", f" {PS_total/1000000:,.2f}M")
    col6.metric("Perda de Peso Morto (DWL)", f" {DWL/1000000:,.2f}M")

    # ==========================================
    # 5. GERAÇÃO DO GRÁFICO
    # ==========================================
    st.markdown("---")
    
    # Vetor de quantidades para desenhar a linha da demanda (em milhares)
    q_plot_k = np.linspace(0, (Q_pc * 1.2)/1000, 500)
    p_demand = d0 - d1 * (q_plot_k * 1000) # Use original d1 with scaled Q for plotting function
    p_demand = np.maximum(p_demand, 0) # Evita preços negativos no gráfico

    # Cria a figura para o Streamlit (usando subplots é mais seguro em ambientes web)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Linhas principais
    ax.plot(q_plot_k, p_demand, label='Demanda (D)', color='blue')
    ax.axhline(y=s0, color='red', linestyle='-', label='Custo Marginal (MC)')
    
    # Ponto de equilíbrio (em milhares)
    ax.plot(Q/1000, P, marker='o', markersize=8, color='black', label=f'Equilíbrio Cournot (Q={Q/1000:,.0f}k, P={P:,.2f})')
    
    # Linhas tracejadas para mostrar as coordenadas do ponto (em milhares)
    ax.vlines(x=Q/1000, ymin=0, ymax=P, color='gray', linestyle='--')
    ax.hlines(y=P, xmin=0, xmax=Q/1000, color='gray', linestyle='--')

    # Área de Perda de Peso Morto (DWL) (em milhares)
    dwl_x_k = [val / 1000 for val in [Q, Q_pc, Q, Q]] # Scale x-coordinates for DWL
    dwl_y = [P, s0, s0, P]
    ax.fill(dwl_x_k, dwl_y, color='purple', alpha=0.3, label='Perda de Peso Morto (DWL)')

    # Configurações visuais do gráfico
    ax.set_title("Gráfico de Demanda e Custo Marginal (Modelo de Cournot)")
    ax.set_xlabel("Quantidade Total (Q, em milhares)")
    ax.set_ylabel("Preço (P)")
    ax.set_ylim(0, d0 * 1.1)
    ax.set_xlim(0, (Q_pc * 1.2)/1000)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Exibe o gráfico final no Streamlit
    st.pyplot(fig)
