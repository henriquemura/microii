import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Simulador Cournot", layout="wide")

st.title("Simulador do Modelo de Cournot")
st.markdown("Baseado na planilha *Cournot* do arquivo **CartelDWL.xlsm**.")
st.markdown("Este aplicativo calcula o Equilíbrio de Nash-Cournot não cooperativo para n firmas idênticas competindo em quantidade.")

# Sidebar para Variáveis Exógenas
st.sidebar.header("Variáveis Exógenas")
st.sidebar.markdown("Demanda: $P = d_0 - d_1Q$")
st.sidebar.markdown("Custo Marginal (MC): $s_0$")

d0 = st.sidebar.number_input("Intercepto da Demanda (d0)", value=3300.0, step=100.0)
d1 = st.sidebar.number_input("Inclinação da Demanda (d1)", value=0.0005, format="%.5f", step=0.0001)
s0 = st.sidebar.number_input("Custo Marginal (s0)", value=1000.0, step=100.0)
n = st.sidebar.number_input("Número de firmas (n)", value=3, min_value=1, step=1)

# Validação simples
if s0 >= d0:
    st.error("O Custo Marginal (s0) deve ser menor que o intercepto da demanda (d0) para haver produção.")
    st.stop()

# Cálculos Analíticos do Modelo de Cournot
# Pela teoria de Cournot, a quantidade individual de cada firma (qi) é:
q = (d0 - s0) / (d1 * (n + 1))
# A quantidade total no mercado (Q) é n * q:
Q = n * q
# Preço de equilíbrio (P):
P = d0 - d1 * Q

# Cálculo para Concorrência Perfeita (P = MC) para medir a Perda de Peso Morto (DWL)
Q_pc = (d0 - s0) / d1

# Medidas de Bem-Estar (Welfare Surplus)
# Excedente do Consumidor (CS)
CS = 0.5 * (d0 - P) * Q
# Excedente do Produtor individual (Lucro) e Total (PS)
PS_firm = (P - s0) * q
PS_total = PS_firm * n
# Perda de Peso Morto (DWL)
DWL = 0.5 * (P - s0) * (Q_pc - Q)

# Exibição dos Resultados
st.header("Solução de Cournot (Equilíbrio)")
col1, col2, col3 = st.columns(3)
col1.metric("Preço (P)", f"kr {P:,.2f}")
col2.metric("Quantidade por Firma (q)", f"{q:,.0f}")
col3.metric("Quantidade Total (Q)", f"{Q:,.0f}")

st.header("Análise de Bem-Estar (Welfare)")
col4, col5, col6 = st.columns(3)
col4.metric("Excedente do Consumidor (CS)", f"kr {CS:,.2f}")
col5.metric("Excedente do Produtor Total (PS)", f"kr {PS_total:,.2f}")
col6.metric("Perda de Peso Morto (DWL)", f"kr {DWL:,.2f}")

st.markdown("---")

# Gráfico Interativo
st.header("Representação Gráfica")

# Gerando pontos para as curvas
q_plot = np.linspace(0, Q_pc * 1.2, 500)
p_demand = d0 - d1 * q_plot
# O preço não pode ser negativo no gráfico
p_demand = np.maximum(p_demand, 0)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot da Demanda
ax.plot(q_plot, p_demand, label='Demanda (D)', color='blue')
# Plot do Custo Marginal
ax.axhline(y=s0, color='red', linestyle='-', label='Custo Marginal (MC)')

# Ponto de Equilíbrio Cournot
ax.plot(Q, P, marker='o', markersize=8, color='black', label=f'Equilíbrio Cournot (Q={Q:,.0f}, P={P:,.2f})')
ax.vlines(x=Q, ymin=0, ymax=P, color='gray', linestyle='--')
ax.hlines(y=P, xmin=0, xmax=Q, color='gray', linestyle='--')

# Configurações do Gráfico
ax.set_title("Gráfico de Demanda e Custo Marginal", fontsize=14)
ax.set_xlabel("Quantidade (Q)")
ax.set_ylabel("Preço (P)")
ax.set_ylim(0, d0 * 1.1)
ax.set_xlim(0, Q_pc * 1.2)
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
