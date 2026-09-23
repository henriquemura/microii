import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define constants for the model
P_DEMAND_INTERCEPT = 100 # Original demand intercept A in P = A - (Q1 + Q2)
Q2_MIN = 0
Q2_MAX = 75 # Based on the original ipywidgets IntText max value

st.set_page_config(layout="wide") # Use wide layout for better plot display
st.title('Decisão de producação em Cournot')

# Initialize session state for Q2 if not already present
if 'q2_value' not in st.session_state:
    st.session_state.q2_value = 0

st.sidebar.header('Controles de Quantidade da Empresa 2 (Q2)')

# Create columns for buttons and number_input in the sidebar
col1, col2, col3 = st.sidebar.columns([1, 2, 1])

with col1:
    if st.button('-5', key='minus_button_key'):
        st.session_state.q2_value = max(Q2_MIN, st.session_state.q2_value - 5)

with col2:
    # Display current Q2 value and allow direct input
    st.session_state.q2_value = st.number_input(
        'Q2 esperado:',
        min_value=Q2_MIN,
        max_value=Q2_MAX,
        value=st.session_state.q2_value,
        step=5,
        key='q2_input_widget' # Unique key for this widget
    )

with col3:
    if st.button('+5', key='plus_button_key'):
        st.session_state.q2_value = min(Q2_MAX, st.session_state.q2_value + 5)

# Get the current Q2 value from session state
q2 = st.session_state.q2_value

# --- Gráfico 1: Decisão de Produção da Empresa 1 ---
st.subheader(f' Decisão de Produção da Empresa 1 (Com Q2 esperado = {q2})')

# Function to calculate demand and marginal revenue for Firm 1
intercepto_p = max(0, P_DEMAND_INTERCEPT - q2)
q1_vals_plot1 = np.linspace(0, P_DEMAND_INTERCEPT, 200) # Plot range up to P_DEMAND_INTERCEPT

p_demanda = np.maximum(0, intercepto_p - q1_vals_plot1)
p_rmg = np.maximum(0, intercepto_p - 2 * q1_vals_plot1)

cmg = 0  # Marginal cost
q1_otimo = max(0, (P_DEMAND_INTERCEPT - q2) / 2) # Optimal Q1 for Firm 1

fig1, ax1 = plt.subplots(figsize=(9, 6))
ax1.plot(q1_vals_plot1, p_demanda, label=f'Demanda Residual $D_1({q2})$', color='#1f77b4', linewidth=2)
ax1.plot(q1_vals_plot1, p_rmg, label=f'Receita Marginal $RMg_1({q2})$', color='#ff7f0e', linestyle='--', linewidth=2)
ax1.axhline(y=cmg, color='black', linestyle='-', label='Custo Marginal ($CMg_1 = 0$)')

ax1.axvline(x=q1_otimo, color='red', linestyle=':', alpha=0.7)
ax1.scatter([q1_otimo], [cmg], color='red', s=70, zorder=5)
ax1.annotate(f'Ótimo: $Q_1 = {q1_otimo:.1f}$', (q1_otimo, cmg), textcoords="offset points", xytext=(10,10), fontweight='bold', color='red')

ax1.set_xlabel('Quantidade da Empresa 1 ($Q_1$)')
ax1.set_ylabel('Preço / RMg / CMg ($P$)')
ax1.autoscale_view(True,True,True) # Auto-adjust axes
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)
plt.tight_layout()
st.pyplot(fig1) # Display the plot in Streamlit
plt.close(fig1) # Close the figure to free up memory


# --- Gráfico 2: Curvas de Reação e Equilíbrio de Cournot ---
st.subheader('Curvas de Reação das Empresas e Equilíbrio de Cournot')

fig2, ax2 = plt.subplots(figsize=(9, 6))

q_range_plot2 = np.linspace(0, P_DEMAND_INTERCEPT, 200)

# Firm 1 Reaction Curve: Q1 = (A - Q2) / 2
# Plot (Q2, Q1) on (xlabel=Q2, ylabel=Q1) axes
reaction1 = np.maximum(0, (P_DEMAND_INTERCEPT - q_range_plot2) / 2) # This is Q1 as a function of Q2
ax2.plot(q_range_plot2, reaction1, label=f'Curva de Reação da Empresa 1 ($Q_1 = ({P_DEMAND_INTERCEPT} - Q_2)/2$)', color='blue')

# Firm 2 Reaction Curve: Q2 = (A - Q1) / 2
# To plot this on (xlabel=Q2, ylabel=Q1) axes, we need Q1 in terms of Q2:
# Q1 = A - 2 * Q2
reaction2_Q1 = np.maximum(0, P_DEMAND_INTERCEPT - 2 * q_range_plot2) # This is Q1 as a function of Q2
ax2.plot(q_range_plot2, reaction2_Q1, label=f'Curva de Reação da Empresa 2 ($Q_1 = {P_DEMAND_INTERCEPT} - 2Q_2$)', color='green')

# Cournot-Nash Equilibrium point
cournot_q = P_DEMAND_INTERCEPT / 3
ax2.plot(cournot_q, cournot_q, 'ro', markersize=8, label=f'Equilíbrio de Cournot-Nash ($Q_1=Q_2={cournot_q:.1f}$)')
ax2.annotate(f'C-N: ({cournot_q:.1f}, {cournot_q:.1f})', (cournot_q, cournot_q), textcoords="offset points", xytext=(10, -15), fontweight='bold', color='red')

# Current point (q2, q1_otimo) plotted on (Q2, Q1) axes
ax2.plot(q2, q1_otimo, 'kx', markersize=10, label=f'Ponto Atual ($Q_2={q2:.1f}, Q_1={q1_otimo:.1f}$)')
ax2.annotate(f'Atual: ({q2:.1f}, {q1_otimo:.1f})', (q2, q1_otimo), textcoords="offset points", xytext=(10, 10), fontweight='bold', color='black')

ax2.set_xlabel('Quantidade da Empresa 2 ($Q_2$)')
ax2.set_ylabel('Quantidade da Empresa 1 ($Q_1$)')
ax2.set_xlim(0, P_DEMAND_INTERCEPT) # Fixed limits for clarity
ax2.set_ylim(0, P_DEMAND_INTERCEPT) # Fixed limits for clarity
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)
