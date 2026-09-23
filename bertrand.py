%%writefile edgeworth_streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simular_edgeworth_streamlit():
    st.title('Paradoxo de Edgeworth: Simulação Interativa')
    st.write('Explore o modelo de Bertrand com restrição de capacidade (Paradoxo de Edgeworth).')

    # Valores padrão para custo e capacidade
    custo_c = 5

    st.sidebar.header('Parâmetros da Simulação')
    capacidade_k = st.sidebar.number_input('Capacidade (K):', min_value=10, max_value=90, value=60, step=5)
    p1 = st.sidebar.number_input('Preço Empresa 1 ($p_1$):', min_value=5.0, max_value=40.0, value=10.0, step=0.5)
    p2 = st.sidebar.number_input('Preço Empresa 2 ($p_2$):', min_value=5.0, max_value=40.0, value=10.0, step=0.5)

    # Função de cálculo de lucro com racionamento de capacidade
    def calc(p_a, p_b, cap):
        p_min = min(p_a, p_b)
        q_tot = max(0, 100 - p_min)
        if p_a == p_b:
            return (p_a - custo_c) * min(q_tot / 2, cap)
        elif p_a < p_b:
            q_a = min(q_tot, cap)
            return (p_a - custo_c) * q_a
        else:
            q_b = min(q_tot, cap)
            residual = max(0, 100 - p_a - q_b)
            return (p_a - custo_c) * min(residual, cap)

    pi1 = calc(p1, p2, capacidade_k)
    pi2 = calc(p2, p1, capacidade_k)

    # Criação do gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    p_vals = np.linspace(5, 40, 150)
    pi1_vals = [calc(p, p2, capacidade_k) for p in p_vals]

    ax.plot(p_vals, pi1_vals, label=f'Lucro Emp. 1 (com $p_2$ = {p2})', color='#1f77b4', linewidth=2)
    ax.axvline(x=p1, color='red', linestyle='--', label=f'Preço atual Emp. 1 ($p_1$ = {p1})')

    ax.set_title(f'Função de Lucro da Empresa 1\nLucro Emp. 1 = {pi1:.2f} | Lucro Emp. 2 = {pi2:.2f}', fontsize=14)
    ax.set_xlabel('Preço da Empresa 1 ($p_1$)', fontsize=12)
    ax.set_ylabel(r'Lucro ($\pi_1$)', fontsize=12) # Corrected backslash for LaTeX
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

    st.markdown(f"""
    ### Resultados Atuais:
    *   **Custo Marginal (c):** {custo_c}
    *   **Capacidade (K):** {capacidade_k}
    *   **Preço Empresa 1 ($p_1$):** {p1}
    *   **Preço Empresa 2 ($p_2$):** {p2}
    *   **Lucro Empresa 1 ($\pi_1$):** {pi1:.2f}
    *   **Lucro Empresa 2 ($\pi_2$):** {pi2:.2f}
    """)

if __name__ == '__main__':
    simular_edgeworth_streamlit()
