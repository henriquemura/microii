# =====================================================================
#  Cara ou coroa — recompensas esperadas em estratégias mistas
#  Adaptação do jogo de prevenção de ataque de Fiani, Teoria dos Jogos,
#  cap. 5 (Figura 5.5, Gráficos 5.1 e 5.2, pp. 195-198):
#     Azul (defensor, linhas)       -> Jogador 1
#     Vermelho (atacante, colunas)  -> Jogador 2
#     Porto Sul                     -> Cara
#     Porto Norte                   -> Coroa
#  Notação do livro:
#     p = Pr(Jogador 1 jogar Cara)  (eixo horizontal)
#     q = Pr(Jogador 2 jogar Cara)  (eixo vertical do Gráfico 5.2)
#  Matrizes: A[i, j] = recompensa do J1, B[i, j] = recompensa do J2,
#            i = estratégia do J1 (linha: 0 = Cara, 1 = Coroa), j = do J2 (coluna).
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

EST = ["Cara", "Coroa"]
COR1, COR2, COR_EQ = "#c62828", "#1565c0", "#2e7d32"


def matriz_padrao():
    """Jogo do livro: o J1 (defensor) ganha 1 quando as escolhas coincidem
    e perde 1 quando diferem; o J2 (atacante) recebe o oposto."""
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    return A, -A


# ---------------------------------------------------------------------
# Recompensas esperadas
# ---------------------------------------------------------------------
def esperadas_j2(B, p):
    """Recompensa esperada do J2 ao jogar Cara e ao jogar Coroa, dado p."""
    p = np.asarray(p, float)
    return B[0, 0] * p + B[1, 0] * (1 - p), B[0, 1] * p + B[1, 1] * (1 - p)


def esperadas_j1(A, q):
    """Recompensa esperada do J1 ao jogar Cara e ao jogar Coroa, dado q."""
    q = np.asarray(q, float)
    return A[0, 0] * q + A[0, 1] * (1 - q), A[1, 0] * q + A[1, 1] * (1 - q)


def rev(M, p, q):
    """Recompensa esperada para quaisquer estratégias mistas (p do J1, q do J2)."""
    return (p * q * M[0, 0] + p * (1 - q) * M[0, 1]
            + (1 - p) * q * M[1, 0] + (1 - p) * (1 - q) * M[1, 1])


# ---------------------------------------------------------------------
# Indiferença, melhores respostas e equilíbrios
# ---------------------------------------------------------------------
def p_indiferenca(B):
    """p que deixa o J2 indiferente entre Cara e Coroa."""
    den = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    return None if np.isclose(den, 0) else (B[1, 1] - B[1, 0]) / den


def q_indiferenca(A):
    """q que deixa o J1 indiferente entre Cara e Coroa."""
    den = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    return None if np.isclose(den, 0) else (A[1, 1] - A[0, 1]) / den


def melhor_resposta(e_cara, e_coroa, tol=1e-9):
    if e_cara > e_coroa + tol:
        return "Cara"
    if e_coroa > e_cara + tol:
        return "Coroa"
    return "indiferente"


def equilibrios(A, B):
    """Puros: lista de (i, j). Misto: (p*, q*) com 0 < p*, q* < 1."""
    puros = [(i, j) for i in range(2) for j in range(2)
             if A[i, j] >= A[1 - i, j] and B[i, j] >= B[i, 1 - j]]
    p, q = p_indiferenca(B), q_indiferenca(A)
    misto = (p, q) if (p is not None and q is not None and 0 < p < 1 and 0 < q < 1) else None
    return puros, misto


def _curva_mr(f, x_ind):
    """Correspondência de melhor resposta (prob. de Cara) em função da prob. do outro."""
    if x_ind is None or not (0 < x_ind < 1):
        c, k = f(0.5)
        v = 1.0 if c > k else 0.0
        return [0, 1], [v, v]
    c0, k0 = f(0.0)
    v0 = 1.0 if c0 > k0 else 0.0
    return [0, x_ind, x_ind, 1], [v0, v0, 1 - v0, 1 - v0]


def _frac(x):
    for n, d in [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5), (2, 5), (3, 5), (4, 5)]:
        if np.isclose(x, n / d):
            return f"{n}/{d}"
    return f"{x:.2f}"


# ---------------------------------------------------------------------
# Gráfico 5.1 — recompensas esperadas, no estilo do livro
# ---------------------------------------------------------------------
def _grafico_recompensas(ax, x, e_cara, e_coroa, xv, x_ind, jog, outro, var):
    lo = min(e_cara.min(), e_coroa.min(), 0); hi = max(e_cara.max(), e_coroa.max(), 0)
    pad = 0.12 * (hi - lo if hi > lo else 1)
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)
    ax.plot([0, 0], [lo - pad, hi + pad], color="k", lw=1.2)          # eixo vertical esquerdo
    ax.plot([1, 1], [lo - pad, hi + pad], color="k", lw=1.2)          # eixo vertical direito
    ax.plot([0, 1.04], [0, 0], color="k", lw=1.0)                     # eixo horizontal
    ax.text(1.06, 0, var, va="center", fontsize=12, style="italic")
    ax.text(0, hi + pad * 1.3, f"Recompensa\nesperada do\n{jog}", ha="center", va="bottom", fontsize=8.5)
    ax.text(1, hi + pad * 1.3, f"Recompensa\nesperada do\n{jog}", ha="center", va="bottom", fontsize=8.5)

    ax.plot(x, e_cara, ls="--", color=COR1, lw=2.2)
    ax.plot(x, e_coroa, ls="-", color=COR2, lw=2.2)
    ax.plot(x, np.maximum(e_cara, e_coroa), color="k", lw=7, alpha=0.10, solid_capstyle="butt")
    i_c, i_k = int(len(x) * 0.18), int(len(x) * 0.80)
    ax.text(x[i_c], e_cara[i_c], f"  {jog} joga Cara", color=COR1, fontsize=10, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.text(x[i_k], e_coroa[i_k], f"{jog} joga Coroa  ", color=COR2, fontsize=10, va="bottom", ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    for v in sorted({e_cara[0], e_coroa[0]}):
        ax.text(-0.02, v, f"{v:g}", ha="right", va="center", fontsize=9)
    for v in sorted({e_cara[-1], e_coroa[-1]}):
        ax.text(1.02, v, f"{v:g}", ha="left", va="center", fontsize=9)
    ax.text(-0.02, 0, "0", ha="right", va="center", fontsize=9)

    if x_ind is not None and 0 < x_ind < 1:
        y = float(np.interp(x_ind, x, e_cara))
        ax.scatter(x_ind, y, s=70, color=COR_EQ, zorder=6)
        ax.plot([x_ind, x_ind], [0, y], color=COR_EQ, ls=":", lw=1.2)
        ax.text(x_ind, -pad * 0.55 if y >= 0 else pad * 0.3, _frac(x_ind), ha="center", fontsize=10, color=COR_EQ)

    yc, yk = float(np.interp(xv, x, e_cara)), float(np.interp(xv, x, e_coroa))
    ax.plot([xv, xv], [lo - pad * 0.6, hi + pad * 0.6], color="0.35", lw=1.2)
    ax.scatter([xv], [yc], s=60, color=COR1, edgecolor="k", zorder=7)
    ax.scatter([xv], [yk], s=60, color=COR2, edgecolor="k", zorder=7)
    lado = -1 if xv > 0.75 else 1
    ha = "right" if lado < 0 else "left"
    ax.annotate(f"Cara: {yc:.2f}", (xv, yc), xytext=(8 * lado, 6), textcoords="offset points",
                ha=ha, color=COR1, fontsize=9, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.annotate(f"Coroa: {yk:.2f}", (xv, yk), xytext=(8 * lado, -14), textcoords="offset points",
                ha=ha, color=COR2, fontsize=9, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.text(xv, lo - pad * 0.95, f"{var} = {xv:.2f}", ha="center", fontsize=9, color="0.25")
    ax.set_xlim(-0.12, 1.14); ax.set_ylim(lo - pad * 1.3, hi + pad * 3.2)
    ax.set_xticks([]); ax.set_yticks([])
    return melhor_resposta(yc, yk)


def plot_grafico_5_1(B, p):
    """Recompensas esperadas do J2, dada a estratégia mista do J1 (Gráfico 5.1)."""
    x = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    c, k = esperadas_j2(B, x)
    mr = _grafico_recompensas(ax, x, c, k, p, p_indiferenca(B), "Jogador 2", "Jogador 1", "p")
    ax.set_title("Recompensas esperadas do Jogador 2, dada a estratégia mista do Jogador 1\n"
                 f"p = Pr(Jogador 1 jogar Cara) = {p:.2f}   →   melhor resposta do Jogador 2: {mr}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_grafico_5_2(B, p, q=None):
    """Melhores respostas do J2: q como função de p (Gráfico 5.2)."""
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    xs, ys = _curva_mr(lambda t: esperadas_j2(B, t), p_indiferenca(B))
    xx, yy = [], []
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        n = max(2, int(30 * np.hypot(x1 - x0, y1 - y0)))
        xx += list(np.linspace(x0, x1, n)); yy += list(np.linspace(y0, y1, n))
    ax.plot(xx, yy, ls="", marker="o", ms=3.5, color=COR1, label="melhor resposta do Jogador 2")
    ax.plot([0, 1, 1], [1, 1, 0], color="0.6", ls="--", lw=0.8)
    pi = p_indiferenca(B)
    c, k = esperadas_j2(B, p)
    mr = melhor_resposta(float(c), float(k))
    if mr == "indiferente":
        ax.plot([p, p], [0, 1], color=COR_EQ, lw=5, alpha=0.35, label="qualquer q é melhor resposta")
    else:
        ax.scatter(p, 1.0 if mr == "Cara" else 0.0, s=110, color=COR_EQ, edgecolor="k", zorder=6,
                   label=f"melhor resposta em p = {p:.2f}")
    ax.axvline(p, color="0.35", lw=1)
    if q is not None:
        ax.scatter(p, q, s=55, marker="s", color="0.2", zorder=7, label=f"(p, q) escolhidos = ({p:.2f}, {q:.2f})")
    ticks = [0, 1] + ([pi] if pi is not None and 0 < pi < 1 else [])
    ax.set_xticks(sorted(ticks), [_frac(t) if 0 < t < 1 else f"{t:g}" for t in sorted(ticks)])
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.03, 1.08); ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("p = Pr(Jogador 1 jogar Cara)"); ax.set_ylabel("q = Pr(Jogador 2 jogar Cara)")
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.set_title("Melhores respostas do Jogador 2", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Extensões: Jogador 1 e equilíbrio
# ---------------------------------------------------------------------
def plot_recompensas_j1(A, q):
    x = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    c, k = esperadas_j1(A, x)
    mr = _grafico_recompensas(ax, x, c, k, q, q_indiferenca(A), "Jogador 1", "Jogador 2", "q")
    ax.set_title("Recompensas esperadas do Jogador 1, dada a estratégia mista do Jogador 2\n"
                 f"q = Pr(Jogador 2 jogar Cara) = {q:.2f}   →   melhor resposta do Jogador 1: {mr}",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_equilibrio(A, B, p, q):
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    xs, ys = _curva_mr(lambda t: esperadas_j2(B, t), p_indiferenca(B))          # q(p)
    ax.plot(xs, ys, color=COR1, lw=3, label="MR do Jogador 2: q(p)")
    xs2, ys2 = _curva_mr(lambda t: esperadas_j1(A, t), q_indiferenca(A))        # p(q)
    ax.plot(ys2, xs2, color=COR2, lw=3, ls="--", label="MR do Jogador 1: p(q)")
    puros, misto = equilibrios(A, B)
    pts = [(1.0 - i, 1.0 - j) for i, j in puros] + ([misto] if misto else [])
    for k_, (pp, qq) in enumerate(pts):
        ax.scatter(pp, qq, s=260, marker="*", color="gold", edgecolor="k", zorder=6,
                   label="equilíbrio de Nash" if k_ == 0 else None)
    ax.scatter(p, q, s=60, marker="s", color="0.2", zorder=7, label="(p, q) escolhidos")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("p = Pr(Jogador 1 jogar Cara)"); ax.set_ylabel("q = Pr(Jogador 2 jogar Cara)")
    ax.set_title("Melhores respostas dos dois jogadores", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def resumo(A, B, p, q):
    c, k = esperadas_j2(B, p)
    linhas = [f"Dado p = {p:.2f}:  E2(Cara) = {float(c):.3f}   E2(Coroa) = {float(k):.3f}   "
              f"→ melhor resposta do J2: {melhor_resposta(float(c), float(k))}",
              f"REV do Jogador 1 com (p, q) = ({p:.2f}, {q:.2f}): {rev(A, p, q):.3f}",
              f"REV do Jogador 2 com (p, q) = ({p:.2f}, {q:.2f}): {rev(B, p, q):.3f}"]
    puros, misto = equilibrios(A, B)
    linhas.append("Equilíbrios em estratégias puras: " +
                  ("; ".join(f"(J1 {EST[i]}, J2 {EST[j]})" for i, j in puros) if puros else "nenhum"))
    if misto:
        linhas.append(f"Equilíbrio em estratégias mistas: p* = {misto[0]:.3f}, q* = {misto[1]:.3f}  "
                      f"(REV1 = {rev(A, *misto):.3f}, REV2 = {rev(B, *misto):.3f})")
    return "\n".join(linhas)


FORMULAS = """Matriz (J1 nas linhas, J2 nas colunas): célula (i, j) = (a_ij , b_ij)

E2(Cara | p)  = b11·p + b21·(1 - p)
E2(Coroa | p) = b12·p + b22·(1 - p)
p* (indiferença do J2) = (b22 - b21) / (b11 - b21 - b12 + b22)

REV2 = p·q·b11 + p·(1 - q)·b12 + (1 - p)·q·b21 + (1 - p)·(1 - q)·b22

No jogo do livro (b11 = -1, b12 = 1, b21 = 1, b22 = -1):
E2(Cara | p)  = 1 - 2p
E2(Coroa | p) = 2p - 1
REV2 = pq(-1) + p(1 - q)(1) + (1 - p)q(1) + (1 - p)(1 - q)(-1) = (2p - 1)(1 - 2q)
→ com p = 1/2, REV2 = 0 para qualquer q"""
