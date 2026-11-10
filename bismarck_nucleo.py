# =====================================================================
#  BATALHA DO MAR DE BISMARCK (1943) — Haywood (1954), "Military
#  Decision and Game Theory", JORSA 2(4): 365-385.
#  Núcleo do modelo: geografia estilizada, clima, detecção, danos,
#  solução do jogo de soma zero 2x2 e gráficos (matplotlib).
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# 1. Geografia estilizada (coordenadas aproximadas lon/lat)
# ---------------------------------------------------------------------
RABAUL = (152.20, -4.20)
LAE = (147.00, -6.73)

NOVA_BRETANHA = [(148.45, -5.70), (148.80, -5.40), (149.60, -5.50), (150.20, -5.40),
                 (150.80, -5.20), (151.50, -4.90), (152.00, -4.30), (152.40, -4.20),
                 (152.20, -4.60), (151.90, -5.00), (151.60, -5.60), (151.00, -6.00),
                 (150.20, -6.20), (149.40, -6.10), (148.60, -5.95)]

NOVA_GUINE = [(144.50, -4.00), (145.80, -4.90), (146.80, -5.70), (147.60, -6.00),
              (147.85, -6.15), (147.90, -6.45), (147.55, -6.75), (147.00, -6.72),
              (147.15, -7.10), (147.60, -7.60), (148.30, -8.30), (149.30, -9.20),
              (150.30, -10.3), (144.50, -10.3)]

# Previsão do estado-maior aliado (Gen. Kenney): chuva e baixa visibilidade ao norte da Nova Bretanha
ZONA_CHUVA = [(148.50, -3.00), (153.30, -3.00), (153.30, -4.05), (152.30, -4.05),
              (151.60, -4.60), (150.50, -5.00), (149.30, -5.25), (148.50, -5.30)]

ROTAS = {
    "Norte": np.array([RABAUL, (151.50, -4.40), (150.50, -4.80), (149.50, -5.00),
                       (148.60, -5.10), (148.10, -5.85), (148.15, -6.40), (147.70, -6.90), LAE]),
    "Sul": np.array([RABAUL, (152.55, -4.50), (152.30, -5.20), (151.80, -6.15),
                     (150.80, -6.55), (149.50, -6.65), (148.60, -6.65), (147.80, -7.00), LAE]),
}
ESTRATEGIAS = ["Norte", "Sul"]
N_DIAS = 3  # qualquer das rotas leva três dias

# Matriz do artigo (p. 370): dias de bombardeio. Chave = (reconhecimento aliado, rota japonesa)
MATRIZ_ARTIGO = {("Norte", "Norte"): 2, ("Norte", "Sul"): 2,
                 ("Sul", "Norte"): 1, ("Sul", "Sul"): 3}


def parametros_padrao():
    """Parâmetros estilizados, calibrados para que os valores esperados
    reproduzam aproximadamente a matriz 2-2 / 1-3 de Haywood."""
    return {
        # prob. de avistar o comboio num dia, dado cobertura x clima
        # dia 1 ao norte: chuva forte (frente sobre Rabaul); dia 2: chuva
        "p_conc_chuvaforte": 0.15, "p_conc_chuva": 0.80, "p_conc_limpo": 0.95,
        "p_esp_chuvaforte": 0.02, "p_esp_chuva": 0.08, "p_esp_limpo": 0.50,
        "p_lae": 0.90,             # dia 3 em tempo limpo, perto das bases aliadas
        # prob. de afundar um navio por dia de bombardeio
        "q_transp_rasante": 0.60, "q_destr_rasante": 0.30,   # skip bombing (surpresa tática)
        "q_transp_conv": 0.10, "q_destr_conv": 0.05,         # bombardeio em altitude
        "n_transp": 8, "n_destr": 8, "tropas": 6900, "frac_morte": 0.45,
    }


# ---------------------------------------------------------------------
# 2. Geometria das rotas
# ---------------------------------------------------------------------
def _comprimentos(pts):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def ponto_na_rota(rota, s):
    """s em [0,1] = fração do percurso (0 = Rabaul, 1 = Lae)."""
    pts = ROTAS[rota]
    L = _comprimentos(pts)
    alvo = np.asarray(s) * L[-1]
    return np.interp(alvo, L, pts[:, 0]), np.interp(alvo, L, pts[:, 1])


def trecho(rota, s0, s1, n=80):
    x, y = ponto_na_rota(rota, np.linspace(s0, s1, n))
    return np.column_stack([x, y])


_PATH_CHUVA = Path(ZONA_CHUVA)


def clima_por_dia(rota):
    """Dia com mais da metade do trecho na zona de chuva = 'chuva'; no dia 1
    (saída de Rabaul, sob a frente de mau tempo) = 'chuva forte'."""
    out = []
    for d in range(N_DIAS):
        frac = _PATH_CHUVA.contains_points(trecho(rota, d / N_DIAS, (d + 1) / N_DIAS, 200)).mean()
        out.append(("chuva forte" if d == 0 else "chuva") if frac > 0.5 else "limpo")
    return out


# ---------------------------------------------------------------------
# 3. Detecção e dias de bombardeio
# ---------------------------------------------------------------------
def probs_deteccao(recon, rota, P):
    clima = clima_por_dia(rota)
    cob = "conc" if recon == rota else "esp"
    ps = []
    for d, c in enumerate(clima, start=1):
        p = P[f"p_{cob}_{c.replace(' ', '')}"]
        if d == N_DIAS and c == "limpo":
            p = max(p, P["p_lae"])
        ps.append(p)
    return ps, clima


def dist_dias_bombardeio(ps):
    """Avistado no dia d  =>  B = 4 - d dias de bombardeio (discretização do artigo)."""
    sobrev, dist = 1.0, {}
    for d, p in enumerate(ps, start=1):
        dist[N_DIAS + 1 - d] = sobrev * p
        sobrev *= 1 - p
    dist[0] = sobrev
    return dist


def dias_esperados(ps):
    return sum(b * pr for b, pr in dist_dias_bombardeio(ps).items())


def qs(P, rasante):
    return ((P["q_transp_rasante"], P["q_destr_rasante"]) if rasante
            else (P["q_transp_conv"], P["q_destr_conv"]))


def matriz_payoff(P, modo):
    """Linhas = Aliados (maximizador), colunas = comboio japonês (minimizador)."""
    A = np.zeros((2, 2))
    for i, r in enumerate(ESTRATEGIAS):
        for j, c in enumerate(ESTRATEGIAS):
            A[i, j] = (MATRIZ_ARTIGO[(r, c)] if modo == "artigo"
                       else dias_esperados(probs_deteccao(r, c, P)[0]))
    return A


# ---------------------------------------------------------------------
# 4. Solução do jogo de soma zero 2x2
# ---------------------------------------------------------------------
def resolver_jogo(A):
    A = np.asarray(A, float)
    rmin, cmax = A.min(axis=1), A.max(axis=0)
    res = {"maximin": rmin.max(), "minimax": cmax.min(),
           "i": int(rmin.argmax()), "j": int(cmax.argmin())}
    if np.isclose(res["maximin"], res["minimax"]):
        res.update(tipo="sela", valor=res["maximin"],
                   p=1.0 if res["i"] == 0 else 0.0, q=1.0 if res["j"] == 0 else 0.0)
    else:
        a, b, c, d = A.ravel()
        den = a - b - c + d
        res.update(tipo="mista", p=(d - c) / den, q=(d - b) / den, valor=(a * d - b * c) / den)
    return res


# ---------------------------------------------------------------------
# 5. Simulação de uma travessia e Monte Carlo
# ---------------------------------------------------------------------
def simular(recon, rota, P, modo="artigo", rasante=True, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    ps, clima = probs_deteccao(recon, rota, P)
    qt, qd = qs(P, rasante)
    nT, nD = P["n_transp"], P["n_destr"]
    if modo == "artigo":
        B = MATRIZ_ARTIGO[(recon, rota)]
        d, u = N_DIAS + 1 - B, 0.30
        # danos ESPERADOS por dia de bombardeio
        t_dia = [nT * (1 - qt) ** k * qt for k in range(B)]
        d_dia = [nD * (1 - qd) ** k * qd for k in range(B)]
    else:
        d = next((k for k, p in enumerate(ps, start=1) if rng.random() < p), None)
        B = 0 if d is None else N_DIAS + 1 - d
        u = rng.random()
        t_dia, d_dia, vt, vd = [], [], nT, nD
        for _ in range(B):
            kt, kd = rng.binomial(vt, qt), rng.binomial(vd, qd)
            t_dia.append(kt); d_dia.append(kd); vt -= kt; vd -= kd
    T = float(sum(t_dia))
    return {"recon": recon, "rota": rota, "modo": modo, "rasante": rasante,
            "ps": ps, "clima": clima, "dia_avist": d, "B": B,
            "s_av": None if d is None else (d - 1 + u) / N_DIAS,
            "t_dia": t_dia, "d_dia": d_dia, "T": T, "D": float(sum(d_dia)),
            "tropas_perdidas": T * P["tropas"] / nT * P["frac_morte"]}


def monte_carlo(recon, rota, P, rasante=True, n=10_000, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    ps, _ = probs_deteccao(recon, rota, P)
    det = rng.random((n, N_DIAS)) < np.array(ps)
    primeiro = np.where(det.any(axis=1), det.argmax(axis=1) + 1, 0)
    B = np.where(primeiro > 0, N_DIAS + 1 - primeiro, 0)
    qt, qd = qs(P, rasante)
    T = rng.binomial(P["n_transp"], 1 - (1 - qt) ** B)
    D = rng.binomial(P["n_destr"], 1 - (1 - qd) ** B)
    return {"B": B, "T": T, "D": D, "tropas": T * P["tropas"] / P["n_transp"] * P["frac_morte"]}



# ---------------------------------------------------------------------
# 5b. Ícones (marcadores desenhados como Path): navio e avião
# ---------------------------------------------------------------------
def _poligono(pts):
    pts = list(pts) + [pts[0]]
    return pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY]


def _composto(*polys):
    v, c = [], []
    for p in polys:
        pv, pc = _poligono(p)
        v += pv; c += pc
    return Path(v, c)


# navio visto de lado: casco + superestrutura + chaminé
NAVIO = _composto([(-1.0, 0.0), (1.0, 0.0), (0.78, -0.32), (-0.85, -0.32)],
                  [(-0.45, 0.0), (0.35, 0.0), (0.35, 0.28), (-0.45, 0.28)],
                  [(-0.05, 0.28), (0.12, 0.28), (0.12, 0.55), (-0.05, 0.55)])

# avião visto de cima, nariz para cima
AVIAO = _composto([(0, 1.0), (0.09, 0.78), (0.09, 0.28), (0.95, 0.05), (0.95, -0.09),
                   (0.09, -0.05), (0.07, -0.58), (0.34, -0.74), (0.34, -0.86), (0, -0.80),
                   (-0.34, -0.86), (-0.34, -0.74), (-0.07, -0.58), (-0.09, -0.05),
                   (-0.95, -0.09), (-0.95, 0.05), (-0.09, 0.28), (-0.09, 0.78)])

COR_TRANSP, COR_DESTR, COR_AFUND = "#8d5524", "#37474f", "#bdbdbd"
COR_RECON, COR_BOMB = "#1f3a93", "#b71c1c"


def _painel_frota(ax, res, P):
    """Quadro com os 16 navios do comboio: afundados em cinza com X vermelho."""
    nT, nD = P["n_transp"], P["n_destr"]
    kT = int(round(min(res["T"], nT))); kD = int(round(min(res["D"], nD)))
    x0, y0, dx = 150.05, -7.72, 0.38
    ax.add_patch(Rectangle((x0 - 0.3, y0 - 0.55), 3.35, 0.95, fc="white", ec="0.5",
                           alpha=0.92, zorder=11))
    aprox = "≈ " if res["modo"] == "artigo" else ""
    ax.text(x0 - 0.2, y0 + 0.24, f"Comboio japonês ao chegar a Lae ({aprox}{nT - kT} transp. e "
            f"{aprox}{nD - kD} destr. restantes)", fontsize=7.5, zorder=12, va="center")
    for linha, (n, k, cor, nome) in enumerate([(nT, kT, COR_TRANSP, "transp."),
                                                (nD, kD, COR_DESTR, "destr.")]):
        y = y0 - 0.08 - linha * 0.32
        ax.text(x0 - 0.2, y, nome, fontsize=7, va="center", zorder=12)
        for i in range(n):
            x = x0 + 0.42 + i * dx * 0.85
            afund = i < k
            ax.scatter(x, y, marker=NAVIO, s=520, color=COR_AFUND if afund else cor,
                       edgecolor="none", zorder=12)
            if afund:
                ax.scatter(x, y + 0.02, marker="x", s=30, color="#d32f2f", lw=1.3, zorder=13)


# ---------------------------------------------------------------------
# 6. Gráficos
# ---------------------------------------------------------------------
def plot_mapa(res, P):
    fig, ax = plt.subplots(figsize=(12, 7.4))
    ax.set_facecolor("#d6e6f2")
    ax.add_patch(Polygon(ZONA_CHUVA, closed=True, facecolor="#7a8a99", alpha=0.30,
                         hatch="..", edgecolor="#56636f", lw=0.8, zorder=1))
    for poly in (NOVA_GUINE, NOVA_BRETANHA):
        ax.add_patch(Polygon(poly, closed=True, facecolor="#e9dfc4",
                             edgecolor="#8a7a55", lw=1.2, zorder=2))
    for txt, xy, kw in [("MAR DE BISMARCK", (149.3, -3.6), dict(fontsize=13, color="#2c4a66", weight="bold")),
                        ("chuva e baixa visibilidade", (148.7, -3.95), dict(fontsize=9, color="#3b4a57", style="italic")),
                        ("MAR DE SALOMÃO", (148.3, -7.75), dict(fontsize=11, color="#2c4a66", weight="bold")),
                        ("tempo limpo", (150.3, -7.05), dict(fontsize=9, color="#3b4a57", style="italic")),
                        ("Nova Bretanha", (149.7, -5.85), dict(fontsize=10, color="#5a4a25")),
                        ("Nova Guiné", (145.3, -7.4), dict(fontsize=11, color="#5a4a25"))]:
        ax.text(*xy, txt, zorder=4, **kw)

    rota, recon = res["rota"], res["recon"]
    outra = ROTAS["Sul" if rota == "Norte" else "Norte"]
    ax.plot(outra[:, 0], outra[:, 1], ls="--", color="gray", alpha=0.5, lw=1.5, zorder=3)

    # aviões de reconhecimento: concentrados numa rota, esparsos na outra
    for r in ESTRATEGIAS:
        s = np.linspace(0.06, 0.94, 8) if r == recon else np.array([0.35, 0.75])
        x, y = ponto_na_rota(r, s)
        ax.scatter(x, y + (0.24 if r == "Norte" else -0.24), marker=AVIAO, s=420,
                   color=COR_RECON, edgecolor="white", lw=0.4, zorder=6)

    # limites e clima de cada dia
    for k in range(1, N_DIAS):
        x, y = ponto_na_rota(rota, k / N_DIAS)
        ax.scatter(x, y, marker=NAVIO, s=900, color=COR_DESTR, edgecolor="white", lw=0.5, zorder=8)
        ax.annotate(f"fim do dia {k}", (x, y), xytext=(6, -12 if rota == "Norte" else 8), textcoords="offset points", fontsize=8, zorder=7)
    for k, c in enumerate(res["clima"]):
        x, y = ponto_na_rota(rota, (k + 0.5) / N_DIAS)
        ax.annotate(f"D{k+1}: {c}\np={res['ps'][k]:.2f}", (x, y),
                    xytext=(0, -26 if rota == "Sul" else 14), textcoords="offset points",
                    ha="center", fontsize=8, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))

    # trajeto: cinza antes de ser avistado, vermelho sob bombardeio
    if res["B"] > 0:
        s_av = res["s_av"]
        pre, pos = trecho(rota, 0, s_av), trecho(rota, s_av, 1)
        ax.plot(*pre.T, color="#333333", lw=3.2, zorder=5)
        ax.plot(*pos.T, color="#c0392b", lw=4.5, solid_capstyle="round", zorder=5)
        xa, ya = ponto_na_rota(rota, s_av)
        ax.scatter(xa, ya, s=1100, facecolor="none", edgecolor="gold", lw=2.5, zorder=9)
        ax.scatter(xa, ya, marker=NAVIO, s=900, color=COR_DESTR, edgecolor="white", lw=0.5, zorder=9)
        for k in range(res["B"]):
            sk = s_av + (k + 0.5) * (1 - s_av) / res["B"]
            x, y = ponto_na_rota(rota, sk)
            ax.scatter(x, y, marker="*", s=260, color="#ff6f00", edgecolor="#b71c1c", lw=0.8, zorder=9)
            ax.scatter(x + 0.25, y + 0.25, marker=AVIAO, s=750, color=COR_BOMB,
                       edgecolor="white", lw=0.4, zorder=10)
            fmt = "{:.1f}" if res["modo"] == "artigo" else "{:.0f}"
            ax.annotate(f"bombardeio {k+1}\n{fmt.format(res['t_dia'][k])} transp. | "
                        f"{fmt.format(res['d_dia'][k])} destr.",
                        (x, y), xytext=(0, 18 if rota == "Sul" else -34), textcoords="offset points",
                        ha="center", fontsize=8, color="#7b1d12", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#fff3e0", ec="#e65100", alpha=0.9))
    else:
        r = ROTAS[rota]
        ax.plot(r[:, 0], r[:, 1], color="#333333", lw=3.2, zorder=5)
        ax.text(145.0, -4.6, "Comboio NÃO avistado:\nchegou intacto a Lae", fontsize=11,
                color="darkgreen", weight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkgreen", alpha=0.9))

    _painel_frota(ax, res, P)
    for nome, (x, y) in {"Rabaul": RABAUL, "Lae": LAE}.items():
        ax.scatter(x, y, s=90, color="black", marker="s", zorder=10)
        ax.annotate(nome, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=11, weight="bold", zorder=10)

    leg = [Line2D([], [], color="#333333", lw=3, label="trajeto não detectado"),
           Line2D([], [], color="#c0392b", lw=4, label="trajeto sob bombardeio"),
           Line2D([], [], ls="--", color="gray", label="rota alternativa"),
           Line2D([], [], marker=NAVIO, ls="", ms=20, mfc=COR_DESTR, mec="w", label="comboio japonês (fim de cada dia)"),
           Line2D([], [], marker="o", ls="", ms=14, mfc="none", mec="gold", mew=2.5, label="comboio avistado"),
           Line2D([], [], marker=AVIAO, ls="", ms=16, mfc=COR_RECON, mec="w", label="reconhecimento aliado"),
           Line2D([], [], marker=AVIAO, ls="", ms=16, mfc=COR_BOMB, mec="w", label="bombardeio aliado (1 por dia)")]
    ax.legend(handles=leg, loc="lower left", fontsize=7.5, framealpha=0.92)
    ax.set_xlim(144.8, 153.4); ax.set_ylim(-8.3, -3.1); ax.set_aspect("equal")
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude (°)")
    tec = "rasante (skip bombing)" if res["rasante"] else "convencional"
    if res["modo"] == "artigo":
        sub = f"{res['B']} dia(s) de bombardeio (matriz do artigo)"
    else:
        sub = (f"nesta travessia: {res['B']} dia(s) de bombardeio  "
               f"(média esperada {dias_esperados(res['ps']):.2f})")
    ax.set_title(f"Aliados: reconhecimento no {recon}  ×  Comboio japonês: rota {rota}\n"
                 f"{sub}  [{tec}]", fontsize=11)
    fig.tight_layout()
    return fig


def plot_matriz(A, recon, rota, titulo):
    sol = resolver_jogo(A)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.imshow(A, cmap="Reds", vmin=0, vmax=3.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{A[i, j]:.2f}".rstrip("0").rstrip(".") + " dias",
                    ha="center", va="center", fontsize=14)
        ax.text(1.6, i, f"mín linha = {A[i].min():.2f}", va="center", fontsize=9)
    for j in range(2):
        ax.text(j, 1.68, f"máx coluna = {A[:, j].max():.2f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["Japão: Norte", "Japão: Sul"])
    ax.set_yticks([0, 1], ["Aliados: Norte", "Aliados: Sul"])
    ax.xaxis.tick_top()
    ax.add_patch(Rectangle((ESTRATEGIAS.index(rota) - 0.5, ESTRATEGIAS.index(recon) - 0.5),
                           1, 1, fill=False, ec="royalblue", lw=4))
    if sol["tipo"] == "sela":
        ax.scatter(sol["j"], sol["i"] - 0.3, marker="*", s=260, color="gold", edgecolor="k")
        sub = (f"Ponto de sela: maximin = minimax = {sol['valor']:.2f} "
               f"({ESTRATEGIAS[sol['i']]}, {ESTRATEGIAS[sol['j']]})")
    else:
        sub = (f"Sem sela (maximin {sol['maximin']:.2f} < minimax {sol['minimax']:.2f}). Mista: "
               f"P(Aliados N)={sol['p']:.2f}, P(Japão N)={sol['q']:.2f}, v={sol['valor']:.2f}")
    ax.set_title(f"{titulo}\n{sub}", fontsize=9.5, pad=28)
    ax.set_xlim(-0.5, 2.4); ax.set_ylim(1.95, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_monte_carlo(mc, P):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, chave, n, cor, rot in [(axs[0], "B", 3, "#c0392b", "dias de bombardeio"),
                                   (axs[1], "T", P["n_transp"], "#6d4c41", "transportes afundados"),
                                   (axs[2], "D", P["n_destr"], "#455a64", "destróieres afundados")]:
        v = mc[chave]
        freq = np.bincount(v, minlength=n + 1) / len(v)
        ax.bar(range(n + 1), freq, color=cor, alpha=0.85)
        ax.axvline(v.mean(), color="k", ls="--", lw=1.2)
        ax.set_title(f"{rot}\nmédia = {v.mean():.2f}", fontsize=10)
        ax.set_xticks(range(n + 1)); ax.set_ylabel("frequência")
    fig.suptitle(f"Monte Carlo com {len(mc['B']):,} travessias".replace(",", "."), fontsize=11)
    fig.tight_layout()
    return fig


def resumo_texto(res, P):
    nT, nD = P["n_transp"], P["n_destr"]
    est = " esperados" if res["modo"] == "artigo" else ""
    linhas = [f"Clima por dia na rota {res['rota']}: " + ", ".join(res["clima"]),
              "Prob. de avistamento por dia: " + ", ".join(f"{p:.2f}" for p in res["ps"]),
              (f"Avistado no dia {res['dia_avist']}  ->  {res['B']} dia(s) de bombardeio"
               if res["dia_avist"] else "Comboio não avistado  ->  0 dia de bombardeio"),
              f"Transportes afundados{est}: {res['T']:.1f} de {nT}",
              f"Destróieres afundados{est}: {res['D']:.1f} de {nD}",
              f"Tropas perdidas{est}: {res['tropas_perdidas']:,.0f} de {P['tropas']:,}".replace(",", "."),
              f"Dias de bombardeio esperados (modelo estocástico): {dias_esperados(res['ps']):.3f}"]
    return "\n".join(linhas)
