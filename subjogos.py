"""
Laboratório de equilíbrio perfeito em subjogos — três jogos sequenciais do
livro Teoria dos Jogos (Ronaldo Fiani):

  1. Prevenção da entrada com escolha de capacidade (Figura 6.10)
  2. Jogo da centopeia (Exercício 6.6)
  3. Jogo da regulação (Figura 6.8)

Para cada jogo o app mostra:
  * a árvore com os subjogos sinalizados;
  * o teste de um perfil subjogo a subjogo (recorte, letras usadas, jogada,
    desvios unilaterais), acendendo na sigla as letras de cada subjogo;
  * a forma estratégica com melhores respostas, EN e EPS;
  * a indução reversa passo a passo;
  * a lista de equilíbrios de Nash em estratégias puras.

Rodar localmente:   streamlit run app.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Definição dos jogos (informação perfeita, 2 jogadores)
# ---------------------------------------------------------------------------


@dataclass
class Game:
    key: str
    title: str
    source: str
    players: list[str]
    short: list[str]
    root: str
    # nó -> {"pl": jogador, "sj": nº do subjogo, "desc": texto,
    #        "acts": [(código, rótulo, filho), ...]}
    nodes: dict
    terminals: dict            # terminal -> (u0, u1)
    test_order: list[str]      # ordem de teste (de baixo para cima)
    sig_tokens: list[str]      # tokens da sigla: ids de nós ou separadores
    sig_help: str
    same_rank: list[list[str]] = field(default_factory=list)
    rankdir: str = "LR"
    lesson: str = ""


GAMES: dict[str, Game] = {}

GAMES["entrada"] = Game(
    key="entrada",
    title="Prevenção da entrada (capacidade flexível × inflexível)",
    source="Fiani, Figura 6.10(a)",
    players=["Dominante", "Desafiante"],
    short=["Dom", "Des"],
    root="n0",
    nodes={
        "n0": {"pl": 0, "sj": 1, "desc": "na raiz",
               "acts": [("F", "Flexível", "n1"), ("I", "Inflexível", "n2")]},
        "n1": {"pl": 1, "sj": 2, "desc": "após Flexível",
               "acts": [("E", "Entra", "n3"), ("N", "Não entra", "t1")]},
        "n2": {"pl": 1, "sj": 3, "desc": "após Inflexível",
               "acts": [("E", "Entra", "n4"), ("N", "Não entra", "t4")]},
        "n3": {"pl": 0, "sj": 4, "desc": "após Flexível + Entra",
               "acts": [("A", "Acomoda", "t2"), ("L", "Luta", "t3")]},
        "n4": {"pl": 0, "sj": 5, "desc": "após Inflexível + Entra",
               "acts": [("A", "Acomoda", "t5"), ("L", "Luta", "t6")]},
    },
    terminals={"t1": (10, 0), "t2": (7, 3), "t3": (2, -1),
               "t4": (8, 0), "t5": (-2, 3), "t6": (-1, -1)},
    test_order=["n3", "n4", "n1", "n2", "n0"],
    sig_tokens=["(", "n0", "·", "n3", "n4", ", ", "n1", "n2", ")"],
    sig_help="Dominante: raiz · SJ4 · SJ5 | Desafiante: SJ2 · SJ3",
    lesson=(
        "Com capacidade flexível, a ameaça de lutar não é crível (7 > 2). "
        "A capacidade inflexível muda os payoffs pós-entrada e torna a luta "
        "a melhor resposta (−1 > −2): o Dominante ganha ao reduzir as próprias "
        "opções (valor do comprometimento = 8 − 7 = 1)."
    ),
)

GAMES["centopeia"] = Game(
    key="centopeia",
    title="Jogo da centopeia",
    source="Fiani, Exercício 6.6",
    players=["A", "B"],
    short=["A", "B"],
    root="n0",
    nodes={
        "n0": {"pl": 0, "sj": 1, "desc": "no nó 1",
               "acts": [("S", "Sai", "t0"), ("C", "Continua", "n1")]},
        "n1": {"pl": 1, "sj": 2, "desc": "no nó 2",
               "acts": [("S", "Sai", "t1"), ("C", "Continua", "n2")]},
        "n2": {"pl": 0, "sj": 3, "desc": "no nó 3",
               "acts": [("S", "Sai", "t2"), ("C", "Continua", "n3")]},
        "n3": {"pl": 1, "sj": 4, "desc": "no nó 4",
               "acts": [("S", "Sai", "t3"), ("C", "Continua", "tF")]},
    },
    terminals={"t0": (3, 0), "t1": (2, 4), "t2": (5, 2),
               "t3": (4, 7), "tF": (9, 5)},
    test_order=["n3", "n2", "n1", "n0"],
    sig_tokens=["(", "n0", "n2", ", ", "n1", "n3", ")"],
    sig_help="A: nó 1 · nó 3 | B: nó 2 · nó 4   (S = sai, C = continua)",
    same_rank=[["n0", "t0"], ["n1", "t1"], ["n2", "t2"], ["n3", "t3"]],
    lesson=(
        "Em cada nó, quem decide sai porque antecipa que o outro sairá no nó "
        "seguinte. O resultado (3, 0) é Pareto-dominado por (5, 2), (4, 7) e "
        "(9, 5): a racionalidade sequencial destrói o ganho conjunto."
    ),
)

GAMES["regulacao"] = Game(
    key="regulacao",
    title="Jogo da regulação",
    source="Fiani, Figura 6.8",
    players=["Empresa", "Regulador"],
    short=["Emp", "Reg"],
    root="n0",
    nodes={
        "n0": {"pl": 0, "sj": 1, "desc": "na raiz",
               "acts": [("E", "Investimento elevado", "n1"),
                        ("B", "Investimento baixo", "n2")]},
        "n1": {"pl": 1, "sj": 2, "desc": "após investimento elevado",
               "acts": [("R", "Remunera", "t1"), ("N", "Não remunera", "t2")]},
        "n2": {"pl": 1, "sj": 3, "desc": "após investimento baixo",
               "acts": [("R", "Remunera", "t3"), ("N", "Não remunera", "t4")]},
    },
    terminals={"t1": (2, 1), "t2": (-2, 2), "t3": (1, -2), "t4": (-1, -1)},
    test_order=["n1", "n2", "n0"],
    sig_tokens=["(", "n0", ", ", "n1", "n2", ")"],
    sig_help="Empresa: raiz | Regulador: SJ2 · SJ3",
    lesson=(
        "O Regulador não consegue se comprometer a remunerar depois que o "
        "investimento foi feito (2 > 1). Antecipando isso, a Empresa investe "
        "pouco: (−1, −1), embora (2, 1) fosse melhor para ambos. É o problema "
        "de hold-up."
    ),
)

# ---------------------------------------------------------------------------
# Motor de cálculo
# ---------------------------------------------------------------------------


def is_term(x: str) -> bool:
    return x.startswith("t")


def children(g: Game, n: str) -> list[str]:
    return [c for _, _, c in g.nodes[n]["acts"]]


def act_label(g: Game, n: str, code: str) -> str:
    return next(lab for c, lab, _ in g.nodes[n]["acts"] if c == code)


def subtree(g: Game, n: str) -> list[str]:
    """Nós de decisão do subjogo que começa em n (inclui n)."""
    if is_term(n):
        return []
    out = [n]
    for c in children(g, n):
        out += subtree(g, c)
    return out


def subtree_all(g: Game, n: str) -> list[str]:
    """Nós de decisão e terminais do subjogo que começa em n."""
    if is_term(n):
        return [n]
    out = [n]
    for c in children(g, n):
        out += subtree_all(g, c)
    return out


def play(g: Game, start: str, prof: dict) -> list[str]:
    path, x = [start], start
    while not is_term(x):
        x = next(c for code, _, c in g.nodes[x]["acts"] if code == prof[x])
        path.append(x)
    return path


def plans(g: Game, i: int, r: str) -> list[dict]:
    ns = [n for n in subtree(g, r) if g.nodes[n]["pl"] == i]
    opts = [[a[0] for a in g.nodes[n]["acts"]] for n in ns]
    return [dict(zip(ns, combo)) for combo in itertools.product(*opts)]


def test_subgame(g: Game, r: str, prof: dict, pay: dict) -> dict:
    path = play(g, r, prof)
    t = path[-1]
    res = []
    for i in (0, 1):
        pls = plans(g, i, r)
        if not pls or not pls[0]:
            res.append(None)
            continue
        alts = []
        for pl in pls:
            p2 = {**prof, **pl}
            tt = play(g, r, p2)[-1]
            alts.append({"plan": pl, "t": tt, "v": pay[tt][i]})
        best = max(alts, key=lambda a: a["v"])
        res.append({"alts": alts, "best": best, "cur": pay[t][i],
                    "gain": best["v"] - pay[t][i]})
    ne = all(x is None or x["gain"] <= 0 for x in res)
    return {"path": path, "t": t, "res": res, "ne": ne}


def analyze(g: Game, prof: dict, pay: dict) -> dict:
    subs = {n: test_subgame(g, n, prof, pay) for n in g.nodes}
    return {"subs": subs, "NE": subs[g.root]["ne"],
            "SPE": all(s["ne"] for s in subs.values()),
            "u": pay[play(g, g.root, prof)[-1]]}


def backward(g: Game, pay: dict):
    val, choice, steps = {}, {}, []

    def v(x):
        return pay[x] if is_term(x) else val[x]

    for n in g.test_order:  # a ordem de teste já é de baixo para cima
        i = g.nodes[n]["pl"]
        opts = [(code, lab, c, v(c)) for code, lab, c in g.nodes[n]["acts"]]
        best = max(o[3][i] for o in opts)
        winners = [o for o in opts if o[3][i] == best]
        code, lab, c, vv = winners[0]
        choice[n] = code
        val[n] = vv
        steps.append({"node": n, "opts": opts, "choice": code,
                      "tie": len(winners) > 1, "value": vv})
    return choice, val, steps


def route_to(g: Game, t: str) -> str:
    """Sequência de ações da raiz até o terminal t (para rotular payoffs)."""
    def go(x, acc):
        if x == t:
            return acc
        if is_term(x):
            return None
        for code, lab, c in g.nodes[x]["acts"]:
            r = go(c, acc + [lab])
            if r is not None:
                return r
        return None
    return " → ".join(go(g.root, []))


def sigla(g: Game, prof: dict) -> str:
    return "".join(prof[t] if t in g.nodes else t for t in g.sig_tokens)


def fmt(v: float) -> str:
    v = int(v) if float(v).is_integer() else round(v, 2)
    return f"−{-v}" if v < 0 else str(v)


def fpay(p) -> str:
    return f"({fmt(p[0])}, {fmt(p[1])})"


# ---------------------------------------------------------------------------
# Desenho da árvore (Graphviz)
# ---------------------------------------------------------------------------

C_SJ = "#BA7517"
C_SJ_FILL = "#FAEEDA"
C_BLUE = "#378ADD"
C_ORANGE = "#D85A30"
C_FADE = "#C8C6BD"
C_EDGE = "#5F5E5A"
P_FILL = ["#EEEDFE", "#E1F5EE"]
P_LINE = ["#534AB7", "#0F6E56"]


def tree_dot(g: Game, pay: dict, prof: dict | None, focus: str | None) -> str:
    """focus=None: visão geral com todos os subjogos aninhados.
    focus=nó: destaca apenas esse subjogo e o perfil."""
    inside = set(subtree_all(g, focus)) if focus else set()
    path = set(play(g, focus, prof)) if (focus and prof) else set()
    end = play(g, focus, prof)[-1] if (focus and prof) else None

    def faded(x):
        return focus is not None and x not in inside

    def node_decl(x):
        if is_term(x):
            col = C_ORANGE if x == end else ("#B4B2A9" if faded(x) else "#2C2C2A")
            font = "Helvetica-Bold" if x == end else "Helvetica"
            return (f'"{x}" [shape=plaintext, label="{fpay(pay[x])}", '
                    f'fontcolor="{col}", fontname="{font}"];')
        nd = g.nodes[x]
        pl = nd["pl"]
        if faded(x):
            return (f'"{x}" [shape=box, style="rounded,filled", fillcolor="#F1EFE8", '
                    f'color="{C_FADE}", fontcolor="#B4B2A9", '
                    f'label="{g.short[pl]}\\nSJ{nd["sj"]}"];')
        pen = 2.5 if x == focus else 1
        col = C_SJ if x == focus else P_LINE[pl]
        return (f'"{x}" [shape=box, style="rounded,filled", fillcolor="{P_FILL[pl]}", '
                f'color="{col}", penwidth={pen}, fontcolor="{P_LINE[pl]}", '
                f'label="{g.short[pl]}\\nSJ{nd["sj"]}"];')

    def rank_lines(members):
        out = []
        for grp in g.same_rank:
            if all(m in members for m in grp):
                out.append("{rank=same; " + "; ".join(f'"{m}"' for m in grp) + ";}")
        return out

    lines = [
        "digraph G {",
        f'rankdir={g.rankdir}; bgcolor="white"; nodesep=0.3; ranksep=0.45;',
        'node [fontname="Helvetica", fontsize=22];',
        'edge [fontname="Helvetica", fontsize=18];',
        'graph [fontname="Helvetica", fontsize=18];',
    ]

    # Subjogos aninhados (como na Figura 6.8 do livro). Na visão geral todos
    # aparecem tracejados; no teste, só o subjogo em foco aparece (os demais
    # clusters ficam invisíveis, o que mantém a árvore sempre no mesmo lugar).
    def emit(n):
        nd = g.nodes[n]
        if focus is None:
            style = (f'label="SJ{nd["sj"]}"; labeljust=r; fontcolor="{C_SJ}"; '
                     f'fontname="Helvetica-Bold"; style="dashed,rounded"; color="{C_SJ}";')
        elif n == focus:
            style = (f'label="SJ{nd["sj"]}"; labeljust=r; fontcolor="{C_SJ}"; '
                     f'fontname="Helvetica-Bold"; style="filled,dashed,rounded"; '
                     f'color="{C_SJ}"; fillcolor="{C_SJ_FILL}";')
        else:
            style = 'label=""; style=invis;'
        out = [f"subgraph cluster_{n} {{", style, node_decl(n)]
        members = [n]
        for c in reversed(children(g, n)):
            if is_term(c):
                out.append(node_decl(c))
                members.append(c)
        out += rank_lines(members)
        for c in reversed(children(g, n)):
            if not is_term(c):
                out += emit(c)
        out.append("}")
        return out

    lines += emit(g.root)

    for n, nd in g.nodes.items():
        for code, lab, c in nd["acts"]:
            label = f"{lab} ({code})"
            if focus is None:
                attr = f'color="{C_EDGE}", fontcolor="{C_EDGE}"'
            elif faded(n):
                attr = f'color="{C_FADE}", fontcolor="{C_FADE}"'
            elif n in path and c in path:
                attr = f'color="{C_ORANGE}", penwidth=4, fontcolor="{C_ORANGE}"'
            elif prof and prof[n] == code:
                attr = f'color="{C_BLUE}", penwidth=2.5, fontcolor="{C_BLUE}"'
            else:
                attr = f'color="{C_EDGE}", fontcolor="{C_EDGE}"'
            lines.append(f'"{n}" -> "{c}" [label="{label}", {attr}];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Componentes de interface
# ---------------------------------------------------------------------------

CSS = """
<style>
.sig {font-family: ui-monospace, Menlo, Consolas, monospace; text-align: center;
      margin: .3rem 0 .6rem 0;}
.sig .tk {display: inline-block; text-align: center; vertical-align: top; margin: 0 1px;}
.sig .ch {display: block; font-size: 32px; line-height: 1.2; padding: 0 5px; border-radius: 6px;}
.sig .nb {display: block; font-size: 12px; opacity: .65; height: 16px;}
.sig .on .ch {background: #FAEEDA; color: #854F0B;}
.sig .off {opacity: .25;}
.sf {border-collapse: collapse; margin: .5rem 0; font-variant-numeric: tabular-nums;}
.sf th, .sf td {border: 1px solid rgba(128,128,128,.35); padding: 6px 10px; text-align: center;}
.sf th {background: rgba(128,128,128,.10);}
.sf td.ne {background: rgba(29,158,117,.20);}
.sf td.spe {box-shadow: inset 0 0 0 3px #BA7517;}
.sf td.cur {outline: 2px dashed #378ADD; outline-offset: -7px;}
.sf .br {text-decoration: underline; text-decoration-thickness: 2px; font-weight: 700;}
[data-testid="stGraphVizChart"] g.node text {font-size: 22px !important; font-family: Helvetica, Arial, sans-serif !important;}
[data-testid="stGraphVizChart"] g.edge text {font-size: 18px !important; font-family: Helvetica, Arial, sans-serif !important;}
[data-testid="stGraphVizChart"] g.cluster text {font-size: 18px !important; font-family: Helvetica, Arial, sans-serif !important;}
.ok {color: #1D9E75; font-weight: 600;}
.no {color: #E24B4A; font-weight: 600;}
</style>
"""


def sigla_html(g: Game, prof: dict, active: set[str]) -> str:
    """Sigla com o número do subjogo sob cada letra; acende as letras ativas."""
    parts = []
    for t in g.sig_tokens:
        if t in g.nodes:
            cls = "on" if t in active else "off"
            parts.append(f'<span class="tk {cls}"><span class="ch">{prof[t]}</span>'
                         f'<span class="nb">SJ{g.nodes[t]["sj"]}</span></span>')
        else:
            parts.append(f'<span class="tk"><span class="ch">{t.strip()}</span>'
                         f'<span class="nb"></span></span>')
    return f'<div class="sig">{"".join(parts)}</div>'


def plan_text(g: Game, plan: dict) -> str:
    """Plano condicional: uma instrução por nó ("se chegar aqui, faço isso")."""
    return "; ".join(f"{act_label(g, n, a)} {g.nodes[n]['desc']} (SJ{g.nodes[n]['sj']})"
                     for n, a in plan.items())


def path_text(g: Game, start: str, prof: dict) -> str:
    """Caminho efetivamente percorrido a partir de start."""
    return " → ".join(act_label(g, n, prof[n]) for n in play(g, start, prof)
                      if not is_term(n))


def strategic_form_html(g: Game, pay: dict, prof: dict) -> str:
    P0, P1 = plans(g, 0, g.root), plans(g, 1, g.root)

    def name(i, pl):
        toks = []
        for t in g.sig_tokens:
            if t in g.nodes and g.nodes[t]["pl"] == i:
                toks.append(pl[t])
            elif t == "·" and i == 0 and toks:
                toks.append("·")
        return "".join(toks)

    h = (f'<table class="sf"><tr><th>{g.short[0]} \\ {g.short[1]}</th>'
         + "".join(f"<th>{name(1, b)}</th>" for b in P1) + "</tr>")
    for a in P0:
        h += f"<tr><th>{name(0, a)}</th>"
        for b in P1:
            p = {**a, **b}
            u = pay[play(g, g.root, p)[-1]]
            br0 = all(pay[play(g, g.root, {**a2, **b})[-1]][0] <= u[0] for a2 in P0)
            br1 = all(pay[play(g, g.root, {**a, **b2})[-1]][1] <= u[1] for b2 in P1)
            an = analyze(g, p, pay)
            cls = " ".join(c for c, ok in (("ne", an["NE"]), ("spe", an["SPE"]),
                                           ("cur", p == prof)) if ok)
            h += (f'<td class="{cls}">(<span class="{"br" if br0 else ""}">{fmt(u[0])}</span>, '
                  f'<span class="{"br" if br1 else ""}">{fmt(u[1])}</span>)</td>')
        h += "</tr>"
    return h + "</table>"


def all_profiles(g: Game) -> list[dict]:
    return [{**a, **b} for a in plans(g, 0, g.root) for b in plans(g, 1, g.root)]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Equilíbrio perfeito em subjogos", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("Jogo")
    gkey = st.radio("Escolha o jogo", list(GAMES),
                    format_func=lambda k: GAMES[k].title)
    g = GAMES[gkey]

    with st.expander("Editar payoffs"):
        if st.button("Restaurar valores do livro", key=f"reset_{gkey}"):
            for t, p in g.terminals.items():
                for i in (0, 1):
                    st.session_state[f"{gkey}_{t}_{i}"] = float(p[i])
        pay = {}
        for t, p in g.terminals.items():
            st.markdown(f"**{route_to(g, t)}**")
            for i in (0, 1):
                st.session_state.setdefault(f"{gkey}_{t}_{i}", float(p[i]))
            c1, c2 = st.columns(2)
            u0 = c1.number_input(g.short[0], step=1.0, key=f"{gkey}_{t}_0")
            u1 = c2.number_input(g.short[1], step=1.0, key=f"{gkey}_{t}_1")
            pay[t] = (u0, u1)

    # perfis ordenados: EPS, depois EN, depois os demais
    profs = all_profiles(g)
    tagged = []
    for p in profs:
        an = analyze(g, p, pay)
        tag = 0 if an["SPE"] else (1 if an["NE"] else 2)
        tagged.append((tag, sigla(g, p), p))
    tagged.sort(key=lambda x: (x[0], x[1]))
    labels = [f"{s}  {'★ EPS' if t == 0 else ('EN' if t == 1 else '')}"
              for t, s, _ in tagged]

    st.header("Perfil a testar")
    only_ne = st.checkbox("Mostrar só equilíbrios de Nash", value=False)
    idx = [k for k, (t, _, _) in enumerate(tagged) if (t < 2 or not only_ne)]
    choice = st.selectbox("Perfil (sigla)", idx, format_func=lambda k: labels[k],
                          key=f"prof_{gkey}")
    prof = tagged[choice][2]
    st.caption(g.sig_help)

st.title("Equilíbrio perfeito em subjogos")
st.markdown(f"**{g.title}** — {g.source}. Payoffs na ordem ({g.players[0]}, {g.players[1]}).")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Teste por subjogos", "Forma estratégica", "Indução reversa", "Equilíbrios"])

# --- Aba 1: teste por subjogos --------------------------------------------
with tab1:
    steps = ["Visão geral"] + [f"SJ{g.nodes[n]['sj']}" for n in g.test_order] + ["Resumo"]
    step = st.radio("Passo (de baixo para cima)", steps, horizontal=True,
                    key=f"step_{gkey}")
    k = steps.index(step)
    focus = g.test_order[k - 1] if 0 < k < len(steps) - 1 else None
    active = set(subtree(g, focus)) if focus else set(g.nodes)

    st.markdown(sigla_html(g, prof, active), unsafe_allow_html=True)
    st.caption(g.sig_help)

    col_tree, col_txt = st.container(), st.container()
    with col_tree:
        st.graphviz_chart(tree_dot(g, pay, prof if focus else None, focus),
                          width="stretch")
        if focus:
            st.caption("Azul: instrução do perfil · Laranja: caminho jogado · "
                       "Área amarela: o subjogo em teste")
        else:
            st.caption("Cada caixa tracejada é um subjogo: começa num nó e "
                       "contém tudo o que vem depois dele.")

    with col_txt:
        if step == "Visão geral":
            st.subheader(f"Os {len(g.nodes)} subjogos")
            for n in sorted(g.nodes, key=lambda x: g.nodes[x]["sj"]):
                nd = g.nodes[n]
                st.markdown(f"- **SJ{nd['sj']}** começa {nd['desc']} "
                            f"({g.players[nd['pl']]})")
            st.markdown(f"**Perfil em teste:** `{sigla(g, prof)}`")
            st.markdown("Cada letra é a instrução para um nó (o número sob a "
                        "letra indica qual). Avance pelos passos: em cada um "
                        "só acendem as letras dos nós dentro do recorte.")
        elif step == "Resumo":
            an = analyze(g, prof, pay)
            rows = []
            for n in sorted(g.nodes, key=lambda x: g.nodes[x]["sj"]):
                T = an["subs"][n]
                who = ", ".join(g.players[i] for i, x in enumerate(T["res"])
                                if x and x["gain"] > 0)
                rows.append({"Subjogo": f"SJ{g.nodes[n]['sj']} ({g.nodes[n]['desc']})",
                             "Resultado": fpay(pay[T["t"]]),
                             "Teste": "passa" if T["ne"] else f"falha ({who})"})
            st.subheader(f"Resumo de {sigla(g, prof)}")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if an["SPE"]:
                st.success("Passou em todos os subjogos: é equilíbrio perfeito.")
            elif an["NE"]:
                st.warning("É equilíbrio de Nash (passa em SJ1), mas falha em "
                           "algum subjogo: não é perfeito.")
            else:
                st.error("Falha já em SJ1: nem é equilíbrio de Nash.")
        else:
            nd = g.nodes[focus]
            T = test_subgame(g, focus, prof, pay)
            st.subheader(f"SJ{nd['sj']} — começa {nd['desc']}")
            st.markdown("**1. Recorte:** só vale a área amarela; o resto da "
                        "árvore fica apagado.")
            st.markdown("**2. Letras usadas** (acesas na sigla, azul na árvore):")
            st.markdown("\n".join(
                f"- {g.players[g.nodes[n]['pl']]} em SJ{g.nodes[n]['sj']}: "
                f"**{act_label(g, n, prof[n])}** (letra {prof[n]})"
                for n in subtree(g, focus)))
            jog = " → ".join(act_label(g, n, prof[n]) for n in T["path"] if not is_term(n))
            st.markdown(f"**3. Jogar a partir do início do recorte:** {jog} → "
                        f"**{fpay(pay[T['t']])}**")
            st.markdown("**4. Testar desvios** (muda só as letras de um jogador; "
                        "as do outro ficam fixas). Cada plano é uma lista de instruções "
                        "condicionais, uma por nó; o caminho jogado mostra quais delas "
                        "foram de fato usadas.")
            for i, x in enumerate(T["res"]):
                if x is None:
                    continue
                st.markdown(f"*{g.players[i]}* recebe hoje **{fmt(x['cur'])}**. "
                            "Alternativas:")
                # todas as combinações das letras deste jogador no recorte;
                # a combinação atual (a do perfil) vem marcada
                linhas = []
                for a in x["alts"]:
                    atual = all(prof[n] == v for n, v in a["plan"].items())
                    cam = path_text(g, focus, {**prof, **a["plan"]})
                    linhas.append(
                        f"- Plano: {plan_text(g, a['plan'])}  \n"
                        f"  Caminho jogado: {cam} → {fpay(pay[a['t']])} → recebe "
                        f"**{fmt(a['v'])}**" + ("  ← **atual**" if atual else ""))
                st.markdown("\n".join(linhas))
                st.caption(f"{len(x['alts'])} combinações = 2^{len(x['alts']).bit_length() - 1} "
                           "(duas ações em cada nó deste jogador dentro do recorte).")
                if x["gain"] > 0:
                    st.markdown(f'<span class="no">Ganha +{fmt(x["gain"])} desviando → falha.</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<span class="ok">Nenhuma alternativa dá mais → não desvia.</span>',
                                unsafe_allow_html=True)
            if T["ne"]:
                st.success(f"SJ{nd['sj']}: é equilíbrio de Nash ✓")
            else:
                st.error(f"SJ{nd['sj']}: não é equilíbrio de Nash ✗")

# --- Aba 2: forma estratégica ---------------------------------------------
with tab2:
    st.subheader("Forma estratégica")
    st.markdown(strategic_form_html(g, pay, prof), unsafe_allow_html=True)
    st.caption("Sublinhado = melhor resposta · verde = equilíbrio de Nash · "
               "borda dourada = perfeito em subjogos · contorno azul = perfil "
               "selecionado na barra lateral.")
    st.caption(g.sig_help)

# --- Aba 3: indução reversa -----------------------------------------------
with tab3:
    choice_bi, val_bi, steps_bi = backward(g, pay)
    st.subheader("Indução reversa (de baixo para cima)")
    lines = []
    for s in steps_bi:
        n = s["node"]
        nd = g.nodes[n]
        i = nd["pl"]
        opts = "; ".join(f"{lab} → {fmt(v[i])}" for _, lab, _, v in s["opts"])
        tie = " (empate: adotada a primeira ação)" if s["tie"] else ""
        lines.append(f"SJ{nd['sj']} — {g.players[i]} {nd['desc']}: {opts}  "
                     f"⇒ {act_label(g, n, s['choice'])}{tie}, valor {fpay(s['value'])}")
    st.code("\n".join(lines), language=None)
    spe_prof = {n: choice_bi[n] for n in g.nodes}
    st.markdown(f"**EPS:** `{sigla(g, spe_prof)}` → resultado "
                f"**{fpay(val_bi[g.root])}**")
    st.graphviz_chart(tree_dot(g, pay, spe_prof, g.root), width="stretch")
    if g.lesson:
        st.info(g.lesson)

# --- Aba 4: equilíbrios -----------------------------------------------------
with tab4:
    st.subheader("Equilíbrios de Nash em estratégias puras")
    rows = []
    for p in all_profiles(g):
        an = analyze(g, p, pay)
        if not an["NE"]:
            continue
        fails = []
        for n, T in an["subs"].items():
            if not T["ne"]:
                who = ", ".join(g.players[i] for i, x in enumerate(T["res"])
                                if x and x["gain"] > 0)
                fails.append(f"SJ{g.nodes[n]['sj']} ({who})")
        rows.append({"Perfil": sigla(g, p), "Resultado": fpay(an["u"]),
                     "EPS": "★ sim" if an["SPE"] else "não",
                     "Subjogos em que falha": "; ".join(sorted(fails)) or "—"})
    df = pd.DataFrame(rows).sort_values(["EPS", "Perfil"], ascending=[False, True])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"{len(rows)} equilíbrio(s) de Nash em estratégias puras; "
               f"{sum(r['EPS'] == '★ sim' for r in rows)} perfeito(s).")
