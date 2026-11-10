### 1. O jogo de Haywood

Jogadores: Kenney (linhas, maximiza dias de bombardeio) e o comandante japonês (colunas, minimiza).
Estratégias: Norte (N) e Sul (S) para ambos.

```latex
A = \begin{pmatrix} a_{NN} & a_{NS} \\ a_{SN} & a_{SS} \end{pmatrix}
  = \begin{pmatrix} 2 & 2 \\ 1 & 3 \end{pmatrix}
```

Maximin (Kenney) e minimax (Japão):

```latex
\underline{v} = \max_i \min_j a_{ij} = \max\{2,\,1\} = 2
\qquad
\overline{v} = \min_j \max_i a_{ij} = \min\{2,\,3\} = 2
```

Como `\underline{v} = \overline{v} = 2`, há ponto de sela em (N, N): estratégias puras "casadas" (*matched strategies*). A mesma solução sai por dominância iterada: para o Japão (minimizador) a coluna N domina fracamente a coluna S (2 ≤ 2 e 1 ≤ 3); eliminada S, Kenney prefere N (2 > 1).

### 2. Modelo estocástico de detecção

O comboio leva 3 dias. No dia d, a probabilidade de ser avistado depende da cobertura (concentrada/esparsa) e do clima no trecho (chuva/limpo): `p_d`. Se o primeiro avistamento ocorre no dia d, há `B = 4 - d` dias de bombardeio (discretização implícita no artigo).

```latex
\Pr(B = 4-d) = p_d \prod_{k<d} (1-p_k), \quad d = 1,2,3
\qquad
\Pr(B = 0) = \prod_{k=1}^{3} (1-p_k)
```

```latex
E[B] = 3p_1 + 2(1-p_1)p_2 + (1-p_1)(1-p_2)p_3
```

A matriz do modelo é `a_{ij} = E[B \mid i, j]`. Com os parâmetros padrão:

```latex
E[B] \approx \begin{pmatrix} 2{,}02 & 2{,}23 \\ 1{,}06 & 2{,}95 \end{pmatrix}
```

o que reproduz o ponto de sela em (N, N) com valor ≈ 2.

### 3. Danos

Cada navio ainda flutuando é afundado num dia de bombardeio com probabilidade q (independente). Após b dias:

```latex
T \mid B=b \;\sim\; \text{Binomial}\big(n_T,\; 1-(1-q_T)^{b}\big)
\qquad
E[T \mid B=b] = n_T\,[1-(1-q_T)^{b}]
```

Afundamentos esperados no dia k (k = 1,…,b): `n_T (1-q_T)^{k-1} q_T`.

O *skip bombing* (surpresa tática citada por Haywood) entra como q alto; o bombardeio convencional em altitude, como q baixo. Isso mostra o ponto central do artigo: a decisão japonesa foi correta dado o jogo (minimax); a derrota veio de um parâmetro que eles desconheciam — a eficácia por dia de bombardeio —, não da escolha de rota.

### 4. Solução mista (quando não há sela)

Para a matriz 2×2 `(a, b; c, d)` sem ponto de sela, com `D = a - b - c + d`:

```latex
p^{*} = \frac{d-c}{a-b-c+d}, \qquad
q^{*} = \frac{d-b}{a-b-c+d}, \qquad
v = \frac{ad-bc}{a-b-c+d}
```

`p*` = probabilidade de Kenney jogar N; `q*` = probabilidade de o Japão jogar N. Exemplo: leve "Concentrado + chuva" para 0,90. Então E[B|N,N] ≈ 2,9 > E[B|N,S] ≈ 2,2: a chuva deixa de proteger o comboio, maximin ≈ 2,2 < minimax ≈ 2,9 e o equilíbrio passa a ser misto.

### 5. Calibração (estilizada)

Parâmetros padrão: 8 transportes, 8 destróieres, ≈ 6.900 soldados (números históricos do comboio). Geografia, clima e probabilidades são estilizados, calibrados para reproduzir a matriz 2-2 / 1-3; não são estimativas históricas.
