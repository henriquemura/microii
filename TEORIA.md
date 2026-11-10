### 1. O jogo de Haywood

Jogadores: Aliados — força aérea comandada pelo Gen. George Kenney — (linhas, maximizam dias de bombardeio) e o comandante japonês (colunas, minimiza).
Estratégias: Norte (N) e Sul (S) para ambos.

```latex
A = \begin{pmatrix} a_{NN} & a_{NS} \\ a_{SN} & a_{SS} \end{pmatrix}
  = \begin{pmatrix} 2 & 2 \\ 1 & 3 \end{pmatrix}
```

Maximin (Aliados) e minimax (Japão):

```latex
\underline{v} = \max_i \min_j a_{ij} = \max\{2,\,1\} = 2
\qquad
\overline{v} = \min_j \max_i a_{ij} = \min\{2,\,3\} = 2
```

Como `\underline{v} = \overline{v} = 2`, há ponto de sela em (N, N): estratégias puras "casadas" (*matched strategies*). A mesma solução sai por dominância iterada: para o Japão (minimizador) a coluna N domina fracamente a coluna S (2 ≤ 2 e 1 ≤ 3); eliminada S, os Aliados preferem N (2 > 1).

### 2. Modelo estocástico de detecção

O comboio leva 3 dias. No dia d, a probabilidade de ser avistado depende da cobertura (concentrada/esparsa) e do clima no trecho: `p_d`. Na rota Norte, o dia 1 (saída de Rabaul, sob a frente de mau tempo) é de chuva forte, o dia 2 de chuva e o dia 3, perto de Lae, limpo. Na rota Sul, os três dias são limpos. Se o primeiro avistamento ocorre no dia d, há `B = 4 - d` dias de bombardeio (discretização implícita no artigo).

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
E[B] \approx \begin{pmatrix} 1{,}97 & 2{,}23 \\ 1{,}03 & 2{,}95 \end{pmatrix}
```

o que reproduz o ponto de sela em (N, N) com valor ≈ 2. Além da média, a distribuição também segue o artigo: em (N, N), com p = (0,15; 0,80; 0,95),

```latex
\Pr(B=3)=0{,}15 \quad \Pr(B=2)=0{,}68 \quad \Pr(B=1)=0{,}16 \quad \Pr(B=0)=0{,}01
```

ou seja, o caso histórico (avistado ao fim do dia 1, dois dias de bombardeio) é o resultado mais provável.

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

`p*` = probabilidade de os Aliados jogarem N; `q*` = probabilidade de o Japão jogar N. Exemplo: no modo estocástico, leve "Concentrado + chuva forte (dia 1)" para 0,60. Então E[B|N,N] ≈ 2,52 > E[B|N,S] ≈ 2,23: a chuva forte deixa de proteger o comboio, maximin ≈ 2,23 < minimax ≈ 2,52 e o equilíbrio passa a ser misto, com p* ≈ 0,87, q* ≈ 0,33 e v ≈ 2,32.

### 5. Calibração (estilizada)

Parâmetros padrão: 8 transportes, 8 destróieres, ≈ 6.900 soldados (números históricos do comboio). Geografia, clima e probabilidades são estilizados, calibrados para reproduzir a matriz 2-2 / 1-3; não são estimativas históricas.
