# Szemerédi's Regularity Lemma

## The problem it solves

Counting a fixed subgraph $H$ in a graph $G$ is easy when $G$ is random-looking — edges spread
evenly across all large vertex sets — and intractable for arbitrary dense graphs. The Regularity
Lemma removes the gap: it shows every large dense graph can be approximated, to any fixed
tolerance, by a *bounded-complexity* weighted blueprint of random-looking blocks. Counting,
extremal, and removal results for random graphs then transfer to all dense graphs.

## The key idea

Partition the vertices into a bounded number of equal blocks so that between almost every pair of
blocks the edges are quasirandom ($\varepsilon$-regular). The partition is found by an
**energy/index increment** argument: each block-pair carries a squared-density weight; refining
never lowers the total, an irregular pair refined along its own counterexample strictly raises it,
and since the total is bounded above by $1$, only a bounded number of refinements can occur.

## Definitions

For disjoint $X,Y\subseteq V(G)$, density $d(X,Y)=e(X,Y)/(|X|\,|Y|)$.

**$\varepsilon$-regular pair.** $(A,B)$ is $\varepsilon$-regular if for all $X\subseteq A$,
$Y\subseteq B$ with $|X|\ge\varepsilon|A|$, $|Y|\ge\varepsilon|B|$:
$\;|d(X,Y)-d(A,B)|\le\varepsilon$.

**$\varepsilon$-regular partition.** $V=V_0\cup V_1\cup\dots\cup V_k$ with exceptional set
$|V_0|\le\varepsilon|V|$, equal blocks $|V_1|=\dots=|V_k|$, and all but at most $\varepsilon k^2$ of
the pairs $(V_i,V_j)$, $1\le i<j\le k$, $\varepsilon$-regular.

**Energy / index.** With $n=|V|$,
$$q(A,B)=\frac{|A||B|}{n^2}\,d(A,B)^2=\frac{e(A,B)^2}{|A||B|\,n^2},\qquad
q(P)=\sum_{i<j}q(V_i,V_j),$$
extended to partitions with an exceptional set by treating $V_0$ as singletons.

## The theorem

**Regularity Lemma.** For every $\varepsilon>0$ and every integer $m\ge1$ there is an integer
$M=M(\varepsilon,m)$ such that every graph of order $\ge m$ has an $\varepsilon$-regular partition
$\{V_0,V_1,\dots,V_k\}$ with $m\le k\le M$. The proof gives a tower-type bound, with tower height
proportional to $\varepsilon^{-5}$.

## Proof (energy increment)

**(0) $q(P)\le1$.** Since $d^2\le1$,
$q(P)\le n^{-2}\sum_{i<j}|V_i||V_j|\le n^{-2}\cdot\tfrac12\big(\sum_i|V_i|\big)^2\le1.$

**(1) Refinement does not decrease $q$.** For partitions $\mathcal C$ of $C$, $\mathcal D$ of $D$,
the inequality $\sum_i e_i^2/\mu_i\ge(\sum_i e_i)^2/\sum_i\mu_i$ (Cauchy–Schwarz, $a_i=\sqrt{\mu_i}$,
$b_i=e_i/\sqrt{\mu_i}$) with $\mu_{ij}=|C_i||D_j|$, $e_{ij}=e(C_i,D_j)$ gives
$$\sum_{i,j}\frac{e_{ij}^2}{|C_i||D_j|}\ge\frac{e(C,D)^2}{|C||D|}\quad\Longrightarrow\quad
q(\mathcal C,\mathcal D)\ge q(C,D).$$
Summing over pairs, $q(P')\ge q(P)$ when $P'$ refines $P$.

**(2) An irregular pair gains $\varepsilon^4|C||D|/n^2$.** If $(C,D)$ is not $\varepsilon$-regular,
take witnesses $C_1\subseteq C$, $D_1\subseteq D$, $|C_1|\ge\varepsilon|C|$, $|D_1|\ge\varepsilon|D|$,
$|\eta|>\varepsilon$ where $\eta=d(C_1,D_1)-d(C,D)$. Refine $\{C_1,C_2\},\{D_1,D_2\}$. With
$c=|C|,d=|D|,e=e(C,D),c_i=|C_i|,d_j=|D_j|$, applying Cauchy–Schwarz to the off-$(1,1)$ terms,
$$n^2 q(\mathcal C,\mathcal D)\ge\frac{e_{11}^2}{c_1d_1}+\frac{(e-e_{11})^2}{cd-c_1d_1}.$$
Substituting $e_{11}=\tfrac{c_1d_1}{cd}e+\eta c_1d_1$, the cross terms cancel and the $e^2$ terms
recombine to $e^2/(cd)$, leaving
$$n^2 q(\mathcal C,\mathcal D)\ge\frac{e^2}{cd}+\eta^2 c_1d_1\ge n^2 q(C,D)+\varepsilon^4 cd,$$
using $\eta^2>\varepsilon^2$, $c_1\ge\varepsilon c$, $d_1\ge\varepsilon d$. Hence
$q(\mathcal C,\mathcal D)\ge q(C,D)+\varepsilon^4|C||D|/n^2$.

**(3) An irregular partition gains $\varepsilon^5/2$.** It is enough to prove the case
$0<\varepsilon\le\tfrac14$, since a stronger partition at tolerance $1/4$ handles larger
$\varepsilon$. Suppose $P=\{C_0,C_1,\dots,C_k\}$ has equal blocks of size $c$,
$|C_0|\le\varepsilon n$, and more than $\varepsilon k^2$ irregular pairs. For each, take the
$2\times2$ witness-cut of (2); let
$\mathcal C$ be the common refinement (each $C_i$ shattered by its $\le k-1$ cuts into
$\le2^{k-1}$ cells, so $k\le|\mathcal C|\le k2^{k-1}$). By (1) and (2),
$$q(\mathcal C)\ge q(P)+\varepsilon k^2\cdot\varepsilon^4\frac{c^2}{n^2}
=q(P)+\varepsilon^5\Big(\frac{kc}{n}\Big)^2\ge q(P)+\frac{\varepsilon^5}{2},$$
since $kc=n-|C_0|\ge\tfrac34 n$ gives $(kc/n)^2\ge9/16>\tfrac12$. Re-slicing $\mathcal C$ into
equal blocks of size $\lfloor c/4^k\rfloor$ (slivers to $C_0$) is a further refinement, so by (1)
it preserves the gain; it yields between $k$ and $k4^k$ blocks and adds $\le n/2^k$ vertices to
the exceptional set.

**(4) Termination + bound.** Put $s_0=2/\varepsilon^5$ and $s=\lceil s_0\rceil$. Choose
$k_0\ge m$ with $2^{k_0-1}\ge s/\varepsilon$, set $f(x)=x4^x$, and take
$$M=\max\{f^{\,s}(k_0),\,2k_0/\varepsilon\}.$$
If $n\le M$, the singleton partition has $m\le n\le M$ parts and is regular. If $n>M$, start with
$k_0$ equal blocks and $|C_0|<k_0\le\varepsilon n/2$. While the partition is irregular, (3) raises
$q$ by $\ge\varepsilon^5/2$; since $q\in[0,1]$, this happens at most $s_0$ times, hence at most $s$
integer rounds. The number of real blocks never drops below $k_0$, so the exceptional set grows by at most
$s n/2^{k_0}\le\varepsilon n/2$ in total, and stays $\le\varepsilon n$. The block count evolves by
$x\mapsto x4^x$ for at most $s$ rounds, so it is $\le f^s(k_0)\le M$, a tower-type bound of height
$\propto\varepsilon^{-5}$. $\qquad\blacksquare$

## The payoff: reduced graph + Embedding/Counting Lemma

**Reduced graph.** Given an $\varepsilon$-regular partition and a density floor $d$, let $R$ have
one vertex per block and an edge $V_iV_j$ whenever $(V_i,V_j)$ is $\varepsilon$-regular with
$d(V_i,V_j)\ge d$. Then $|V(R)|\le M(\varepsilon,m)$.

**Most degrees are large.** If $(A,B)$ is $\varepsilon$-regular of density $d$ and $|Y|>\varepsilon|B|$,
then $\#\{x\in A:\deg(x,Y)<(d-\varepsilon)|Y|\}\le\varepsilon|A|$.

**Embedding (Key) Lemma.** Let $\delta=d-\varepsilon$ and $\varepsilon_0=\delta^{\Delta}/(2+\Delta)$.
Build $G$ by blowing up each vertex of $R$ to $m$ vertices and each edge to an $\varepsilon$-regular
pair of density $\ge d$. If $\varepsilon\le\varepsilon_0$ and $H$ has $h$ vertices and maximum degree
$\Delta$, then $H\subseteq R(t)$ with $t-1\le\varepsilon_0 m$ implies $H\subseteq G$, with
$H\to G>\big((\delta^{\Delta}-\Delta\varepsilon)m\big)^{h}$ copies. Proof: embed vertices one by
one, confining each unplaced $v_j$ to a candidate set $C_j$; placing $v_i$ shrinks each unplaced
neighbour's candidate set by a factor $>\delta$, and the most-degrees-large fact removes at most
$\varepsilon m$ bad choices for each future neighbour. The choice
$\varepsilon_0=\delta^\Delta/(2+\Delta)$ leaves room for those $\Delta\varepsilon m$ bad choices and
the previously used vertices in the same cluster.

**Consequence (removal / counting).** Run a classical extremal theorem (Turán, Erdős–Stone,
König–Hall) on the bounded $R$ and lift via the Embedding Lemma; conversely, $o(n^{v(H)})$ copies
of $H$ implies all copies sit on within-block / irregular / sparse pairs, removable by $o(n^2)$
edges — the graph removal lemma. The triangle case yields Roth's theorem (positive density forces
$3$-term arithmetic progressions).
