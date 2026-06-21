# A Time-Space Lower Bound for Parity Learning

## Problem

**Parity learning.** An unknown $x \in \{0,1\}^n$ is chosen uniformly at random. A learner sees a
stream of samples $(a_1,b_1),(a_2,b_2),\dots$, where each $a_t$ is uniform over $\{0,1\}^n$ and
$b_t = a_t \cdot x \pmod 2$. A past sample is unavailable unless stored. Two extreme algorithms bound
the resource trade: Gaussian elimination uses $O(n)$ samples and $\Theta(n^2)$ memory; candidate
enumeration uses $n+o(n)$ memory and $2^{\Theta(n)}$ samples. The result below shows this trade is
forced: there is no sub-quadratic-memory learner with sub-exponential samples.

## Key idea

Model the learner as a **branching program**: width $d = 2^{\text{memory}}$, length $m =$ number of
samples. The proof has two parts.

1. **Affine programs.** If the program's vertices carry affine-subspace labels with a soundness
   invariant, then "learning $x$" is forced to be "shrinking the label's dimension," and a
   *dimension-of-intersection* progress measure shows that reaching a low-dimensional label is
   $2^{-\Omega(n^2)}$-improbable: the overlap between the learner's locked-in linear constraints and
   the target's orthogonal space rises by at most one per step, and each rise needs the fresh random
   sample $a_t$ to fall in an exponentially small coset.

2. **Reduction.** A *general* branching program is simulated by an accurate affine one. The
   conditional distribution of $x$ at any memory state is a near-uniform mixture over affine
   subspaces — by Fourier analysis, the inner product $a\cdot x$ is a strong extractor, so the
   mixture is uniform unless some linear test is pinned to a constant. Grouping those subspaces into
   few near-uniform representatives keeps the simulation's width at $2^{O(n^2)}$, and the accuracy is
   propagated *additively* across the $2^{\Theta(n)}$ layers.

Balancing the dimension threshold $k = 4n/5$, the carving parameter $r = (\tfrac12+2\alpha)n$, and the
budgets makes the two parts cancel against the width, yielding the bound for memory $< cn^2$, $c <
\tfrac1{20}$.

## Definitions

For $a,x \in \{0,1\}^n$, $a\cdot x = \sum_i a_i x_i \bmod 2$. $A(n)$ = affine subspaces of $\{0,1\}^n$;
$U_w$ = uniform distribution on $w \in A(n)$; $U_n = U_{\{0,1\}^n}$. $|P-Q|_1$ = total $\ell_1$
distance. For a function $P:\{0,1\}^n\to\mathbb{R}$, $\widehat P(a) = \mathbb{E}_x[P(x)(-1)^{a\cdot x}]$.

**Branching program for parity learning.** A layered directed multigraph with $m+1$ layers of width
$\le d$; one start vertex; each non-leaf vertex has one out-edge per $(a,b)\in\{0,1\}^n\times\{0,1\}$;
each leaf labeled by an affine subspace $w(v)\in A(n)$ (the output guess "$x\in w(v)$"). A stream
traces a computation-path start $\to$ leaf. Width $\leftrightarrow$ memory $\log_2 d$; length
$\leftrightarrow$ samples $m$.

**Affine branching program.** Every vertex $v$ carries $w(v)\in A(n)$, with: (start) $w(\text{start})
= \{0,1\}^n$; (soundness) for edge $e=(u,v)$ labeled $(a,b)$, $w(e):= w(u)\cap\{x':a\cdot x'=b\}
\subseteq w(v)$. Soundness $\Rightarrow$ $x\in w(v)$ along the honest path always; success prob $= 1$.

**$\epsilon$-accurate.** For the layer-$t$ reached vertex $V_t$ and $y_t\sim U_{w(V_t)}$:
$|P_{V_t,x} - P_{V_t,y_t}|_1 \le \epsilon$.

## Main theorem

**Theorem (formal).** For any $c < \tfrac1{20}$ there is $\alpha>0$ such that: let $x\sim U_n$, $m\le
2^{\alpha n}$, and let $B$ be a branching program of length $m$ and width $\le 2^{cn^2}$ for parity
learning whose output is always an affine subspace of dimension $\le \tfrac{3n}{5}$. Then
$$
\Pr[\,x \in B\text{'s output}\,] \le O(2^{-\alpha n}).
$$

**Corollary (headline).** For any $c<\tfrac1{20}$ there is $\alpha>0$ such that any parity-learning
algorithm using $\le c n^2$ memory bits and $\le 2^{\alpha n}$ samples outputs $\tilde x$ with
$\Pr[\tilde x = x] \le O(2^{-\alpha n})$. Equivalently, learning parity with fewer than $\sim n^2/25$
memory bits requires an exponential number of samples.

## Proof

### Step 1 — distributions over affine subspaces (the Fourier / extractor core)

Let $W\in A(n)$ be a random affine subspace; $\mathbb{E}_W[U_W]$ is a mixture of uniform-on-subspace
distributions.

**Lemma 1 (mixing).** Let $r\ge n/2$. If for all $a\neq\vec0$, $b\in\{0,1\}$,
$\Pr_W[\forall x\in W: a\cdot x=b]\le 2^{-r}$, then $|\mathbb{E}_W[U_W]-U_n|_1 < 2^{-(r-n/2)}$.

*Proof.* For affine $w$, $\widehat{U_w}(a)= 2^{-n}$ if $a\cdot x\equiv0$ on $w$, $-2^{-n}$ if
$\equiv1$, else $0$. Hence $\widehat{\mathbb{E}_W[U_W]}(a)= 2^{-n}\!\big(\Pr_W[a\cdot x\equiv0\text{ on }
W]-\Pr_W[a\cdot x\equiv1\text{ on }W]\big)$, with the $a=\vec0$ coefficient $=2^{-n}=\widehat{U_n}
(\vec0)$. For $a\neq\vec0$, $\widehat{U_n}(a)=0$ and $|\widehat{\mathbb{E}_W[U_W]}(a)|\le 2^{-n}\cdot
2^{-r}$, so $\sum_{a\neq\vec0}(\widehat{\mathbb{E}_W[U_W]}(a)-\widehat{U_n}(a))^2 < 2^n(2^{-n}2^{-r})^2
= 2^{-n-2r}$. By Cauchy–Schwarz and Parseval, $(\mathbb{E}_x|P-U_n|)^2\le \mathbb{E}_x(P-U_n)^2 =
\sum_a(\widehat P-\widehat{U_n})^2 < 2^{-n-2r}$ (with $P=\mathbb{E}_W[U_W]$). Thus $|P-U_n|_1 = 2^n
\mathbb{E}_x|P-U_n| < 2^n\cdot 2^{-(n+2r)/2}=2^{-(r-n/2)}$. $\square$

**Lemma 2 (capture).** Let $r\ge n/2$. There exists $s\in A(n)$ with (1) $\Pr_W[W\subseteq s]\ge
2^{-\sum_{i=0}^{n-\dim s-1}(r-i/2)}$ and (2) $|\mathbb{E}_{W\mid W\subseteq s}[U_W]-U_s|_1 <
2^{-(r-n/2)}$.

*Proof.* Induction on $n$. Base $n=0$: $s=\{\vec0\}$. Step: if Lemma 1's hypothesis holds, take
$s=\{0,1\}^n$. Else $\exists a\neq\vec0,b$ with $\Pr_W[\forall x\in W:a\cdot x=b]>2^{-r}$; set
$u=\{x:a\cdot x=b\}$ ($\dim u=n-1$), so $\Pr_W[W\subseteq u]>2^{-r}$. Apply the hypothesis to $W'=
W\mid(W\subseteq u)$ over $u\cong\{0,1\}^{n-1}$ with $(n-1, r-\tfrac12)$, getting $s\subseteq u$.
(1): $\Pr_W[W\subseteq s]=\Pr_W[W\subseteq u]\Pr_{W'}[W'\subseteq s] > 2^{-r}\cdot 2^{-\sum_{i=0}^{n-1-
\dim s-1}(r-1/2-i/2)} = 2^{-\sum_{i=0}^{n-\dim s-1}(r-i/2)}$. (2): $\mathbb{E}_{W\mid W\subseteq s}[U_W]=
\mathbb{E}_{W'\mid W'\subseteq s}[U_{W'}]$, inherited. $\square$

**Lemma 3 (grouping — the only export).** Let $r\ge n/2$. There is a partial $\sigma:A(n)\to A(n)$ with
(1) $\Pr_W[W\notin\mathrm{dom}\,\sigma]\le 2^{-2n}$; (2) $w\subseteq\sigma(w)$; (3) for all $s\in
\mathrm{image}\,\sigma$, $|\mathbb{E}_{W\mid\sigma(W)=s}[U_W]-U_s|_1 < 2^{-(r-n/2)}$; (4) for every $k$,
$\#\{s\in\mathrm{image}\,\sigma:\dim s\ge k\}\le 4n\cdot 2^{\sum_{i=0}^{n-k-1}(r-i/2)}$.

*Proof.* Repeatedly apply Lemma 2: $W_0=W\to s_0$, set $\sigma(w)=s_0$ for $w\subseteq s_0$; $W_1=W\mid
(W\not\subseteq s_0)\to s_1$; …; stop when $\Pr_W[W\notin\mathrm{dom}\,\sigma]\le2^{-2n}$. The $s_i$ are
distinct. (1) by the stopping rule; (2),(3) by Lemma 2. (4): each carve producing $\dim s_i\ge k$
captures $\ge 2^{-\sum_{i=0}^{n-k-1}(r-i/2)}$ of the remaining mass (more dimension $\Rightarrow$ fewer
sum terms $\Rightarrow$ larger fraction), so after $\le 4n\cdot 2^{\sum_{i=0}^{n-k-1}(r-i/2)}$ such
carves the residual is $\le2^{-2n}$. $\square$

### Step 2 — affine lower bound (dimension-of-intersection)

**Lemma 4.** Let $k<n$, $P$ a length-$m$ affine program with $\dim w(u)\ge k$ for all $u$, and $v$ a
vertex with $\dim w(v)=k$. Then $\Pr[\text{path reaches }v]\le m^{\,n-k}\cdot 2^{\sum_{j=0}^{n-k-1}
(n-2k-j)}$.

*Proof.* Let $s=\{a:\exists b,\forall x'\in w(v),a\cdot x'=b\}$ (orthogonal to $w(v)$, $\dim s=n-k$).
Let $S_i$ be orthogonal to $w(V_i)$; soundness gives $S_i\subseteq\mathrm{span}(S_{i-1}\cup\{a_i\})$.
Set $Z_i=\dim(S_i\cap s)$: $Z_0=0$, $Z_i\le Z_{i-1}+1$, and reaching $v$ needs some $Z_i=n-k$. A rise
($Z_i>Z_{i-1}$) requires $\exists a\in S_{i-1}$ with $a\oplus a_i\in s$; for fixed $a$, $\Pr[a\oplus
a_i\in s]=2^{\dim s-n}=2^{-k}$. Distinct possibilities form cosets of $S_{i-1}\cap s$, numbering
$2^{\dim S_{i-1}-Z_{i-1}}\le 2^{n-k-Z_{i-1}}$, so conditioned on $x,a_1,\dots,a_{i-1}$ (fixing
$Z_{i-1}$), $\Pr[\text{rise at }i]\le 2^{n-k-Z_{i-1}}\cdot2^{-k}=2^{n-2k-Z_{i-1}}$. For a fixed tuple
$i_1<\dots<i_{n-k}$ of rise-steps, $\Pr\le\prod_{j=1}^{n-k}2^{n-2k-(j-1)}=2^{\sum_{j=0}^{n-k-1}(n-2k-j)}$.
Union bound over $<m^{n-k}$ tuples gives the claim. $\square$

### Step 3 — reduction general $\to$ accurate affine

**Lemma 5.** Let $k'<n$. Let $B$ be a length-$m$, width-$d$ parity-learning program, all leaves in the
last layer, output dimension $\le k'$, success $\beta$. Let $n/2\le r\le n$, $\epsilon=4m\cdot
2^{-(r-n/2)}$. Then there is an $\epsilon$-accurate length-$m$ affine program $P$ with: (1) for every
$k$, $\#$dimension-$k$ vertices $\le 4n\cdot2^{\sum_{i=0}^{n-k-1}(r-i/2)}\cdot dm$; (2) for $k'<k<n$,
$\Pr[\dim(\text{output})<k]\ge \beta-\epsilon-2^{-(k-k')}$.

*Proof sketch (full induction).* Build $P$ layer by layer with inductive hypothesis: $\exists U_j$ over
layer $j$ with $y_j\sim U_{w(U_j)}$ and $|P_{V_j,x}-P_{U_j,y_j}|_1\le \tfrac{\epsilon_j}{2}$,
$\epsilon_j=4j\cdot2^{-(r-n/2)}$ (the surrogate form keeps errors additive across $m=2^{\Theta(n)}$
layers). Base $j=0$: label start $\{0,1\}^n$, $U_0=V_0$, distance $0$. Step: from $U_{j-1}$, $y_{j-1}$,
draw $a\sim U$, $b=a\cdot y_{j-1}$, follow edge to $V$; $W=w(U_{j-1})\cap\{a\cdot x'=b\}$. For each
$B$-vertex $v$, apply Lemma 3 to $W_v=W\mid(V=v)$ to get $\sigma_v$; split $v$ into copies $(v,s)$,
$s\in\mathrm{image}\,\sigma_v$ (label $s$, or $\{0,1\}^n$ for the $*$ leftover), routing incoming edge
$e=(u,v)$ to $(v,\sigma_v(w(e)))$ — soundness holds since $w(e)\subseteq\sigma_v(w(e))$. Set $U_j=
(V,\sigma_V(W))$. With $y'_j\sim U_W$: (3a) $|P_{U_j,y'_j}-P_{U_j,y_j}|_1\le 2\cdot2^{-(r-n/2)}$ by
Lemma 3(3) plus the $2^{-2n}$ leftover; (3b) $|P_{V_j,x}-P_{U_j,y'_j}|_1\le 2(j-1)\cdot2^{-(r-n/2)}$ by
the shared transformation $T$ (draw $a$, $b=a\cdot z$, follow edge) with $T(V_{j-1},x)\sim(V_j,x)$,
$T(U_{j-1},y_{j-1})\sim(U_j,y'_j)$ — $T$ cannot increase $\ell_1$ — and the inductive hypothesis.
Triangle: $|P_{V_j,x}-P_{U_j,y_j}|_1\le 2j\cdot2^{-(r-n/2)}=\epsilon_j/2$. Accuracy of $P$ follows from
the hypothesis at each layer. Property (1) is Lemma 3(4) times $dm$. Property (2): with $V_m=(V,S)$,
$\Pr[x\in w(V)]=\beta$ and $\epsilon$-accuracy give $\Pr[y_m\in w(V)]\ge\beta-\epsilon$; since $\dim
w(V)\le k'$, if $\dim w(V_m)\ge k$ then $\Pr[y_m\in w(V)\mid\cdot]\le2^{k'-k}$, so $\beta-\epsilon\le
\Pr[\dim w(V_m)<k]+2^{k'-k}$. $\square$

### Step 4 — assemble the constants

Let $B$ have length $m=2^{\alpha n}$, width $d=2^{cn^2}$, output dimension $\le k'=\tfrac{3n}{5}$,
success $\beta$. Set $r=(\tfrac12+2\alpha)n$, $k=\tfrac{4n}{5}$ (so $n-k=\tfrac n5$, $n-2k=-\tfrac{3n}
{5}$). Then $\epsilon=4\cdot2^{-\alpha n}$ and $2^{-(k-k')}=2^{-n/5}$, so by Lemma 5(2)
$\Pr[\dim(\text{output})<k]\ge\beta-5\cdot2^{-\alpha n}$. Make all dimension-$k$ vertices leaves;
every vertex has $\dim\ge k$. By Lemma 5(1) and Lemma 4,
$$
\Pr[\text{reach a dim-}k\text{ vertex}]\le \Big(4n\cdot2^{\sum_{i=0}^{n-k-1}(r-i/2)}\cdot dm\Big)
\Big(m^{n-k}\cdot2^{\sum_{i=0}^{n-k-1}(n-2k-i)}\Big).
$$
Substituting and combining (the cancellation $\tfrac12-\tfrac35=-\tfrac1{10}$ in the linear terms;
$-\tfrac i2 - i = -\tfrac32 i$ in the quadratic terms):
$$
= 4nm\cdot2^{cn^2}\cdot2^{(n-k)(3\alpha n-\frac1{10}n)}\cdot2^{-\frac34(n-k)(n-k-1)}
= 4nm\cdot2^{\,n^2\left(c+\frac35\alpha-\frac1{20}+\frac{3}{20n}\right)}.
$$
Hence if $\alpha<\tfrac53(\tfrac1{20}-c)$, this is $2^{-\Omega(n^2)}$, so $\beta-5\cdot2^{-\alpha n}\le
2^{-\Omega(n^2)}$, i.e. $\beta\le O(2^{-\alpha n})$. $\blacksquare$

The corollary is the special case $k'=0$ (the learner outputs a single guess $\tilde x$, a dimension-0
subspace), and the constant is rounded: any $c<\tfrac1{20}$ works, so $\sim n^2/25$ memory bits is a
clean sufficient threshold below which exponentially many samples are needed.

## Remarks

- **Why the conjectured constant $n^2/4$ is too strong.** A Gaussian-elimination variant keeps, at
  step $k$, a $k\times n$ matrix whose first $k$ columns form the identity, costing $\tfrac{n^2}{4}+
  O(n)$ memory; restricting to samples supported on the first $\tfrac{3n}{4}$ coordinates lets one
  solve a size-$\tfrac{3n}{4}$ problem with $2^{n/4}$ samples and $\tfrac{9}{64}n^2+o(n^2)\approx
  0.14\,n^2$ memory, beating $n^2/4$. So the right statement is the qualitative trade with a smaller
  constant, here $c<\tfrac1{20}$.

- **Cryptographic corollary.** Sharing $x\in\{0,1\}^n$, encrypt a bit $M$ as $M\oplus(a\cdot x)$ for
  fresh public uniform $a$ (cost $n$/bit). An adversary with $<\tfrac{n^2}{25}$ memory watching up to
  $2^{\Omega(n)}$ uses cannot recover $x$ beyond exponentially small probability; since $a\cdot x$ is
  a strong extractor, it cannot predict the next pad bit beyond exponentially small advantage. This is
  a bounded-storage scheme whose encryption time is *not* forced to be linear in the adversary's
  memory.
