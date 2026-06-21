# Context: sensitivity, degree, and the geometry of large induced subgraphs of the hypercube

## Research question

A Boolean function is a map $f:\{0,1\}^n \to \{0,1\}$. Theoretical computer science measures its "complexity" with a whole family of quantities, and a striking empirical fact about that family is that almost all of its members are **polynomially related**: each is bounded above by a fixed power of any other. Block sensitivity, certificate complexity, deterministic decision-tree depth, the degree of the representing real polynomial, randomized and quantum query complexity, approximate degree — these all live in one equivalence class, mutually polynomial.

One natural measure refuses to join. The **sensitivity** $s(f)$ — the largest number of single-coordinate flips, at any one input, that change the value of $f$ — is the most local, most elementary measure of all. It is trivial that $s(f)\le bs(f)$ (a single flipped bit is a block of size one). The reverse direction is the whole problem: is there an absolute constant $C$ with
$$bs(f)\le s(f)^C$$
for every Boolean function? Three decades of effort produced no polynomial bound at all; the best known upper bound on block sensitivity in terms of sensitivity is **exponential**. The goal is to decide whether sensitivity belongs to the same polynomial equivalence class as everything else, or sits genuinely apart.

## Background

**The complexity measures.** For $x\in\{0,1\}^n$ and $S\subseteq[n]$, write $x^S$ for $x$ with all coordinates in $S$ flipped. The local sensitivity $s(f,x)=\#\{i: f(x)\ne f(x^{\{i\}})\}$, and $s(f)=\max_x s(f,x)$. The local block sensitivity $bs(f,x)$ is the largest number of pairwise-disjoint blocks $B_1,\dots,B_k\subseteq[n]$ with $f(x)\ne f(x^{B_i})$ for each $i$, and $bs(f)=\max_x bs(f,x)$. Block sensitivity was introduced by Nisan (1989) precisely to capture parallel-computation cost: for a CREW PRAM, the number of steps to compute $f$ is $\Theta(\log bs(f))$. Every $f$ has a **unique multilinear real polynomial** that agrees with it on $\{0,1\}^n$; its degree is $\deg(f)$. Certificate complexity $C(f)$ is the size of the smallest partial assignment forcing the output. Decision-tree depth $D(f)$ is the worst-case number of adaptive bit-queries.

**The web of polynomial relations.** The known relations (Nisan; Nisan–Szegedy; Buhrman–de Wolf survey) form a tight cluster: $bs(f)\le C(f)\le D(f)$, $D(f)\le bs(f)^2 \cdot(\text{lower-order})$, $\deg(f)\le D(f)$, and the two directions tying degree to block sensitivity. Nisan and Szegedy (1992/94), by symmetrizing the representing polynomial and applying the Markov brothers' inequality from approximation theory, proved
$$\deg(f)\ge \sqrt{bs(f)/2}, \qquad\text{equivalently}\qquad bs(f)\le 2\,\deg(f)^2 .$$
Tal (2013) sharpened the constant to $bs(f)\le \deg(f)^2$. So degree and block sensitivity are quadratically tied, in a way that is fully proved. Sensitivity is the only node of this graph with no polynomial edge back to the cluster.

**The hypercube and its spectrum.** $Q^n$ is the graph on $\{0,1\}^n$ with an edge between vectors differing in exactly one coordinate. It is bipartite (even vs. odd weight), $n$-regular, vertex-transitive. The adjacency matrix of $Q^n$ is the $n$-fold tensor sum of $\bigl[\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\bigr]$; its eigenvalues are $n-2k$ for $k=0,\dots,n$, i.e. the integers $-n,-n+2,\dots,n-2,n$, with the multiplicity of $n-2k$ equal to $\binom{n}{k}$. The $2^{n-1}$ even-weight vertices form an independent set, so the half-cube can be totally edgeless.

**The motivating combinatorial phenomenon (Chung–Füredi–Graham–Seymour, 1988).** Take strictly more than half the vertices of $Q^n$. The even vertices alone (exactly half) induce no edges; but adding even one more vertex forces structure. CFGS proved that any induced subgraph $H$ on more than $2^{n-1}$ vertices must contain a vertex of degree at least $\bigl(\tfrac12-o(1)\bigr)\log_2 n$. Their argument is a clever counting/averaging induction (one shows a subgraph of $Q^n$ with average degree $\bar d$ has at least $2^{\bar d}$ vertices, then localizes per direction), and it bottoms out at a **logarithmic** lower bound. In the same paper they exhibit a $(2^{n-1}+1)$-vertex induced subgraph whose maximum degree is exactly $\lceil\sqrt n\,\rceil$, by a set-system construction. So the truth, if there is a clean one, lies between $\log_2 n$ and $\sqrt n$.

**The bridge (Gotsman–Linial, 1992).** Identify $f:\{0,1\}^n\to\{\pm1\}$ with a $2$-coloring of $Q^n$. Multiplying $f$ by the parity function $(-1)^{\sum x_i}$ produces $g=f\cdot\chi_{[n]}$ that flips the color on every odd vertex; under this twist, sensitivity becomes adjacency-counting in an induced subgraph, and the top Fourier coefficient becomes the average of $g$. Concretely $s(g,x)=n-s(f,x)$ and $\widehat g(S)=\widehat f([n]\setminus S)$, so $\mathbb E[g]=\widehat g(\varnothing)=\widehat f([n])$, which is nonzero exactly when $\deg(f)=n$. Writing $\Gamma(H)=\max\{\Delta(H),\Delta(Q^n-H)\}$ where $Q^n-H$ is the complementary induced subgraph, this gives an exact equivalence (for any monotone $h:\mathbb N\to\mathbb R$):
$$\bigl[\forall H,\ |V(H)|\ne 2^{n-1}\Rightarrow \Gamma(H)\ge h(n)\bigr] \iff \bigl[\forall f,\ s(f)\ge h(\deg(f))\bigr].$$
This converts the sensitivity question into a pure question about how dense an induced subgraph of the hypercube on more than half the vertices must be somewhere.

## Baselines

**Counting / averaging on the cube (CFGS, 1988).** The state of the art for *lower-bounding* the max degree of a large induced subcube. Core mechanism: a density inequality ($\bar d$ average degree $\Rightarrow \ge 2^{\bar d}$ vertices), applied direction-by-direction and summed. It is purely combinatorial and self-contained, and yields $\Delta(H)\gtrsim\tfrac12\log_2 n$. The same paper's $\lceil\sqrt n\rceil$-degree construction is the opposite-direction baseline: it pins the *upper* side of the truth at $\sqrt n$.

**Symmetrization + Markov inequality (Nisan–Szegedy, 1992).** The tool that ties degree to block sensitivity. Average the representing polynomial over the symmetric group to get a univariate polynomial of no larger degree; its values on integer points are pinned by $f$'s behavior on a sensitive block, and a polynomial that moves that much over $[0,t]$ must have degree $\gtrsim\sqrt t$. Gives $bs(f)\le 2\deg(f)^2$.

**Direct spectral bounds via the plain adjacency matrix.** A classical fact: for any graph $H$, $\Delta(H)\ge \lambda_1(H)$, the largest adjacency eigenvalue. Combined with eigenvalue interlacing — a principal submatrix's eigenvalues interlace the parent's — this transfers eigenvalue information of $Q^n$ to every large induced subgraph. For the ordinary $0/1$ adjacency matrix of $Q^n$, the eigenvalues are the integers $n, n-2, \dots, -n$ with multiplicities $\binom{n}{k}$ concentrating near zero.

**Direct attacks on $s$ vs $bs$.** Lower bounds (separations): Rubinstein (1995) built $f$ with $bs(f)=\tfrac12 s(f)^2$, a quadratic gap; Virza (2011) and Ambainis–Sun (2011) improved the constant but stayed quadratic. Upper bounds: Kenyon–Kutin (2004) got $bs(f)=O\!\bigl(e^{s(f)}\sqrt{s(f)}\bigr)$.

## Evaluation settings

This is a theorem to be proved, so the "evaluation" is correctness and sharpness, not benchmarking. The yardsticks already in place:
- **Sharpness target on the cube side.** The CFGS $(2^{n-1}+1)$-vertex construction with max degree $\lceil\sqrt n\rceil$: any lower bound on $\Delta(H)$ better than $\sqrt n$ would contradict it, so $\sqrt n$ is the bound to aim for.
- **Sharpness target on the Boolean side.** The AND-of-ORs function $\bigwedge_{i=1}^{m}\bigvee_{j=1}^{m} x_{ij}$ on $n=m^2$ variables has $\deg(f)=m^2=n$ and $s(f)=m=\sqrt n$, so any relation $s(f)\ge h(\deg(f))$ cannot beat $h(d)=\sqrt d$.
- **The reduction interface.** Gotsman–Linial's equivalence is the protocol: a cube lower bound $\Gamma(H)\ge h(n)$ for all $H$ with $|V(H)|\ne 2^{n-1}$ is *equivalent* to $s(f)\ge h(\deg(f))$ for all $f$. Plugging the desired $h(n)=\sqrt n$ through it connects the graph theorem to the conjecture; then the already-proven $bs(f)\le\deg(f)^2$ closes the loop to $bs$ vs $s$.

## Code framework

The natural artifact here is a theorem with a complete proof, not a program. Still, the proof is constructive enough to express its skeleton as the small symbolic objects it manipulates; the "scaffold" below names the pieces that already exist before the key idea and leaves the central object empty.

```python
import numpy as np

# Pre-existing objects -------------------------------------------------

def hypercube_adjacency(n):
    """0/1 adjacency matrix of Q^n: vertices in {0,1}^n, edge iff Hamming distance 1.
    Eigenvalues are the integers n, n-2, ..., -n (multiplicity C(n,k) at n-2k)."""
    # tensor-sum of [[0,1],[1,0]]; standard, already known
    A = np.array([[0, 1], [1, 0]])
    M = A.copy()
    for _ in range(n - 1):
        I = np.eye(M.shape[0], dtype=int)
        M = np.block([[M, I], [I, M]])
    return M

def largest_eigenvalue(M):
    """lambda_1 of a symmetric matrix (real)."""
    return np.max(np.linalg.eigvalsh(M))

def max_degree(adj):
    """Maximum degree Delta of the graph with this 0/1 adjacency matrix."""
    return int(adj.sum(axis=1).max())

# Known inequality: for a symmetric matrix supported on the edges of a graph
# H with entries bounded by 1 in absolute value, Delta(H) >= lambda_1.
# Known tool: Cauchy interlacing -- a principal submatrix's eigenvalues
# interlace those of the parent symmetric matrix.

# The object to be discovered ------------------------------------------

def matrix_on_cube_edges(n):
    """A symmetric matrix indexed by V(Q^n), supported on the edges of Q^n.
    The choice of its entries is the object we will design here."""
    # TODO: the object we will define here
    pass

def main_bound(n):
    """Lower bound on the max degree of any (2^{n-1}+1)-vertex induced
    subgraph of Q^n, obtained from matrix_on_cube_edges(n) via the known
    spectral inequality and interlacing."""
    # TODO: assemble the argument from the object above
    pass
```
