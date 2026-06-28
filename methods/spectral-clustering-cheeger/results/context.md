# Context: cutting a graph into two well-separated clusters

## Research question

Given a graph $G=(V,E)$ — a social network, a mesh, a similarity graph between data points — we
want to split the vertices into two groups that are *internally well-connected* but *weakly
connected to each other*: a good "bottleneck" cut. The natural way to score a candidate set
$S \subseteq V$ is by how few edges leave it relative to how big it is. Counting boundary edges
$\partial(S) = \{(u,v)\in E : u\in S, v\notin S\}$ and normalizing by either vertex count or degree
volume gives two closely related measures:

$$\theta(S) = \frac{|\partial(S)|}{|S|}\quad\text{(isoperimetric ratio)},\qquad
\phi(S) = \frac{|\partial(S)|}{\min(d(S),\,d(V\setminus S))}\quad\text{(conductance)},$$

where $d(S)=\sum_{u\in S} d(u)$ is the total degree (volume) of $S$. The graph's conductance is
$\phi_G = \min_{\varnothing\ne S\subsetneq V} \phi(S)$, while the vertex-count variant is
$\theta_G=\min_{|S|\le n/2}\theta(S)$. A small value means there is a genuine bottleneck; a large
value means the graph is an expander with no good cut.

The question is how to find a cut with small conductance in polynomial time, given that
$\phi_G$ is a minimum over all $2^{|V|}$ subsets.

For this programming task, the deliverable is a single self-contained C++17 program that reads the
graph instance from standard input and writes the required output to standard output.

## Background

**The graph Laplacian and its quadratic form.** For a weighted graph $G=(V,E,w)$ define the
degree matrix $D$ with $D(u,u)=d(u)=\sum_v w_{u,v}$, the (weighted) adjacency matrix $A$, and the
Laplacian $L = D - A$. The single fact that makes $L$ central is its quadratic form: for any
$x\in\mathbb{R}^{V}$,

$$x^{T} L x = \sum_{(u,v)\in E} w_{u,v}\,(x(u)-x(v))^{2}.$$

This is a sum of squared differences across edges, so $L$ is positive semidefinite, $L\mathbf{1}=0$
(the all-ones vector is in the kernel), and $x^{T}Lx$ measures how much $x$ "stretches" across
edges. Writing $L=\sum_{(u,v)\in E} w_{u,v}(\delta_u-\delta_v)(\delta_u-\delta_v)^{T}$ exhibits it
as a sum of one rank-one term per edge. As an operator, $(Lx)(u)=\sum_{(u,v)\in E}w_{u,v}(x(u)-x(v))$.

**Eigenvalues as optimization (Courant–Fischer).** For a symmetric matrix $M$ with eigenvalues
$\lambda_1\le\lambda_2\le\cdots\le\lambda_n$ and eigenvectors $\psi_1,\dots,\psi_n$,

$$\lambda_i = \min_{x\,\perp\,\psi_1,\dots,\psi_{i-1}} \frac{x^{T}Mx}{x^{T}x},$$

with the minimizer equal to $\psi_i$. The quantity $x^{T}Mx/x^{T}x$ is the **Rayleigh quotient**.
For the Laplacian, $\lambda_1=0$ with $\psi_1=\mathbf{1}$, so the first informative eigenvalue is

$$\lambda_2 = \min_{x\,\perp\,\mathbf{1}} \frac{x^{T}Lx}{x^{T}x}
            = \min_{x\,\perp\,\mathbf{1}} \frac{\sum_{(u,v)\in E} w_{u,v}(x(u)-x(v))^{2}}{\sum_u x(u)^2}.$$

**Algebraic connectivity (Fiedler 1973).** Miroslav Fiedler studied exactly this second-smallest
Laplacian eigenvalue, naming it the *algebraic connectivity* $a(G)$ and characterizing it
variationally as $a(G)=\min\{x^{T}Lx : x^{T}x=1,\ x^{T}\mathbf{1}=0\}$. He proved that $a(G)=0$ if
and only if $G$ is disconnected (and more generally the multiplicity of the eigenvalue $0$ equals
the number of connected components), and that $a(G)\le v(G)\le e(G)$ — the algebraic connectivity
lower-bounds the *vertex* connectivity, which lower-bounds the *edge* connectivity. He computed it
for basic families: $a(K_n)=n$, $a(P_n)=2(1-\cos(\pi/n))$ for the path, $a(K_{p,q})=\min(p,q)$ for
complete bipartite graphs. The qualitative message — $\lambda_2$ detects whether and how strongly
the graph hangs together — is the first theoretical justification for reading global connectivity
off a single eigenvalue, and Fiedler suggested using the corresponding *Fiedler vector* to bisect a graph.

**Eigenvector graph drawing (Hall 1970).** Hall asked how to lay a graph out on a line so that
adjacent vertices land near one another: minimize $\sum_{(u,v)\in E}(x(u)-x(v))^2=x^{T}Lx$ subject
to a scale constraint $\|x\|^2=1$ and a centering constraint $\mathbf{1}^{T}x=0$ (without which the
trivial all-equal placement wins). The solution is the eigenvector $\psi_2$ of $\lambda_2$. This is
the same Rayleigh-quotient relaxation seen from the embedding side: vertices that the graph wants
close get close coordinates, so the coordinate *separates* loosely-connected regions.

**The continuous ancestor (Cheeger 1970).** On a compact Riemannian manifold $M$, the *Cheeger
isoperimetric constant* is $h(M)=\inf_E \frac{S(E)}{\min(V(A),V(B))}$, the infimum over smooth
hypersurfaces $E$ cutting $M$ into pieces $A,B$, of cut area over the smaller volume — a continuous
"sparsest cut." Cheeger proved a lower bound on the first nonzero eigenvalue $\lambda_1$ of the
Laplace–Beltrami operator in terms of it, $\lambda_1(M)\ge h(M)^2/4$: a small spectral gap forces a
geometric bottleneck. Buser (1982) proved a reverse inequality. The graph Laplacian is the discrete
analogue of the Laplace–Beltrami operator, and $\theta_G$/$\phi_G$ are the discrete analogues of
$h(M)$, which invites a discrete version of these bounds.

## Baselines

**Brute-force / combinatorial sparsest cut.** Directly minimizing $\phi(S)$ over subsets is exact
but exponential; the decision problem is NP-hard. Local-search and max-flow/min-cut heuristics
(Kernighan–Lin–style swaps, flow-based balanced cuts) give cuts but without a certificate that they
are close to $\phi_G$, and flow-based balanced separators are expensive.

**Spectral bounds for partitioning (Donath–Hoffman 1973).** Donath and Hoffman derived lower bounds
on the weight of a balanced partition in terms of the eigenvalues of the connectivity matrix, the
first use of eigenvalues to *bound* partition quality (for unit edge weights).

**Algebraic connectivity as a proxy (Fiedler 1973).** Fiedler's $a(G)=\lambda_2$ gives
$a(G)=0\iff$ disconnected and $a(G)\le v(G)$, so a small $\lambda_2$ is *necessary* for a sparse
vertex cut. Fiedler-vector bisection partitions the graph by the sign or a threshold of the Fiedler
vector.

**Continuous Cheeger inequality (Cheeger 1970, Buser 1982).** $\lambda_1(M)\ge h(M)^2/4$ is the
lower-bound direction for manifolds and the Laplace–Beltrami operator, with Buser proving the
complementary upper bound in the smooth setting.

## Evaluation settings

The natural yardsticks are graphs where the sparsest cut or a useful spectral proxy is known or
computable:

- **Structured families with closed-form spectra**: paths $P_n$ and cycles $C_n$ (where the relevant
  second eigenvalue is $\Theta(1/n^2)$ while $\phi_G=\Theta(1/n)$, so the bottleneck scale is
  proportional to the square root of the spectral scale), complete graphs $K_n$, complete bipartite $K_{p,q}$,
  hypercubes (where $\theta\ge 1$ is exactly tight), and grids.
- **Two-cluster planted instances**: two dense blobs joined by a few edges, where the intended cut
  is obvious and one checks whether the spectral cut recovers it.
- **Meshes / finite-element graphs**: where balanced separators are needed and the partition quality
  is measured by boundary size at fixed balance.

The metrics are the conductance $\phi(S)$ (or isoperimetric ratio $\theta(S)$) of a returned cut,
the spectral gap $\lambda_2$ (or normalized $\nu_2$), and the ratio between the achieved cut quality
and the available spectral certificate.

## Code framework

The program reads a weighted undirected graph from stdin as `n m`, followed by `m` lines `u v w`
for 0-indexed endpoints and a positive edge weight. It prints stdout in the final-answer format:
first a line with the two computed numeric quantities and `lower=... upper=...`, then the size of
the returned vertex set, then the vertices of that set in ascending order.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    struct Edge {
        int u, v;
        double w;
    };

    vector<Edge> edges;
    edges.reserve(m);
    for (int i = 0; i < m; ++i) {
        int u, v;
        double w;
        cin >> u >> v >> w;
        edges.push_back({u, v, w});
    }

    // TODO: implement algorithm.
    double nu2 = 0.0;
    double best_phi = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    vector<int> cut;

    cout.setf(std::ios::fixed);
    cout << setprecision(6);
    cout << nu2 << " " << best_phi << "  lower=" << lower << " upper=" << upper << "\n";
    cout << cut.size() << "\n";
    for (size_t i = 0; i < cut.size(); ++i)
        cout << cut[i] << (i + 1 < cut.size() ? ' ' : '\n');
    if (cut.empty()) cout << "\n";

    return 0;
}
```
