The thing that wouldn't let go of me is that parity learning has two completely different algorithms living at opposite ends of one resource axis. A learner wants to recover an unknown $x \in \{0,1\}^n$, chosen uniformly at random, from a stream of samples $(a_t, b_t)$ with $a_t$ uniform over $\{0,1\}^n$ and $b_t = a_t \cdot x \pmod 2$, where a sample once seen is gone unless it is deliberately stored. One way to solve it: gather $n$ independent linear equations $a_t \cdot x = b_t$ and run Gaussian elimination — only $O(n)$ samples, but I am holding an $n \times n$ system over $\mathbb{F}_2$, so $\Theta(n^2)$ bits of memory. The other way: hold a single candidate $x'$ in $n$ bits, draw samples until one refutes it, advance — tiny memory, but in the worst case an exponential number of samples before the true $x$ is the lone survivor. So one can buy down samples with memory, or buy down memory with samples, and the question is whether that trade is *forced*. Could a clever algorithm in the middle — linear memory, the size of one sample — still learn $x$ from polynomially many samples? The embarrassing state of affairs is that there was no non-trivial sample lower bound for *any* learning problem even with memory capped at the length of a single sample.

The obvious tools do not reach this. The statistical-query bound says parity is hard because for any single bounded statistic $\chi$ of a labeled example, the value $P_\chi(f) = \Pr[\chi = 1]$ has variance only about $2^{-n}$ over a uniformly random target parity (at bottom the pairwise orthogonality of the $2^n$ characters), so any one bounded look learns almost nothing; Steinhardt, Valiant and Wager turned that into a communication statement, that a learner compressing each example to $b$ bits before the next arrives can be simulated by $2^b m$ statistical queries and therefore needs exponentially many samples. But a communication bound constrains bits *per sample* — each example squeezed in isolation. A genuine memory-bounded learner reads every sample in full; the only thing it cannot do is carry a large *accumulated* state across samples, and that state may depend adaptively on the entire history. Memory is cumulative and adaptive, not per-sample, and that is precisely the gap SVW could not close (they handled only learners whose memory states are subspaces). The classical time-space tradeoff literature for *computing* functions does not help either: those results only reach sub-quadratic time, far short of the exponential sample regime, and worse, they store the input for free — re-readable, uncharged — which is exactly the wrong assumption for learning, where fresh randomness is cheap but memory of past randomness is expensive. That asymmetry is the entire phenomenon, and the computation literature throws it away.

So I model the learner directly, in the most general and least-assuming way, as a *branching program*: a layered directed multigraph with $m+1$ layers, width $\le d$, one start vertex, every non-leaf vertex carrying one outgoing edge per possible sample $(a,b) \in \{0,1\}^n \times \{0,1\}$, and leaves labeled by an output guess. A vertex is a memory state, so $\log_2 d$ is the memory in bits; a layer is a time step, so $m$ is the number of samples. The learner gets unbounded computation; only memory and samples are charged, so a lower bound here binds every algorithm, uniform or not. I propose, and prove, a time-space lower bound in this model: any branching program of width $2^{cn^2}$ and length $2^{\alpha n}$ that learns parity — with $c < \tfrac{1}{20}$ and a suitable $\alpha > 0$ — succeeds only with exponentially small probability. Equivalently, learning parity with fewer than $\sim n^2/25$ memory bits requires an exponential number of samples. The whole argument turns on letting the linear algebra of parity lead, and it has two parts that have to be made to cancel against each other.

The first part handles *affine* branching programs, where every vertex $v$ carries an affine subspace label $w(v) \in A(n)$ with a soundness invariant: the start is labeled $\{0,1\}^n$, and along an edge $e=(u,v)$ labeled $(a,b)$ the restricted subspace $w(e) := w(u) \cap \{x' : a\cdot x' = b\}$ satisfies $w(e) \subseteq w(v)$. The reason this label is natural is that after seeing $(a_1,b_1),\dots,(a_t,b_t)$ the set of consistent strings $\{x' : a_i \cdot x' = b_i\}$ is exactly an affine subspace, the honest knowledge state, whose dimension starts at $n$ and drops by at most one per step. Soundness forces $x \in w(v)$ along the honest path always — by induction, $x \in w(u)$ and the honest edge $(a, a\cdot x)$ give $x \in w(u) \cap \{a\cdot x' = a\cdot x\} = w(e) \subseteq w(v)$ — so an affine program never excludes $x$ and has success probability $1$. The only way it can be a *good* learner is to drive the dimension of its label down, which makes "learning $x$" a purely geometric event. To bound how rarely a low-dimensional vertex is reached I track the right progress measure: for a target vertex $v$ with $\dim w(v) = k$, let $s$ be the space of linear tests constant on $w(v)$ (orthogonal to $w(v)$, $\dim s = n-k$), let $S_i$ be the space orthogonal to the label at the $i$-th vertex on the path, and watch $Z_i = \dim(S_i \cap s)$ — the overlap between the constraints the learner has locked in and the constraints defining $v$. Reaching $v$ forces $s \subseteq S_{\text{last}}$, so $Z$ must climb all the way from $0$ to $n-k$; soundness gives $S_i \subseteq \mathrm{span}(S_{i-1} \cup \{a_i\})$, so $Z$ rises by at most one per step; and a rise requires some $a \in S_{i-1}$ with $a \oplus a_i \in s$, which for fixed $a$ has probability $2^{\dim s - n} = 2^{-k}$ since $a_i$ is fresh and uniform. Counting distinct possibilities as cosets of $S_{i-1} \cap s$ — there are $2^{\dim S_{i-1} - Z_{i-1}} \le 2^{n-k-Z_{i-1}}$ of them — the per-step rise probability is at most $2^{n-2k-Z_{i-1}}$, which not only is exponentially small but gets *smaller* as $Z$ climbs. Multiplying the $n-k$ rises with their telescoping exponents and union-bounding over the fewer than $m^{n-k}$ choices of rise-steps gives a reach probability of at most $m^{n-k} \cdot 2^{\sum_{j=0}^{n-k-1}(n-2k-j)}$, which with $k$ a constant fraction above $n/2$ (so $n-2k < 0$) is $2^{-\Theta(n^2)}$ unless the sample-count exponent $\log_2 m$ is large enough to eat it.

The second part is the hard one, the part the prior framework could not get past: a *general* branching program vertex is not an affine subspace, it is an arbitrary memory state, and the conditional distribution of $x$ given that the path reached it can be a messy distribution. So I reduce the general case to the affine one. The conditional distribution of $x$ at a vertex is, inductively, a convex combination — a *mixture* — of uniform distributions over affine subspaces, because each incoming edge restricts a near-uniform-on-a-subspace distribution to a sub-subspace. The key fact, exactly the inner product acting as a strong extractor and most transparent in Fourier language, is that such a mixture is essentially uniform unless some linear test is pinned to a constant. For an affine $w$ the Fourier coefficient $\widehat{U_w}(a)$ equals $2^{-n}$ if $a\cdot x \equiv 0$ on $w$, $-2^{-n}$ if $\equiv 1$, and $0$ otherwise, so for a random $W$ the mixture's coefficient at $a \neq \vec 0$ is $2^{-n}$ times the difference of two "$a\cdot x$ is pinned" probabilities — the mixture differs from uniform *only* through pinned linear tests (Lemma 1, giving $\ell_1$ distance $< 2^{-(r-n/2)}$ when no test is pinned above $2^{-r}$, the $n/2$ being the benign Cauchy–Schwarz slack). Recursing — if some test $a\cdot x = b$ is pinned, condition $W$ on lying inside that $(n-1)$-dimensional subspace $u$ and recurse with parameter $r - \tfrac12$ — yields a single affine subspace $s$ that captures a non-negligible chunk of the mixture and on which the conditional mixture is near-uniform (Lemma 2, capture probability $\ge 2^{-\sum_{i=0}^{n-\dim s-1}(r-i/2)}$, the per-level $1/2$ decrement chosen so the arithmetic closes). Carving repeatedly until the leftover mass is below $2^{-2n}$ groups almost every affine subspace into a representative containing it, with few groups of each dimension — at most $4n \cdot 2^{\sum_{i=0}^{n-k-1}(r-i/2)}$ of dimension $\ge k$ (Lemma 3, the only export of this detour, since dimension exactly the threshold is the worst case). With grouping in hand I turn any general program $B$ into an *accurate affine* program $P$ layer by layer: at each $B$-vertex I split it into one copy per representative subspace in its mixture, label the copy by that subspace, and route incoming edges to the copy whose representative contains the honestly restricted subspace. The decisive bookkeeping is that the accuracy error must be propagated *additively* across the $2^{\Theta(n)}$ layers — a multiplicative $(1+\delta)$ per layer would explode — which I secure with a surrogate form of the inductive hypothesis: a probabilistic transformation $T$ (draw fresh $a$, set $b = a\cdot z$, follow the edge) maps the true pair $(V_{j-1}, x)$ to $(V_j, x)$ and the surrogate pair $(U_{j-1}, y_{j-1})$ to $(U_j, y'_j)$, and since $T$ cannot increase $\ell_1$ distance, the carried error grows only by the per-layer grouping cost $2 \cdot 2^{-(r-n/2)}$, for a total $\epsilon = 4m \cdot 2^{-(r-n/2)}$ (Lemma 5). Accuracy then transfers $B$'s success into $P$ reaching a low-dimensional vertex: if $B$ outputs a dimension-$\le k'$ subspace with success $\beta$, then $P$ reaches a dimension-$<k$ vertex with probability at least $\beta - \epsilon - 2^{-(k-k')}$.

What makes it all close is the choice of constants so the two halves cancel. With $B$ of length $m = 2^{\alpha n}$, width $d = 2^{cn^2}$, output dimension $\le k' = \tfrac{3n}{5}$, I set the carving parameter $r = (\tfrac12 + 2\alpha)n$ and the threshold $k = \tfrac{4n}{5}$, so $n-k = \tfrac n5$ and $n-2k = -\tfrac{3n}{5}$. Then $\epsilon = 4\cdot 2^{-\alpha n}$ and the slack $2^{-(k-k')} = 2^{-n/5}$, so $P$ reaches a dimension-$<k$ vertex with probability $\ge \beta - 5\cdot 2^{-\alpha n}$. Multiplying the grouping vertex count $4n\cdot 2^{\sum(r-i/2)}\cdot dm$ by the affine reach bound $m^{n-k}\cdot 2^{\sum(n-2k-i)}$ and simplifying — the cancellation $\tfrac12 - \tfrac35 = -\tfrac1{10}$ in the linear terms and $-\tfrac i2 - i = -\tfrac32 i$ in the quadratic terms — collapses the whole reach probability to $4nm \cdot 2^{n^2(c + \frac35\alpha - \frac1{20} + \frac{3}{20n})}$. So whenever $c + \tfrac35\alpha - \tfrac1{20} < 0$, that is $\alpha < \tfrac53(\tfrac1{20} - c)$, this is $2^{-\Omega(n^2)}$, which forces $\beta \le O(2^{-\alpha n})$. The two corners I started from are exactly the two things this forbids improving: Gaussian elimination pays the quadratic memory, candidate enumeration pays the exponential samples, and nothing lives below quadratic memory and below exponential samples. It also gives a clean cryptographic reading — encrypting a bit as $M \oplus (a\cdot x)$ for fresh public uniform $a$ is a bounded-storage scheme secure against a memory-$\tfrac{n^2}{25}$ adversary watching exponentially many uses, and unlike the older bounded-storage template its encryption time is not forced to be linear in the adversary's memory.

Here is the final result, with the full proof.

```
Definitions. For a, x ∈ {0,1}^n, a·x = Σ_i a_i x_i mod 2. A(n) = affine subspaces of {0,1}^n;
U_w = uniform distribution on w ∈ A(n); U_n = U_{{0,1}^n}. |P−Q|_1 = total ℓ1 distance.
For P:{0,1}^n→ℝ, P̂(a) = E_x[P(x)(−1)^{a·x}].

Branching program for parity learning. A layered directed multigraph with m+1 layers of width
≤ d; one start vertex; each non-leaf vertex has one out-edge per (a,b)∈{0,1}^n×{0,1}; each leaf
labeled by an affine subspace w(v)∈A(n) (output guess "x∈w(v)"). A stream traces a
computation-path start → leaf. Width ↔ memory log2 d; length ↔ samples m.

Affine branching program. Every vertex v carries w(v)∈A(n), with: (start) w(start)={0,1}^n;
(soundness) for edge e=(u,v) labeled (a,b), w(e):= w(u)∩{x':a·x'=b} ⊆ w(v). Soundness ⇒
x∈w(v) along the honest path always; success prob = 1.

ε-accurate. For the layer-t reached vertex V_t and y_t∼U_{w(V_t)}: |P_{V_t,x} − P_{V_t,y_t}|_1 ≤ ε.

────────────────────────────────────────────────────────────────────────────
THEOREM (formal). For any c < 1/20 there is α>0 such that: let x∼U_n, m ≤ 2^{αn}, and let B
be a branching program of length m and width ≤ 2^{cn^2} for parity learning whose output is
always an affine subspace of dimension ≤ 3n/5. Then

        Pr[ x ∈ B's output ] ≤ O(2^{−αn}).

COROLLARY (headline). For any c < 1/20 there is α>0 such that any parity-learning algorithm
using ≤ cn^2 memory bits and ≤ 2^{αn} samples outputs x̃ with Pr[x̃ = x] ≤ O(2^{−αn}).
Equivalently, learning parity with fewer than ~ n^2/25 memory bits requires an exponential
number of samples.

────────────────────────────────────────────────────────────────────────────
PROOF.

Step 1 — distributions over affine subspaces (the Fourier / extractor core).
Let W∈A(n) be a random affine subspace; E_W[U_W] is a mixture of uniform-on-subspace
distributions.

Lemma 1 (mixing). Let r ≥ n/2. If for all a≠0, b∈{0,1}, Pr_W[∀x∈W: a·x=b] ≤ 2^{−r}, then
|E_W[U_W] − U_n|_1 < 2^{−(r−n/2)}.
  Proof. For affine w, Û_w(a)= 2^{−n} if a·x≡0 on w, −2^{−n} if ≡1, else 0. Hence
  Ê_W[U_W](a)= 2^{−n}(Pr_W[a·x≡0 on W] − Pr_W[a·x≡1 on W]), with the a=0 coefficient
  = 2^{−n} = Û_n(0). For a≠0, Û_n(a)=0 and |Ê_W[U_W](a)| ≤ 2^{−n}·2^{−r}, so
  Σ_{a≠0}(Ê_W[U_W](a) − Û_n(a))^2 < 2^n(2^{−n}2^{−r})^2 = 2^{−n−2r}. By Cauchy–Schwarz and
  Parseval, (E_x|P−U_n|)^2 ≤ E_x(P−U_n)^2 = Σ_a(P̂−Û_n)^2 < 2^{−n−2r} (with P=E_W[U_W]). Thus
  |P−U_n|_1 = 2^n E_x|P−U_n| < 2^n·2^{−(n+2r)/2} = 2^{−(r−n/2)}. □

Lemma 2 (capture). Let r ≥ n/2. There exists s∈A(n) with (1) Pr_W[W⊆s] ≥
2^{−Σ_{i=0}^{n−dim s−1}(r−i/2)} and (2) |E_{W|W⊆s}[U_W] − U_s|_1 < 2^{−(r−n/2)}.
  Proof. Induction on n. Base n=0: s={0}. Step: if Lemma 1's hypothesis holds, take s={0,1}^n.
  Else ∃a≠0,b with Pr_W[∀x∈W:a·x=b] > 2^{−r}; set u={x:a·x=b} (dim u=n−1), so Pr_W[W⊆u] >
  2^{−r}. Apply the hypothesis to W'= W|(W⊆u) over u≅{0,1}^{n−1} with (n−1, r−1/2), getting
  s⊆u. (1): Pr_W[W⊆s] = Pr_W[W⊆u] Pr_{W'}[W'⊆s] > 2^{−r}·2^{−Σ_{i=0}^{n−1−dim s−1}(r−1/2−i/2)}
  = 2^{−Σ_{i=0}^{n−dim s−1}(r−i/2)}. (2): E_{W|W⊆s}[U_W] = E_{W'|W'⊆s}[U_{W'}], inherited. □

Lemma 3 (grouping — the only export). Let r ≥ n/2. There is a partial σ:A(n)→A(n) with
(1) Pr_W[W∉dom σ] ≤ 2^{−2n}; (2) w⊆σ(w); (3) for all s∈image σ, |E_{W|σ(W)=s}[U_W] − U_s|_1 <
2^{−(r−n/2)}; (4) for every k, #{s∈image σ: dim s ≥ k} ≤ 4n·2^{Σ_{i=0}^{n−k−1}(r−i/2)}.
  Proof. Repeatedly apply Lemma 2: W_0=W→s_0, set σ(w)=s_0 for w⊆s_0; W_1=W|(W⊄s_0)→s_1; …;
  stop when Pr_W[W∉dom σ] ≤ 2^{−2n}. The s_i are distinct. (1) by the stopping rule; (2),(3)
  by Lemma 2. (4): each carve producing dim s_i ≥ k captures ≥ 2^{−Σ_{i=0}^{n−k−1}(r−i/2)} of
  the remaining mass (more dimension ⇒ fewer sum terms ⇒ larger fraction), so after
  ≤ 4n·2^{Σ_{i=0}^{n−k−1}(r−i/2)} such carves the residual is ≤ 2^{−2n}. □

Step 2 — affine lower bound (dimension-of-intersection).

Lemma 4. Let k<n, P a length-m affine program with dim w(u) ≥ k for all u, and v a vertex with
dim w(v)=k. Then Pr[path reaches v] ≤ m^{n−k}·2^{Σ_{j=0}^{n−k−1}(n−2k−j)}.
  Proof. Let s={a:∃b,∀x'∈w(v),a·x'=b} (orthogonal to w(v), dim s=n−k). Let S_i be orthogonal
  to w(V_i); soundness gives S_i ⊆ span(S_{i−1}∪{a_i}). Set Z_i=dim(S_i∩s): Z_0=0, Z_i ≤
  Z_{i−1}+1, and reaching v needs some Z_i=n−k. A rise (Z_i>Z_{i−1}) requires ∃a∈S_{i−1} with
  a⊕a_i∈s; for fixed a, Pr[a⊕a_i∈s]=2^{dim s−n}=2^{−k}. Distinct possibilities form cosets of
  S_{i−1}∩s, numbering 2^{dim S_{i−1}−Z_{i−1}} ≤ 2^{n−k−Z_{i−1}}, so conditioned on
  x,a_1,…,a_{i−1} (fixing Z_{i−1}), Pr[rise at i] ≤ 2^{n−k−Z_{i−1}}·2^{−k}=2^{n−2k−Z_{i−1}}.
  For a fixed tuple i_1<…<i_{n−k} of rise-steps, Pr ≤ ∏_{j=1}^{n−k}2^{n−2k−(j−1)} =
  2^{Σ_{j=0}^{n−k−1}(n−2k−j)}. Union bound over < m^{n−k} tuples gives the claim. □

Step 3 — reduction general → accurate affine.

Lemma 5. Let k'<n. Let B be a length-m, width-d parity-learning program, all leaves in the
last layer, output dimension ≤ k', success β. Let n/2 ≤ r ≤ n, ε=4m·2^{−(r−n/2)}. Then there
is an ε-accurate length-m affine program P with: (1) for every k, # dimension-k vertices ≤
4n·2^{Σ_{i=0}^{n−k−1}(r−i/2)}·dm; (2) for k'<k<n, Pr[dim(output)<k] ≥ β−ε−2^{−(k−k')}.
  Proof sketch (full induction). Build P layer by layer with inductive hypothesis: ∃U_j over
  layer j with y_j∼U_{w(U_j)} and |P_{V_j,x}−P_{U_j,y_j}|_1 ≤ ε_j/2, ε_j=4j·2^{−(r−n/2)} (the
  surrogate form keeps errors additive across m=2^{Θ(n)} layers). Base j=0: label start
  {0,1}^n, U_0=V_0, distance 0. Step: from U_{j−1}, y_{j−1}, draw a∼U, b=a·y_{j−1}, follow
  edge to V; W=w(U_{j−1})∩{a·x'=b}. For each B-vertex v, apply Lemma 3 to W_v=W|(V=v) to get
  σ_v; split v into copies (v,s), s∈image σ_v (label s, or {0,1}^n for the * leftover),
  routing incoming edge e=(u,v) to (v,σ_v(w(e))) — soundness holds since w(e)⊆σ_v(w(e)). Set
  U_j=(V,σ_V(W)). With y'_j∼U_W: (3a) |P_{U_j,y'_j}−P_{U_j,y_j}|_1 ≤ 2·2^{−(r−n/2)} by Lemma
  3(3) plus the 2^{−2n} leftover; (3b) |P_{V_j,x}−P_{U_j,y'_j}|_1 ≤ 2(j−1)·2^{−(r−n/2)} by the
  shared transformation T (draw a, b=a·z, follow edge) with T(V_{j−1},x)∼(V_j,x),
  T(U_{j−1},y_{j−1})∼(U_j,y'_j) — T cannot increase ℓ1 — and the inductive hypothesis.
  Triangle: |P_{V_j,x}−P_{U_j,y_j}|_1 ≤ 2j·2^{−(r−n/2)}=ε_j/2. Accuracy of P follows from the
  hypothesis at each layer. Property (1) is Lemma 3(4) times dm. Property (2): with V_m=(V,S),
  Pr[x∈w(V)]=β and ε-accuracy give Pr[y_m∈w(V)] ≥ β−ε; since dim w(V) ≤ k', if dim w(V_m) ≥ k
  then Pr[y_m∈w(V)|·] ≤ 2^{k'−k}, so β−ε ≤ Pr[dim w(V_m)<k]+2^{k'−k}. □

Step 4 — assemble the constants.

Let B have length m=2^{αn}, width d=2^{cn^2}, output dimension ≤ k'=3n/5, success β. Set
r=(1/2+2α)n, k=4n/5 (so n−k=n/5, n−2k=−3n/5). Then ε=4·2^{−αn} and 2^{−(k−k')}=2^{−n/5}, so
by Lemma 5(2) Pr[dim(output)<k] ≥ β−5·2^{−αn}. Make all dimension-k vertices leaves; every
vertex has dim ≥ k. By Lemma 5(1) and Lemma 4,

  Pr[reach a dim-k vertex] ≤ (4n·2^{Σ_{i=0}^{n−k−1}(r−i/2)}·dm)·(m^{n−k}·2^{Σ_{i=0}^{n−k−1}(n−2k−i)}).

Substituting and combining (the cancellation 1/2−3/5=−1/10 in the linear terms; −i/2−i=−3i/2
in the quadratic terms):

  = 4nm·2^{cn^2}·2^{(n−k)(3αn−n/10)}·2^{−(3/4)(n−k)(n−k−1)}
  = 4nm·2^{ n^2 ( c + (3/5)α − 1/20 + 3/(20n) ) }.

Hence if α < (5/3)(1/20 − c), this is 2^{−Ω(n^2)}, so β−5·2^{−αn} ≤ 2^{−Ω(n^2)}, i.e.
β ≤ O(2^{−αn}). ∎

The corollary is the special case k'=0 (the learner outputs a single guess x̃, a dimension-0
subspace), and the constant is rounded: any c<1/20 works, so ~ n^2/25 memory bits is a clean
sufficient threshold below which exponentially many samples are needed.

────────────────────────────────────────────────────────────────────────────
Remarks.

• Why the conjectured constant n^2/4 is too strong. A Gaussian-elimination variant keeps, at
  step k, a k×n matrix whose first k columns form the identity, costing n^2/4 + O(n) memory;
  restricting to samples supported on the first 3n/4 coordinates lets one solve a size-3n/4
  problem with 2^{n/4} samples and (9/64)n^2 + o(n^2) ≈ 0.14 n^2 memory, beating n^2/4. So the
  right statement is the qualitative trade with a smaller constant, here c < 1/20.

• Cryptographic corollary. Sharing x∈{0,1}^n, encrypt a bit M as M⊕(a·x) for fresh public
  uniform a (cost n/bit). An adversary with < n^2/25 memory watching up to 2^{Ω(n)} uses cannot
  recover x beyond exponentially small probability; since a·x is a strong extractor, it cannot
  predict the next pad bit beyond exponentially small advantage. This is a bounded-storage
  scheme whose encryption time is not forced to be linear in the adversary's memory.
```
