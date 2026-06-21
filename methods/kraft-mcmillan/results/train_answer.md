We want to encode the symbols of a source $X$ over an alphabet $\{x_1,\dots,x_m\}$ with probabilities $p_i$ as strings over a $D$-ary channel alphabet $\{0,1,\dots,D-1\}$, short strings for frequent symbols and long ones for rare, so that the expected length $L = \sum_i p_i \ell_i$ is small. The difficulty is not the wish for short codewords but the stream: we transmit codewords back-to-back with no separators, and the receiver gets one long $D$-ary string from which the original symbol sequence must be recovered unambiguously. Morse code, the motivating artifact, cheats here by inserting an inter-letter space — a whole extra alphabet symbol spent purely on punctuation, which is exactly the waste we want to remove. So the real question comes first, before any optimization: which length profiles $(\ell_1,\dots,\ell_m)$ are actually realizable by a decodable code? Handing out very short codewords to everyone must create ambiguity, so there is a budget on the $\ell_i$ — but what is it, and does it depend on how strong a decodability requirement we impose? The prior art does not answer this. Shannon–Fano splitting and the asymptotic typical-set counting of the noiseless coding theorem give constructions and rates of order $H_D(X)$, but neither isolates an exact, finite, per-symbol condition on the lengths; the Sardinas–Patterson test decides whether a *given* code is uniquely decodable but says nothing about which length profiles admit one. What is missing is a tight, exact condition $\Phi(\ell_1,\dots,\ell_m)$ that is necessary and sufficient for a decodable code to exist, and a settled answer to whether demanding instant decoding shrinks the realizable set.

The method is the Kraft–McMillan inequality: the achievable lengths of *any* decodable $D$-ary code are exactly those obeying the single budget
$$\sum_i D^{-\ell_i} \le 1,$$
and this one inequality both characterizes feasibility and forces $L \ge H_D(X)$. To see why, fix the decodability conditions, which nest strictly: prefix-free (no codeword is a prefix of another, so each codeword's end is recognizable instantly with zero lookahead) $\subset$ uniquely decodable (the concatenation map on sequences is injective, but decoding may require reading the whole string) $\subset$ nonsingular. That the middle containment is strict matters: over $\{0,1\}$ the code $\{10,00,11,110\}$ is uniquely decodable yet not prefix-free, so demanding prefix codes is a priori a real restriction, and one should worry it costs achievable lengths.

The necessity of the budget for prefix codes — Kraft — is a leaf count on the $D$-ary tree. Place the code on the tree where each node has $D$ children and a codeword is the root-to-node path; prefix-free means the codewords form an antichain, no one an ancestor of another. Let $\ell_{\max}=\max_i \ell_i$. A codeword at depth $\ell_i$ owns exactly $D^{\ell_{\max}-\ell_i}$ descendants at depth $\ell_{\max}$ (each of the $\ell_{\max}-\ell_i$ extra levels multiplies by $D$), and because no codeword is an ancestor of another these descendant sets are disjoint. There are only $D^{\ell_{\max}}$ nodes at depth $\ell_{\max}$ in all, so $\sum_i D^{\ell_{\max}-\ell_i}\le D^{\ell_{\max}}$; dividing by $D^{\ell_{\max}}$ — which cancels the arbitrary truncation level, as it must — gives $\sum_i D^{-\ell_i}\le 1$. The converse is sufficiency: given positive lengths obeying the budget, sort them $\ell_1\le\cdots\le\ell_m$ and number the depth-$\ell_{\max}$ leaves lexicographically. Each depth-$\ell_j$ node is an aligned block of $D^{\ell_{\max}-\ell_j}$ bottom leaves; before placing $\ell_j$ the earlier codewords have consumed $U_{j-1}=\sum_{i<j}D^{\ell_{\max}-\ell_i}$ leaves, and since every earlier $\ell_i\le\ell_j$ each earlier block size is a multiple of the current one, so $U_{j-1}$ is aligned to a depth-$\ell_j$ node. The budget gives $U_{j-1}+D^{\ell_{\max}-\ell_j}\le\sum_i D^{\ell_{\max}-\ell_i}\le D^{\ell_{\max}}$, so the aligned block starting at $U_{j-1}$ exists and avoids all earlier blocks; choosing its node as the next codeword and iterating builds a prefix code. The sorting is the load-bearing design choice: it is what makes each cumulative offset a multiple of the current block size, so the next codeword's digits are well defined. Equivalently — and this is the picture that makes the sub-probability flavor literal — map each codeword to the half-open $D$-adic interval of width $D^{-\ell_i}$ in $[0,1)$ whose expansions start with it; prefix-free is exactly disjointness of these intervals, their widths sum to $\le 1$, and laying the sorted intervals end to end realizes the code.

The worry — that the larger uniquely-decodable class might escape this budget and let us use shorter codewords — is what McMillan's argument kills, and the surprise is that it costs nothing. A uniquely decodable code has no tree to count on; the only handle is that the extension map is injective. The trick is to raise $S=\sum_i D^{-\ell_i}$ to the $k$-th power, because a single codeword exposes nothing but $S^k$ is a sum over all source $k$-tuples, which is exactly what injectivity controls:
$$S^k=\Big(\sum_x D^{-\ell(x)}\Big)^k=\sum_{x_1,\dots,x_k}D^{-(\ell(x_1)+\cdots+\ell(x_k))}=\sum_{m'=k\,\ell_{\min}}^{k\,\ell_{\max}}a(m')\,D^{-m'},$$
where $a(m')$ counts the source $k$-tuples whose encoding has total length exactly $m'$. (A zero-length codeword is excluded outright, since inserting that symbol into any sequence would leave the channel string unchanged and break injectivity, so $\ell_{\min}\ge 1$.) Now unique decodability pays: the $a(m')$ encodings are distinct $D$-ary strings of length $m'$, and there are only $D^{m'}$ such strings, so $a(m')\le D^{m'}$. Substituting, the $D^{m'}$ and $D^{-m'}$ cancel term by term and we are merely counting the admissible total lengths:
$$S^k\le\sum_{m'=k\,\ell_{\min}}^{k\,\ell_{\max}}1=k(\ell_{\max}-\ell_{\min})+1\le k\,\ell_{\max}.$$
The left side is exponential in $k$ if $S>1$; the right is linear; an exponential cannot stay under a line forever, so $S>1$ is impossible. Cleanly, $S\le(k\,\ell_{\max})^{1/k}\to 1$ as $k\to\infty$ while $S$ is independent of $k$, forcing $\sum_i D^{-\ell_i}\le 1$ for every uniquely decodable code. The same budget that the tree argument gave for instantaneous codes therefore binds the entire larger class, so the achievable length sets coincide: prefix codes lose nothing, and we may restrict to comma-free, lookahead-free codes with a clear conscience.

With the feasible set pinned down exactly, the coding problem becomes a constrained minimization of $L=\sum_i p_i\ell_i$ subject to $\sum_i D^{-\ell_i}\le 1$, and the answer is the entropy. A Lagrangian relaxation ignoring integrality wants $D^{-\ell_i}=p_i$, i.e. $\ell_i^*=\log_D(1/p_i)$, giving $L^*=H_D(X)$ — the hunch that the lengths behave like a distribution, made exact. The clean lower bound holds for every feasible integer profile: treat $D^{-\ell_i}$ as a sub-probability, normalize $r_i=D^{-\ell_i}/c$ with $c=\sum_j D^{-\ell_j}$, and the gap splits into two nonnegative pieces,
$$L-H_D(X)=\sum_i p_i\log_D\frac{p_i}{D^{-\ell_i}}=D(p\,\|\,r)+\log_D\frac{1}{c}\ge 0,$$
since relative entropy $D(p\,\|\,r)\ge 0$ by Gibbs' inequality and the budget $c\le 1$ makes $\log_D(1/c)\ge 0$. The budget is doing essential work here — without $c\le 1$ that second term could go negative and the bound would fail — and equality, after restricting to the positive-probability alphabet, needs both $p_i=r_i$ and $c=1$, forcing $D^{-\ell_i}=p_i$, exactly the dyadic case. To approach the floor with integer lengths, round up: the Shannon lengths $\ell_i=\lceil\log_D(1/p_i)\rceil$ are positive on a non-degenerate support, are feasible because $D^{-\ell_i}\le p_i$ sums to $\le\sum_i p_i=1$, and from $\log_D(1/p_i)\le\ell_i<\log_D(1/p_i)+1$ give $H_D(X)\le L<H_D(X)+1$. The stray $+1$ is pure rounding overhead, so spread it: encode $n$ i.i.d. symbols as one super-symbol and divide by $n$ to get $H_D(X)\le L_n<H_D(X)+1/n\to H_D(X)$ (and convergence to the entropy rate for a stationary source). The entropy is thus both the floor no decodable code can beat and the limit blocking attains.

```python
from fractions import Fraction
from math import log, ceil

def kraft_sum(lengths, D):
    # S = sum_i D^{-l_i}: exact rational arithmetic keeps the budget exact.
    if D < 2 or any(not isinstance(l, int) or l <= 0 for l in lengths):
        raise ValueError("use D >= 2 and positive integer lengths")
    return sum(Fraction(1, D ** l) for l in lengths)

def kraft_feasible(lengths, D):
    # Necessary AND sufficient for a decodable D-ary code (prefix == UD here).
    return kraft_sum(lengths, D) <= 1

def base_D_digits(index, length, D):
    digits = [0] * length
    for pos in range(length - 1, -1, -1):
        digits[pos] = index % D
        index //= D
    return tuple(digits)

def build_prefix_code(lengths, D):
    # Constructive converse: sorted lengths give aligned depth-l_max leaf blocks.
    if not kraft_feasible(lengths, D):
        raise ValueError("lengths violate the Kraft budget")
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    code = [None] * len(lengths)
    l_max = max(lengths, default=0)
    next_leaf = 0
    for i in order:
        block = D ** (l_max - lengths[i])
        assert next_leaf % block == 0      # sorted lengths make the block aligned
        code[i] = base_D_digits(next_leaf // block, lengths[i], D)
        next_leaf += block
    return code

def shannon_lengths(p, D):                 # l_i = ceil(log_D(1/p_i))
    if any(pi <= 0 or pi >= 1 for pi in p):
        raise ValueError("use a non-degenerate positive-probability support")
    return [ceil(log(1.0 / pi, D)) for pi in p]

def entropy_D(p, D):
    return sum(pi * log(1.0 / pi, D) for pi in p if pi > 0)

def expected_length(lengths, p):
    return sum(pi * l for pi, l in zip(p, lengths))

# D = 2, dyadic p = (1/2, 1/4, 1/8, 1/8):
#   shannon_lengths -> (1, 2, 3, 3); kraft_sum = 1 (budget exactly filled);
#   build_prefix_code -> [(0,), (1, 0), (1, 1, 0), (1, 1, 1)];
#   expected_length = 1.75 = H_2(X)  (equality, since p is D-adic).
```
