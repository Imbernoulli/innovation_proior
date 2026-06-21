We are handed a finite ground set $V$ of size $m$, a value function $f: 2^V \to \mathbb{R}_{\ge 0}$ on its subsets, and a hard budget: choose a set $S$ with at most $k$ elements that makes $f(S)$ as large as possible. Maximum coverage is the picture to keep in mind — $m$ subsets of a universe $U$, pick $k$ of them to cover the most of $U$ — and many other selection problems share the shape: place $k$ sensors, summarize with $k$ documents, site $k$ facilities. Even maximum coverage is NP-hard, so computing the exact best $S$ in polynomial time is hopeless, and exhaustive enumeration over the $\binom{m}{k}$ candidate sets is exponential and useless past toy sizes. What we actually want is something cheap that comes with a *worst-case multiplicative guarantee*: a constant $\alpha$ such that the returned set always satisfies $f(S) \ge \alpha\, f(\mathrm{OPT})$ on every instance — the property that separates a principled algorithm from a heuristic that merely tends to work. The existing options do not deliver this. Set-cover greedy (Johnson, Lovász, Chvátal) is well understood but solves the *covering* cousin — cover all of $U$ at minimum cost — and gives only a logarithmic $H_n \approx \ln n$ ratio, not a constant fraction under a hard budget $k$. Local search, random restarts, and swap heuristics return *a* set, sometimes a good one, but without a specified exchange rule and a proof they carry no worst-case ratio at all; extra compute just enlarges the search. And the obvious instinct — grow $S$ greedily, one element at a time taking whatever helps most right now — has no guarantee on a generic combinatorial objective, since one can usually cook up an instance where the locally best early picks paint the search into a corner and it finishes arbitrarily far from OPT. So the real question is what structure in $f$ could stop that failure.

The structure is exactly that $f$ is non-negative, monotone, and submodular, and I propose to lean on it with the plainest possible rule: the submodular greedy algorithm. Start from $S \leftarrow \emptyset$ and repeat $k$ times the single step
$$S \leftarrow S + \arg\max_{e \notin S}\; \big[\,f(S + e) - f(S)\,\big],$$
adding each round the element of maximum marginal gain $f(e \mid S) = f(S+e) - f(S)$. That argmax-marginal move is the entire method; what makes it more than a heuristic is that the two structural properties of $f$ turn it into a provable $(1 - 1/e)$ approximation. Monotonicity means $A \subseteq B \Rightarrow f(A) \le f(B)$, so enlarging $S$ never hurts and the only thing stopping us is the budget; submodularity is the diminishing-returns inequality $f(A + e) - f(A) \ge f(B + e) - f(B)$ for $A \subseteq B$, $e \notin B$, so a marginal gain only shrinks as the conditioning set grows. The derivation that converts these into a guarantee is the load-bearing part. Let $S_i$ be the greedy set after $i$ additions, $S_0 = \emptyset$, and let $O = \mathrm{OPT}$ with $|O| \le k$ (if $O$ is empty, monotonicity already makes $\emptyset$ optimal and we are done). The greedy choice does at least as well as adding any *single* fixed element $o \in O$, since that is one of the candidates it maximizes over (and if $o$ is already in $S_i$ its marginal is $0$, which monotonicity keeps harmless). I deliberately do not try to pick the "best" $o$ — I have no handle on which it is — so instead I average. The left side $f(S_{i+1}) - f(S_i)$ does not depend on $o$, hence it is at least the average of the $|O|$ single-element marginals; because every marginal is non-negative and $|O| \le k$, replacing $|O|$ by $k$ in the denominator only weakens the bound and is therefore safe:
$$f(S_{i+1}) - f(S_i) \;\ge\; \frac{1}{k} \sum_{o \in O}\big[\, f(S_i + o) - f(S_i)\,\big].$$
Now submodularity collapses that sum of *individual* marginals into the single *joint* gain $f(S_i \cup O) - f(S_i)$. Order $O$ as $o_1, \dots, o_r$ and telescope the joint gain into a sum where the $j$-th term is the marginal of $o_j$ added on top of $S_i$ *plus the earlier* $o$'s — a set strictly larger than $S_i$ — so diminishing returns makes each such term no larger than the marginal of $o_j$ over $S_i$ alone. Termwise this gives $f(S_i \cup O) - f(S_i) \le \sum_{o \in O}[f(S_i + o) - f(S_i)]$, exactly the direction needed; the unscaled sum dominates the joint gain, so after the $1/k$ averaging the joint gain survives scaled by $1/k$. Monotonicity then closes the chain: $S_i \cup O \supseteq O$, so $f(S_i \cup O) \ge f(O) = f(\mathrm{OPT})$, and combining the three steps,
$$f(S_{i+1}) - f(S_i) \;\ge\; \frac{1}{k}\big[\,f(S_i \cup O) - f(S_i)\,\big] \;\ge\; \frac{1}{k}\big[\,f(\mathrm{OPT}) - f(S_i)\,\big].$$
This is the whole ballgame: every greedy step closes at least a $1/k$ fraction of the *remaining gap* to OPT, a geometric contraction. Writing the gap $a_i = f(\mathrm{OPT}) - f(S_i)$ and rearranging gives $a_{i+1} \le (1 - 1/k)\, a_i$, so after the $k$ rounds the algorithm runs, $a_k \le (1 - 1/k)^k\, a_0$. Since $f \ge 0$ we have $a_0 = f(\mathrm{OPT}) - f(\emptyset) \le f(\mathrm{OPT})$, and using the tangent inequality $\log(1 - x) \le -x$ with $x = 1/k$ gives $k \log(1 - 1/k) \le -1$, i.e. $(1 - 1/k)^k \le e^{-1}$. Hence
$$f(S_{\text{greedy}}) \;\ge\; \big[\,1 - (1 - 1/k)^k\,\big]\, f(\mathrm{OPT}) \;\ge\; \Big(1 - \frac{1}{e}\Big)\, f(\mathrm{OPT}) \;\approx\; 0.632\, f(\mathrm{OPT}),$$
on *every* instance, with no restarts and no luck. Nothing magical produced $e$ — it is just where a $(1 - 1/k)$ contraction compounded $k$ times lands. The two structural facts are each indispensable: submodularity supplies the termwise domination and monotonicity supplies $f(S_i \cup O) \ge f(\mathrm{OPT})$; drop either and the chain breaks, which pins down precisely the function class on which the method lives.

This $1 - 1/e$ is not an artifact of a loose proof — it is genuinely achievable, so the rule cannot be argued up to a better constant. Split a universe into $k$ blocks $B_1, \dots, B_k$ of size $N = k^k$ (chosen so the slice sizes below are integers); the optimal sets are $O_i = B_i$, covering $kN$ together. In each block carve disjoint slices $P_{i,t}$ of size $(N/k)(1 - 1/k)^{t-1}$ for $t = 1, \dots, k$, leaving a remainder $R_i$ of size $N(1 - 1/k)^k$, and add cross-block sets $G_t = \bigcup_i P_{i,t}$. Initially every $O_i$ and $G_1$ has marginal $N$; after $G_1, \dots, G_{t-1}$ are chosen, every $O_i$ has marginal $N(1 - 1/k)^{t-1}$ from its future slices plus $R_i$, and $G_t$ has the identical marginal $k \cdot (N/k)(1 - 1/k)^{t-1}$ from one fresh slice in each of the $k$ blocks. A tie-breaking greedy can therefore be sent to $G_1, \dots, G_k$, covering $kN[1 - (1 - 1/k)^k]$, while OPT takes $O_1, \dots, O_k$ and covers $kN$; the ratio is exactly $1 - (1 - 1/k)^k \to 1 - 1/e$, and arbitrarily small perturbations break the ties without moving the limit. The single pressure point is the constraint itself: I divided by $k$ because under a cardinality budget every element of OPT is a comparable feasible single move, so greedy's one move can be smeared uniformly over all of $O$. The cardinality constraint is the uniform matroid of rank $k$, the favorable special case where that averaging is cleanest. Under a general matroid, feasibility depends on what is already in $S$, the uniform average over $k$ OPT elements is replaced by a matroid-exchange pairing, and the same plain greedy rule guarantees only $1/2$ rather than $1 - 1/e$.

The code mirrors the proof exactly: a value oracle, a coverage instance to make it concrete, a tiny exhaustive checker so the result can be compared against OPT on a toy case, and the selector whose only real decision is to scan the available elements and add the largest marginal until the budget is spent or no positive marginal remains.

```python
from itertools import combinations
from math import e


class SetFunction:
    """Value oracle over a ground set of element indices."""
    def __init__(self, ground_set):
        self.ground_set = list(ground_set)

    def value(self, S):
        raise NotImplementedError

    def marginal(self, e, S):
        return self.value(S | {e}) - self.value(S)


class Coverage(SetFunction):
    """f(S) = | union of the subsets indexed by S |."""
    def __init__(self, sets):
        super().__init__(range(len(sets)))
        self.sets = [set(s) for s in sets]

    def value(self, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(covered)

    def marginal(self, e, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(self.sets[e] - covered)


def exact_best(f, k):
    """Exhaustive optimum for tiny instances, used only as a checker."""
    best_S, best_value = set(), f.value(set())
    limit = min(k, len(f.ground_set))
    for r in range(limit + 1):
        for combo in combinations(f.ground_set, r):
            S = set(combo)
            value = f.value(S)
            if value > best_value:
                best_S, best_value = S, value
    return best_S, best_value


def select(f, k):
    """Grow S by repeatedly adding the element with maximum marginal gain."""
    if k < 0:
        raise ValueError("k must be non-negative")

    S = set()
    for _ in range(min(k, len(f.ground_set))):
        best_e, best_gain = None, 0
        for e in f.ground_set:
            if e in S:
                continue
            g = f.marginal(e, S)
            if best_e is None or g > best_gain:
                best_e, best_gain = e, g
        if best_e is None or best_gain <= 0:
            break
        S.add(best_e)
    return S


if __name__ == "__main__":
    sets = [
        {1, 2, 3, 8},
        {1, 2, 3, 4, 5},
        {1, 4, 6, 7},
        {5, 6, 7, 8},
        {2, 3, 9},
    ]
    f = Coverage(sets)
    k = 2

    chosen = select(f, k)
    opt, opt_value = exact_best(f, k)
    chosen_value = f.value(chosen)
    ratio = 1.0 if opt_value == 0 else chosen_value / opt_value

    assert chosen_value >= (1 - 1 / e) * opt_value
    print("selected", sorted(chosen), "value", chosen_value)
    print("optimum ", sorted(opt), "value", opt_value)
    print("ratio   ", round(ratio, 3))
```
