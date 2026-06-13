# Context: estimating item frequencies in a high-volume stream in sublinear space

## Research question

A vector $a$ of dimension $n$ is presented implicitly and incrementally. It starts at zero, and a stream of updates $(i_t, c_t)$ arrives: the $t$-th update means $a_{i_t} \leftarrow a_{i_t} + c_t$, all other coordinates unchanged. At any moment we want to answer questions about the current $a$ — most basically, a **point query** $\mathcal{Q}(i)$ returning (an approximation of) $a_i$, the frequency of item $i$. The catch is scale: $n$ may be $2^{32}$ (every IP address) or larger, the stream may run to billions of updates at multi-gigabit line rate, and we are allowed space only **polylogarithmic in $n$** — far too little to store $a$ explicitly. Because the summary is so much smaller than $a$, exact answers are impossible; the goal is an approximation with a provable, tunable guarantee. The standard contract is a pair $(\varepsilon, \delta)$: the answer should be within an additive $\varepsilon$-fraction of some norm of $a$ with probability at least $1-\delta$, and the space and per-update time should depend on $\varepsilon, \delta$ as mildly as possible.

What a good solution must achieve: (1) space proportional to $1/\varepsilon$ rather than $1/\varepsilon^2$ — the quadratic dependence is crippling when $\varepsilon = 0.01$; (2) update time sublinear in the size of the summary, so it keeps up with the stream; (3) only weak, cheap-to-evaluate randomness (not strong $k$-wise independence that is awkward in hardware); (4) one structure that serves many query types — point, range, inner product, heavy hitters, quantiles; (5) explicit, small constants, not hidden in big-Oh; and (6) support for deletions (negative $c_t$) and for merging summaries computed at different sites.

## Background

The data-stream model crystallized in the late 1990s and early 2000s around exactly this tension: massive input, one pass, tiny memory, approximate answers with parameters $\varepsilon$ (error) and $\delta$ (failure probability). Two cases of the update stream are distinguished. In the **cash-register** case all $c_t > 0$ (counts only increase); in the **turnstile** case $c_t$ may be negative, with a *strict* sub-case where every current $a_i$ remains $\ge 0$ (insertions and deletions of real items) and a *general* sub-case where coordinates may go negative (e.g. one vector minus another).

The load-bearing concept underneath every sketch of this era is the **random linear projection**. A sketch is a small number of inner products $\langle a, r\rangle$ between the data vector and random vectors $r$ defined implicitly by hash functions. Linearity is what makes these structures composable: the sketch of $a+b$ is the sum of the sketches (same hashes), and the sketch of $\lambda a$ is $\lambda$ times the sketch. That single property lets the counter array process turnstile updates (a deletion is just a negative increment) and lets distributed sites merge by adding summaries. The distinction between non-negative and general signed vectors matters: a general signed vector can cancel inside a counter, whereas a non-negative one cannot.

The second load-bearing tool is the theory of **limited-independence hash families**. A family is *$k$-wise independent* if any $k$ keys land independently. For the clean analysis, each row needs a hash into $w$ columns with collision probability $\Pr[h(i)=h(k)] \le 1/w$ for $i \ne k$; pairwise independence is exactly what lets that collision bound be used one item-pair at a time. Carter–Wegman affine hashing over a prime field gives the standard cheap primitive, and implementations reduce its output to a table column. How much independence a sketch *needs* — pairwise, four-wise, or more — turns out to be the hinge on which both correctness and hardware-friendliness swing, and reducing that requirement is itself a goal.

An elementary fact about hashing into a counter array: in a **non-negative** stream, when several items collide in the same counter, their masses only ever **add**. A counter that an item $i$ touches therefore holds $a_i$ *plus* the masses of everything else that hashed there. By contrast, the relevant lower-bound result of the time (Saks–Sun and related) showed an $\Omega(1/\varepsilon^2)$ space lower bound for estimating frequency moments $F_k = \sum_i a_i^k$ and for $L_2$-type quantities — a warning that anything routed through an $L_2$ norm or a small-dimension embedding is likely stuck at $1/\varepsilon^2$.

## Baselines

**AMS sketch / tug-of-war (Alon, Matias, Szegedy 1996).** To estimate $F_2 = \|a\|_2^2$, attach a random sign $s(i) \in \{-1,+1\}$ to each item and maintain a single counter $z = \sum_i a_i\, s(i)$; each update $(i,c)$ does $z \mathrel{{+}{=}} c\cdot s(i)$. Then $\mathbb{E}[z^2] = \sum_i a_i^2 + \sum_{i\ne k} a_i a_k \,\mathbb{E}[s(i)s(k)] = \|a\|_2^2$, because the cross terms vanish in expectation when the signs are pairwise independent. The estimator $z^2$ is unbiased; controlling its **variance** requires $\mathbb{E}[s(i)s(j)s(k)s(l)]$ to factor, i.e. **four-wise independence**, and averaging $O(1/\varepsilon^2)$ independent copies (plus a median of means for the $\delta$ amplification) drives the error to $\varepsilon\|a\|_2^2$. Gaps it leaves: space $\propto 1/\varepsilon^2$; it needs four-wise-independent hashes, which are heavier to evaluate, particularly in hardware; and it is built to estimate a **norm / inner product**, not to answer per-item point queries.

**Count Sketch (Charikar, Chen, Farach-Colton 2002).** This adapts the tug-of-war idea to per-item frequencies and to repetition for confidence. Keep a $d \times w$ table of counters; for each row $j$ use a hash $h_j$ to pick a column and a *second* hash $g_j(i)\in\{-1,+1\}$ to pick a sign, and on update $(i,c)$ add $c\cdot g_j(i)$ to $C[j,h_j(i)]$. To estimate $a_i$, read $g_j(i)\cdot C[j,h_j(i)]$ in each row and take the **median** across rows. The signs *symmetrize* the collision noise: an item $k$ colliding with $i$ contributes $g_j(i)g_j(k)\,a_k$, which is $+a_k$ or $-a_k$ with equal probability, so each row's estimate is **unbiased**. Unbiasedness is why the median (not the minimum) is the right aggregator, and why the analysis goes through the **variance** of a row's estimate, $\approx \|a\|_2^2/w$. Setting that to $(\varepsilon\|a\|_2)^2$ forces $w \propto 1/\varepsilon^2$, and the guarantee is in terms of the **$L_2$** norm: $|\hat a_i - a_i| \le \varepsilon\|a\|_2$ with probability $1-\delta$ using $d \approx \log(1/\delta)$ rows. Gaps: the $1/\varepsilon^2$ width, inherited from the variance/$L_2$ route; the need for sign hashes in addition to the index hashes; and a two-sided estimator whose error guarantee is in the $L_2$ norm.

**Bloom filters and multistage / counting filters.** The older idea of hashing an item into several independent tables and combining the per-table readings — taking an AND for set membership, or a min for counts (the "multistage filter" of Estan–Varghese in the networking literature) — was deployed with only limited independence and worked well in practice. These were heuristics without a clean $(\varepsilon,\delta)$ characterization: the practical behavior was observed but never given a provable accuracy guarantee.

## Evaluation settings

The natural testbeds are high-rate streams where frequencies are heavily skewed: IP packet traces keyed by source/destination address or flow, and large query/click logs. The implicit vector dimension $n$ is the key space size ($2^{32}$ for IPv4 addresses, larger for flows). The query workloads are point queries (per-item frequency), range queries (sum of $a_i$ over an interval of the ordered domain), inner-product / self-join-size queries (estimating $a\cdot b$ and $F_2 = a\cdot a$), heavy hitters (all $i$ with $a_i \ge \phi\|a\|_1$), and $\phi$-quantiles of the cardinality. The quality metrics are: observed per-query error measured against $\|a\|_1$ (or $\|a\|_2$ for the $L_2$-style baselines); space in bytes for fixed target accuracy; and update throughput in updates per second. Skewed (Zipfian, parameter $z$) synthetic streams are the standard stress test, since the whole point is that mass concentrates on few items. The natural reference points are the AMS and Count Sketch space/accuracy trade-offs above.

## Code framework

The primitives that already exist: a random-number generator to seed hash functions, an affine hash over a large prime field whose output can be reduced to a column, a two-dimensional integer counter array, and ordinary array addition. The open slots are the parameter rule, the update rule, the estimator, and the way compatible summaries compose.

```python
import math
import random
from typing import List, Optional, Tuple


HashParams = List[Tuple[int, int]]


class FrequencySketch:
    def __init__(
        self,
        epsilon: float,
        delta: float,
        seed: Optional[int] = None,
        hash_params: Optional[HashParams] = None,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.width = self._choose_width(epsilon)
        self.depth = self._choose_depth(delta)
        self.prime = 2**31 - 1
        self.total_mass = 0
        self.counts = [[0] * self.width for _ in range(self.depth)]
        self.hash_params = hash_params or self._draw_hashes(seed)

    @staticmethod
    def _choose_width(epsilon: float) -> int:
        # TODO
        pass

    @staticmethod
    def _choose_depth(delta: float) -> int:
        # TODO
        pass

    def _draw_hashes(self, seed: Optional[int]) -> HashParams:
        # TODO
        pass

    def _hash(self, row: int, item: int) -> int:
        # TODO
        pass

    def copy_empty(self) -> "FrequencySketch":
        # TODO
        pass

    def compatible(self, other: "FrequencySketch") -> bool:
        # TODO
        pass

    def _require_compatible(self, other: "FrequencySketch") -> None:
        # TODO
        pass

    def update(self, item: int, count: int = 1) -> None:
        # TODO
        pass

    def estimate(self, item: int) -> int:
        # TODO
        pass

    def merge(self, other: "FrequencySketch") -> "FrequencySketch":
        # TODO
        pass

    def merge_in_place(self, other: "FrequencySketch") -> None:
        # TODO
        pass

    def inner_product(self, other: "FrequencySketch") -> int:
        # TODO
        pass
```
