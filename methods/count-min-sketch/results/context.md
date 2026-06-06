# Context: estimating item frequencies in a high-volume stream in sublinear space

## Research question

A vector $a$ of dimension $n$ is presented implicitly and incrementally. It starts at zero, and a stream of updates $(i_t, c_t)$ arrives: the $t$-th update means $a_{i_t} \leftarrow a_{i_t} + c_t$, all other coordinates unchanged. At any moment we want to answer questions about the current $a$ — most basically, a **point query** $\mathcal{Q}(i)$ returning (an approximation of) $a_i$, the frequency of item $i$. The catch is scale: $n$ may be $2^{32}$ (every IP address) or larger, the stream may run to billions of updates at multi-gigabit line rate, and we are allowed space only **polylogarithmic in $n$** — far too little to store $a$ explicitly. Because the summary is so much smaller than $a$, exact answers are impossible; the goal is an approximation with a provable, tunable guarantee. The standard contract is a pair $(\varepsilon, \delta)$: the answer should be within an additive $\varepsilon$-fraction of some norm of $a$ with probability at least $1-\delta$, and the space and per-update time should depend on $\varepsilon, \delta$ as mildly as possible.

What a good solution must achieve: (1) space proportional to $1/\varepsilon$ rather than $1/\varepsilon^2$ — the quadratic dependence is crippling when $\varepsilon = 0.01$; (2) update time sublinear in the size of the summary, so it keeps up with the stream; (3) only weak, cheap-to-evaluate randomness (not strong $k$-wise independence that is awkward in hardware); (4) one structure that serves many query types — point, range, inner product, heavy hitters, quantiles; (5) explicit, small constants, not hidden in big-Oh; and (6) support for deletions (negative $c_t$) and for merging summaries computed at different sites.

## Background

The data-stream model crystallized in the late 1990s and early 2000s around exactly this tension: massive input, one pass, tiny memory, approximate answers with parameters $\varepsilon$ (error) and $\delta$ (failure probability). Two cases of the update stream are distinguished. In the **cash-register** case all $c_t > 0$ (counts only increase); in the **turnstile** case $c_t$ may be negative, with a *non-negative* sub-case where the final $a_i$ are guaranteed $\ge 0$ (insertions and deletions of real items) and a *general* sub-case where they may go negative (e.g. one vector minus another).

The load-bearing concept underneath every sketch of this era is the **random linear projection**. A sketch is a small number of inner products $\langle a, r\rangle$ between the data vector and random vectors $r$ defined implicitly by hash functions. Linearity is what makes these structures composable: the sketch of $a+b$ is the sum of the sketches (same hashes), and the sketch of $\lambda a$ is $\lambda$ times the sketch. That single property delivers both turnstile updates (a deletion is just a negative increment, processed identically to an insertion) and distributed merging (sites sketch their shards and the coordinator adds the sketches).

The second load-bearing tool is the theory of **limited-independence hash families**. A family is *$k$-wise independent* if any $k$ keys land independently; the Carter–Wegman construction $h(x) = ((ax+b) \bmod p) \bmod w$ with $p$ prime and $a,b$ chosen at random is 2-universal (pairwise independent), and pairwise independence is exactly what lets one bound collision probabilities: $\Pr[h(i)=h(k)] \le 1/w$ for $i \ne k$. How much independence a sketch *needs* — pairwise, four-wise, or more — turns out to be the hinge on which both correctness and hardware-friendliness swing, and reducing that requirement is itself a goal.

A diagnostic that shapes everything: in a **non-negative** stream, when several items collide in the same counter, their masses only ever **add**. A counter that an item $i$ touches therefore holds $a_i$ *plus* the masses of everything else that hashed there — never less. The error is **one-sided** (an overestimate), and its size in a single counter is governed by the *expected colliding mass*, a first-moment quantity. By contrast, the relevant lower-bound result of the time (Saks–Sun and related) showed an $\Omega(1/\varepsilon^2)$ space lower bound for estimating frequency moments $F_k = \sum_i a_i^k$ and for $L_2$-type quantities — a warning that anything routed through an $L_2$ norm or a small-dimension embedding is likely stuck at $1/\varepsilon^2$.

## Baselines

**AMS sketch / tug-of-war (Alon, Matias, Szegedy 1996).** To estimate $F_2 = \|a\|_2^2$, attach a random sign $s(i) \in \{-1,+1\}$ to each item and maintain a single counter $z = \sum_i a_i\, s(i)$; each update $(i,c)$ does $z \mathrel{{+}{=}} c\cdot s(i)$. Then $\mathbb{E}[z^2] = \sum_i a_i^2 + \sum_{i\ne k} a_i a_k \,\mathbb{E}[s(i)s(k)] = \|a\|_2^2$, because the cross terms vanish in expectation when the signs are pairwise independent. The estimator $z^2$ is unbiased; controlling its **variance** requires $\mathbb{E}[s(i)s(j)s(k)s(l)]$ to factor, i.e. **four-wise independence**, and averaging $O(1/\varepsilon^2)$ independent copies (plus a median of means for the $\delta$ amplification) drives the error to $\varepsilon\|a\|_2^2$. Gaps it leaves: space $\propto 1/\varepsilon^2$; it needs four-wise-independent hashes, which are heavier to evaluate, particularly in hardware; and it is built to estimate a **norm / inner product**, not to answer per-item point queries.

**Count Sketch (Charikar, Chen, Farach-Colton 2002).** This adapts the tug-of-war idea to per-item frequencies and to repetition for confidence. Keep a $d \times w$ table of counters; for each row $j$ use a hash $h_j$ to pick a column and a *second* hash $g_j(i)\in\{-1,+1\}$ to pick a sign, and on update $(i,c)$ add $c\cdot g_j(i)$ to $C[j,h_j(i)]$. To estimate $a_i$, read $g_j(i)\cdot C[j,h_j(i)]$ in each row and take the **median** across rows. The signs *symmetrize* the collision noise: an item $k$ colliding with $i$ contributes $g_j(i)g_j(k)\,a_k$, which is $+a_k$ or $-a_k$ with equal probability, so each row's estimate is **unbiased**. Unbiasedness is why the median (not the minimum) is the right aggregator, and why the analysis goes through the **variance** of a row's estimate, $\approx \|a\|_2^2/w$. Setting that to $(\varepsilon\|a\|_2)^2$ forces $w \propto 1/\varepsilon^2$, and the guarantee is in terms of the **$L_2$** norm: $|\hat a_i - a_i| \le \varepsilon\|a\|_2$ with probability $1-\delta$ using $d \approx \log(1/\delta)$ rows. Gaps: the $1/\varepsilon^2$ width, inherited from the variance/$L_2$ route; the need for sign hashes in addition to the index hashes; and a guarantee scaled by $\|a\|_2$, which can be loose relative to $\|a\|_1$ on skewed data where a few heavy items dominate.

**Bloom filters and multistage / counting filters.** The older idea of hashing an item into several independent tables and combining the per-table readings — taking an AND for set membership, or a min for counts (the "multistage filter" of Estan–Varghese in the networking literature) — already showed that *only limited independence* and a *min across rows* can give strong practical behavior. These were heuristics without a clean $(\varepsilon,\delta)$ characterization, but they carry the intuition that one-sided error plus a min, over a few weakly-random tables, is a powerful and cheap combination.

## Evaluation settings

The natural testbeds are high-rate streams where frequencies are heavily skewed: IP packet traces keyed by source/destination address or flow, and large query/click logs. The implicit vector dimension $n$ is the key space size ($2^{32}$ for IPv4 addresses, larger for flows). The query workloads are point queries (per-item frequency), range queries (sum of $a_i$ over an interval of the ordered domain), inner-product / self-join-size queries (estimating $a\cdot b$ and $F_2 = a\cdot a$), heavy hitters (all $i$ with $a_i \ge \phi\|a\|_1$), and $\phi$-quantiles of the cardinality. The quality metrics are: observed per-query error measured against $\|a\|_1$ (or $\|a\|_2$ for the $L_2$-style baselines); space in bytes for fixed target accuracy; and update throughput in updates per second. Skewed (Zipfian, parameter $z$) synthetic streams are the standard stress test, since the whole point is that mass concentrates on few items. The natural reference points are the AMS and Count Sketch space/accuracy trade-offs above.

## Code framework

The primitives that already exist: a random-number generator to seed hash functions, and a Carter–Wegman pairwise-independent hash $h(x) = ((ax+b)\bmod p)\bmod w$ with $p = 2^{31}-1$ a Mersenne prime (so the mod can be done by a shift-and-add). The data structure is a 2-D array of integer counters plus, per row, the two hash parameters $(a_j, b_j)$. What does not yet exist is the choice of estimator, the rule relating $(w,d)$ to $(\varepsilon,\delta)$, and the query procedures — those are the empty slots.

```c
#define MOD 2147483647        /* 2^31 - 1, a Mersenne prime */
#define HL  31

/* pairwise-independent (2-universal) hash of x with parameters a,b, mod 2^31-1 */
long hash31(long long a, long long b, long long x) {
  long long r = a * x + b;
  r = ((r >> HL) + r) & MOD;    /* fast reduction mod 2^31-1 */
  return (long) r;
}

typedef struct {
  int depth, width;            /* d rows, w columns */
  int count;                   /* running ||a||_1 */
  int **counts;                /* depth x width counter array */
  unsigned int *hasha, *hashb; /* per-row hash parameters */
} CM_type;

/* choose width w and depth d from the accuracy/confidence targets */
void params_from_eps_delta(double eps, double delta, int *width, int *depth) {
  /* TODO: the rule relating (w,d) to (eps,delta) is exactly what must be derived */
}

CM_type *CM_Init(int width, int depth, int seed) {
  /* allocate counts[depth][width]; draw hasha[j], hashb[j] per row from the prng */
  /* TODO */
}

void CM_Update(CM_type *cm, unsigned int item, int diff) {
  /* TODO: which cells get touched on each update, and by how much */
}

int CM_PointEst(CM_type *cm, unsigned int query) {
  /* TODO: how to combine the d per-row readings into one estimate */
}

int CM_InnerProd(CM_type *cm1, CM_type *cm2) {
  /* TODO: estimate a . b from two compatible sketches */
}
```
