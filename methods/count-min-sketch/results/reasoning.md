OK, let me think this through from scratch. I have a stream of updates building up a vector $a$ — item $i_t$ gets bumped by $c_t$, so $a_{i_t} \mathrel{{+}{=}} c_t$ — and the domain is enormous, $n = 2^{32}$ say, every IP address. I can't store $a$. I get one pass and polylog space. And I want to answer, at any moment, "how big is $a_i$?" — the frequency of item $i$. The contract everyone uses in this world is $(\varepsilon,\delta)$: be within an $\varepsilon$-fraction of something with probability $1-\delta$. So really I want a tiny summary plus an estimator $\hat a_i$, and a theorem that says $\hat a_i$ is close to $a_i$.

The thing gnawing at me is the space. The good sketches I know about all cost $1/\varepsilon^2$. For $\varepsilon = 0.01$ that's $10^4$ — a factor that decides whether the structure fits in cache or not, whether it keeps up with a multi-gigabit link or not. If I could get to $1/\varepsilon$ that would be the whole ballgame. So before I invent anything, let me figure out *where the square comes from* in the methods I already have, because that's the enemy.

Start with the cleanest ancestor, the tug-of-war sketch — Alon, Matias, Szegedy. Their problem is $F_2 = \|a\|_2^2$, not point queries, but the mechanism is the seed of everything. Give each item a random sign $s(i) \in \{-1,+1\}$ and keep one running counter $z = \sum_i a_i\, s(i)$; an update $(i,c)$ just does $z \mathrel{{+}{=}} c\,s(i)$. Then $z^2 = \sum_{i,k} a_i a_k\, s(i)s(k)$, and if the signs are pairwise independent the cross terms $s(i)s(k)$, $i\ne k$, have mean zero, so $\mathbb{E}[z^2] = \sum_i a_i^2 = \|a\|_2^2$. Beautiful — an unbiased estimator of the $L_2$ norm in *one counter*. So why isn't it free? Because $z^2$ is just unbiased, it's *noisy*. To make it usable I have to drive down its variance, and the variance of $z^2$ involves the fourth moment $\mathbb{E}[s(i)s(j)s(k)s(l)]$ — which is why AMS needs **four-wise** independent signs, and why I have to average $O(1/\varepsilon^2)$ independent copies to squeeze the relative error to $\varepsilon$. There's my square: it comes straight out of a **variance** argument. Variance $\Rightarrow$ Chebyshev $\Rightarrow$ averaging $1/\varepsilon^2$ copies. And there's the second cost: four-wise independence, which is a pain to evaluate, especially if anyone ever wants this in hardware.

Now the closer ancestor for *my* problem — point queries — is Count Sketch, Charikar–Chen–Farach-Colton. They take the tug-of-war trick and (a) spread it over a table so it gives *per-item* answers and (b) repeat it for confidence. A $d \times w$ grid of counters; row $j$ has an index hash $h_j$ picking a column and a sign hash $g_j(i)\in\{-1,+1\}$; update $(i,c)$ adds $c\,g_j(i)$ to $C[j,h_j(i)]$. To read $a_i$ off row $j$, multiply back by the sign: $g_j(i)\,C[j,h_j(i)]$. What's in that cell? It's $g_j(i)\big(g_j(i)a_i + \sum_{k:\,h_j(k)=h_j(i)} g_j(k)a_k\big) = a_i + \sum_{k\ne i,\, h_j(k)=h_j(i)} g_j(i)g_j(k)\,a_k$. Each colliding $k$ contributes $\pm a_k$ with equal probability — the signs *symmetrize* the noise. So each row gives an **unbiased** estimate of $a_i$, and because it's unbiased and symmetric, the right way to combine the $d$ rows is the **median**: the median of unbiased, roughly-symmetric estimates kills the tails. Good. But now I ask the same question — where's the square? The noise in one row is $\sum_{k} (\pm)a_k$ over the colliding items; its *expectation* is zero (that's the point of the signs), so the row's accuracy is governed by its **variance**, which is $\sum_{k\ne i} a_k^2\,\Pr[h_j(k)=h_j(i)] \approx \|a\|_2^2 / w$. To make the standard deviation $\varepsilon\|a\|_2$ I need $\|a\|_2^2/w \approx \varepsilon^2\|a\|_2^2$, i.e. $w \approx 1/\varepsilon^2$. Same square, same root cause: I made the estimator unbiased, so accuracy lives in the *variance*, so I pay $1/\varepsilon^2$. And the guarantee comes out scaled by $\|a\|_2$.

So both ancestors square because they're **unbiased**, and unbiasedness sends me to a second-moment (variance) analysis. Stare at that. The signs are the thing that buys unbiasedness. The signs are *also* the thing that forces four-wise independence in AMS and that double the hashing in Count Sketch. What if I just... don't symmetrize? Throw the signs away.

Let me see what breaks. Drop $g_j$. Now an update $(i,c)$ just adds $c$ to $C[j,h_j(i)]$ in each row — a plain counter table. And suppose the stream is **non-negative**: all $a_i \ge 0$ (true for the headline applications — packet counts, click counts; an IP flow can't have negative bytes). Then the cell $C[j,h_j(i)]$ holds $a_i + \sum_{k\ne i,\,h_j(k)=h_j(i)} a_k$, and every term in that collision sum is $\ge 0$. So the cell is **never an underestimate**: $C[j,h_j(i)] \ge a_i$, always, with certainty, in every row. The error is **one-sided**. I traded the unbiasedness away and got something that looks much worse — a biased, always-too-big estimator — but watch what one-sidedness does for me.

First, the estimator. I have $d$ readings, each of which is $a_i$ plus a non-negative junk term. The *smallest* of them has the least junk. So the natural estimate is
$$\hat a_i = \min_{j} C[j,h_j(i)].$$
And because every reading is $\ge a_i$, the minimum is too: $\hat a_i \ge a_i$ with certainty. No probabilistic underestimate to worry about, ever. That's already a cleaner promise than the median gives — half of Count Sketch's guarantee (the "no underestimate" half) is *free* and *deterministic* here. The min is right precisely *because* the error is one-sided; a median would be throwing away the structure. Wait — let me make sure min beats averaging too. If I averaged the rows I'd be averaging in the collision junk from *every* row; a row that happened to have a bad collision would pollute the average. The min instead *selects* the row that got lucky and ignores the rest. With one-sided error, selecting the luckiest reading is exactly the right move. Min it is.

Now the part that matters: with the bias, does the analysis escape the square? The junk in one row is $X_{i,j} = \sum_{k\ne i} I_{i,j,k}\, a_k$, where $I_{i,j,k} = \mathbf{1}[h_j(k)=h_j(i)]$. This is a sum of *non-negative* terms, and crucially I only need its **expectation**, not its variance — because I'm going to bound the probability it's large by a first-moment argument. That's the whole point of one-sidedness: a non-negative random variable can be controlled by Markov, which only wants $\mathbb{E}[X]$.

Let me compute $\mathbb{E}[I_{i,j,k}]$. If $h_j$ is drawn from a pairwise-independent family mapping into $w$ buckets, then for $i \ne k$, $\Pr[h_j(i)=h_j(k)] \le 1/w$. That is *all* I need from the hash — pairwise independence, nothing stronger. (Contrast AMS's four-wise: I'm asking for a first moment, so I only need *pairs* to behave.) So by linearity of expectation,
$$\mathbb{E}[X_{i,j}] = \sum_{k\ne i} a_k\,\Pr[h_j(k)=h_j(i)] \le \frac{1}{w}\sum_{k\ne i} a_k \le \frac{\|a\|_1}{w}.$$
There's the $\|a\|_1$ — falls right out of summing non-negative collision masses, no $L_2$ in sight. So the expected error in a single row is $\|a\|_1/w$. I want that error, after combining, to be $\varepsilon\|a\|_1$. Notice $w$ sits *linearly* under $1/w$ — to make $\|a\|_1/w$ around $\varepsilon\|a\|_1$ I want $w \approx 1/\varepsilon$, **not** $1/\varepsilon^2$. The square is gone. It's gone because I never formed a variance; the expected error is a first-order quantity in the bias, and bias scales like $1/w$, not $1/\sqrt{w}$.

But I have to be careful — $\mathbb{E}[X_{i,j}] \le \|a\|_1/w$ is the *expected* junk; a single row could still be unlucky and have junk much bigger than $\varepsilon\|a\|_1$. That's what the $d$ rows and the min are for. Let me set $w = e/\varepsilon$ — I'll justify the $e$ in a second — so that $\mathbb{E}[X_{i,j}] \le \frac{\varepsilon}{e}\|a\|_1$. Then for a single row, by Markov,
$$\Pr\big[X_{i,j} > \varepsilon\|a\|_1\big] = \Pr\big[X_{i,j} > e\cdot \tfrac{\varepsilon}{e}\|a\|_1\big] \le \Pr\big[X_{i,j} > e\,\mathbb{E}[X_{i,j}]\big] < \frac1e.$$
So each row independently fails — overshoots by more than $\varepsilon\|a\|_1$ — with probability under $1/e$. The estimate $\hat a_i = \min_j$ exceeds $a_i + \varepsilon\|a\|_1$ only if **every** row fails, and the rows are independent because I drew the $d$ hash functions independently:
$$\Pr\big[\hat a_i > a_i + \varepsilon\|a\|_1\big] = \Pr\big[\forall j:\, X_{i,j} > \varepsilon\|a\|_1\big] < \Big(\frac1e\Big)^{d} = e^{-d}.$$
I want this $\le \delta$, so $e^{-d} \le \delta \Rightarrow d \ge \ln(1/\delta)$. Take $d = \lceil \ln(1/\delta)\rceil$. And there it is, the whole guarantee:
$$a_i \le \hat a_i \le a_i + \varepsilon\|a\|_1 \quad\text{with probability } \ge 1-\delta,$$
the left inequality holding *with certainty*, using $w = \lceil e/\varepsilon\rceil$ columns and $d = \lceil\ln(1/\delta)\rceil$ rows, $wd \approx \frac{e}{\varepsilon}\ln(1/\delta)$ counters, and **only pairwise-independent hashing**.

Now let me pay the debt — *why $e$* in $w = e/\varepsilon$. It wasn't arbitrary; it's the choice that minimizes the space. Let me keep it general: pick a slack factor $b > 1$, set $w = e/\varepsilon$... no, let me redo this honestly with $b$ as the Markov factor. Say I want each row's expected junk to be a $1/b$-fraction of the target, i.e. $\mathbb{E}[X] \le \frac{\varepsilon}{b}\|a\|_1$, which means $w = b/\varepsilon$. Markov then gives per-row failure $\Pr[X > \varepsilon\|a\|_1] = \Pr[X > b\,\mathbb{E}[X]] < 1/b$, so $d$ rows fail together with probability $b^{-d}$, and to hit $\delta$ I need $d = \log_b(1/\delta) = \ln(1/\delta)/\ln b$. The total counter count is
$$wd = \frac{b}{\varepsilon}\cdot\frac{\ln(1/\delta)}{\ln b} = \frac{\ln(1/\delta)}{\varepsilon}\cdot\frac{b}{\ln b}.$$
So I want to minimize $b/\ln b$ over $b > 1$. Differentiate: $\frac{d}{db}\frac{b}{\ln b} = \frac{\ln b - 1}{(\ln b)^2} = 0 \Rightarrow \ln b = 1 \Rightarrow b = e$. So $b=e$ is the optimum, $w = e/\varepsilon$, $d = \ln(1/\delta)$, and $wd = \frac{e}{\varepsilon}\ln(1/\delta)$ counters — adding the two hash words per row, the cost is $\big(2 + \frac{e}{\varepsilon}\big)\ln(1/\delta)$ words. The $e$ in the width is just where $b/\ln b$ bottoms out. (In practice any integer $b$ — say $w = 2/\varepsilon$, $w = 4/\varepsilon$ — works too and is sometimes nicer to compute with; it just isn't quite space-optimal.)

Let me sanity-check the update and query cost. An update touches one cell per row: compute $d$ hashes, $d$ increments — $O(d)$, *independent of $w$*. Since $d \approx \ln(1/\delta)$ is a tiny constant (often under 10), updates are effectively constant-time and fast enough for line rate. A point query is also $O(d)$: $d$ hashes and a min. Good — that's the "update should be fast" desideratum.

Now, the hash. I need a concrete pairwise-independent family that's cheap. Carter–Wegman: pick a prime $p$, draw $a, b$ at random in $[0,p)$, and use $h(x) = ((ax+b) \bmod p) \bmod w$. For $x \ne y$ the pair $(ax+b, ay+b) \bmod p$ is uniform over distinct pairs, which gives collision probability $\le 1/w$ — exactly the pairwise property I leaned on. Pick $p = 2^{31}-1$, a Mersenne prime, because then $\bmod\,p$ is almost free: for a 62-bit product $r = ax+b$, $r \bmod (2^{31}-1) = ((r \gg 31) + r) \,\&\, (2^{31}-1)$, possibly one subtraction — a shift and an add instead of a division. So one row's hash is a multiply, an add, a shift-add, and a mod-$w$. That's it. No four-wise machinery, because I only ever used a first moment.

Let me make sure I haven't quietly assumed the easy case everywhere. The whole one-sided story rested on $a_k \ge 0$ so the junk $X_{i,j}$ is non-negative and the cell is an overestimate. What if the stream has true deletions and some $a_i$ can go *negative* — the general turnstile case? Then a colliding $k$ with $a_k < 0$ can pull a cell *below* $a_i$, so $C[j,h_j(i)] \ge a_i$ no longer holds, and the min is no longer a valid upper bound — in fact min would systematically *under*-shoot, picking out the row where negative junk dragged the cell down most. Wall. The min is wrong for signed data. But I don't have to abandon the table — I just need the right aggregator for *symmetric* noise. With signed $a$, the junk in a row is $\sum_k (\pm a_k)$-ish and is roughly symmetric around $a_i$, so the **median** of the $d$ readings is the robust estimator (now I do want the median — but for the *signed* case, and the analysis there wants a few more rows for the median to concentrate). So: non-negative stream $\to$ min; general signed stream $\to$ median. The same counter table serves both; only the read-out changes.

Step back and look at what I've actually built: a $d\times w$ array of counters, $C[j,k] = \sum_{i:\,h_j(i)=k} a_i$. That's a **linear function of $a$** — $C[j,k] = \langle a, r_{j,k}\rangle$ where $r_{j,k}[i] = \mathbf{1}[h_j(i)=k]$. Linearity is a gift I should spend. (1) A deletion is just an update with $c_t < 0$; the structure handles insertions and deletions identically — turnstile streams for free (the *counters* update fine even when I have to switch to the median for reading). (2) If two sites each sketch their own sub-stream with the *same* hashes, the sketch of the combined stream is the entrywise sum of the two sketches. So I can sketch shards in parallel and merge by addition — distributed aggregation for free. (3) Scaling: $\text{sketch}(\lambda a) = \lambda\,\text{sketch}(a)$. Anything I can phrase as a linear combination of vectors, I can phrase on the sketches.

Linearity also hands me **inner products / join sizes** almost immediately. I want $a\cdot b = \sum_i a_i b_i$ — in databases, the size of the equi-join of two relations on an attribute. Sketch both with the same hashes. Look at one row and dot the two count arrays:
$$(\widehat{a\odot b})_j = \sum_{k=1}^{w} C_a[j,k]\,C_b[j,k] = \sum_i a_i b_i + \sum_{\substack{p\ne q\\ h_j(p)=h_j(q)}} a_p b_q.$$
The first sum is exactly $a\cdot b$; the second is collision cross-talk. For non-negative $a,b$ the cross-talk is $\ge 0$, so each row over-estimates and again I take the **min over rows**. Its expectation: $\mathbb{E}\big[\sum_{p\ne q} \mathbf{1}[h_j(p)=h_j(q)] a_p b_q\big] = \sum_{p\ne q}\Pr[h_j(p)=h_j(q)]\,a_p b_q \le \frac1w \sum_{p\ne q} a_p b_q \le \frac{1}{w}\|a\|_1\|b\|_1$. With $w = e/\varepsilon$ this is $\frac{\varepsilon}{e}\|a\|_1\|b\|_1$, and the same Markov-then-min-over-$d$-rows argument gives error $\le \varepsilon\|a\|_1\|b\|_1$ with probability $1-\delta$. Compare AMS, which estimates inner products too but with error $\varepsilon\|a\|_2\|b\|_2$ at $1/\varepsilon^2$ space — same shape of result, the $L_1$ version, at $1/\varepsilon$ space. Self-join $a\cdot a$ estimates $F_2$. The pattern is identical because the mechanism is identical: non-negative collision junk, bound its mean, Markov, min.

Now the harder applications, the ones a raw point-query sketch can't do directly: **range queries**, **heavy hitters**, **quantiles**. A range query wants $\sum_{i=l}^{r} a_i$. Naively that's $(r-l+1)$ point queries, and the errors *add* — useless for a wide range. I need to query *aggregates*, not individual items. The clean trick: impose a binary hierarchy on the domain $1..n$. At level $k$, group the domain into dyadic blocks of $2^k$ consecutive items and define a *derived* vector
$$a^k[j] = \sum_{i=j\cdot 2^k}^{(j+1)2^k - 1} a_i,$$
the total mass of the $j$-th block at level $k$. Keep a Count-Min sketch of *each* level's derived vector. There are $\log n$ levels. An update to item $i$ now touches one block per level — its enclosing dyadic block at each scale — so the update walks up the hierarchy, $O(\log n)$ sketches touched. Why dyadic? Because any interval $[l,r]$ decomposes into at most $2\log n$ **dyadic blocks** (the standard segment-tree cover): greedily take the largest aligned power-of-two block that fits, at most two per level. So a range query becomes at most $2\log n$ point queries — *on aggregates*, each answered by the appropriate level's sketch — and now the error is controlled, $\le 2\log n$ times the per-query error, which I can absorb by tightening $\varepsilon$. Space goes to $O\big(\frac{\log^2 n}{\varepsilon}\log\frac1\delta\big)$. One refinement that pays for itself: at the *coarse* levels there are very few blocks — fewer than $wd$ of them — so I might as well store those counts **exactly** rather than sketch them; a dyadic level with only a handful of nodes is cheaper kept exact, and it's also error-free there.

Heavy hitters — all items with $a_i \ge \phi\|a\|_1$ — fall out of the same hierarchy by a **descent**. A heavy item's mass is contained in its ancestor block at every level, so *every* ancestor block is also heavy (mass $\ge \phi\|a\|_1$). So I never need to examine an item whose ancestor was light. Start at the root; for each block whose estimated count is $\ge \phi\|a\|_1$, recurse into its two children; prune any block below threshold. Because at most $1/\phi$ blocks per level can be heavy, the descent touches $O(\frac1\phi \log n)$ nodes, and at the leaves it outputs the heavy items — with the guarantee that nothing with frequency below $(\phi-\varepsilon)\|a\|_1$ is output, since the sketch only over-estimates by $\varepsilon\|a\|_1$. Quantiles are the same hierarchy read the other way: a $\phi$-quantile is the point where the prefix mass $\sum_{i\le j} a_i$ first reaches $\phi\|a\|_1$, and prefix sums are range queries from the left, so I **binary-search** over $j$ using range-sum estimates — halve the interval, ask "is the prefix mass past $\phi\|a\|_1$ yet?", descend — $O(\log n)$ range queries per quantile.

One more practical lever, for the positive-only case where I'll never need deletions. When I update item $i$, I currently bump all $d$ cells by $c$. But the *estimate* $\hat a_i$ is the min of those cells; bumping a cell that's already far above the min does nothing for $\hat a_i$ and only adds junk that hurts *other* items sharing that cell. So instead: compute $\hat a_i$, then set each of $i$'s cells to $\max\big(C[j,h_j(i)],\ \hat a_i + c\big)$ — raise each cell only up to what the new estimate needs, no higher. This **conservative update** (the minimal-increment idea, from the multistage-filter line, Estan–Varghese) never increases any estimate beyond the standard procedure, so $\hat a_i \ge a_i$ still holds and the error bound is never worse — and empirically it cuts error substantially because it stops pouring mass into cells gratuitously. The catch: it breaks linearity (the $\max$ is non-linear), so deletions and merges are off the table, and each update now costs a query first — roughly doubling per-update work. A pure-insertion, accuracy-hungry deployment takes that trade; a turnstile one doesn't.

Let me write the core down as real code. The counter table, the pairwise hash, the increment-all-rows update, and the min read-out — the rest (hierarchy, inner product, conservative update) hangs off these.

```c
#include <stdlib.h>
#include "prng.h"          /* pairwise-independent hash setup */

#define MOD 2147483647     /* 2^31 - 1, Mersenne prime */
#define HL  31

/* pairwise-independent hash of x with random (a,b), reduced mod 2^31-1.
   The shift-add is the fast reduction for a Mersenne-prime modulus. */
long hash31(long long a, long long b, long long x) {
  long long r = a * x + b;
  r = ((r >> HL) + r) & MOD;
  return (long) r;
}

typedef struct {
  int depth, width;          /* d rows, w columns */
  int count;                 /* running ||a||_1 = total mass seen */
  int **counts;              /* depth x width counter array */
  unsigned int *hasha, *hashb;
} CM_type;

/* w = ceil(e/eps), d = ceil(ln(1/delta)) : derived from Markov (e) + union over rows (ln 1/delta) */
CM_type *CM_Init(int width, int depth, int seed) {
  CM_type *cm = malloc(sizeof(CM_type));
  prng_type *prng = prng_Init(-abs(seed), 2);
  cm->depth = depth; cm->width = width; cm->count = 0;
  cm->counts = calloc(depth, sizeof(int *));
  cm->counts[0] = calloc((size_t)depth * width, sizeof(int));
  cm->hasha = calloc(depth, sizeof(unsigned int));
  cm->hashb = calloc(depth, sizeof(unsigned int));
  for (int j = 0; j < depth; j++) {
    cm->hasha[j] = prng_int(prng) & MOD;   /* draw one pairwise hash per row */
    cm->hashb[j] = prng_int(prng) & MOD;
    cm->counts[j] = cm->counts[0] + (j * width);
  }
  return cm;
}

/* update: item i gets +diff in EXACTLY ONE cell per row. O(d), independent of w.
   linear in a -> diff may be negative (turnstile) and sketches may be summed/merged. */
void CM_Update(CM_type *cm, unsigned int item, int diff) {
  cm->count += diff;
  for (int j = 0; j < cm->depth; j++)
    cm->counts[j][hash31(cm->hasha[j], cm->hashb[j], item) % cm->width] += diff;
}

/* point query, non-negative case: every cell is an overestimate, so the MIN
   is the tightest valid upper bound. a_i <= est <= a_i + eps*||a||_1  w.p. 1-delta. */
int CM_PointEst(CM_type *cm, unsigned int q) {
  int ans = cm->counts[0][hash31(cm->hasha[0], cm->hashb[0], q) % cm->width];
  for (int j = 1; j < cm->depth; j++) {
    int c = cm->counts[j][hash31(cm->hasha[j], cm->hashb[j], q) % cm->width];
    if (c < ans) ans = c;            /* min over rows */
  }
  return ans;
}

/* signed (general turnstile) case: collision noise is symmetric, not one-sided,
   so MEDIAN of the per-row readings replaces min (needs a larger depth). */
int CM_PointMed(CM_type *cm, unsigned int q) {
  int *ans = calloc(cm->depth, sizeof(int));
  for (int j = 0; j < cm->depth; j++)
    ans[j] = cm->counts[j][hash31(cm->hasha[j], cm->hashb[j], q) % cm->width];
  int m = median_of(ans, cm->depth);  /* select the middle reading */
  free(ans);
  return m;
}

/* inner product / join size: per-row dot of the two count arrays, then min over rows.
   est = a.b + (collision cross-talk >= 0); error <= eps*||a||_1*||b||_1 w.p. 1-delta. */
int CM_InnerProd(CM_type *c1, CM_type *c2) {  /* assumes same w,d, same hashes */
  int result = 0;
  for (int i = 0; i < c1->width; i++)
    result += c1->counts[0][i] * c2->counts[0][i];
  for (int j = 1; j < c1->depth; j++) {
    int tmp = 0;
    for (int i = 0; i < c1->width; i++)
      tmp += c1->counts[j][i] * c2->counts[j][i];
    if (tmp < result) result = tmp;   /* min over rows */
  }
  return result;
}

/* conservative (minimal) update, positive-only: raise each cell only as high as the
   new estimate needs. Never increases any estimate beyond CM_Update, so the bound
   still holds; empirically much tighter. Breaks linearity -> no deletions/merges. */
void CM_ConservativeUpdate(CM_type *cm, unsigned int item, int c) {
  int est = CM_PointEst(cm, item) + c;
  for (int j = 0; j < cm->depth; j++) {
    int *cell = &cm->counts[j][hash31(cm->hasha[j], cm->hashb[j], item) % cm->width];
    if (est > *cell) *cell = est;     /* max, not += */
  }
  cm->count += c;
}
```

So the causal chain, end to end: I wanted point queries at $1/\varepsilon$ space, found the ancestors stuck at $1/\varepsilon^2$ because their signs make them *unbiased*, which sends accuracy into the *variance* and forces a second-moment ($1/\varepsilon^2$) argument plus four-wise hashing. Dropping the signs on a non-negative stream makes every counter a deterministic *overestimate* — one-sided error — so the right estimator is the **min** over rows, and the right tool is **Markov on the first moment**: expected per-row junk $\le \|a\|_1/w$, so $w = e/\varepsilon$ (the $e$ is where $b/\ln b$ bottoms out) makes one row fail with probability $<1/e$, and $d = \ln(1/\delta)$ independent rows make them *all* fail with probability $<\delta$ — giving $a_i \le \hat a_i \le a_i + \varepsilon\|a\|_1$ w.p. $1-\delta$, at $1/\varepsilon$ space, with **pairwise** hashing only. The structure is a linear projection, which hands back deletions, merging, and $L_1$ inner-product/join estimates for free; a dyadic hierarchy lifts it to range, heavy-hitter, and quantile queries; the signed case swaps min for median; and the conservative-update variant trades linearity for a sharper estimate when only insertions arrive.
