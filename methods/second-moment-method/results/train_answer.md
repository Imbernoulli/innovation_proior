I will present the canonical method known as the second moment method, which closes the one-sided gap left by the first moment when we want to show that a random structure almost surely contains a given configuration. The setup is familiar: we have a nonnegative integer-valued counting random variable X that records how many copies of some configuration appear in a random object, such as the number of 4-cliques in an Erdős–Rényi graph G(n,p) or the number of distinct prime divisors of a random integer. The first moment tells us that if E[X] tends to zero, then X is zero almost always, because Markov's inequality gives Pr[X ≥ 1] ≤ E[X]. That direction is clean and complete. But the converse fails dramatically: a variable can have E[X] diverging to infinity while still being zero with probability approaching one, simply by placing all of its mass on a vanishingly rare event. The second moment method is the standard way to rule out that pathology and convert a large mean into a high-probability existence statement.

The key observation is that a large mean by itself says nothing about how the mass of X is distributed around that mean. If X were tightly concentrated near a large mean, it could not simultaneously be zero most of the time, because zero would be a huge deviation. The missing quantity is therefore the spread of X around its mean, measured by the variance Var[X] = E[(X − E[X])²]. Chebyshev's inequality makes the connection precise: for any λ > 0, Pr[|X − E[X]| ≥ λ√Var[X]] ≤ 1/λ². In particular, the event X = 0 is contained in the event |X − E[X]| ≥ E[X], so Pr[X = 0] ≤ Var[X] / E[X]². Thus the criterion for almost sure existence is not merely that the mean is large, but that the variance is small relative to the mean squared. If Var[X] = o(E[X]²), then Pr[X = 0] → 0, which means X > 0 almost always. The same argument applied to |X − E[X]| ≥ εE[X] shows that X/E[X] → 1 in probability, so we actually obtain the stronger conclusion X ∼ E[X] almost always. In other words, the number of copies is concentrated around its mean.

For combinatorial counts, X is usually a sum of indicator random variables X = Σᵢ 1_{Aᵢ}, where Aᵢ is the event that the i-th potential copy is present. The variance then decomposes as Var[X] = Σᵢ Var[Xᵢ] + Σ_{i≠j} Cov[Xᵢ, Xⱼ]. For an indicator, Var[Xᵢ] = Pr[Aᵢ](1 − Pr[Aᵢ]) ≤ Pr[Aᵢ] = E[Xᵢ], so the diagonal terms sum to at most E[X]. Independent pairs contribute zero covariance, so only dependent pairs matter. Writing i ∼ j when the events are dependent and defining Δ = Σ_{i∼j} Pr[Aᵢ ∧ Aⱼ], we get Var[X] ≤ E[X] + Δ. The criterion Var[X] = o(E[X]²) therefore reduces to two practical conditions: E[X] → ∞ and Δ = o(E[X]²). When the events are symmetric, the computation simplifies further. Fixing any i and defining Δ* = Σ_{j∼i} Pr[Aⱼ | Aᵢ], we have Δ = Δ* · E[X], so the criterion becomes Δ* = o(E[X]). This has an intuitive reading: conditioning on one copy being present must not inflate the expected number of other dependent copies by too much.

A concrete illustration is the appearance of a fixed subgraph H with v vertices and e edges in G(n,p). For each v-set S of vertices, let A_S be the event that the induced subgraph on S contains a copy of H. Then Pr[A_S] is Θ(p^e), and summing over all v-sets gives E[X] = Θ(n^v p^e). Balancing n^v p^e = Θ(1) suggests the threshold p = n^{−v/e}, or equivalently p = n^{−1/ρ(H)} where ρ(H) = e/v is the density of H. Below this threshold the first moment gives E[X] = o(1) and hence X = 0 almost always. For the appearance direction above the threshold, assume H is balanced, meaning every subgraph H′ of H satisfies ρ(H′) ≤ ρ(H). Fixing a copy on S, any dependent T overlaps S in i vertices with 2 ≤ i ≤ v − 1. The part of the T-copy inside the overlap is a subgraph of H on at most i vertices, so by balance it has at most i·e/v edges; the remaining at least e − i·e/v edges must still appear. There are O(n^{v−i}) choices of T for overlap i, and each contributes conditional probability O(p^{e−ie/v}). Summing over i gives Δ* = Σ_{i=2}^{v−1} O((n^v p^e)^{1−i/v}) = o(E[X]) once n^v p^e → ∞, because each exponent 1 − i/v is strictly less than one. Thus the second moment method confirms p = n^{−v/e} as the threshold for any balanced H.

The simplest special case is H = K₄, the complete graph on four vertices. Here v = 4 and e = 6, so the threshold is p = n^{−2/3}. The expected number of 4-cliques is E[X] = C(n,4)p^6 ∼ n^4 p^6 / 24. If p ≪ n^{−2/3}, this tends to zero and 4-cliques are absent almost always. If p ≫ n^{−2/3}, the second moment calculation above shows X > 0 almost always and in fact X ∼ E[X]. A direct computation of Δ* for K₄ gives O(n²p⁵) from pairs of 4-sets sharing two vertices and O(np³) from pairs sharing three vertices, both of which are o(n⁴p⁶) above the threshold. The balancedness assumption is essential: if H has a denser subgraph H₁, then H cannot appear before H₁ does, so the threshold is governed by the densest subgraph rather than by the overall density of H.

The second moment method also applies beautifully outside graph theory. For a random integer x uniform in {1, …, n}, let ν(x) be the number of distinct prime divisors. Writing X as a sum of indicators X_p for primes p up to n^{1/10}, one finds E[X] = ln ln n + O(1) by Mertens's theorem. The diagonal contribution to the variance is also ln ln n + O(1), while the off-diagonal covariances are tiny because divisibility by distinct primes is nearly independent; their total is o(1). Chebyshev then yields the Hardy–Ramanujan theorem: almost all integers up to n have ν(x) within a few √(ln ln n) of ln ln n. This is the same conceptual engine as the graph threshold, only with dependencies so weak that the variance is essentially the mean.

There is one further refinement worth keeping in mind. If Var[X] is only comparable to E[X]² rather than asymptotically smaller, Chebyshev no longer drives Pr[X = 0] to zero. Nevertheless, the Paley–Zygmund inequality, which lower-bounds the chance that a nonnegative variable exceeds a constant fraction of its mean using only the first two moments, gives Pr[X > 0] ≥ E[X]² / E[X²] = 1 / (1 + Var[X]/E[X]²). When Var[X] = O(E[X]²), this supplies a constant lower bound on the probability of existence, which is often enough to bootstrap a sharper argument. Thus the second moment method has two regimes: small variance relative to the mean squared forces high-probability existence and concentration, while merely comparable variance still forces existence with bounded-away-from-zero probability.

The following Python script verifies the threshold phenomenon for 4-cliques in G(n,p). It samples Erdős–Rényi graphs at several values of p relative to the threshold n^{−2/3}, counts 4-cliques in each sample, and compares the empirical frequency of seeing at least one 4-clique and the empirical mean count against the theoretical expectation C(n,4)p^6.

```python
import itertools
import random
import math

def count_4_cliques(n, p):
    # Build G(n,p) as an adjacency bitset for fast edge queries.
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    count = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        mask_a = (1 << b) | (1 << c) | (1 << d)
        if (adj[a] & mask_a) != mask_a:
            continue
        mask_b = (1 << c) | (1 << d)
        if (adj[b] & mask_b) != mask_b:
            continue
        if (adj[c] & (1 << d)) != (1 << d):
            continue
        count += 1
    return count

def simulate(n, c, trials=25):
    p = c * (n ** (-2.0 / 3.0))
    expected = math.comb(n, 4) * (p ** 6)
    total = 0
    positive = 0
    for _ in range(trials):
        x = count_4_cliques(n, p)
        total += x
        positive += (x > 0)
    print(f"n={n}, c={c:.2f}, p={p:.4f}")
    print(f"  theoretical E[X] = {expected:.3f}")
    print(f"  empirical  E[X]  = {total / trials:.3f}")
    print(f"  Pr[X > 0] ≈ {positive / trials:.3f}")
    print()

random.seed(0)
for c in [0.3, 0.7, 1.0, 1.5, 2.5]:
    simulate(n=60, c=c, trials=25)
```

In summary, the second moment method upgrades the first moment by adding variance control. For a count X, once we verify that E[X] grows and that dependencies are weak enough that Var[X] = o(E[X]²), we conclude not only that at least one copy exists almost surely, but that the number of copies is concentrated around its mean. This is the standard tool for establishing appearance thresholds in random graphs and for proving concentration of additive arithmetic functions, and the Paley–Zygmund refinement extends its reach to cases where the variance is only comparable to the mean squared.
