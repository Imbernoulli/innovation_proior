Many design problems carry several conflicting objectives at once — minimize cost while maximizing reliability, minimize weight while maximizing stiffness — and there is no single winner: improving one objective costs another. The honest answer is a whole set, the Pareto-optimal trade-off surface, and a decision-maker wants to see that surface and then choose. A population-based evolutionary algorithm is the natural vehicle, because its unit of work is already a set of candidates, so in principle one run can hold many trade-off solutions and the entire front falls out. The dominant way to drive such a population is to rank its members by Pareto dominance — a solution on a better nondomination front out-selects one on a worse front — and bolt on a density device to spread them along the front. The elitist fast-sorted version of that works well on two or three objectives, but two things about it are structurally wrong. The first is cost: every generation runs a nondominated sort over the combined $2N$ population at $O(MN^2)$, a quadratic tax paid every generation just to *order* the population. The second is deeper. Dominance is only a *partial* order. For one solution to dominate another it must be no worse on *every* one of the $M$ objectives, and as $M$ grows almost any pair has one solution better somewhere and worse elsewhere, so they are mutually nondominated. The population floods into front 1, nearly everybody is rank 1, the rank becomes constant across most of the population and carries almost no information, and the selection that was supposed to drive *convergence* degrades toward the tie-breaking crowding heuristic — which was only ever meant to spread, not to converge. The convergence pressure thins exactly as the problem gets harder, and no amount of speeding up the sort repairs that.

The reason we are ranking vectors at all is that a GA wants a scalar fitness while the objectives form a vector. Dominance is one way to compare vectors, the honest partial order; but there is a much older way to turn a vector of objectives into a scalar — aggregate them, *scalarize*. Pick a weight vector $w$, form a scalar aggregation $g(x\mid w)$, and a Pareto-optimal solution comes out as the optimizer of that scalar subproblem. A scalar gives a *total* order: every pair is comparable, every comparison decisive, and the selection pressure from "is $g(y)$ smaller than $g(x)$?" never degrades, no matter how many objectives are stacked up, because it is a single number. The standard objection is that one scalarization gives only one point, so tracing a front means sweeping the weights and re-solving from scratch, one full optimization per point — wasteful precisely because each subproblem is solved independently and sequentially. But a population dissolves that accounting. I propose MOEA/D — a Multiobjective Evolutionary Algorithm based on Decomposition — which attaches *each population member to its own weight vector*, so there are $N$ scalar subproblems, and evolves all $N$ solutions simultaneously in a single run. The "$N$ separate runs" are no longer separate; they are $N$ members of one co-evolving population, and the Pareto front is no longer an emergent property of dominance ranking but the *collection of optima of $N$ well-spread scalar subproblems*.

Three things have to be made concrete: which scalarization, where the weights come from, and how the subproblems help each other. The weighted sum $g^{ws} = \sum_j w_j f_j(x)$ is disqualified, because minimizing it slides a hyperplane with normal $w$ down until it touches the feasible set, and a hyperplane touches only the *convex* boundary — on a concave dip in the front, no nonnegative weights ever land a solution. I use the Tchebycheff aggregation instead,
$$g^{te}(x \mid w, z^*) = \max_{1 \le i \le M} w_i \, |f_i(x) - z_i^*|,$$
where $z^*$ is the ideal point, the vector of per-objective minima. Its level set $\{f : \max_i w_i |f_i - z_i^*| = c\}$ is an axis-aligned box-corner anchored at $z^*$ — an "L" in two dimensions — and as $c$ shrinks the corner collapses toward $z^*$ along the ray where all weighted deviations $w_i(f_i - z_i^*)$ are equal. The corner of an L can poke into a concave dip that a flat plane slides right over. Underneath sits the correspondence that earns the method: under mild conditions, for any Pareto-optimal point there is a weight that makes it a minimizer of $g^{te}$, and conversely every minimizer for a positive weight is at least weakly Pareto-optimal, so sweeping $w$ sweeps the entire front, convex and concave alike. It is not a clean bijection — minimizers need not be unique and different weights can land on the same point — but every front point is reachable by *some* weight, which is all that is needed. The $\max$ is non-smooth, but a GA takes no gradients, so non-smoothness is free here. The absolute value is a guard for the case where $z^*$ momentarily sits above some objective value — a utopian over-estimate — making $f_i(x) - z_i^*$ negative. And $z^*$ itself is unknown (finding it is $M$ single-objective optimizations), so I estimate it with a running minimum: every time a new solution $y$ is evaluated, $z_j \leftarrow \min(z_j, f_j(y))$. It only ever descends, tracking the best per-objective value seen, so as $z^*$ sinks toward the true ideal all the subproblems sharpen toward the true front together.

The weights come from the Das & Dennis simplex lattice: pick an integer $H$, divide each coordinate into $H$ equal steps, and take every weight whose nonnegative components are multiples of $1/H$ summing to 1. That generates $N = \binom{H+M-1}{M-1}$ vectors spread uniformly over the simplex; for two objectives it is simply $w = (i/H,\, 1-i/H)$ for $i = 0 \ldots H$, giving $N = H+1$ evenly spaced points. The number of weight vectors *is* the population size, and $H$ is the resolution knob. Because each solution is pinned to a distinct, evenly spread weight, the diversity is supplied structurally — there is no crowding distance and no niche radius $\sigma_{\text{share}}$, a real simplification over the dominance-ranked machinery. (One bookkeeping note: a component $w_i = 0$ makes that objective invisible to the $\max$, so zeros are nudged to a tiny $\varepsilon$ and renormalized to keep every objective in play.)

The piece that earns the whole reframing is the collaboration, because if each subproblem ran its own little optimization I would have $N$ independent searches in a population costume and gain nothing. The leverage is a fact about the weights: two weight vectors that are close in weight space define two subproblems with nearly identical aggregations, hence nearly the same optimum, so a solution good for subproblem $i$ is, with high probability, good for the subproblems whose weights sit next to $w^i$. So for each $w^i$ I define a neighborhood $B(i)$, the indices of its $T$ closest weight vectors by Euclidean distance in weight space (including $i$ itself), computed once with a single $O(N^2)$ distance-sort and reused every generation. $T$ controls how local the sharing is: too small and almost no information passes, so I am back to $N$ near-independent searches; too large and "neighbors" stop being similar — distant weights have different optima, so recombining their solutions just injects noise, and at $T=N$ it is a global, structureless search. A modest $T$, around 20, keeps "neighbor" synonymous with "similar subproblem." Each generation sweeps the subproblems in random order. For subproblem $i$, two parents are drawn from $B(i)$ — not from the whole population, because the neighbors are exactly the locally-relevant material — and an offspring $y$ is produced by simulated binary crossover (distribution index $\eta_c$ controlling how tightly children spread around the parents) and polynomial mutation (index $\eta_m$); $y$ is evaluated and the ideal point is updated. Then comes the update that closes the loop, the dual of "parents come from the neighborhood": the offspring was bred to be good around $w^i$, so it might improve any subproblem in $B(i)$. For every neighbor $k \in B(i)$ I ask the only question that matters for that subproblem — its scalar one: if $g^{te}(y \mid w^k, z^*) \le g^{te}(x^k \mid w^k, z^*)$, replace $x^k$ with $y$. Each subproblem is judged purely by its own total order, so the comparison is always decisive, and a single good offspring can improve several neighboring subproblems at once — the information sharing that makes co-evolution beat $N$ separate runs.

Counting the work per generation: there is no global nondominated sort at all. For each of the $N$ subproblems there is a constant number of variation operations and then $T$ scalar comparisons, each a $\max$ over $M$ objectives, for $O(M\,N\,T)$ per generation total — linear in $N$ for fixed $T$ and $M$, against the $O(MN^2)$ sort I was paying, with a clean total-order selection throughout. Both things that nagged me vanish at once. The "$\le$" replacement could in principle let one lucky offspring overwrite every neighbor it beats, but each subproblem still keeps whatever is best *under its own weight*, and the well-spread, overlapping neighborhoods tile the simplex, so a solution only sticks where it is genuinely best for that weight, and the next offspring bred for a slightly different weight displaces it there — the lattice of distinct weights holds the spread, not a separate diversity device. Two refinements finish the design. Drawing parents strictly from $B(i)$ exploits local structure, but if a neighborhood stalls I want an occasional shot of diversity, so with probability $0.9$ the parents come from $B(i)$ and with the small remaining probability from the whole population — cheap insurance, one random branch. And because a solution optimal for a skewed weight can itself be dominated by another subproblem's solution, the per-subproblem store is not guaranteed to be a clean nondominated set; so I also keep an external population $EP$ — for each generated $y$, remove from $EP$ anything $F(y)$ dominates and add $F(y)$ unless some $EP$ member already dominates it — and $EP$ is the reported front. For three or more objectives, where the Tchebycheff spread can be uneven, the aggregation can be swapped for the penalty-based boundary intersection $g^{pbi} = d_1 + \theta d_2$, with $d_1$ the signed distance of $F(x) - z^*$ along the reference direction $w/\|w\|$ and $d_2$ the perpendicular distance off it, which gives a smoother, more even spread for a penalty $\theta > 0$.

```python
import numpy as np


def das_dennis_weights(n_partitions, n_obj):
    # uniform simplex lattice: components are multiples of 1/H summing to 1
    def rec(acc, w, left, depth):
        if depth == n_obj - 1:
            w[depth] = left / n_partitions
            acc.append(w.copy()); return
        for i in range(left + 1):
            w[depth] = i / n_partitions
            rec(acc, w.copy(), left - i, depth + 1)
    acc = []
    rec(acc, np.zeros(n_obj), n_partitions, 0)
    W = np.clip(np.array(acc), 1e-6, None)      # keep every objective visible to max
    return W / W.sum(axis=1, keepdims=True)     # N = C(H+m-1, m-1) vectors


def tchebycheff(F, w, z):
    return np.max(w * np.abs(F - z), axis=-1)   # max_i w_i |f_i - z*_i|


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def ep_update(EP, fy):
    EP = [g for g in EP if not dominates(fy, g)]
    if not any(dominates(g, fy) for g in EP):
        EP.append(fy.copy())
    return EP


def sbx_crossover(p1, p2, xl, xu, eta, pc, rng):
    c = p1.copy()
    if rng.random() <= pc:
        for i in range(len(p1)):
            if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
                x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                u = rng.random()
                if rng.random() <= 0.5:                      # lower child c1
                    beta = 1.0 + 2.0 * (x1 - xl[i]) / (x2 - x1)
                    alpha = 2.0 - beta ** (-(eta + 1))
                    bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha \
                        else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
                    c[i] = 0.5 * ((x1 + x2) - bq * (x2 - x1))
                else:                                        # upper child c2
                    beta = 1.0 + 2.0 * (xu[i] - x2) / (x2 - x1)
                    alpha = 2.0 - beta ** (-(eta + 1))
                    bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha \
                        else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
                    c[i] = 0.5 * ((x1 + x2) + bq * (x2 - x1))
    return np.clip(c, xl, xu)


def polynomial_mutation(x, xl, xu, eta, rng):
    x = x.copy(); pm = 1.0 / len(x)
    for i in range(len(x)):
        if rng.random() <= pm:
            d1 = (x[i] - xl[i]) / (xu[i] - xl[i])
            d2 = (xu[i] - x[i]) / (xu[i] - xl[i])
            u, mp = rng.random(), 1 / (eta + 1)
            if u < 0.5:
                dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** mp - 1
            else:
                dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** mp
            x[i] = np.clip(x[i] + dq * (xu[i] - xl[i]), xl[i], xu[i])
    return x


def moead(F_eval, n_var, xl, xu, n_partitions=99, T=20, n_gen=250,
          eta_c=20, pc=1.0, eta_m=20, prob_nb=0.9, seed=1):
    rng = np.random.default_rng(seed)
    W = das_dennis_weights(n_partitions, 2)                 # N subproblems
    N = len(W)
    B = np.argsort(((W[:, None, :] - W[None, :, :]) ** 2).sum(-1), axis=1)[:, :T]
    X = rng.uniform(xl, xu, (N, n_var))                     # one solution per subproblem
    FV = F_eval(X)
    z = FV.min(axis=0)                                      # ideal-point estimate
    EP = []
    for _ in range(n_gen):
        for i in rng.permutation(N):
            pool = B[i] if rng.random() < prob_nb else np.arange(N)
            k, l = rng.choice(pool, 2, replace=False)
            child = sbx_crossover(X[k], X[l], xl, xu, eta_c, pc, rng)
            y = polynomial_mutation(child, xl, xu, eta_m, rng)
            fy = F_eval(y[None, :])[0]
            z = np.minimum(z, fy)                           # z_j <- min(z_j, f_j(y))
            g_new = tchebycheff(fy,       W[B[i]], z)
            g_old = tchebycheff(FV[B[i]], W[B[i]], z)
            repl = B[i][g_new <= g_old]                     # replace neighbors y beats
            X[repl] = y; FV[repl] = fy
            EP = ep_update(EP, fy)
    return X, FV, np.array(EP)


def zdt1(X):
    f1 = X[:, 0]
    g = 1 + 9 * X[:, 1:].sum(axis=1) / (X.shape[1] - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


if __name__ == "__main__":
    X, F, EP = moead(zdt1, 30, np.zeros(30), np.ones(30),
                     n_partitions=99, n_gen=250, seed=1)
    d = np.abs(F[:, 1] - (1 - np.sqrt(np.clip(F[:, 0], 0, 1))))   # vertical residual to true PF f2=1-sqrt(f1)
    print(f"pop {len(F)}  EP {len(EP)}  mean vert. residual to PF {d.mean():.2e}  f1∈[{F[:,0].min():.2f},{F[:,0].max():.2f}]")
```

For three or more objectives, swap the Tchebycheff aggregation for PBI:

```python
def pbi(F, w, z, theta=5.0):
    wn = w / np.linalg.norm(w, axis=-1, keepdims=True)
    d1 = np.sum((F - z) * wn, axis=-1)                  # along the reference direction
    d2 = np.linalg.norm((F - z) - d1[..., None] * wn, axis=-1)   # perpendicular off it
    return d1 + theta * d2
```
