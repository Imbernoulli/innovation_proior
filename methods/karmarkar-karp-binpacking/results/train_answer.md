We want to pack $n$ pieces of rational sizes in $(0,1]$ into the fewest unit-capacity bins, in polynomial time, with an output count as close to the true minimum $OPT(I)$ as we can manage. Deciding $OPT$ exactly is NP-hard, so the realistic question is how close a polynomial algorithm can get — and here the entire prior literature has the same shape. First-fit-decreasing and its greedy relatives sit between the two crude bounds $SIZE(I) \le OPT(I) \le 2\,SIZE(I) + 1$ (where $SIZE(I) = \sum_i s_i$) and chip away at a worst-case ratio, reaching about $11/9$; the asymptotic scheme of Fernandez de la Vega and Lueker breaks the constant ceiling to give $(1+\varepsilon)\,OPT(I) + O(\varepsilon^{-2})$. But every one of these is a *multiplicative* guarantee $A(I) \le C\cdot OPT(I) + o(OPT)$, and a multiplicative constant — even $1.05$ — wastes a fixed *fraction* of the bins forever. On an instance that truly needs a million bins, a $C = 1.18$ algorithm can throw away a hundred-eighty-thousand of them, and polishing $C$ never changes the shape of the bound. What I actually want is to kill the constant entirely and ask how small an *additive* remainder I can achieve: $A(I) = OPT(I) + (\text{something lower-order})$. Pushing $\varepsilon \to 0$ in the asymptotic scheme to chase this only yields slack that is a genuine power of $OPT$, like $O(OPT^{1-\delta})$, while the runtime explodes — its linear grouping cannot make the number of distinct sizes small and keep the grouping loss small at the same time. The greedy methods, meanwhile, only ever make local placement decisions; they never look at the global combinatorial structure of which pieces can share a bin, which is exactly where fractional optima turn loss into a lower-order term.

I propose the Karmarkar–Karp algorithm, which achieves $A(I) \le OPT(I) + O(\log^2 OPT(I))$ — polylogarithmic additive slack, no multiplicative constant — in polynomial time. It rests on the strongest relaxation available rather than on greedy analysis: the configuration (Gilmore–Gomory cutting-stock) LP. Bucket the pieces into $m$ distinct sizes $s_1,\dots,s_m$ with $b_i$ pieces of size $i$, and let a *configuration* be a multiset of sizes fitting in one bin, a vector $v$ with $v\cdot s \le 1$. Let $A$ be the $m \times q$ matrix whose columns are all configurations. The honest problem is the integer program "choose $x_j \ge 0$ bins of each configuration covering every piece"; dropping integrality gives

$$(\mathrm{I})\quad \min\ \mathbf{1}\cdot x \quad\text{s.t.}\quad A x \ge b,\ x \ge 0, \qquad LIN(I) := \text{opt},$$

with dual $(\mathrm{II})\ \max\ u\cdot b$ s.t. $u^{\mathsf T} A \le \mathbf{1},\ u \ge 0$, where $u_i$ is a *price* per distinct size. Two facts pin $LIN(I)$: an integer packing is itself a feasible integer $x$, so $LIN(I) \le OPT(I)$; and since every column has size-sum $s^{\mathsf T}(\text{col}) \le 1$, we get $LIN(I) = \mathbf 1\cdot x \ge (s^{\mathsf T}A)x \ge s^{\mathsf T} b = SIZE(I)$. So $SIZE(I) \le LIN(I) \le OPT(I)$, and $LIN$ — unlike $OPT$ — is a linear program.

What makes the relaxation earn its keep is how cheaply a fractional optimum rounds to an integer packing. A basic feasible solution to any LP has at most as many nonzero variables as constraints, and $(\mathrm{I})$ has $m$ covering constraints, so a basic $x$ mixes at most $m$ distinct configurations no matter how huge $q$ and $n$ are. Take $\lfloor x_j\rfloor$ bins of each configuration as the principal part; the residual instance $I'$ is covered by the fractional remainders $f_j = x_j - \lfloor x_j\rfloor \in [0,1)$, so $SIZE(I') \le \sum_j f_j < m$. Pack the residual by the better of two ways — one fresh bin per nonzero configuration (at most $m$ bins) or the half-full argument ($\le 2\,SIZE(I') + 1$ bins) — costing $\min(m,\,2\,SIZE(I')+1)$. Writing the min as $SIZE(I') + \min(m - SIZE(I'),\, SIZE(I')+1)$ and bounding a min by an average, the residual costs at most $SIZE(I') + (m+1)/2$. Principal plus residual is then $\sum_j \lfloor x_j\rfloor + \sum_j f_j + (m+1)/2 = \mathbf 1\cdot x + (m+1)/2$, so with $x$ optimal we get the central lever

$$OPT(I) \le LIN(I) + \frac{m+1}{2}.$$

The integer–fractional gap is governed by the number of *distinct sizes* $m$, not the number of pieces $n$. If $m$ were small I would be done; but real instances have $m = \Theta(n)$, where $(m+1)/2 \approx n/2$ is $\Theta(OPT)$, no better than greedy. So two obstacles remain: make $m$ small without moving $OPT$, and solve an LP with astronomically many columns $q$ in polynomial time.

For the first obstacle, the known move is *linear grouping*: sort descending, cut into consecutive groups of $k$, round every piece in a group *up* to the group's largest size (round up so a packing of the grouped instance stays feasible for the originals), and discard the top group, packing its $k$ pieces one per bin. The descending sort makes group $i$ rounded-up dominated piece-for-piece by un-rounded group $i-1$, giving $OPT(J) \le OPT(I) \le OPT(J) + k$ with $m(J) \le n/k$. But the $n/k$-vs-$k$ tradeoff balances at $k \approx \sqrt n$ and stalls at $O(\sqrt n)$ — a square root, not the logarithm I want — because linear grouping treats a fat piece and a tiny piece identically, $k$ of each per group, even though rounding up $k$ tiny pieces costs almost nothing in size while rounding up $k$ fat ones costs a lot. The fix is to group by *size budget*, not piece count. Split the pieces into dyadic size classes $I_r$ with sizes in $(2^{-(r+1)},\,2^{-r}]$, and apply grouping within class $r$ using group size $k\cdot 2^r$. A group of $k\cdot 2^r$ pieces each exceeding $2^{-(r+1)}$ carries total size $> k\cdot 2^r\cdot 2^{-(r+1)} = k/2$, so every group in every class carries roughly the same size $\approx k$ regardless of how small the pieces are. The discarded top group in class $r$ is $k\cdot 2^r$ pieces each below $2^{-r}$, hence at most $k$ bins; class $r$ leaves at most $2\,SIZE(I_r)/k$ distinct sizes. Summing over the $\lceil\log_2(1/a)\rceil$ classes (where $a(I)$ is the smallest size),

$$m(J) \le \frac{2}{k}\,SIZE(I) + \big\lceil\log_2(1/a(I))\big\rceil, \qquad OPT(J) \le OPT(I) \le OPT(J) + k\big\lceil\log_2(1/a(I))\big\rceil.$$

(A cleaner size-budget variant sweeps the sorted pieces and starts a new group whenever the accumulated size reaches $k$, giving $m(J) \le SIZE(I)/k + \ln(1/a)$ with loss $\le 2k(2+\ln(1/a))$.) The $n$ is gone: distinct sizes now scale with $SIZE/k$, the loss with $k\log(1/a)$. To keep $\log(1/a)$ from blowing up I first *eliminate small pieces* — set aside every piece of size $\le g/2$ at threshold $g = 1/SIZE(I)$, pack the rest, and reinsert the small ones greedily, opening a new bin only when forced. If reinsertion opens a bin then every bin but one is filled past $1 - g/2$, giving cost $\le (1+g)\,OPT(I) + 1 = OPT(I) + O(1)$; and afterwards $a(I) > g/2$, so $\log(1/a) = O(\log SIZE) = O(\log OPT)$. The threshold $g = 1/SIZE$ does both jobs at once.

For the second obstacle, the primal $(\mathrm{I})$ has $q \approx \infty$ columns and cannot even be written down, but the dual $(\mathrm{II})$ has only $m$ variables and $q$ constraints — precisely the situation the Grötschel–Lovász–Schrijver ellipsoid method was built for, since it needs not the constraints listed but only a *separation oracle*. Prices $u$ are dual-feasible iff no configuration is overpriced, i.e. iff the knapsack $\max\ v\cdot u$ s.t. $v\cdot s \le 1,\ v \ge 0$ integer has optimum $\le 1$; if it exceeds $1$, the optimal $v$ is exactly the violated configuration-constraint the ellipsoid wants. This pricing subproblem — the same one Gilmore and Gomory solved heuristically by column generation — read as a separation oracle is what makes the method provably polynomial. Knapsack is NP-hard, but the ellipsoid only needs approximate feasibility within its existing tolerance, so round prices down to a grid, $\bar u_i = (t/n)\lfloor n u_i/t\rfloor$, and solve by dynamic programming: with $F(\kappa)$ the minimum total size of pieces with total price $\kappa\cdot(t/n)$,

$$F(0)=0,\qquad F(\kappa) = \min_i\Big[\,F\big(\kappa - n\bar u_i/t\big) + s_i\,\Big],$$

and $\bar u$ is feasible iff $F(\kappa) > 1$ for every $\kappa$ with $\kappa\cdot t/n > 1$. After $M = 4m^2\lceil\ln(n/t)\rceil$ ellipsoid iterations the dual is solved within $t$. The at-most-$M$ configurations returned by the feasibility cuts define a finite "realized" LP indistinguishable to the ellipsoid from the full one, hence within $t$ of $LIN(I)$; a constraint-elimination procedure (partition into $m+1$ groups, drop one disjoint from the critical $m$-set, repeat) prunes to exactly $m$ constraints whose dual yields a basic primal $x = B^{-1}b$ with $\le m$ nonzero configurations of value $\le LIN(I) + h$. The whole fractional-packing subroutine runs in time $T(m,n) = O(m^8\log m\log^2 n + m^4 n\log m\log n)$ — the configuration LP is in $\mathrm P$ despite its astronomical column count, and it hands back exactly the sparse basic solution the rounding lemma needs.

The final design choice is to refuse paying the residual $(m+1)/2 \approx SIZE/k$ all at once: with grouping loss $k\log(1/a)$ pulling $k$ down and the residual pulling $k$ up, a single round balances at $O(\sqrt{OPT})$, the same wall. Instead *iterate*. Pick a small constant $k$ so the residual after one round has $SIZE(R) \le m \le SIZE/k + O(\log) \le SIZE/2 + O(\log)$ — each round roughly halves the size. Then I never pay $SIZE/k$ outright; I pay only the per-round grouping/reinsertion loss $O(\log(1/a)) = O(\log OPT)$, buy the integer $\lfloor x_j\rfloor$ bins (which count honestly toward $OPT$), and recurse on a residual of half the size. The size sequence $SIZE, SIZE/2, SIZE/4,\dots$ gives $O(\log OPT)$ rounds; the integer bins bought across all rounds telescope to $\le LIN(I) \le OPT(I)$ since each round's purchases charge against a shrinking $LIN$; and the final $O(1)$ residual is first-fit. Multiplying the two independent logarithms — one from the number of rounds (geometric size decay), one from the per-round grouping loss ($\log(1/a)$ over the dyadic classes) —

$$A(I) \le OPT(I) + \sum_{\text{rounds}} O(\log OPT) = O(\log OPT)\cdot O(\log OPT) = OPT(I) + O(\log^2 OPT(I)),$$

with the corollary $A(I) \le OPT(I) + O(\log^2 m(I))$ when distinct sizes are few (track $m$ instead of $SIZE$, halving it each round), and a time/error knob: grouping parameter $k = SIZE^\alpha$ trades runtime for $O(OPT^\alpha\log OPT)$ additive error in one round, recovering the square-root regime at $\alpha = 1/2$. To exhibit the algorithm faithfully I solve the same configuration LP the separation oracle implies — column generation whose pricing subproblem is exactly the knapsack oracle, so all $q$ configurations are never enumerated — wrapped in the recursive-rounding outer loop: group sizes, solve the fractional packing, buy $\lfloor x_j\rfloor$ bins per configuration, recurse on the residual, and first-fit the leftovers.

```python
from collections import Counter
import math
from scipy.optimize import linprog

def knapsack_oracle(prices, sizes, counts):
    # separation oracle = pricing subproblem: max-price configuration fitting a
    # unit bin; price > 1  =>  dual-infeasible, this config is a new column.
    G = 1000
    best = [(0.0, ())] * (G + 1)
    for i, (s, u, b) in enumerate(zip(sizes, prices, counts)):
        w = max(1, math.ceil(s * G))
        if w > G:
            continue
        for _ in range(b):                                   # bounded knapsack
            for g in range(G, w - 1, -1):
                if best[g - w][0] + u > best[g][0] + 1e-12:
                    cfg = dict(best[g - w][1]); cfg[i] = cfg.get(i, 0) + 1
                    best[g] = (best[g - w][0] + u, tuple(sorted(cfg.items())))
    price, cfg = max(best, key=lambda t: t[0])
    return price, dict(cfg)

def solve_fractional_packing(sizes, counts):
    # configuration LP min 1.x s.t. Ax>=b, x>=0 by COLUMN GENERATION; pricing =
    # knapsack oracle. Never enumerates all q configurations.
    m = len(sizes)
    columns = [{i: min(counts[i], int(1 // sizes[i]) or 1)} for i in range(m)]
    while True:
        A_ub = [[-(c.get(i, 0)) for c in columns] for i in range(m)]
        res = linprog([1.0] * len(columns), A_ub=A_ub, b_ub=[-c for c in counts],
                      bounds=[(0, None)] * len(columns), method="highs")
        u = [max(0.0, -y) for y in res.ineqlin.marginals]
        price, cfg = knapsack_oracle(u, sizes, counts)
        if price <= 1.0 + 1e-6 or not cfg or cfg in columns:
            return columns, res.x
        columns.append(cfg)

def reduce_distinct_sizes(items, k):
    # grouping: round each group of k up to its max, discard top group.
    s = sorted(items, reverse=True)
    if len(s) <= k:
        return [], s
    top, rest = s[:k], s[k:]
    grouped = []
    for g in range(0, len(rest), k):
        chunk = rest[g:g + k]; grouped += [chunk[0]] * len(chunk)
    return grouped, top

def first_fit(items, cap=1.0):
    bins = []
    for x in sorted(items, reverse=True):
        for bn in bins:
            if sum(bn) + x <= cap + 1e-9:
                bn.append(x); break
        else:
            bins.append([x])
    return bins

def karmarkar_karp(items, cap=1.0):
    # recursive rounding: group, solve fractional, buy floor(x_c), recurse.
    items = [x for x in items if x > 1e-12]
    if not items:
        return []
    if sum(items) <= 1.0 + 1e-9 or len(set(items)) <= 1:
        return first_fit(items, cap)
    k = max(1, int(math.isqrt(max(1, int(sum(items))))))
    grouped, discarded = reduce_distinct_sizes(items, k)
    bins = [[x] for x in discarded]
    if not grouped:
        return bins + first_fit([x for x in items if x not in discarded], cap)
    cnt = Counter(round(x, 6) for x in grouped)
    sizes, counts = list(cnt.keys()), [cnt[s] for s in cnt]
    columns, x = solve_fractional_packing(sizes, counts)
    remaining = Counter({s: counts[i] for i, s in enumerate(sizes)})
    for col, xc in zip(columns, x):
        for _ in range(int(math.floor(xc + 1e-9))):
            bn = []
            for i, a in col.items():
                take = min(a, remaining[sizes[i]])
                bn += [sizes[i]] * take; remaining[sizes[i]] -= take
            if bn:
                bins.append(bn)
    residual = [s for s, r in remaining.items() for _ in range(r)]
    return bins + karmarkar_karp(residual, cap)
```
