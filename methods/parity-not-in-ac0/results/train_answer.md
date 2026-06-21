I want to show, with a quantitative size–depth tradeoff, that the parity function $\mathrm{PARITY}(x_1,\dots,x_n)=x_1\oplus\cdots\oplus x_n$ cannot be computed by polynomial-size, constant-depth, unbounded-fan-in $\{\wedge,\vee,\neg\}$ circuits, the class $AC^0$. The only structural feature I can exploit is shallowness: a depth-$k$ circuit is an alternating tree of $\wedge$- and $\vee$-levels with literals at the leaves, after pushing negations to the inputs by De Morgan and merging adjacent gates of the same type. The strategy is therefore forced: peel off levels one at a time until the circuit becomes depth $2$, where parity is known to be exponentially hard, and argue that the original circuit must already have been huge.

The basic depth-reduction move is simple. A bottom depth-$2$ block is either an AND-of-ORs or an OR-of-ANDs. If I could rewrite an AND-of-ORs block as an equivalent OR-of-ANDs, its top gate changes from $\wedge$ to $\vee$, which then sits next to the $\vee$-level immediately above it; the two like layers merge and the total depth drops by one. Any Boolean function can be written in either normal form, so the rewrite is always possible, but naive distribution is ruinously expensive: an AND of $w$ ORs of fan-in $t$ can explode into $t^w$ terms. I need the block to be *cheap* to switch, and cheapness is captured by decision-tree depth. If a function has a decision tree of depth at most $s$, then reading off its $1$-paths gives a width-$s$ DNF and reading off the negated $0$-paths gives a width-$s$ CNF, so the same function is simultaneously an OR-of-small-ANDs and an AND-of-small-ORs. Thus decision-tree depth is the right currency: I need every bottom block to collapse to a shallow decision tree.

The blocks are not shallow as given, but I can force simplicity by fixing some input variables to constants and studying the induced sub-function on the surviving variables. This is where parity’s rigidity becomes decisive. For any restriction $\rho$, the restricted parity $\mathrm{PARITY}|_\rho$ is still $\pm\mathrm{PARITY}$ on the free variables; it never simplifies, still needs decision-tree depth equal to the number of survivors, and still needs DNF/CNF size $2^{m-1}$ on $m$ free variables. By contrast, a small low-width AND-of-ORs, once most of its variables are fixed, almost surely loses clause after clause because a single literal set the killing way forces an OR gate to a constant. Fixing variables is therefore a wedge that melts any small circuit while leaving parity fully complex. Since no explicit fixing is known to simplify every block, I use the probabilistic method: a random restriction $\rho\in R_p$ keeps each variable free with probability $p$ and otherwise sets it to $0$ or $1$ with equal probability. Small $p$ simplifies the circuit more aggressively but leaves fewer survivors for the parity base case, so $p$ is the quantity to tune.

Furst, Saxe, and Sipser pioneered this random-restriction approach, but their per-gate failure probability is only polynomially small, and a union bound over the $n^k$ gates forces the fan-in constant to grow each round, capping the result at a super-polynomial lower bound. Yao later obtained an exponential bound, but his switch is only approximate: the restricted block agrees with the other normal form on most inputs rather than equaling it, so approximation error must be carried through every layer and the constants become messy. The fix I adopt is the **Switching Lemma**, together with the random-restriction lower-bound argument it powers. It gives an *exact* switch — the restricted block genuinely equals a shallow decision tree, hence a small-width formula of either type — with an exponentially small failure probability of the form $\alpha^s$, where $s$ is the target width and $\alpha=\Theta(pt)$. This is exactly the shape needed: with $s$ logarithmic in the circuit size, the failure beats a union bound over polynomially many gates.

In its clean fixed-star form, the lemma says that for a width-$w$ DNF $F$ and a uniformly random restriction $\rho$ with exactly $r=\sigma n$ stars and $\sigma\le 1/5$, the probability that $F|_\rho$ has decision-tree depth greater than $d$ is at most $(10\,\sigma w)^d$. In Håstad’s independent-$R_p$ minterm form, for an AND of ORs of fan-in at most $t$, the probability that $G|_\rho$ is not an OR of ANDs of width less than $s$ is at most $\alpha^s$, where $\alpha$ is the unique positive root of $\big(1+\frac{4p}{(1+p)\alpha}\big)^t=\big(1+\frac{2p}{(1+p)\alpha}\big)^t+1$, and asymptotically $\alpha=\frac{2}{\ln\phi}\,pt+O(p^2t)<5pt$ with $\phi=\frac{1+\sqrt5}{2}$. The crucial feature is that the bound depends only on the *width* $w$, not on the size of the DNF.

The proof is a compression argument. Build the canonical decision tree of $F|_\rho$ by scanning terms in order, taking the first term not killed by $\rho$, querying all its surviving variables, and recursing off the unique branch that satisfies that term. A bad restriction is one whose canonical tree has a root path longer than $d$; trim that path to a partial assignment $\pi$ fixing exactly $d$ variables while leaving $F|_{\rho\pi}$ non-constant. The naive encoding $\rho\mapsto\rho\pi$ is too costly because pointing out which $d$ of the $n$ variables are the $\pi$-variables consumes about $d\log n$ bits, canceling the gain from $d$ fewer stars. The trick is to encode block by block, recording not $\pi$ itself but $\gamma_i$, the assignment that *satisfies* the $i$-th touched term. Then the encoded restriction makes that term the first satisfied term, so the decoder can locate it without extra information, and only $d\log w$ bits are needed to identify which of the at most $w$ variables inside each term were fixed, plus one bit each to recover $\pi_i$ from $\gamma_i$. This gives an injection from bad restrictions into $R_{r-d}\times\{0,1\}^{d\log w+2d}$, and counting restrictions with exactly $j$ stars yields the bound $(10\,\sigma w)^d$. Håstad’s induction refines the same idea into an exact algebraic constant.

With the lemma in hand, the lower bound assembles cleanly. Start with a depth-$k$ circuit of size $S$ computing parity. A preliminary restriction at $p_0=1/100$ chops every bottom gate to fan-in at most $w=20\log S$: a gate with larger fan-in survives unforced with probability at most $((1+p_0)/2)^L\le 0.505^w\ll 1/S$, so a union bound kills all initially wide gates. Then repeat $k-2$ times: apply a fresh restriction with star-fraction $\sigma=1/(20w)$ and switching target $s=w$. Each bottom width-$w$ block fails to collapse to a depth-$w$ decision tree with probability $(10\,\sigma w)^w=(1/2)^{20\log S}=S^{-20}$, so a union bound over the at most $S$ blocks switches all of them simultaneously. Each switched block becomes a width-$w$ CNF, the adjacent $\vee$-levels merge, and depth drops by one while bottom fan-in stays bounded by $w$. After $k-2$ rounds the circuit is depth $2$ on roughly $m=n/(400\log S)^{k-2}$ live variables and still computes $\pm\mathrm{PARITY}_m$. Lupanov’s base case forces bottom fan-in at least $m$ and size at least $2^{m-1}$ for any depth-$2$ parity circuit, so $20\log S=w\ge m$. Hence $(400\log S)^{k-1}\gtrsim n$, giving $\log S\ge\Omega(n^{1/(k-1)})$ and $S\ge 2^{\Omega(n^{1/(k-1)})}$. For polynomial size, where $\log S=O(\log n)$, this forces $k\ge\Omega(\log n/\log\log n)$, which tends to infinity; therefore no constant depth with polynomial size can compute parity, and $\mathrm{PARITY}\notin AC^0$. This is tight up to constants in the exponent, matching the known depth-$k$ construction of size roughly $n\,2^{n^{1/(k-1)}}$.

The following Python script illustrates the two phenomena on small instances: it verifies that parity remains parity under any restriction, and it empirically checks that random restrictions collapse small-width DNFs with failure probability consistent with the switching bound.

```python
import itertools
import random


def random_restriction(n, p):
    rho = []
    for _ in range(n):
        u = random.random()
        if u < p:
            rho.append('*')
        elif u < (1 + p) / 2:
            rho.append(0)
        else:
            rho.append(1)
    return rho


def random_dnf_table(n, w, num_terms):
    table = 0
    for assignment in range(1 << n):
        value = 0
        for _ in range(num_terms):
            term_value = 1
            vars = random.sample(range(n), w)
            for v in vars:
                lit = random.choice([0, 1])
                if ((assignment >> v) & 1) != lit:
                    term_value = 0
                    break
            if term_value:
                value = 1
                break
        if value:
            table |= (1 << assignment)
    return table


def apply_restriction(table, n, rho):
    free = [i for i, v in enumerate(rho) if v == '*']
    m = len(free)
    new_table = 0
    for x in range(1 << m):
        assignment = 0
        for j, var in enumerate(free):
            if (x >> j) & 1:
                assignment |= (1 << var)
        for i, v in enumerate(rho):
            if v != '*':
                assignment |= (v << i)
        if (table >> assignment) & 1:
            new_table |= (1 << x)
    return new_table, m


def parity_table(n):
    table = 0
    for a in range(1 << n):
        if bin(a).count('1') % 2 == 1:
            table |= (1 << a)
    return table


def dt_depth(table, n, memo=None):
    if memo is None:
        memo = {}
    if n == 0:
        return 0
    const = (table == 0) or (table == (1 << (1 << n)) - 1)
    if const:
        return 0
    key = (table, n)
    if key in memo:
        return memo[key]
    best = n
    for i in range(n):
        t0, t1 = 0, 0
        for a in range(1 << n):
            bit = (table >> a) & 1
            low = a & ((1 << i) - 1)
            high = a >> (i + 1)
            b = low | (high << i)
            if (a >> i) & 1:
                t1 |= (bit << b)
            else:
                t0 |= (bit << b)
        d = 1 + max(dt_depth(t0, n - 1, memo), dt_depth(t1, n - 1, memo))
        if d < best:
            best = d
    memo[key] = best
    return best


def empirical_switch_failure(n, w, p, d, trials, num_terms):
    fail = 0
    for _ in range(trials):
        table = random_dnf_table(n, w, num_terms)
        rho = random_restriction(n, p)
        restricted, m = apply_restriction(table, n, rho)
        if m == 0:
            continue
        if dt_depth(restricted, m) > d:
            fail += 1
    return fail / trials


if __name__ == "__main__":
    random.seed(0)
    n = 6
    parity = parity_table(n)
    for _ in range(200):
        rho = random_restriction(n, 0.5)
        restricted, m = apply_restriction(parity, n, rho)
        if m == 0:
            continue
        expected = parity_table(m)
        same = restricted == expected
        neg = restricted == ((1 << (1 << m)) - 1) ^ expected
        assert same or neg
    print("Parity restriction invariance verified for small instances.")

    n, w, p, d, num_terms = 10, 3, 0.03, 2, 8
    trials = 1000
    emp = empirical_switch_failure(n, w, p, d, trials, num_terms)
    bound = (10 * p * w) ** d
    print(f"n={n}, w={w}, p={p}, d={d}: empirical failure={emp:.4f}, bound={bound:.4f}")
```
