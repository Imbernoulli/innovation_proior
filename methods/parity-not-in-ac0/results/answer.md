# PARITY ∉ AC⁰, via random restrictions and the Switching Lemma

## Problem

Show that $\mathrm{PARITY}(x_1,\dots,x_n)=x_1\oplus\cdots\oplus x_n$ cannot be computed by polynomial-size, constant-depth circuits of unbounded-fan-in $\wedge,\vee$ gates with $\neg$ (the class $AC^0$) — and quantify the size–depth tradeoff.

## Key idea

A **random restriction** drives a wedge between parity and any small circuit. Setting each variable free with probability $p$ and to a random constant otherwise:

- **Parity is invariant**: $\mathrm{PARITY}|_\rho=\pm\mathrm{PARITY}$ on the surviving variables — it never simplifies, and on $m$ free variables it still needs decision-tree depth $m$ and DNF/CNF size $2^{m-1}$.
- **A small low-width depth-2 block collapses**: with high probability $G|_\rho$ has a tiny decision tree, or in Håstad's independent-$R_p$ form all minterms of the switched block are small. A depth-$s$ decision tree is *simultaneously* a width-$s$ DNF and a width-$s$ CNF, and the minterm form directly gives the OR-of-small-ANDs representation needed to merge with the layer above.

Iterating $k-2$ restrictions collapses a depth-$k$ circuit to depth 2 while keeping a $\sigma$-fraction of variables; the surviving function is $\pm\mathrm{PARITY}$, which depth 2 cannot do without exponential size. The engine is a sharp, *exact* switching estimate (Håstad), strictly stronger than the polynomial-failure estimate of Furst–Saxe–Sipser (super-polynomial only) and the approximate, non-sharp estimate of Yao (exponential but messy).

## The Switching Lemma

**Lemma.** Let $F$ be a width-$w$ DNF on $n$ variables and let $\rho$ be a uniformly random restriction with exactly $r=\sigma n$ stars, where $\sigma\le 1/5$. Then for every $0\le d\le r$,
$$
\Pr_\rho\big[\,\mathrm{DTdepth}(F|_\rho)>d\,\big]\ \le\ (10\,\sigma w)^{d}.
$$
In Håstad's independent-$R_p$ parameterization: for an AND of ORs of fan-in $\le t$ hit by $\rho\in R_p$, the probability that $G|_\rho$ is not an OR of ANDs all of size $<s$ is $\le \alpha^s$, where $\alpha$ is the unique positive root of
$$
\Big(1+\tfrac{4p}{(1+p)\alpha}\Big)^{t}=\Big(1+\tfrac{2p}{(1+p)\alpha}\Big)^{t}+1,
\qquad \alpha=\frac{2}{\ln\phi}\,pt+O(p^2t)<5pt,
$$
with $\phi=\tfrac{1+\sqrt5}{2}$ the golden ratio. The bound is independent of the *size* of $F$ — only its width matters.

**Proof (Razborov/Beame encoding form).** Let $B=\{\rho:\mathrm{DTdepth}(F|_\rho)>d\}$. Build the canonical decision tree of $F|_\rho$: scan terms in order, take the first not killed, query all its surviving variables; the unique branch satisfying it is a 1-leaf, recurse on the rest. A bad $\rho$ has a root path of length $>d$; trim it to a partial assignment $\pi$ fixing exactly $d$ variables, with $F|_{\rho\pi}$ non-constant. Encode $\rho$ by $\rho\gamma_1\cdots\gamma_\ell\in R_{r-d}$ — where each $\gamma_i$ is the assignment *satisfying* the $i$-th term touched, so the encoded restriction makes that term the first satisfied term and the decoder can locate it — plus $d\log w+2d$ auxiliary bits (which $\le w$ variables of each term are fixed, and how $\pi$ vs $\gamma$ set them). This is injective $B\hookrightarrow R_{r-d}\times\{0,1\}^{d\log w+2d}$, so
$$
\Pr[\text{bad}]\le\frac{\binom{n}{r-d}2^{n-(r-d)}(4w)^d}{\binom{n}{r}2^{n-r}}
=\frac{\binom{n}{r-d}}{\binom{n}{r}}2^d(4w)^d
\le\Big(\tfrac{r}{n-r}\Big)^d 2^d(4w)^d=\Big(\tfrac{8\sigma w}{1-\sigma}\Big)^d\le(10\sigma w)^d.
$$

**Proof (Håstad's induction, exact constant).** Stronger statement: $G=\bigwedge_{i=1}^q G_i$ (each $G_i$ an OR of fan-in $\le t$), $F$ arbitrary, $\rho\in R_p$; then $\Pr[\mathrm{min}(G)\ge s\mid F|_\rho\equiv1]\le\alpha^s$. Induct on $q$. Base $q=0$: $G\equiv1$, probability $0$. Step: condition on whether $G_1|_\rho\equiv1$.
- If yes, $G|_\rho=\bigwedge_{i\ge2}G_i|_\rho$ with conditioning $(F\wedge G_1)|_\rho\equiv1$; IH on $q-1$ gives $\le\alpha^s$ (the arbitrary $F$ in the hypothesis is exactly what lets the peeled clause fold into the conditioning).
- If no, $G_1=\bigvee_{i\in T}x_i$, $|T|\le t$; every minterm must set some $T$-variable to $1$. Partition minterms by the touched set $Y\subseteq T$ (those vars were stars). $\Pr[\rho_1(Y)=\ast\mid G_1|_{\rho_1}\not\equiv1]=(\tfrac{2p}{1+p})^{|Y|}$, and conditioning on $F|_\rho\equiv1$ only lowers it (Saks: $\Pr[A\mid B\wedge C]\le\Pr[A\mid C]\iff\Pr[B\mid A\wedge C]\le\Pr[B\mid C]$; forcing more stars cannot increase the chance a function is pinned to $1$). For the second factor, the $Y$ assignment has $2^{|Y|}-1$ nonzero sign patterns, variables in $T\setminus Y$ are excluded from the minterm and can be frozen/maximized over, and IH on the remaining product gives $\alpha^{s-|Y|}$. Summing,
$$
\sum_{Y\subseteq T}\big(\tfrac{2p}{1+p}\big)^{|Y|}(2^{|Y|}-1)\alpha^{s-|Y|}
=\alpha^s\Big[\big(1+\tfrac{4p}{(1+p)\alpha}\big)^{|T|}-\big(1+\tfrac{2p}{(1+p)\alpha}\big)^{|T|}\Big]\le\alpha^s,
$$
by the defining equation for $\alpha$ (with $|T|\le t$). Taking $F\equiv1$ recovers the exact switching lemma: if there is no minterm of size $\ge s$, then $G|_\rho$ is an OR of ANDs of width $<s$.

## The PARITY lower bound

**Theorem (Håstad).** A depth-$k$ unbounded-fan-in AND/OR circuit computing $\mathrm{PARITY}_n$ has size $S\ge 2^{\Omega(n^{1/(k-1)})}$. Hence a polynomial-size parity circuit needs depth $\ge \dfrac{\log n}{c+\log\log n}$, so $\mathrm{PARITY}\notin AC^0$. (Tight: parity has depth-$k$ circuits of size $\sim n\,2^{n^{1/(k-1)}}$.)

**Proof.** Normalize to a leveled alternating tree, literals at leaves. One preliminary restriction at $p_0=\tfrac1{100}$ reduces every bottom gate to fan-in $\le w:=20\log S$: a bottom gate of fan-in $L>w$ survives unforced with probability at most $\big(\tfrac{1+p_0}{2}\big)^L\le0.505^w\ll1/S$, so a union bound kills every initially wide bottom gate. Then repeat $k-2$ times: apply $\rho\in R_p$ with $\sigma=\tfrac1{20w}$, target $d=s=w$. Each bottom width-$w$ DNF fails to become a depth-$w$ decision tree with probability $(10\sigma w)^w=(\tfrac12)^{20\log S}=S^{-20}$; union over $\le S$ bottom blocks $\Rightarrow$ all switch with probability $\ge 1-S^{-19}>0$. A depth-$w$ DT is a width-$w$ CNF; merging the adjacent like layers drops depth to $k-1$, and Håstad's bookkeeping keeps bottom fan-in $\le w$ while the number of depth-at-least-2 subcircuits does not increase. After $k-2$ rounds the circuit is depth 2 on
$$
m\approx n\,\sigma^{k-2}=\frac{n}{(400\log S)^{k-2}}
$$
live variables, computing $\pm\mathrm{PARITY}_m$. Lupanov gives both depth-2 size $\ge 2^{m-1}$ and bottom fan-in $\ge m$. The bottom-fan-in form gives $20\log S=w\ge m$, hence $(400\log S)^{k-1}\ge n$; the size shorthand writes the same base case as $S\ge2^{m-1}$ and gives $\log S\ge\Omega(m)$. Thus $\log S\ge\Omega(n^{1/(k-1)})$. For poly size ($\log S=O(\log n)$) this forces $k\ge\Omega(\log n/\log\log n)$. $\qquad\blacksquare$

## Worked spot-check

A small, exact computation confirms the two facts the proof rests on — a small-width DNF collapses to tiny decision-tree depth under a low-$\sigma$ random restriction, while parity always needs full decision-tree depth.

```python
import random
from itertools import product
from functools import lru_cache

def parity(bits):
    s = 0
    for b in bits:
        s ^= b
    return s

def restrict(values, rho):
    """Induced truth table over the free ('*') variables, as a flat output tuple."""
    free = sorted(i for i, v in rho.items() if v == '*')
    table = []
    for assign in product((0, 1), repeat=len(free)):
        full = dict(rho)
        for i, b in zip(free, assign):
            full[i] = b
        table.append(values(tuple(full[i] for i in sorted(full))))
    return tuple(table), free

@lru_cache(maxsize=None)
def dt_depth(table):
    """Optimal decision-tree depth from a 2^m truth table (memoized on subfunction).
    Uses: DTdepth(f) <= t  ==>  f is both a width-t DNF and a width-t CNF."""
    if all(b == table[0] for b in table):
        return 0
    m = len(table).bit_length() - 1
    best = m
    for v in range(m):
        half = 1 << (m - 1 - v)
        t0, t1 = [], []
        for blk in range(0, len(table), 2 * half):
            t0.extend(table[blk:blk + half])
            t1.extend(table[blk + half:blk + 2 * half])
        best = min(best, 1 + max(dt_depth(tuple(t0)), dt_depth(tuple(t1))))
    return best

def random_restriction(n, sigma):
    return {i: ('*' if random.random() < sigma else random.randint(0, 1)) for i in range(n)}

def small_dnf(terms):
    def f(bits):
        return int(any(all(bits[i] == b for i, b in t.items()) for t in terms))
    return f

if __name__ == "__main__":
    random.seed(0)
    n, t = 14, 3
    terms = [{random.randrange(n): random.randint(0, 1) for _ in range(t)} for _ in range(40)]
    f, sigma = small_dnf(terms), 1 / (20 * t)
    depths = [dt_depth(restrict(f, random_restriction(n, sigma))[0]) for _ in range(300)]
    print("small DNF restricted: max DT-depth seen =", max(depths))      # tiny: switching lemma

    for rho_values in product((0, 1, '*'), repeat=6):
        table, free = restrict(parity, dict(enumerate(rho_values)))
        assert dt_depth(table) == len(free)                              # parity never trivializes
    print("PARITY: DT-depth == #free vars on all 3^6 restrictions")
```
