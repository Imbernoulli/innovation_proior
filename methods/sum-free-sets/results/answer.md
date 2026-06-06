# Large sum-free sets in sets of integers, and counting sum-free sets

## Problem

A set is *sum-free* if it has no solution to $x+y=z$. The two targets are:

1. For every set $A$ of $n$ nonzero integers, lower-bound the largest sum-free subset size $s(A)$.
2. Count the sum-free subsets of $[N]=\{1,\dots,N\}$.

## Method

**Random dilation.** The arc $B=(1/3,2/3)\subset\mathbb{T}$ is sum-free, and for a random $\theta$ each nonzero integer $a$ has $\{\theta a\}$ uniformly distributed. Thus
$$A_\theta=\{a\in A:\{\theta a\}\in B\}$$
is sum-free for every $\theta$ and $\mathbb{E}|A_\theta|=n/3$, giving $s(A)\ge n/3$.

**Integrality.** With $f=1_{[1/3,2/3)}-\tfrac13$ and $f_A(x)=\sum_{a\in A}f(ax)$, one has $|A_x|=n/3+f_A(x)$. Since $|A_x|$ is integer-valued, has average $n/3$, and is not constant, its maximum is at least $\lfloor n/3\rfloor+1=\lceil(n+1)/3\rceil$. Hence $s(A)\ge(n+1)/3$.

**Fourier surplus.** The exact expansion is
$$f(x)=-\frac{\sqrt3}{\pi}\sum_{m\ge1}\frac{\chi(m)}m\cos(2\pi m x),$$
where $\chi$ is the nonprincipal character mod $3$. An $L^1$ surplus estimate gives
$$s(A)\ge\frac n3+C\left\|\sum_{a\in A}\cos(2\pi a\,\cdot)\right\|_{L^1(\mathbb{T})}.$$
For structured sets, nonnegative test functions certify $m_A=\max f_A$: if $\varphi\ge0$ and $\int\varphi=1$, then $m_A\ge\int\varphi f_A$. In the case $1\notin A$, with $u=\min A$ and $v$ the smallest element not divisible by $u$,
$$\varphi=(1-\cos 2\pi ux)(1-\cos 2\pi vx)$$
gives
$$\int\varphi f_A=\frac{\sqrt3}{2\pi}\left(2-\frac12 1_A(u+v)\right)>1/3.$$

**Mod-3 descent.** Split $A=A_0\sqcup A_1$ into elements coprime to $3$ and divisible by $3$. Then
$$f_A(x)+f_A(x+1/3)+f_A(x+2/3)=3f_{A_1}(x),$$
so $m_A\ge m_{A_1}$. Also
$$m_A\ge\frac{|A_0|}{6}-\frac{|A_1|}{3}.$$
This reduces the fixed-surplus conjecture $m_A\ge S/3$ to finitely many sizes. For $S=2$, the remaining $n=5,8$ cases are handled by explicit nonnegative trigonometric certificates. Therefore, for coprime positive $A$, either $A=\{1,2\}$ or
$$s(A)\ge\frac{n+2}{3}.$$

**Counting.** The container strategy embeds $[N]$ into $\mathbb{Z}/p\mathbb{Z}$, partitions into arithmetic progressions, and chooses a good common difference $d$ satisfying
$$\left\|\frac{dr}{p}\right\|\le\frac1{4L}\left(\frac{\delta p}{|\widehat A(r)|}\right)^{1/2}$$
on every large Fourier coefficient. The granularization covers all but $\epsilon p$ points of a sum-free set and has at most $\epsilon p^2$ triples; all such containers form a family of size $2^{o(N)}$. A popular-difference/Kneser structure theorem says every large almost-sum-free container is essentially contained in a short interval or is essentially all odd. Pairing arguments discard the exceptional choices, leaving only entirely odd sets and sets inside $\{\lceil(N+1)/3\rceil,\dots,N\}$. Cameron-Erdos count the latter, giving
$$|\mathrm{SF}(N)|\sim c(N)2^{N/2},$$
with $c(N)$ depending only on the parity of $N$.

## Code

```python
import math
from fractions import Fraction
from itertools import combinations

ARC_START = Fraction(1, 3)
ARC_END = Fraction(2, 3)

def chi(m):
    r = m % 3
    return 1 if r == 1 else (-1 if r == 2 else 0)

def is_sum_free(S):
    s = set(S)
    return all((x + y) not in s for x in s for y in s)

def in_selected_arc(t):
    t %= 1
    return ARC_START <= t < ARC_END

def residues_mod_one(A, theta):
    theta = Fraction(theta)
    return [(a, (theta * a) % 1) for a in A]

def select_by_dilation(A, theta):
    return [a for a, t in residues_mod_one(A, theta) if in_selected_arc(t)]

def _candidate_thetas(A):
    points = {Fraction(0, 1)}
    for a in A:
        q = abs(a)
        if q == 0:
            continue
        for j in range(q):
            points.add(((Fraction(j, 1) + ARC_START) / q) % 1)
            points.add(((Fraction(j, 1) + ARC_END) / q) % 1)
    points = sorted(points)
    candidates = set(points)
    for i, x in enumerate(points):
        y = points[(i + 1) % len(points)]
        if i + 1 == len(points):
            y += 1
        candidates.add(((x + y) / 2) % 1)
    return sorted(candidates)

def select_filtered_subset(A):
    best_set, best_theta = [], Fraction(0, 1)
    for theta in _candidate_thetas(A):
        S = select_by_dilation(A, theta)
        if len(S) > len(best_set):
            assert is_sum_free(S)
            best_set, best_theta = S, theta
    return best_set, best_theta

def certificate_integral(A, coeffs):
    total = 0.0
    positive_coeffs = [(n, c) for n, c in coeffs.items() if n > 0]
    for a in A:
        q = abs(a)
        if q == 0:
            continue
        for n, c in positive_coeffs:
            if n % q == 0:
                m = n // q
                total += (chi(m) / m) * c
    return -math.sqrt(3) / (2 * math.pi) * total

def certificate_bound(A):
    if any(a <= 0 for a in A):
        return None
    A = sorted(set(A))
    if not A:
        return None
    u = A[0]
    v = next((x for x in A if x % u != 0), None)
    if v is None:
        return None
    coeffs = {0: 1.0, u: -1.0, v: -1.0}
    coeffs[u + v] = coeffs.get(u + v, 0.0) + 0.5
    coeffs[abs(v - u)] = coeffs.get(abs(v - u), 0.0) + 0.5
    return certificate_integral(A, coeffs)

def count_sumfree_subsets(N):
    elts = range(1, N + 1)
    return sum(is_sum_free(S) for k in range(N + 1) for S in combinations(elts, k))

def baseline_families(N):
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    upper_half = list(range(N // 2 + 1, N + 1))
    return {"odds": odds, "upper_half": upper_half}

def estimate_count(N):
    families = baseline_families(N)
    return {
        "exact": count_sumfree_subsets(N),
        "odd_family_choices": 2 ** len(families["odds"]),
        "upper_half_choices": 2 ** len(families["upper_half"]),
        "scale_2_to_N_over_2": 2 ** (N / 2),
    }

if __name__ == "__main__":
    A = [1, 2, 3, 4, 5, 6, 7]
    S, theta = select_filtered_subset(A)
    print(f"A={A}  forced subset={S}  theta={theta}  n/3={len(A)/3:.3f}")

    B = [2, 3, 5, 7, 11]
    val = certificate_bound(B)
    target = math.sqrt(3) / (2 * math.pi) * 1.5
    assert abs(val - target) < 1e-12
    print(f"certificate={val:.6f}  formula={target:.6f}  >1/3={val > 1/3}")

    for N in range(1, 11):
        data = estimate_count(N)
        ratio = data["exact"] / data["scale_2_to_N_over_2"]
        print(f"N={N:2d}  |SF(N)|={data['exact']:5d}  ratio={ratio:.3f}")
```
