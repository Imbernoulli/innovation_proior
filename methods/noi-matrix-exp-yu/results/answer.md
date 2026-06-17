# Matrix power for walk counting and linear recurrences

## Problem

Count the number of walks of length exactly $N$ from a source $s$ to a target $t$ in a
directed graph on $d$ vertices, or evaluate a $d$-term linear recurrence
$a_n = c_1 a_{n-1} + \dots + c_d a_{n-d}$ at index $N$, with $N$ up to $10^{18}$ and
answers taken modulo $m$.

## Key idea

One step of the process is multiplication by a **fixed $d \times d$ transition matrix
$T$**, identical at every index. Stacking $d$ consecutive states into a column vector
$v_n = (a_n, a_{n-1}, \dots, a_{n-d+1})^\top$, the recurrence becomes $v_n = T\,v_{n-1}$
with

$$T = \begin{pmatrix} c_1 & c_2 & \cdots & c_d \\ 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & 1 & 0 \end{pmatrix}
\quad(\text{companion matrix}),$$

so $v_N = T^{\,N-(d-1)} v_{d-1}$ and $a_N$ is its top entry. For walk counting $T$ is just
the edge-count (adjacency) matrix $G$ with $G[i][j] = $ number of edges $i \to j$; by the
midpoint identity $G^k[i][j] = \sum_l G^{k-1}[i][l]\,G[l][j]$ counts length-$k$ walks, so the
answer is the single entry $(G^N)[s][t]$.

Matrix multiplication $C[i][j] = \sum_k A[i][k]\,B[k][j]$ is **associative**
($(AB)C = A(BC)$, by reordering the finite double sum $\sum_l\sum_k A[i][k]B[k][l]C[l][j]
= \sum_k\sum_l$), so the $N$-fold product $T^N$ is well defined and may be regrouped. A
forward scan computing it as $T \cdot T \cdots T$ costs $N$ products and is hopeless at
$N = 10^{18}$.

**Binary (fast) exponentiation.** Using $T^N = (T^{\lfloor N/2\rfloor})^2$ (times $T$ once
more if $N$ is odd), walk the bits of $N$: keep an accumulator starting at the identity
matrix $I$, square a running power $T, T^2, T^4, \dots$ once per bit, and fold it into the
accumulator on each set bit. This reaches exponent $N$ in $O(\log N)$ matrix products.

**Modulus.** Reduction mod $m$ commutes with $+$ and $\times$, so every matrix entry is kept
in $[0, m)$ throughout; the exact (exponentially large) counts are never formed.

## Complexity

One $d \times d$ product is $O(d^3)$; binary exponentiation does $O(\log N)$ of them, for
$O(d^3 \log N)$ time and $O(d^2)$ space. At $N = 10^{18}$ that is about sixty bit rounds,
with at most one extra fold per round, instead of $10^{18}$ scan steps.

## Code

```python
def walks(adj,s,t,N,m):
    """Number of length-N walks s -> t in a directed graph (adj[i][j] = #edges i->j)."""
    d = len(adj)

    def mul(A, B):
        p, q, r = len(A), len(B), len(B[0])
        C = [[0] * r for _ in range(p)]
        for i in range(p):
            Ai, Ci = A[i], C[i]
            for k in range(q):
                a = Ai[k]
                if a == 0:
                    continue
                Bk = B[k]
                for j in range(r):
                    Ci[j] = (Ci[j] + a * Bk[j]) % m
        return C

    R = [[1 % m if i == j else 0 for j in range(d)] for i in range(d)]
    B = [[x % m for x in row] for row in adj]
    while N > 0:
        if N & 1:
            R = mul(R, B)
        B = mul(B, B)
        N >>= 1
    return R[s][t]


def linrec(coeffs,init,N,m):
    """Evaluate a_n = sum_j coeffs[j]*a_{n-1-j} at index N; init = [a_0, ..., a_{d-1}]."""
    d = len(coeffs)
    if N < d:
        return init[N] % m

    def mul(A, B):
        p, q, r = len(A), len(B), len(B[0])
        C = [[0] * r for _ in range(p)]
        for i in range(p):
            Ai, Ci = A[i], C[i]
            for k in range(q):
                a = Ai[k]
                if a == 0:
                    continue
                Bk = B[k]
                for j in range(r):
                    Ci[j] = (Ci[j] + a * Bk[j]) % m
        return C

    def power(M, exponent):
        R = [[1 % m if i == j else 0 for j in range(d)] for i in range(d)]
        B = [[x % m for x in row] for row in M]
        while exponent > 0:
            if exponent & 1:
                R = mul(R, B)
            B = mul(B, B)
            exponent >>= 1
        return R

    T = [[0] * d for _ in range(d)]
    T[0] = [c % m for c in coeffs]
    for i in range(1, d):
        T[i][i - 1] = 1
    v0 = [[init[d - 1 - i] % m] for i in range(d)]
    vN = mul(power(T, N - (d - 1)), v0)
    return vN[0][0]
```
