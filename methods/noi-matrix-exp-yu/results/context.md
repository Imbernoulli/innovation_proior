## Problem

You are given a directed graph on $d$ vertices, where each ordered pair $(i, j)$ carries a nonnegative integer count of edges from vertex $i$ to vertex $j$. Fix a source vertex $s$, a target vertex $t$, and an integer $N$. Count the number of **walks of length exactly $N$** from $s$ to $t$ — sequences $s = u_0 \to u_1 \to \dots \to u_N = t$ where each $u_{r-1} \to u_r$ is an edge, walks that revisit vertices or reuse edges are allowed and counted with multiplicity, and each step's choice is independent.

A second input form gives a $d$-term linear recurrence
$$a_n = c_1\, a_{n-1} + c_2\, a_{n-2} + \dots + c_d\, a_{n-d}$$
with fixed integer coefficients $c_1, \dots, c_d$ and $d$ given initial terms $a_0, a_1, \dots, a_{d-1}$ (a generalized Fibonacci sequence), and you must evaluate $a_N$.

The length or index $N$ can be as large as $10^{18}$. To keep the numbers bounded, every answer is reported modulo a given positive integer $m$ (the exact counts can grow extremely quickly, so the value modulo $m$ is what is required).

## Code framework

For the walk-counting form, the graph is an edge-count table `adj`, where
`adj[i][j]` is the number of edges from vertex `i` to vertex `j`; `s`, `t`,
`N`, and `m` are integers. For the recurrence form, `coeffs = [c_1, ..., c_d]`
and `init = [a_0, ..., a_{d-1}]` are lists, with `N` and `m` again integers.

```python
def walks(adj,s,t,N,m):
    """Number of length-N walks s -> t in a directed graph (adj[i][j] = #edges i->j)."""
    # TODO
    pass


def linrec(coeffs,init,N,m):
    """Evaluate a_n = sum_j coeffs[j]*a_{n-1-j} at index N; init = [a_0, ..., a_{d-1}]."""
    # TODO
    pass
```
