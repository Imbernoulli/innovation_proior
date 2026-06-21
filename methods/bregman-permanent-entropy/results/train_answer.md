The problem is to bound the permanent of an n by n zero-one matrix A in terms of its row sums r_i. The permanent counts the permutations sigma for which every entry a_{i, sigma(i)} equals one, so it is fundamentally a counting problem rather than an algebraic one. Unlike the determinant, the permanent has no sign cancellations to exploit, and exact computation is #P-hard, so a direct attack is out of the question. The most naive bound simply multiplies the row sums, giving per(A) at most the product of r_i. That corresponds to letting every row choose one of its ones independently, which ignores the global constraint that the chosen columns must all be distinct. Because that constraint is real, the product bound is too weak by roughly a factor of e per row.

Entropy gives a natural language for counting. If M is the set of valid permutations and f is chosen uniformly from M, then the Shannon entropy H(f) equals log of the size of M. A first attempt uses subadditivity of entropy, writing H(f) as at most the sum of the entropies of the individual coordinates f(i). Each coordinate takes at most r_i values, so each entropy is at most log r_i, and subadditivity reproduces exactly the trivial product bound. The reason it fails to improve is that subadditivity throws away all dependence between rows; the very interference that makes the permanent small is discarded.

The sharper estimate is the Bregman–Minc inequality, proved here through Radhakrishnan's entropy argument. The key move is to keep the dependence between rows by using the chain rule instead of subadditivity. For any fixed order of revealing the rows, the chain rule expresses H(f) as a sum of conditional entropies, each measuring the uncertainty left in f(k) once the earlier rows are known. Those conditionals can be much smaller than log r_k because earlier rows may already occupy some of row k's columns. The difficulty is that, for a single fixed reveal order, one cannot guarantee how many of row k's neighbors have been used. The solution is to randomize the reveal order.

Choose a uniformly random permutation tau of the rows and average the chain-rule decomposition over tau. Now fix a row k and a fixed matching f. A neighbor column of row k is unavailable at k's turn exactly when the row that owns that column under f appears before k in tau. Therefore the number N_k of still-available neighbors of k depends only on the relative position of k among the r_k rows that own k's neighbors, including k itself. Because tau is uniform, k is equally likely to occupy any of the r_k relative positions. When k is in relative position m, exactly m minus one of its neighbors are gone, so N_k equals r_k minus m plus one. Thus, averaged over tau, N_k is uniform on the set {1, 2, ..., r_k}, independently of the particular matching f.

Conditioned on the past revealed rows, f(k) can take at most N_k values, so its conditional entropy is at most log N_k. Averaging over the random order and the uniform matching gives, for each row k, an expected contribution of (1/r_k) times the sum from i=1 to r_k of log i, which is (1/r_k) log(r_k!). Summing these contributions over all rows bounds H(f) by the sum of (1/r_k) log(r_k!), and exponentiating base two converts the entropy bound into the desired bound on the number of matchings: per(A) is at most the product over i of (r_i!)^{1/r_i}. The bound is tight, achieved for example by a block-diagonal matrix consisting of disjoint copies of K_{d,d}.

```python
import math
from itertools import permutations
import numpy as np


def bregman_minc_bound(A):
    """Return the Bregman-Minc upper bound on the permanent of a 0/1 matrix."""
    n = len(A)
    bound = 1.0
    for i in range(n):
        r = sum(A[i])
        if r == 0:
            return 0.0
        bound *= math.factorial(r) ** (1.0 / r)
    return bound


def permanent_brute_force(A):
    """Exact permanent for small matrices by enumerating permutations."""
    n = len(A)
    total = 0
    for sigma in permutations(range(n)):
        prod = 1
        for i in range(n):
            prod *= A[i][sigma[i]]
        total += prod
    return total


# Tight example: two disjoint copies of K_{3,3}
A = np.zeros((6, 6), dtype=int)
A[:3, :3] = 1
A[3:, 3:] = 1

print("Bregman-Minc bound:", bregman_minc_bound(A))
print("Exact permanent:    ", permanent_brute_force(A))

# A non-tight example with varying row sums
B = np.array([
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
], dtype=int)
print("B bound:", bregman_minc_bound(B))
print("B exact: ", permanent_brute_force(B))
```
