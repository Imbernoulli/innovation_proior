I call the object we are studying a More-Sums-Than-Differences set, or MSTD set for short. The problem is simple to state: take a finite set of integers A and form its sumset A+A={a+a':a,a' in A} and its difference set A-A={a-a':a,a' in A}. Because addition commutes but subtraction does not, a pair of distinct elements contributes one sum but two differences, so my naive instinct is that |A-A| should be at least |A+A|. I want to know whether sets with |A+A|>|A-A| exist, whether I can write down explicit infinite families of them, and how common they are among all subsets of an interval {0,...,n-1}.

The first thing I notice is that the difference set is always symmetric around zero: if c is in A-A then -c is also in A-A, because c=a-a' forces -c=a'-a. This means the missing values in A-A come in plus-minus pairs, so each structural omission in the difference set costs two from the maximum possible budget. The sumset has no such forced symmetry. A value near one end of the sum range can be missing without anything being forced at the other end. So the real contest is not about who has more potential elements, but about who misses fewer values, and the difference set pays a symmetry tax that the sumset does not.

This asymmetry is the lever that makes MSTD sets possible. The cleanest way I know to exploit it is to start from a symmetric set, which is automatically balanced. If A* is symmetric about a center a*, meaning A*=a*-A*, then A*+A*=a*+(A*-A*), so |A*+A*|=|A*-A*|. I then adjoin a single new element m. If m adds at least one fresh sum but no fresh difference, the skeleton tips from balanced to sum-dominant. For the classical Conway seed A1={0,2,3,4,7,11,12,14}, the skeleton A*={0,2,3,7,11,12,14} is symmetric about 14, and adjoining 4 adds the sum 8 but no new difference, giving |A1+A1|=26 and |A1-A1|=25.

This one-off example generalizes to an infinite family. I fix a step m, a hole d with 1<=d<=m-1 and d not equal to m/2, and a length k large enough that the construction closes. I let B=[0,m-1]\{d}, L={m-d,2m-d,...,km-d}, and a*=(k+1)m-2d. The skeleton A*=B union L union (a*-B) is symmetric about a*, hence balanced. Adjoining m gives A=A* union {m}. I check that 2m is a fresh sum because no two elements of A* add to 2m, while every difference m-a for a in A* already appears in A*-A*. Therefore |A+A|=|A*+A*|+1>|A*-A*|=|A-A|. Taking (m,d,k)=(4,1,3) recovers Conway's A1.

I can amplify the same seed to realize any desired imbalance. If A has |A+A|=s and |A-A|=t, and I form A_N=A+bA+b^2 A+...+b^{N-1}A for a base b larger than twice the diameter of A, then no carries occur between digit levels. A sum in A_N+A_N is determined independently per digit, so |A_N+A_N|=s^N and |A_N-A_N|=t^N. The imbalance therefore compounds geometrically. By stacking shifted copies of A1 at spacing 29, or by deleting an interior point, I obtain MSTD sets whose imbalance |A+A|-|A-A| is any prescribed integer, positive, negative, or zero.

Beyond explicit constructions, I also ask about density. It turns out that a positive proportion of all subsets of {0,...,n-1} are MSTD. The reason is that for a uniformly random subset, every sum or difference in the middle of the range has on the order of n/4 independent representations, so it is present with overwhelming probability. Only fringe values near 0 or near n-1 can plausibly be missing. I pin the fringe to force a plus-minus pair of missing differences while forcing only a single missing sum, and then let the middle be random; almost every choice of the middle inherits the engineered imbalance. Since a fixed positive fraction of all subsets share my prescribed fringe, the proportion of MSTD subsets is bounded below by a positive constant. A concrete fringe is L={0,2,3,7,8,9,10} and U={n-11,n-10,n-9,n-8,n-6,n-3,n-2,n-1}; then U-L omits n-7, so the difference set loses plus and minus (n-7), while the sumset only loses 1.

It is worth emphasizing that this positive-proportion result does not contradict the naive instinct. If I average |S+S| and |S-S| over all 2^n subsets, the difference set still wins on average by about four, because a typical potential difference has roughly twice as many independent chances to be realized as the corresponding sum. The average and the tail can point in opposite directions: most sets are slightly difference-dominant, yet a positive fraction are strongly sum-dominant.

The following Python script verifies the small seed, constructs the infinite family for a few parameters, confirms the positive-proportion gadget for a random middle, and illustrates that the imbalance can be amplified by base-b digit stacking.

```python
from itertools import product

def sumset(A):
    return {a + b for a in A for b in A}

def diffset(A):
    return {a - b for a in A for b in A}

def verify(A):
    s = sumset(A)
    d = diffset(A)
    return len(s), len(d), len(s) - len(d), sorted(s), sorted(d)

# Conway seed
A1 = {0, 2, 3, 4, 7, 11, 12, 14}
s1, d1, imb1, S1, D1 = verify(A1)
print(f"A1: |A+A|={s1}, |A-A|={d1}, imbalance={imb1}")
print("missing sums:", [x for x in range(min(S1), max(S1)+1) if x not in S1])
print("missing diffs:", [x for x in range(min(D1), max(D1)+1) if x not in D1])

# Parametric family

def mstd_family(m, d, k):
    B = set(range(m)) - {d}
    L = {(i + 1) * m - d for i in range(k)}
    astar = (k + 1) * m - 2 * d
    Astar = B | L | {astar - b for b in B}
    return Astar | {m}

for params in [(4, 1, 3), (5, 2, 4), (6, 1, 4)]:
    A = mstd_family(*params)
    s, d, imb, _, _ = verify(A)
    print(f"params={params}: |A+A|={s}, |A-A|={d}, imbalance={imb}")

# Base-b digit stacking amplification

def digit_stack(A, b, N):
    return {sum(a * (b ** i) for i, a in enumerate(tup))
            for tup in product(A, repeat=N)}

A2 = mstd_family(4, 1, 3)
s2, d2, _, _, _ = verify(A2)
B = digit_stack(A2, b=100, N=2)
sB, dB, imbB, _, _ = verify(B)
print(f"stacked N=2: |A+A|={sB}, |A-A|={dB}, imbalance={imbB}")
print(f"predicted |A+A|={s2**2}, |A-A|={d2**2}")

# Positive-proportion gadget with a random middle
import random

def gadget(n, seed=0):
    random.seed(seed)
    L = {0, 2, 3, 7, 8, 9, 10}
    U = {n - 11, n - 10, n - 9, n - 8, n - 6, n - 3, n - 2, n - 1}
    R = {i for i in range(11, n - 11) if random.random() < 0.5}
    return L | R | U

n = 200
A = gadget(n)
s, d, imb, S, D = verify(A)
print(f"gadget n={n}: |A+A|={s}, |A-A|={d}, imbalance={imb}")
print("expected |A+A| ~=", 2 * n - 2, ", |A-A| <=", 2 * n - 3)
```
