I will now present the core method and its main conclusions in my own words. The canonical name for the technique is random circle dilation to a sum-free arc, used for extremal lower bounds and sharpened by Fourier certificates together with a mod-3 descent, while the enumeration of sum-free subsets of an interval is handled by a granular container method combined with a Kneser-type structure theorem.

Start with the extremal question. Given a finite set A of n nonzero integers, I want a large subset B of A with no solution to x + y = z, where x, y, and z are allowed to repeat. The naive approach of deleting one element per forbidden triple fails immediately because triples overlap and there can be quadratically many of them. Instead, I look for a structural rule that keeps many elements automatically and makes sum-freeness automatic.

The key idea is to map the integers into a group where I already understand sum-free regions, and then pull one back. The circle group T = R / Z is ideal because the map a maps to theta a modulo 1 is a homomorphism for every real theta. On the circle, the open middle-third arc B = (1/3, 2/3) is sum-free: if u and v both lie in B, then u + v modulo 1 lies in (2/3, 4/3), which reduces to (2/3, 1) union (0, 1/3) and never lands back in B. Therefore, for any theta, the preimage A_theta = {a in A : theta a mod 1 lies in B} is automatically a sum-free subset of A.

Now I choose theta at random. For each nonzero a, the fractional part {theta a} is uniformly distributed on T, so each element of A is included with probability exactly 1/3. Linearity of expectation gives E|A_theta| = n/3, so some theta yields |A_theta| greater than or equal to n/3. This is the basic averaging bound, and it already certifies a definite fraction.

To improve it, I switch to the half-open arc [1/3, 2/3) and define f(x) = 1_[1/3,2/3)(x) - 1/3 and f_A(x) = sum over a in A of f(ax). Then |A_x| = n/3 + f_A(x). The function f_A has mean zero, but at x = 0 the set A_0 is empty. Since the average is n/3 and the function is integer-valued and not constant, the maximum must be at least floor(n/3) + 1, which equals ceil((n+1)/3). This gives the Alon-Kleitman strict gain: s(A) is at least (n+1)/3.

The remaining challenge is to push past this single integer jump. The Fourier expansion of f is f(x) = -(sqrt(3)/pi) sum over m at least 1 of chi(m)/m cos(2 pi m x), where chi is the nonprincipal Dirichlet character modulo 3. Thus f_A equals -(sqrt(3)/pi) F_A, where F_A is built from cosines at multiples of elements of A. A large L^1 mass of F_A forces a positive maximum of f_A, and Mobius sifting of the higher harmonics converts this into a lower bound in terms of the plain Littlewood sum sum over a in A of cos(2 pi a dot).

For structured cases where the Fourier mass is small, I use nonnegative trigonometric test functions phi with integral 1. Since m_A, the maximum of f_A, is at least the integral of phi f_A, a well-chosen phi gives a concrete certificate. For example, if 1 is not in A and u is the minimum element of A while v is the smallest element not divisible by u, the product phi(x) = (1 - cos(2 pi u x))(1 - cos(2 pi v x)) is nonnegative, has mean 1, and yields an integral larger than 1/3.

The mod-3 descent handles the rigid case where 1 lies in A. Split A into A_0, the elements coprime to 3, and A_1, the multiples of 3. The three-shift identity f_A(x) + f_A(x+1/3) + f_A(x+2/3) = 3 f_{A_1}(x) implies m_A is at least m_{A_1}, so the divisible part inherits the same problem at a smaller scale. A residue-class estimate also gives m_A at least |A_0|/6 - |A_1|/3. Together these reduce proving a fixed surplus to finitely many base cases, which are verified by explicit trigonometric polynomials. The conclusion is that for coprime positive A, either A equals {1, 2} or s(A) is at least (n+2)/3.

On the counting side, the goal is to show that the number of sum-free subsets of [N] = {1, ..., N} is asymptotically c(N) 2^{N/2}, where c(N) depends only on the parity of N. The lower bound is immediate: every subset of the odd numbers is sum-free, and every subset of the strict upper half {floor(N/2)+1, ..., N} is sum-free; each family already contributes about 2^{N/2} examples.

For the upper bound, I use a container strategy. Embed [N] into Z/pZ for a prime p in [2N, 4N], so no equation x + y = z from [N] wraps around. Partition the group into arithmetic progressions of common difference d and length L, and keep only the dense progressions to form a granularization A'. The difference d is chosen to align with the large Fourier coefficients of A. This ensures that A' covers all but an epsilon-fraction of A and contains at most epsilon p^2 additive triples. The family of all such granularizations has size 2^{o(N)}.

A structural theorem for large almost-sum-free sets then says that every such container is essentially either a short interval or almost entirely odd. Pairing arguments discard the mixed cases: if an odd container contains an even element, or an interval container contains a small element, then many disjoint forbidden pairs arise and the number of choices becomes o(2^{N/2}). Consequently, almost all sum-free sets are either all odd or contained in {ceil((N+1)/3), ..., N}. Cameron-Erdos counted the latter family, yielding that |SF(N)| is asymptotic to c(N) 2^{N/2}.

The code below illustrates the basic mechanism on concrete small examples. It generates random integer sets, computes a random dilation into the middle-third arc, verifies sum-freeness, compares the obtained size against the theoretical lower bounds, and exhaustively enumerates sum-free subsets of [N] for small N to confirm the growth rate.

```python
import random
from itertools import combinations


def is_sum_free(S):
    s = set(S)
    for x in s:
        for y in s:
            if x + y in s:
                return False
    return True


def random_dilation_sum_free(A, trials=10000):
    best = set()
    for _ in range(trials):
        theta = random.random()
        B = {a for a in A if (1 / 3) <= (theta * a) % 1 < (2 / 3)}
        if len(B) > len(best):
            best = B
    return best


def count_sum_free(N):
    return sum(
        1
        for r in range(N + 1)
        for S in combinations(range(1, N + 1), r)
        if is_sum_free(S)
    )


for n in (5, 8, 12):
    A = list(range(1, n + 1))
    B = random_dilation_sum_free(A)
    lower_bound = (n + 3) // 3
    print(
        f"n={n}, dilation size={len(B)}, sum-free={is_sum_free(B)}, "
        f"bound={lower_bound}"
    )

for N in range(1, 11):
    print(f"N={N}, |SF(N)|={count_sum_free(N)}")
```
