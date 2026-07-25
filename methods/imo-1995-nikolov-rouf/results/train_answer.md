I will solve the following counting problem using what I call the roots-of-unity generating-function filter. Let p be an odd prime, and consider the set {1, 2, ..., 2p}. I want the number of p-element subsets A of this set whose elements sum to a multiple of p. I will denote this number by N.

The difficulty is that the constraint has two separate parts: I must choose exactly p elements, and the sum of those elements must be congruent to 0 modulo p. A direct combinatorial argument that handles both conditions at once is not obvious, so I will encode both pieces of information into a single algebraic object and then extract the desired coefficient with a standard projection technique.

I introduce two formal variables, x and y. The variable x will keep track of the size of a subset, and the variable y will keep track of the sum of its elements. For each integer k from 1 to 2p, I include a factor 1 + x y^k. The term 1 corresponds to leaving k out of the subset, and the term x y^k corresponds to including k, contributing one to the size and k to the sum. Multiplying these factors over all k gives the two-variable generating function

F(x, y) = ∏_{k=1}^{2p} (1 + x y^k).

By construction, the coefficient of x^m y^s in F is exactly the number of m-element subsets of {1, ..., 2p} whose elements sum to s. Therefore the answer N is the sum of the coefficients of x^p y^s over all s divisible by p.

The remaining obstacle is to isolate the terms whose y-exponent is a multiple of p without expanding the entire product. The natural tool is the orthogonality of the p-th roots of unity. Let ω = e^{2πi/p}. For any integer s, the sum ∑_{j=0}^{p-1} ω^{j s} equals p if p divides s and equals 0 otherwise. This means that averaging the values of a polynomial over y = ω^j for j = 0, 1, ..., p-1 projects onto the part supported on exponents divisible by p.

Applying this to F, I obtain

N = (1/p) ∑_{j=0}^{p-1} [x^p] F(x, ω^j),

where [x^p] denotes the coefficient of x^p. The problem is now reduced to evaluating the coefficient of x^p in F(x, ω^j) for each j.

The case j = 0 is easy because ω^0 = 1, so every y^k becomes 1 and

F(x, 1) = ∏_{k=1}^{2p} (1 + x) = (1 + x)^{2p}.

The coefficient of x^p is therefore the central binomial coefficient (2p choose p). This term counts all p-subsets without regard to the sum.

Now fix a nontrivial index j, so 1 ≤ j ≤ p-1. Because p is prime, ω^j is a primitive p-th root of unity. As k runs from 1 to 2p, the residues k mod p run through each residue class 0, 1, ..., p-1 exactly twice: the first p numbers form one complete residue system, and the next p numbers form another. Consequently the exponents jk mod p also run through every residue class twice, merely permuted by the multiplication by j. Therefore

F(x, ω^j) = [∏_{r=0}^{p-1} (1 + x ω^r)]^2.

The inner product is independent of j, which explains why all nontrivial roots contribute the same value. To evaluate it, I use the factorization z^p - 1 = ∏_{r=0}^{p-1} (z - ω^r). I rewrite each factor as 1 + x ω^r = (-x)((-1/x) - ω^r), so

∏_{r=0}^{p-1} (1 + x ω^r) = (-x)^p [(-1/x)^p - 1].

Since p is odd, (-x)^p = -x^p, and the right-hand side simplifies to 1 + x^p. Thus for every nontrivial j,

F(x, ω^j) = (1 + x^p)^2 = 1 + 2 x^p + x^{2p},

and the coefficient of x^p is exactly 2.

Putting the two cases together,

N = (1/p)[(2p choose p) + (p-1) · 2]
  = [(2p choose p) + 2p - 2] / p
  = [(2p choose p) - 2] / p + 2.

This is the desired closed form. It is an integer because Wolstenholme-style reasoning, or a direct modular computation, gives (2p choose p) ≡ 2 (mod p). Indeed,

(2p choose p) = [(p+1)(p+2)···(2p)] / [1·2···p] = 2 ∏_{i=1}^{p-1} (p+i)/i,

and each factor (p+i)/i is congruent to 1 modulo p, so the whole binomial coefficient is congruent to 2 modulo p. Hence p divides (2p choose p) - 2 and the formula yields an integer.

A quick check for small primes confirms the formula. For p = 3, the set is {1, 2, 3, 4, 5, 6} and there are 8 triples whose sum is divisible by 3, while [(6 choose 3) - 2]/3 + 2 = (20 - 2)/3 + 2 = 8. For p = 5 the formula gives (252 - 2)/5 + 2 = 52, and for p = 7 it gives (3432 - 2)/7 + 2 = 492. The two boundary subsets {1, ..., p} and {p+1, ..., 2p} are the visible sources of the final +2 correction; both have sums divisible by p, and the algebraic collapse of the nontrivial-root contribution to the coefficient 2 in (1 + x^p)^2 matches them.

The method therefore reduces a delicate congruence-counting problem to a clean generating-function evaluation followed by a roots-of-unity average. The key structural fact that makes the computation tractable is that the ground set {1, ..., 2p} consists of exactly two full copies of the residue classes modulo p.

The final artifact of this analysis is the closed-form count itself, valid for every odd prime $p$:

$$N \;=\; \frac{1}{p}\left[\binom{2p}{p} + 2(p-1)\right] \;=\; \frac{\binom{2p}{p}-2}{p} + 2.$$

Here $\binom{2p}{p}$ is the raw count of $p$-element subsets with no congruence condition at all, contributed by the trivial root $j=0$; the constant $2$ attached to each of the $p-1$ nontrivial roots is the coefficient $[x^p](1+x^p)^2$, and it is exactly what forces the formula to be an integer, since $\binom{2p}{p}\equiv 2 \pmod p$ by the factor-by-factor congruence $(p+i)/i \equiv 1 \pmod p$ for $1 \le i \le p-1$. I read the two-part structure of the answer directly off the derivation: the bulk term $\binom{2p}{p}/p$ is the generic density of one residue class among all $p$-subsets, and the additive $+2$ is the exact correction contributed by the two boundary blocks $\{1,\dots,p\}$ and $\{p+1,\dots,2p\}$ — whose sums are each forced to be divisible by $p$ — a correction the root-of-unity average reproduces algebraically rather than by separately carving those two subsets out of the count.
