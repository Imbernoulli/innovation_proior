# LLL synthesis notes

## Sources retrieved (all three classes)
- PRIMARY: Lenstra, Lenstra, Lovász 1982 "Factoring polynomials with rational coefficients", Math. Ann. 261, 515-534. Got the author's own clean PDF (math.leidenuniv.nl/~hwl). Section 1 "Reduced Bases for Lattices" read in full: (1.1)-(1.38). This is the original LLL.
- BACKGROUND: Gram-Schmidt (in the paper, (1.2)/(1.3)); Gauss/Lagrange 2D reduction (cryptohack/gitbook + arXiv 2504.12948 history); Hermite (1850)/Minkowski (1896)/Korkine-Zolotarev reduction lineage (Lagrange 1773, Gauss 1801 from binary quadratic forms); SVP / successive minima (paper cites Cassels [4]).
- THIRD-PARTY: kel.bz LLL intuition (GS as measuring tool, mu formula, delta swap, potential), MIT 6.876 L4 + OCW scribe, Wikipedia pseudocode (matches Fig.1).

## The intellectual move
A lattice L has infinitely many bases (related by GL_n(Z), det ±1). No canonical "shortest basis" and SVP is hard in high dim. BUT Gram-Schmidt gives, for any basis, an orthogonal sequence b*_1..b*_n with the key facts:
- d(L) = prod |b*_i| (determinant is basis-invariant), and
- any lattice vector x has |x| >= |b*_i| for the largest i with nonzero coefficient.
So the b*_i lengths are the "real" lengths; a good basis is one where the |b*_i| don't collapse too fast.

Two demands define "reduced":
1. size-reduced: |mu_{i,j}| <= 1/2 for all j<i. mu_{i,j} = <b_i, b*_j>/<b*_j, b*_j>. Achieved by b_i <- b_i - round(mu_{i,j}) b_j (lattice-preserving), from j=i-1 down to 1.
2. Lovász condition: |b*_i + mu_{i,i-1} b*_{i-1}|^2 >= delta |b*_{i-1}|^2, equivalently |b*_i|^2 >= (delta - mu_{i,i-1}^2)|b*_{i-1}|^2 (since b*_i ⟂ b*_{i-1}, the LHS = |b*_i|^2 + mu^2 |b*_{i-1}|^2). delta = 3/4 in the paper (any 1/4<delta<1).

The geometric meaning: b*_i + mu_{i,i-1} b*_{i-1} is the projection of b_i onto the orthogonal complement of span(b_1..b_{i-2}); b*_{i-1} is the projection of b_{i-1} there. So in that 2D projected plane it's exactly Gauss/Lagrange 2D reduction. If the condition fails, swapping b_{i-1},b_i makes the new b*_{i-1} = old (b*_i + mu_{i,i-1} b*_{i-1}), whose squared length is < delta times the old |b*_{i-1}|^2 -- a strict shrink of that GS length.

## Potential / termination (1.23-1.25)
d_i = det(<b_j,b_l>)_{1<=j,l<=i} = Gram determinant of first i vectors = prod_{j<=i} |b*_j|^2. Define D = prod_{i=1}^{n-1} d_i. A swap at index k only changes d_{k-1} (the others are determinants of sublattices unaffected), and d_{k-1} -> d_{k-1}*(|new b*_{k-1}|^2/|old b*_{k-1}|^2) < (3/4) d_{k-1}. So each swap multiplies D by < 3/4. Size-reduction never changes any b*_i, so never changes D. If basis is integral, all d_i are positive integers so D >= 1; initially d_i <= B^i so D <= B^{n(n-1)/2}. Hence number of swaps <= log_{4/3}(B^{n(n-1)/2}) = O(n^2 log B). k increments at most n-1 more than it decrements, so total steps bounded -> terminates. Lower bound on d_i via Minkowski: each rank-i sublattice has m(L)<=lambda, d_i >= (3/4)^{i(i-1)/2} m(L)^i.

## Output guarantees (Prop 1.6, 1.11)
From (1.5)+(1.4): |b*_i|^2 >= (3/4 - 1/4)|b*_{i-1}|^2 = (1/2)|b*_{i-1}|^2, so |b*_j|^2 <= 2^{i-j}|b*_i|^2.
(1.7) |b_j|^2 <= 2^{i-1}|b*_i|^2.
(1.9) |b_1| <= 2^{(n-1)/4} d(L)^{1/n}.
(1.11) |b_1|^2 <= 2^{n-1} |x|^2 for all nonzero x in L, i.e. |b_1| <= 2^{(n-1)/2} lambda_1(L). [first vector within 2^{(n-1)/2} of shortest]
Proof of 1.11: x = sum r_i b_i = sum r'_i b*_i, r_i in Z; if i is largest index with r_i != 0 then r'_i = r_i (integer, nonzero), so |x|^2 >= r'^2 |b*_i|^2 >= |b*_i|^2 >= 2^{-(n-1)} |b_1|^2 by (1.7) with j=1.

## Algorithm (1.15)-(1.22), Fig. 1
k from 2; at k: size-reduce b_k against b_{k-1} (get |mu_{k,k-1}|<=1/2). Then:
- Case 1 (Lovász fails, k>1): |b*_k + mu_{k,k-1} b*_{k-1}|^2 < (3/4)|b*_{k-1}|^2 -> swap b_{k-1},b_k; update b*,mu via (1.22); k -> max(k-1,2).
- Case 2 (Lovász holds or k=1): finish size-reducing b_k against b_{k-2}..b_1; k -> k+1. terminate when k=n+1.

Swap update formulae (1.22): with mu=mu_{k,k-1}, B=B_k + mu^2 B_{k-1} (B_i=|b*_i|^2):
new mu_{k,k-1} = mu B_{k-1}/B; new B_k = B_{k-1} B_k / B; new B_{k-1} = B; and for the rows: cross-update mu_{k-1,j},mu_{k,j} swap for j<k-1; for i>k, (mu_{i,k-1}, mu_{i,k}) <- [[mu, 1-mu*newmu... ]] -- the 2x2 matrix [[newmu, 1],[1 - mu*newmu, -mu]] acting on (mu_{i,k-1},mu_{i,k}). Verbatim Fig.1:
  (mu_{i,k-1}; mu_{i,k}) <- [[1, mu_{k,k-1}^new... ]] -- code uses the standard integer/fraction recomputation; simplest faithful code recomputes GS exactly.

## Design choices -> why
- Use GS lengths not raw lengths: GS lengths are basis-invariant in product and lower-bound lattice vectors; raw lengths aren't.
- |mu|<=1/2 threshold: rounding to nearest integer drives |mu| into [-1/2,1/2]; can't do better with one integer subtraction; makes b_i as orthogonal to b*_j as a lattice move allows.
- delta=3/4 specifically: any delta in (1/4,1) gives polynomial time with factor 4/(4delta-1) in the bounds; 3/4 gives clean factor 2 (=4/(4*3/4-1)=4/2) and the shrink-per-swap factor 3/4 < 1. delta->1 gives better basis but the potential shrink factor delta gets closer to 1 so more iterations (and at delta=1 termination proof fails). 3/4 is the convenient middle.
- adjacent-only swap (not arbitrary pair): keeps the potential argument simple -- only d_{k-1} changes; mirrors Euclidean/Gauss 2D step; keeps it polynomial.
- size-reduce only against b_{k-1} before the test, full reduction only in Case 2: Cuppen's improvement, saves a factor n -- the test only needs mu_{k,k-1}.
- integral Gram matrix / d_i denominators: keeps all arithmetic exact in integers, controls bit-length O(n log B).

## Canonical code structure
Faithful self-contained Python using Fraction for exact GS. Functions: gram_schmidt (returns b* and mu), size-reduce step (round mu, update b and mu row), Lovász test with delta=Fraction(3,4), swap + recompute GS, k pointer max(k-1,2)/k+1. Mirrors Fig.1 / Wikipedia pseudocode. This matches widely-used reference implementations (fraction-exact LLL).
