# Synthesis: MSTD sets — principled constructions + positive-proportion proof

## The pain point / research question
Finite set A subset Z. Sumset A+A, difference set A-A. Because + is commutative (a+b=b+a)
and - is not (a-b != b-a unless a=b), each unordered pair {a,b} of distinct elements gives
ONE sum but TWO differences. So naively |A-A| should >= |A+A| "usually". An MSTD
("more sums than differences") set has |A+A| > |A-A|. Two questions:
(1) explicit constructions of (infinite families of) MSTD sets;
(2) how common are they? Nathanson's stated instinct: rare; "vast majority satisfy |A-A|>|A+A|".

## Key counting identity that opens the door
Differences are removed in pairs: if c not-in A-A then -c not-in A-A (A-A symmetric about 0).
Sums have no such forced symmetry: a sum can be "missing" alone. So |A-A| is always odd, and
the budget of missing sums vs missing differences is the real battleground. Max possible value
is 2n-1 for both; MSTD requires (missing differences) > (missing sums). Symmetric sets
(A = a*-A) are exactly balanced: A+A = a*+(A-A) so |A+A|=|A-A|. Arithmetic progressions are
symmetric -> balanced.

## Conway's seed example
A1 = {0,2,3,4,7,11,12,14}. VERIFIED: A+A=[0,28]\{1,20,27} (|A+A|=26),
A-A=[-14,14]\{+-6,+-13} (|A-A|=25). MSTD by 1. Marica {0,1,2,4,7,8,12,14,15} 30 vs 29.

## Construction #1 — Nathanson: symmetric base + single perturbation
A1\{4} = {0,2}u{3,7,11}u(14-{0,2}) is symmetric about 14 (balanced). Adjoin element 4:
 - 8 = 4+4 is a NEW sum not present -> |A+A| up by >=1.
 - new differences 4-A* are ALREADY in A*-A* (1=3-2, 4=7-3, ...) -> |A-A| unchanged.
Net: MSTD by 1. Theorem 1 (general 1-dim): m>=4, 1<=d<=m-1, d!=m/2, k>=3 (d<m/2) or k>=4 (d>m/2).
 B=[0,m-1]\{d}, L={m-d,2m-d,...,km-d}, a*=(k+1)m-2d, A*=BuLu(a*-B), A=A*u{m}.
 A* symmetric about a* (balanced); 2m in A+A but 2m not-in A*+A*; A*-{m} subset A*-A* so A-A=A*-A*.
 m=4,d=1,k=3 recovers A1 (VERIFIED; Thm1(4,1,5) MSTD 38 vs 37). Thms 3,4: 2-D and d-D GAP versions.

## Construction #2 — Tao's group->lattice->integers lift
Find MSTD in finite abelian group G=Z/m1 x...x Z/md. Counting Thm 10: G=Z/nZ x Z/2Z has MSTD
sets ((0,1) not-in A-A so |A-A|<=2n-1; A+A=G for almost all A in family Omega). Thicken each
group element to box LambdaG(0,t) -> lattice MSTD Bt (Thm 8); collapse to Z via
psi(a)=sum ai m^{i-1}, m large (Thm 9), preserving cardinalities. Base-expansion view.

## Base-expansion / digit view (Hegarty; MO sec 5)
A MSTD, form An = A + bA + ... + b^{n-1}A, large b. Digits don't interact:
|An+An|=|A+A|^n, |An-An|=|A-A|^n, so |An+An|=|An-An|^{log s/log d}. A=S4 u (S4+20) gives 91 vs 83,
exponent log91/log83=1.0208 (best known). Prescribed imbalance: S_{2k+1}=S1+{0,29,...,29k}
gives diff 2k+1 (VERIFIED k=1,2,3 -> 3,5,7). S_{2k}=S_{2k+1}\{29}. Negative x:
Sx={0,...,|x|+1}u{2|x|+2} (VERIFIED). Range of |A+A|-|A-A| is all of Z.

## The methodological leap — Martin-O'Bryant: positive proportion are MSTD
Thm 1: positive proportion of subsets of {0,...,n-1} are sum-dominant (also diff-dom, also
balanced), each > c 2^n, n>=15. Thm 13: #sum-dominant >= (2x10^-7) 2^n.
IDEA (fringe-controlled): random S (each elt w.p. 1/2) realizes almost every possible sum and
difference; a middle target k has ~n/4-|.|/4 representations -> overwhelmingly present. So
whether k IS/ISN'T in S+S / S-S is decided only near the FRINGE. FIX the fringe to kill more
potential differences than sums, let middle be random; w.p. positive the middle fills the rest,
leaving the fringe-engineered imbalance.
Machinery: P[k not-in A+A] ~ (3/4)^{~k/2}: events X_j X_{k-j}=0 over ~k/2 pairs are INDEPENDENT
(j, k-j distinct); P[pair both in]=1/4. For differences X_j X_{k+j}=0 NOT all independent for
small k (j,k+j,2k+j chain) but probabilities already tiny -> pick independent subset J,
|J|>=(b-a)/3, bound (3/4)^{(b-a)/3} (Lemma 10). Prop 8: P[middle block subset A+A] >
1-6(2^-|L|+2^-|U|). Prop 12: P[all of {-(n-l-u),..} subset A-A] > 1-4 2^{-(|L|+|U|)} -
(n/2)(3/4)^{(n-l-u)/3}.
Explicit fringe (Thm 13, n>=23): L={0,2,3,7,8,9,10},
U={n-11,n-10,n-9,n-8,n-6,n-3,n-2,n-1}. VERIFIED:
 - U-L omits n-7 -> A-A misses +-(n-7) -> |A-A|<=2n-3.
 - L+L={0,...,20}\{1} (only 1 forced missing on left); L+U={n-11,...,n+9}; U+U={2n-22,...,2n-2};
   Prop 8 fills {21,...,n-12}u{n+10,...,2n-23} -> A+A={0,...,2n-2}\{1}, |A+A|=2n-2.
 - 2n-2=|A+A| > |A-A|<=2n-3, prob >=119/128 -> >= 2^{n-22} 119/128 > (2x10^-7)2^n.
 VERIFIED n=60 random middle -> 118 vs 117, MSTD.
Averages (Thm 3,16,17): E|S+S|~2n-11, E|S-S|~2n-7 (expected 10 missing sums, 6 missing diffs).
The factor ~2 (missing sums vs missing diffs) is exactly commutativity. So on AVERAGE diffs win
by 4 (Nathanson right), yet a positive proportion buck the average.

## Miller-Orosz-Scheinerman: denser explicit families (C/r^4)
Pn-set: A+A and A-A contain all but first/last n possible elements. Random set is Pn whp. Take
Pn MSTD seed A=LuR (L subset [1,n], R subset [n+1,2n]), e.g. {1,2,3,5,8,9,13,15,16} (30 vs 29
VERIFIED). Insert a middle block: A'=LuMuR', R'=R+m. Lemma 2.1: if A' stays Pn then added sums
= added diffs = 2m -> still MSTD. Vary M (no run of >k missing) -> >= C/r^4 of subsets of
{1,...,r} are MSTD; far denser than 2^{-r/2} symmetric-perturbation families, but still not a
positive proportion.

## Design-decision -> why
- Start symmetric (balanced): symmetry forces |A+A|=|A-A|, the zero point a perturbation tips.
- Perturb by ONE element making new sum 2m but no new difference (m-A* pre-absorbed via 0 in A*,
  B-B, L-L). Adding an element that also makes a new difference would cancel.
- Perturbed AP B=[0,m-1]\{d}: APs are canonical balanced sets; hole {d}+ladder L let one fresh
  sum appear with no fresh difference.
- Fringe-fix + random middle: middle provably irrelevant (high rep count); edge is a finite,
  n-independent gadget, isolating it is the leverage.
- Kill +-(n-7) but only sum 1: differences come in +- pairs so one omitted difference costs 2
  from the 2n-1 budget; engineer net loss in the difference budget.
- Base-b digits (large b): separate coordinates so digit sums/diffs don't carry; per-coordinate
  imbalance multiplies.

## Source URLs
- Nathanson, "Sets with more sums than differences", arXiv:math/0608148
- Martin & O'Bryant, "Many sets have more sums than differences", arXiv:math/0608131
- Miller, Orosz, Scheinerman, "Explicit constructions of infinite families of MSTD sets", arXiv:0809.4621
- Zhao, arXiv:0908.4442 (denser via bidirectional ballot sequences; not central)
