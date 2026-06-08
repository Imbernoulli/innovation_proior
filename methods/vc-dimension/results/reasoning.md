Let me start from the thing that actually goes wrong, because the textbook law of large numbers looks like it should already settle everything and it doesn't. I have a class S of decision rules — think of them as events, subsets A of the input space X — and an unknown distribution P. I draw a sample x₁,…,x_l, and for each rule A I can compute its empirical frequency ν_A = n_A/l, the fraction of the sample that lands in A. Bernoulli tells me ν_A → P_A in probability for each fixed A. Chebyshev even gives me a clean rate: the count n_A is binomial, so P(|ν_A − P_A| > ε) ≤ P_A(1−P_A)/(lε²) ≤ 1/(4lε²), and that 1/4 is uniform in P because P_A(1−P_A) never exceeds 1/4. Beautiful, distribution-free, done — for one event fixed in advance.

But that is not what I do. I look at the sample, I compute everybody's empirical error, and I *pick* the rule that fits best. The rule I walk away with is selected precisely because its empirical frequency came out favorable. So its empirical frequency is no longer an unbiased snapshot of its probability — I cherry-picked it. The quantity that decides whether fitting the data buys me anything about the truth is not |ν_A − P_A| for a fixed A; it is the worst case over the whole class,

    π^(l) = sup_{A∈S} |ν_A − P_A|.

If π^(l) is small, then *every* rule's empirical frequency is close to its probability, including whichever one I end up choosing, so a small empirical error really does mean a small true error. If π^(l) can be large, the rule that looks best on the sample can be terrible underneath. So the object of study is this supremum, and I want to know when it goes to zero.

A wrong move tripped up smart people around me, and I have to see exactly why it fails. The objection goes: "Look, the bound P(|ν_A − P_A| > ε) ≤ 1/(4lε²) is true for *every* A. In particular it's true for the A your algorithm happens to choose. So choosing after the fact changes nothing." That feels airtight and it is false. The probability of randomly bumping into a person with some rare disease in a city is tiny; the probability conditioned on my having deliberately walked into the clinic for that disease is much larger — and the clinic is in the same city. Selecting the empirically-best rule is exactly that deliberate walk. The per-event bound is a statement about a *fixed* A and a random sample; the chosen A is a function of the sample, so it can sit exactly where the deviation is largest. The honest way to cover a rule chosen a posteriori is to control the deviation *simultaneously* over the whole class — to demand uniform convergence over S. That phrase is the right one: I am playing on the non-compactness of the class (an infinite ball of linear rules has no finite cover by accident), and the thing I need is a sup that converges, not a pointwise one.

So how do I control a sup over infinitely many events? The finite case I know how to do. If S had only N rules, I'd just union-bound: P(at least one of N events deviates by ε) ≤ N · 1/(4lε²), or in the realizable form, the chance some bad rule survives all l points is ≤ N(1−κ)^l, giving a sample size l ≥ (log N − log η)/κ. The number of rules enters only through log N, which is wonderfully mild. The trouble is that for the classes I actually care about — half-spaces in Rⁿ, the perceptron's hypotheses — there's a continuum of them. N = ∞ and the union bound is vacuous.

My first reflex is to discretize: cover the parameter ball with an ε-net, count how many rules I need, union-bound over those. But the count depends on ε and on the ambient dimension in a crude way, and worse, the "right" granularity depends on the distribution — two rules that look different in parameter space may be indistinguishable on the data, and vice versa. I'm bounding the wrong thing. The cardinality of S is infinite, but that infinity is mostly illusory from the sample's point of view.

That's the crack to push on. Stare at a fixed sample x₁,…,x_l. A rule A only ever interacts with the sample through which points it contains — through the subsample it induces, the labeling (A(x₁),…,A(x_l)) ∈ {0,1}^l. Two rules that contain exactly the same sample points are, as far as this sample is concerned, the same event: same ν_A, same role in the sup. And there are at most 2^l possible labelings of l points, no matter how enormous S is. So on a given sample the number of *genuinely distinct* events is

    Δ^S(x₁,…,x_l) = number of distinct subsamples that sets in S induce on x₁,…,x_l,

which I'll call the index of S with respect to the sample, and Δ^S ≤ 2^l always. The supremum over the continuum S collapses, on the sample, to a maximum over at most Δ^S distinct events. To make a bound that doesn't reference the particular sample, take the worst case:

    m^S(l) = max over all samples of size l of Δ^S(x₁,…,x_l),

the growth function of S. This is the candidate to replace N. If I can get a bound of the shape m^S(2l) · (something exponentially small), then whether uniform convergence holds turns entirely on how fast m^S grows.

But there's a gap I glossed: in the union bound, N multiplied the per-event tail because each event's deviation |ν_A − P_A| was a fixed random variable I could bound. Here P_A is unknown and A ranges over an infinite class — I can't even enumerate the events to union-bound over until I'm on a finite sample, and the sup involves the unknown P_A. I need a way to get rid of P_A and confine the whole problem to finitely many observed points. 

The trick is to introduce a second, independent sample — a ghost sample. Draw 2l points instead of l. Call the first half x₁,…,x_l (frequencies ν′_A) and the second half x_{l+1},…,x_{2l} (frequencies ν″_A). Now look at the two-sample deviation

    ρ^(l) = sup_{A∈S} |ν′_A − ν″_A|.

The unknown P_A has vanished — ρ only involves observed frequencies. The bet is that controlling ρ controls π. Intuitively: if ν′_A is far from P_A (so π is large), then the second half's frequency ν″_A, being a fresh estimate, is near P_A by the law of large numbers, so ν′_A is also far from ν″_A. Let me make that precise.

Set Q = {π^(l) > ε} (bad first half) and C = {ρ^(l) ≥ ε/2}. I claim that for l large enough, P(C) ≥ ½ P(Q). Write P(C) as an integral over the 2l-sample, factor it as first-half × second-half by Fubini, and restrict the outer integral to the event Q. On Q there is, by definition, some event A₀ with |ν′_{A₀} − P_{A₀}| > ε. Condition on that first half. For this fixed A₀, if the second half satisfies |ν″_{A₀} − P_{A₀}| ≤ ε/2, then by the triangle inequality |ν′_{A₀} − ν″_{A₀}| > ε − ε/2 = ε/2, so ρ ≥ ε/2 and we're in C. Therefore, on Q,

    P(C | first half) ≥ P( |ν″_{A₀} − P_{A₀}| ≤ ε/2 ) = 1 − P( |ν″_{A₀} − P_{A₀}| > ε/2 ).

That last probability is a single-event Chebyshev bound on the second half: P(|ν″_{A₀} − P_{A₀}| > ε/2) ≤ 4 P_{A₀}(1−P_{A₀})/(ε²l) < 1/(ε²l). For l ≥ 2/ε² this is ≤ 1/2, so the conditional probability of C is ≥ 1/2. Integrating over Q,

    P(C) ≥ ½ P(Q),   for l ≥ 2/ε².

So if I can show P(C) = P(ρ^(l) ≥ ε/2) → 0, then P(Q) = P(π^(l) > ε) → 0 at half the rate. Every bit of the work now lives on the two-sample deviation ρ, which only sees the 2l observed points — and on those 2l points there are at most m^S(2l) distinct events. That is the payoff of the ghost sample: it traded the unknown P_A for a finite, sample-confined problem.

Now bound P(ρ^(l) ≥ ε/2). Here's the second idea: the 2l points were drawn i.i.d., so the joint distribution is invariant under any permutation of the 2l indices. In particular, randomly reassigning which points are in the "first half" and which in the "second half" doesn't change the probability law. So I can write P(ρ ≥ ε/2) as an average over all (2l)! permutations T of the integrand θ(ρ(T·X) − ε/2), integrated against P:

    P(ρ^(l) ≥ ε/2) = ∫ (1/(2l)!) Σ_T θ( ρ^(l)(T X_{2l}) − ε/2 ) dP.

Two payoffs from working permutation-by-permutation. First, the supremum over S inside ρ depends only on the labelings the events induce on the 2l points, so I can replace S by a finite subsystem S′ containing one representative per distinct induced subsample — and |S′| = Δ^S(x₁,…,x_{2l}) ≤ m^S(2l). The sup over the infinite class becomes a max over ≤ m^S(2l) representatives, and θ(sup …) ≤ Σ over the representatives:

    (1/(2l)!) Σ_T θ( ρ(T X_{2l}) − ε/2 ) ≤ Σ_{A∈S′} [ (1/(2l)!) Σ_T θ( |ν′_A − ν″_A|(T X_{2l}) − ε/2 ) ].

Second, for a *fixed* event A and a fixed multiset of 2l points, the inner average is a purely combinatorial quantity. Suppose m of the 2l points belong to A. A permutation T just chooses which l of the 2l points form the first half. Among those l, say k belong to A, so ν′_A = k/l and ν″_A = (m−k)/l, and |ν′_A − ν″_A| = |2k − m|/l. The fraction of equipartitions with |2k/l − m/l| ≥ ε/2 is the hypergeometric tail

    Γ = Σ_{k : |2k − m| ≥ εl/2}  C(m,k) C(2l−m, l−k) / C(2l, l).

This is drawing l balls without replacement from an urn of 2l with m marked, and asking the probability the marked count k deviates from its mean m/2 by ≥ εl/4. I want Γ ≤ 2 e^{−ε²l/8}. Sampling without replacement is at least as concentrated as sampling with replacement (negative association), so I can bound it by a binomial tail; a Hoeffding/Chernoff bound for the sum of l bounded exchangeable indicators with deviation εl/4 over a range… let me be careful with the constant. The relevant deviation of k from its mean is εl/4; normalizing by the l draws, that's a deviation of ε/4 in the proportion; a two-sided exponential bound of the form 2 exp(−2 l (ε/4)²) = 2 exp(−ε²l/8). The factor 8 is exactly (ε/2) squared, where the ε/2 came from the threshold in C — splitting ε once for the triangle inequality and once into the proportion. So Γ ≤ 2 e^{−ε²l/8}, uniform over A and over the point configuration. (This is the one piece the symmetry hands me without any reference to P — it's a finite combinatorial count.)

Put the two together. Each representative contributes ≤ 2 e^{−ε²l/8}, and there are Δ^S(x₁,…,x_{2l}) ≤ m^S(2l) of them:

    (1/(2l)!) Σ_T θ( ρ(T X_{2l}) − ε/2 ) ≤ 2 Δ^S(x₁,…,x_{2l}) e^{−ε²l/8} ≤ 2 m^S(2l) e^{−ε²l/8}.

Integrating over P (the bound is already uniform in the point configuration, so the integral just passes through),

    P(ρ^(l) ≥ ε/2) ≤ 2 m^S(2l) e^{−ε²l/8}.

And feeding this into the symmetrization lemma P(Q) ≤ 2 P(C):

    P(π^(l) > ε) ≤ 4 m^S(2l) e^{−ε²l/8},   for l ≥ 2/ε².

There it is — the whole class controlled by a single quantity, m^S(2l), times an exponential in l. The 4 is bookkeeping: a 2 from the hypergeometric tail and a 2 from the ghost-sample lemma. No reference to P anywhere; it's distribution-free. A sufficient condition for uniform convergence is now obvious: if m^S(2l) grows slower than e^{ε²l/8} for every ε — in particular if it's polynomial — the right side goes to zero.

So everything reduces to: how fast can m^S grow? Trivially m^S(l) ≤ 2^l. For the pathological class of all subsets of [0,1], m^S(l) = 2^l exactly — every labeling is realizable — and then 2^{2l} e^{−ε²l/8} blows up, no convergence. That matches the diagnostic that uniform convergence really fails there. For the fixed-threshold linear half-spaces {x : ⟨w,x⟩ ≥ 1} in Rⁿ, the cell-counting recurrence in the n-dimensional parameter space gives at most Φ(n,r) = Σ_{k=0}^{n} C(r,k) distinct labelings of r points, polynomial of degree n. So at least two behaviors are possible: full 2^l, or polynomial. The question that decides learnability is whether anything lives *between* — a class whose growth function is, say, 2^{√l} or 2^l for a while and then polynomial.

I suspect there's nothing in between: m^S(l) is either identically 2^l, or eventually polynomial — a sharp dichotomy. And the threshold should be combinatorial. Let me find the dividing line. Say S can shatter a set of points if it induces all 2^k labelings on those k points — Δ^S = 2^k there, the class is as expressive as possible on that set. Let d be the largest k for which some set of k points can be shattered; for r ≤ d there's a sample with Δ^S = 2^r, but for any sample of size > d no full shattering occurs. The claim is that this d controls the polynomial degree. In the index-recurrence it's cleanest to track n = d+1, the *first* sample size at which full shattering becomes impossible — the first r where m^S(r) ≠ 2^r.

To prove the dichotomy I need a lemma connecting "the index is large" to "a large subset is shattered." 

Lemma. If for some sample x₁,…,x_i and some n with 1 ≤ n ≤ i we have Δ^S(x₁,…,x_i) ≥ Φ(n,i), where Φ obeys the Pascal recurrence Φ(n,i) = Φ(n,i−1) + Φ(n−1,i−1) with Φ(0,i)=1, Φ(n,n)=2ⁿ, then some subsample of size n is shattered, i.e. there exist n of the points on which S induces all 2ⁿ labelings.

I'll prove it by induction on i (and n). Base cases: n = 1, the condition Δ^S ≥ Φ(1,i) = i+1 ≥ 2 means at least two labelings, so some single point is contained by one set and excluded by another — that one point is shattered; and n = i, where Φ(i,i) = 2^i forces Δ^S = 2^i, the whole sample is shattered. Now suppose the lemma holds for all samples shorter than r and all n, but fails for a sample X_r = x₁,…,x_r and some n < r: so Δ^S(X_r) ≥ Φ(n,r) yet no subsample of size n is shattered.

Look at the restriction to X_{r−1} = x₁,…,x_{r−1} (drop the last point). Every subsample (labeling) t that S induces on X_{r−1} is of one of two types when I bring x_r back in. Type one: only one extension of t to the full sample occurs — either t alone (with x_r excluded) or t with x_r appended, but not both. Type two: both extensions t and t∪{x_r} occur. Let a = number of type-one labelings, b = number of type-two labelings. Then the labelings on X_{r−1} number

    Δ^S(X_{r−1}) = a + b,

while on the full X_r every type-one labeling contributes 1 and every type-two contributes 2:

    Δ^S(X_r) = a + 2b.

Now here is the key. Collect the sets that realize the type-two labelings into a subfamily S″. For S″, a labeling t on X_{r−1} comes with both extensions, meaning x_r can be freely included or excluded — x_r is "free." So if S″ shattered some subset of size n−1 inside X_{r−1}, then adding x_r to that subset would be shattered by S — a set of size n shattered, contradicting our assumption. Therefore S″ shatters no subset of size n−1 among x₁,…,x_{r−1}. By the induction hypothesis applied to the shorter sample X_{r−1} with parameter n−1 (the contrapositive: no (n−1)-shattered subset forces the index below Φ),

    b = Δ^{S″}(X_{r−1}) < Φ(n−1, r−1).

And S itself induces no n-shattered subset of X_{r−1} (it induces none anywhere by assumption), so again by induction on the shorter sample,

    a + b = Δ^S(X_{r−1}) < Φ(n, r−1).

Adding,

    Δ^S(X_r) = (a + b) + b < Φ(n, r−1) + Φ(n−1, r−1) = Φ(n, r),

by the Pascal recurrence. That contradicts Δ^S(X_r) ≥ Φ(n,r). So the lemma holds. (I was careful with the two counts: it's a+b on the smaller sample and a+2b on the full one, and the "extra" b is precisely the doubly-extendable family whose freedom on x_r is what would manufacture a shattered set one size bigger.)

Now the dichotomy drops out as the contrapositive. Suppose m^S is not identically 2^r, and let n be the first r at which m^S(r) ≠ 2^r — so sets of size n can no longer all be shattered; equivalently d = n−1 is the largest shatterable size, the capacity. Take any sample of size r > n. If its index reached Φ(n,r), the lemma (with parameter n) would extract a shattered subsample of size n, contradicting m^S(n) ≠ 2ⁿ. So Δ^S(X_r) < Φ(n,r) for every r > n, hence

    m^S(r) ≤ Φ(n,r) for r > n.

And Φ(n,r) = Σ_{k=0}^{n} C(r,k) ≤ rⁿ + 1 for r ≥ 0, n > 0 — polynomial of degree n = d+1. So:

    m^S(r) is either identically 2^r, or bounded by rⁿ + 1 where n = d+1 and d is the largest shatterable size.

That d — the size of the largest set S can shatter — is the capacity. It is not the cardinality of S (which is infinite), not a metric entropy, not the dimension of the parameter space directly; it's a purely combinatorial number measuring how rich S is on finite sets. And the lemma says shattering is the *only* obstruction to polynomial growth: if you can't shatter sets bigger than d, your index is forced below Φ(d+1,·), polynomial of degree d+1. There's no middle regime. This is the dichotomy I suspected. In this index argument the polynomial exponent is n = d+1, the first unshatterable sample size; a sharper count of the same projections shaves the exponent to d with m^S(r) ≤ Σ_{k=0}^{d} C(r,k), but the degree d+1 version already buys the dichotomy and everything downstream.

Now close the loop on convergence. Plug the polynomial growth into the deviation bound. Write the exponent supplied by the index argument as n = d+1. If the capacity d is finite, m^S(2l) ≤ (2l)ⁿ + 1, so

    P(π^(l) > ε) ≤ 4 [ (2l)ⁿ + 1 ] e^{−ε²l/8} → 0 as l → ∞,

for every ε > 0, with no dependence on P. So a finite capacity is a distribution-free sufficient condition for uniform convergence. If I want almost-sure convergence, sum the tails: for l > 2/ε² the series Σ_l 4[(2l)ⁿ + 1] e^{−ε²l/8} converges (polynomial times decaying exponential), so by Borel–Cantelli P(π^(l) → 0) = 1. And to size a sample: I want 4 m^S(2l) e^{−ε²l/8} ≤ η with m^S(l) ≤ lⁿ + 1, which I can solve to

    l ≥ (16/ε²) ( n log(16n/ε²) − log(η/4) ),

a clean log-linear-in-n sample complexity. The capacity (through n = d+1) plays exactly the role log N played in the finite case — but it's finite for half-spaces even though N is infinite.

Sanity checks. Rays {x ≤ a} on the line: a sample of r distinct points sorted, the events are nested, and they cut out exactly r + 1 distinct subsamples (the empty prefix up to the whole sample), so m^S(r) = r + 1 — polynomial of degree 1, capacity 1 (I can shatter one point but not two: two points can't both be {first labeled in, second out} since rays are ordered). So uniform convergence holds with probability one, and writing A = {x ≤ a}, P_A = F(a), ν_A = F_l(a), this is precisely P(sup_a |F_l(a) − F(a)| → 0) = 1 — Glivenko's theorem falls out as the special case where the capacity is 1. Fixed-threshold linear half-spaces {x : ⟨w,x⟩ ≥ 1} in Rⁿ have the parameter-space cell count Φ(n,r), hence capacity n; homogeneous half-spaces {x : ⟨w,x⟩ ≥ 0} in Rⁿ also have capacity n, with a central-arrangement count of the same degree; affine half-spaces {x : ⟨w,x⟩ + b ≥ 0} have one more free threshold parameter and capacity n+1. The "all subsets of [0,1]" class shatters every finite set, capacity ∞, m^S = 2^l, the bound never converges — and indeed uniform convergence fails. The cases line up.

That settles sufficiency. But is finite capacity *necessary*? The bound is one-directional — maybe some infinite-capacity classes still converge. To get a necessary-and-sufficient criterion I need to track not the worst-case index m^S but its typical size under P. Introduce the entropy

    H^S(l) = E log₂ Δ^S(x₁,…,x_l),

the expected log-number of distinct labelings on a random l-sample. From the obvious sub-multiplicativity of the index on a concatenated sample, Δ^S(x₁,…,x_k, x_{k+1},…,x_l) ≤ Δ^S(x₁,…,x_k) · Δ^S(x_{k+1},…,x_l), taking logs and expectations gives subadditivity H^S(l₁+l₂) ≤ H^S(l₁) + H^S(l₂). A subadditive sequence has H^S(l)/l → c for some constant c ∈ [0,1] (Fekete). And by an argument like the one for sums of i.i.d. blocks plus Chebyshev, the random variable (1/l) log₂ Δ^S concentrates around c, so it's genuinely the per-sample growth rate.

Claim (necessary and sufficient): uniform convergence in probability of ν to P over S holds if and only if

    H^S(l)/l → 0.

Sufficiency mirrors the earlier argument but now I split by the *random* index rather than the worst case. Redo P(C) ≤ ∫ (1/(2l)!) Σ_T θ(ρ(T X_{2l}) − ε/2) dP, set δ = ε²/16, and split the 2l-sample space into where log₂ Δ^S(X_{2l}) ≤ 2δl and its complement. On the small-index part the integrand is bounded as before by 2 Δ^S e^{−ε²l/8} ≤ 2 · 2^{2δl} e^{−ε²l/8}, and with δ = ε²/16 this is 2 · 2^{ε²l/8} e^{−ε²l/8} = 2 exp(−(1−log 2)ε²l/8) → 0. On the large-index part the integrand is ≤ 1, so it contributes at most the probability that (1/(2l)) log₂ Δ^S(X_{2l}) > δ, which → 0 precisely because H^S(l)/l → c = 0 forces the concentration mass below any positive δ. So both pieces vanish, P(C) → 0, P(Q) → 0.

Necessity is the harder direction, and this is where the shattering lemma comes back to do real work. Suppose c = lim H^S(l)/l > 0; I must exhibit an ε with P(sup|ν − P| > ε) not → 0. Work with the two-sample event C′ = {sup|ν′ − ν″| > 2ε}; a lower bound on P(C′) gives a lower bound on P(Q), because if Q failed on both halves (each half within ε of P) then sup|ν′−ν″| ≤ 2ε, so P(C′) ≤ 2P(Q) − P(Q)², i.e. P(Q) ≥ ½ P(C′). Now use Lemma 1 in reverse: a large index forces a shattered subsample. Pick a small q ∈ (0,1/4). For a sample of length L, set n = ⌊qL⌋. If Δ^S(X_L) ≥ Φ(n,L), then there's a shattered subsample of size n; and since c > 0, choosing q small enough that q log₂(2e/q) < c makes Lemma 4's concentration of (1/L)log₂ Δ^S force P(Δ^S(X_L) ≥ Φ(⌊qL⌋, L)) → 1. Apply this with L = 2l. With high probability the whole two-sample X_{2l} contains a shattered block y of size ⌊2ql⌋. For each partition of the 2l points into the two halves, I can choose a set A that includes exactly the points of y that landed in the first half. Then the contribution of y pushes ν′_A − ν″_A upward; the remaining points outside y add a hypergeometric correction. Classify partitions by r, the number of points of y in the first half. For the typical range 7εl ≤ r ≤ (q+ε)l, Chebyshev's inequality for the without-replacement draw outside y shows that the correction fails to cancel the y-contribution with probability tending to zero, and the fraction of partitions with r in that range tends to one when 0 < ε < q/7. Therefore P(C′) → 1 for a suitable positive ε, and then P(Q) ≥ ½P(C′) stays bounded away from zero (in fact tends to one through the displayed inequality). Uniform convergence fails. So H^S(l)/l → 0 is necessary as well as sufficient.

Let me stand back at the causal chain. The pain was that a rule chosen by fitting is selected for favorable empirical error, so only a uniform — sup over the whole class — control of empirical-vs-true frequency is meaningful; the per-event LLN is impotent against a posteriori selection. The infinite class S looks unboundable until I notice that on any finite sample it produces at most Δ^S ≤ 2^l distinct events, so its worst-case index m^S(l) is the right finite surrogate for the cardinality N. A ghost second sample erases the unknown P and confines the deviation to the 2l observed points; permutation symmetry over those points reduces the sup to a sum over ≤ m^S(2l) representatives, each contributing a hypergeometric tail ≤ 2e^{−ε²l/8}, giving P(π^(l) > ε) ≤ 4 m^S(2l) e^{−ε²l/8}. Then a single combinatorial lemma — that an index exceeding Φ(n,·) forces a shattered subset of size n — proves the growth function obeys a sharp dichotomy: identically 2^l, or polynomial of degree n, where n is the first unshatterable size, one more than the largest shatterable size d. Finite capacity makes m^S polynomial, the exponential kills it, and uniform convergence holds distribution-free with sample size ∝ (n/ε²)·log(n/ε²); the capacity is to the infinite class what log N was to the finite one, with the index argument carrying the harmless off-by-one n=d+1. Glivenko–Cantelli is the capacity-1 instance. And tracking the *expected* log-index instead of its worst case turns the sufficient condition into a tight one: uniform convergence ⇔ H^S(l)/l → 0, with the shattering lemma supplying the necessity.

The clean statements I've landed on:

    Growth function:   m^S(l) = max_{x₁,…,x_l} | { (𝟙_A(x₁),…,𝟙_A(x_l)) : A ∈ S } |.
    Capacity (largest shatterable size):  d = max { l : m^S(l) = 2^l }.
    Index dichotomy:  if d < ∞ and n = d+1 is the first unshatterable size, then m^S(l) ≤ l^n + 1.
    Sharper Sauer-type bound:  m^S(l) ≤ Σ_{k=0}^{d} C(l,k) ≤ (e l / d)^d  for l ≥ d ≥ 1.
    Uniform-convergence bound:  P( sup_{A∈S} |ν_A − P_A| > ε ) ≤ 4 m^S(2l) e^{−ε² l / 8},  l ≥ 2/ε².
    Sample size from m^S(l) ≤ l^n + 1:  l ≥ (16/ε²) ( n log(16 n/ε²) − log(η/4) )  ⇒  P(π^(l) > ε) ≤ η.
    Necessary & sufficient:  uniform convergence ⇔ H^S(l)/l = E log₂ Δ^S(x₁,…,x_l) / l → 0.

Made concrete, the growth function, the Sauer-type bound, and the deviation bound compute the capacity of a finite-VC class and its distribution-free guarantee on the running examples:

```python
import numpy as np
from itertools import product
from math import comb, log, exp

def growth_index(label_fn, S_params, points):
    """Delta^S on a fixed sample: number of distinct labelings sets in S induce."""
    seen = set()
    for theta in S_params:                      # enumerate (or sample) the class
        labels = tuple(int(label_fn(theta, x)) for x in points)
        seen.add(labels)
    return len(seen)                            # <= 2**len(points)

def shatters(label_fn, S_params, points):
    """True iff S realizes all 2^k labelings of these k points (Delta^S = 2^k)."""
    return growth_index(label_fn, S_params, points) == 2 ** len(points)

def capacity(label_fn, S_params, candidate_point_sets):
    """d = largest k for which some k-subset is shattered (the VC dimension)."""
    d = 0
    for pts in candidate_point_sets:            # supply finite sets to test
        if shatters(label_fn, S_params, pts):
            d = max(d, len(pts))
    return d

def sauer_growth_bound(d, l):
    """Sharper polynomial bound on the growth function: m^S(l) <= sum_{k<=d} C(l,k)."""
    if d == float('inf'):
        return 2 ** l
    return sum(comb(l, k) for k in range(d + 1))   # <= (e*l/d)**d for l >= d

def index_growth_bound(n, l):
    """Power bound from the index dichotomy: if m^S first drops below 2^r at n, m^S(l) <= l^n + 1."""
    return l ** n + 1

def uniform_convergence_tail(d, l, eps):
    """Distribution-free bound using the sharper growth estimate from capacity d."""
    assert l >= 2 / eps**2, "symmetrization (ghost-sample) lemma needs l >= 2/eps^2"
    return 4.0 * sauer_growth_bound(d, 2 * l) * exp(-(eps**2) * l / 8.0)

def uniform_convergence_tail_index(n, l, eps):
    """Distribution-free bound using m^S(l) <= l^n + 1."""
    assert l >= 2 / eps**2, "symmetrization (ghost-sample) lemma needs l >= 2/eps^2"
    return 4.0 * index_growth_bound(n, 2 * l) * exp(-(eps**2) * l / 8.0)

def sample_size_from_power_growth(n, eps, eta):
    """l guaranteeing P(sup_A |nu_A - P_A| > eps) <= eta when m^S(l) <= l^n + 1."""
    return (16.0 / eps**2) * (n * log(16.0 * n / eps**2) - log(eta / 4.0))

# --- the running examples line up with the capacity --------------------------
# rays {x <= a}: capacity 1  (m^S(l) = l+1)  -> recovers Glivenko-Cantelli
# homogeneous or fixed-threshold linear half-spaces in R^n: capacity n
# affine half-spaces in R^n: capacity n+1
# all subsets of [0,1]: capacity infinity (m^S = 2^l) -> no uniform convergence
def ray_label(a, x):  return x <= a           # one real parameter a
# capacity(ray_label, np.linspace(-3, 3, 4001), [[0.0], [0.0, 1.0]]) -> 1
```
