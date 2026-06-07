OK, let me think this through from scratch. I have a bipartite graph, the left side L is sitting there known to me, and the right side R is going to walk in one vertex at a time. When a vertex j on the right arrives I see its edges for the first time, and I have to commit it — match it to one of its currently-free left neighbors, or drop it — before I'm allowed to see the next arrival, and I can never take a match back. The graph is promised to have a perfect matching of size n. I want my matching to be as large as I can make it, and I'm going to be judged by the worst case over all graphs and all arrival orders of (what I get) divided by (the best offline matching). So really I want to push that ratio as high as possible.

First let me just get a floor under my feet. The dumbest thing: when j arrives, if it has any free neighbor at all, grab one. What does that buy me? The output is a maximal matching — there's no edge left with both endpoints free, because if there were, the moment that right endpoint arrived it had that free left neighbor and I'd have matched it. And a maximal matching is always at least half of any other matching: take the maximum matching M*, and look at any of its edges {u,v}; if my matching left both u and v free I could have added {u,v}, contradiction, so at least one of u,v is touched by me. Charge each edge of M* to a touched endpoint, each vertex gets charged at most once, so |me| ≥ |M*|/2. So greedy already gives me 1/2. Fine. That's the floor.

Now can any deterministic algorithm do better than 1/2? I should not quietly assume it always grabs an edge, because maybe it tries to refuse early to save vertices for later. But refusal does not save it from a phase adversary. Take two unused left vertices a and b. Present a right vertex adjacent to both. If the algorithm matches it to a, I present the next right vertex adjacent only to a; if it matches to b, I present the next one adjacent only to b; if it refuses the first one, I present the next one adjacent only to a. In every case the deterministic online algorithm gets at most one match from these two arrivals. Offline gets two: it sends the first arrival to the other one of {a,b} and the singleton arrival to its unique neighbor. Repeat this on disjoint pairs. After n/2 phases, OPT = n and the deterministic algorithm has at most n/2. So determinism is pinned at 1/2 from above and below. There's no escaping it without randomness.

So I have to randomize. The obvious randomized move: when j arrives, pick a uniformly random free neighbor. Surely spreading my choices around blunts the adversary, who now can't predict which left vertex I'll burn. Let me stare at this, because it's the natural first attempt and I want to know if it's enough... and it isn't. The trouble is that the coin I flip at each arrival has no memory. Each arrival re-randomizes from scratch, so there's no global structure to my decisions — and the adversary, knowing my algorithm (just not my coins), can build a graph where my choices are *on average* misallocated. Concretely: use the arrival order n, n-1, ..., 1; put ones on the diagonal i=j; and add a dense block where the columns in the second half are all adjacent to the rows in the first half. For the first half of arrivals the algorithm pours its random matches into that dense block, eating up first-half rows; then the sparse diagonal part arrives needing exactly those rows, and many of them are gone. The expected size comes out n/2 + O(log n). Per-step random is essentially as bad as deterministic. So independent local randomness is the wrong kind of randomness. I'm spending my left vertices without any coherent plan, and the adversary exploits exactly that incoherence.

Let me sit with *why* the per-step coin fails, because the failure is the clue. The problem isn't that I'm random, it's that I'm random *anew each step*. There's no consistent notion of which left vertices I'm "protecting" and which I'm "willing to spend early." Each arrival, independently, might burn a left vertex that a later arrival desperately needs. What if instead I commit, once, at the very beginning, to a single global random preference order over the left side — and then every arrival just defers to that one order? Fix a uniformly random total ranking of L. When j arrives, among its free neighbors, take the one of *highest priority* in that ranking (smallest rank number). The ranking is drawn once and never touched again.

Why would one fixed random order beat n fresh coins? Because now my decisions are *correlated through a single object*. A left vertex that ranks high is consistently preferred whenever it's available, so it tends to get used early and deliberately, not by accident; a left vertex that ranks low is consistently held back, so it's still around when a sparse late arrival needs it. There's an implicit self-correction: the order makes me favor, among currently-eligible left vertices, the ones that have been eligible least often so far, because the ones eligible often have probably already been grabbed by some earlier arrival that preferred them. So the algorithm naturally stops over-investing in the "dense" part of the graph — the exact failure mode that sank per-step random and deterministic greedy. Let me call this rule RANKING. Initialization: random permutation σ of L. On each arrival j: if j has a free neighbor, match to the free neighbor minimizing σ.

I should double check this is genuinely different from per-step random and not secretly the same. Per-step random, on each arrival, picks uniformly among *currently free* neighbors — the distribution depends on who's still free, and is re-rolled each time. RANKING fixes the relative order of all left vertices up front, so the choice among free neighbors is *not* uniform and *not* independent across arrivals — it's whatever the one global order dictates. They differ. Good. Now: how good is RANKING? I need to actually pin the ratio.

Let me set up the cleanest worst case to think against. I can assume the graph has a perfect matching: if I remove a vertex outside some maximum matching, OPT stays the same, and the matching on the smaller graph is no larger than the matching on the original graph because the two runs differ by at most one alternating path starting at the removed vertex. So removing those extra vertices can only make the ratio weakly worse, which is exactly where a worst case lives. Number things so the perfect matching is the diagonal: row i ↔ column i. Now I want to argue the matrix can be taken upper-triangular. Here's the move: suppose I zero out everything below the diagonal, i.e. only keep edges with i ≤ j. RANKING run on this sparser graph can be *simulated* as a "refusal" version of RANKING on the original — a variant that sometimes declines to match an arriving vertex even when it could. And a refusal variant can only ever match a subset: by induction over arrivals, the set of free left vertices under refusal is always a superset of the set free under full RANKING, so anything refusal matches, RANKING matches too. Therefore RANKING on the sparser upper-triangular graph does no better than on the original, so the worst case is upper-triangular. Now the perfect matching is the diagonal and column j is adjacent only to rows 1..j.

On an upper-triangular instance, look at index i. Either row i (the left vertex) or column i (the right vertex) — or both — end up matched. Why at least one? Because the diagonal edge (i,i) belongs to the perfect matching; if both endpoints were free, RANKING would have matched column i to row i when column i arrived. So for every i, at least one of {row i, column i} is matched. Let D be the set of i where *both* are matched. Count vertices covered by my matching M: it's n + |D| (each i contributes at least one, the D's contribute two), and edges are half of covered vertices, so |M| = (n + |D|)/2. So E|M| = n/2 + (1/2)·E|D|, and E|D| = Σ_i Pr[row i and column i both matched]. That accounting says exactly how to measure the gain over one-half. I can try to lower-bound D directly, or I can lower-bound the matched vertices on the ranked side; the second route gives a cleaner recurrence and still proves the same matching-size bound because every matched vertex on that side corresponds to exactly one matched edge.

So I want, for each rank position, the probability that the matching reaches that deep. Let me think in terms of the random order on the rows (by the symmetry of RANKING the two sides are interchangeable, so I can think of rows arriving in random order and being matched to highest-ranked available column — same algorithm). Define x_t = the probability, over the random permutation σ, that the ranked-side vertex sitting at position t gets matched. I want a recurrence on x_t. Here's the engine. Fix the vertex v of rank t, and let u be its diagonal partner (v = m*(u)). If v itself is not matched — that event has probability 1 − x_t — then when u arrives it had v as a free neighbor, so u is certainly matched, and matched to a free neighbor whose rank is even *smaller* than t. So "v unmatched" forces "u matched into the top-(t−1) ranks." Let R_{t−1} be the set of arriving-side vertices matched to ranked-side vertices of rank ≤ t−1; its expected size is Σ_{s≤t−1} x_s. The seductive step is: if v is unmatched then u ∈ R_{t−1}, and if u were a *uniformly random* arriving-side vertex independent of R_{t−1}, then Pr[u ∈ R_{t−1}] = E|R_{t−1}|/n, giving

  1 − x_t ≤ (1/n) Σ_{s≤t−1} x_s.

Hold on. Stare at that for a second. I quietly assumed u is independent of R_{t−1}. That's false. R_{t−1} is determined by the whole permutation σ, and u = m*(v) is tied to v, which I placed at rank t in that same σ. The set of "who got matched into the top ranks" and "the partner of the rank-t vertex" are computed from the *same* random order; they're correlated. I can't just multiply E|R_{t−1}|/n. So this clean recurrence, as stated, is wrong — the independence I leaned on doesn't hold. I need to actually engineer the independence rather than assume it.

The fix is to perturb the permutation so that u becomes genuinely independent of the relevant matched-set. Start from a random permutation σ. Build σ′ by picking a ranked-side vertex v uniformly at random, pulling it out, and reinserting it at rank t. Now run RANKING on σ′. Let u = m*(v). The point of reinserting a *uniformly chosen* v at a fixed rank is that u is now uniform and independent of σ (it depends only on which v I happened to pick, which I drew independently of everything). I need the analogue of "v unmatched ⇒ u matched shallow," but transported across the perturbation. So I need: how does removing v and reinserting it at rank t move the matching?

Removing a single vertex and putting it back changes RANKING's matching by at most one alternating path that starts at v. Along that path the ranks of the ranked-side vertices move monotonically upward in the modified ranking: when a vertex loses the option it had, the next choice it can be pushed to is worse in the ranking, because it was already taking the best available option before the perturbation. Concretely, if v is left unmatched in the run on σ′, then for any permutation obtained by moving v to another rank, in particular the original σ, the diagonal partner u is matched to some vertex ṽ with σ(ṽ) ≤ t. Walk it carefully: u is matched in the moved run to a vertex v_i with σ_i(v_i) ≤ σ_i(v′), where v′ was u's match in the σ′ run; moving one vertex changes σ_i(v′) by at most one position, and the shallow-match property gives σ′(v′) < t, so σ_i(v_i) < t + 1, hence σ_i(v_i) ≤ t by integrality. So u ∈ R_t (matched at rank ≤ t). Now u is a uniformly random vertex, independent of σ, hence independent of R_t. Conditional on σ, Pr[u ∈ R_t] = |R_t|/n; take expectations:

  1 − x_t ≤ (1/n) Σ_{s≤t} x_s.

That's the corrected recurrence — the sum runs to t, not t−1, but it's now honestly derived, with independence manufactured by the perturbation rather than wished into being. Now solve the worst lower bound it permits. Let S_t = Σ_{s≤t} x_s. The inequality is 1 − (S_t − S_{t−1}) ≤ (1/n) S_t, i.e. S_t(1 + 1/n) ≥ 1 + S_{t−1}. The competitive ratio is (1/n) Σ_{s≤n} x_s = S_n/n, and the smallest S_n consistent with these constraints comes from making every inequality tight: S_t(1 + 1/n) = 1 + S_{t−1}. Solve that recurrence with S_0 = 0. Write it as S_t = (1 + S_{t−1})/(1 + 1/n) = (1 + S_{t−1})·n/(n+1). Then x_t = S_t − S_{t−1}; unrolling the minimizing sequence gives x_t = (1 − 1/(n+1))^t. Let me check the steady pattern: S_t = Σ_{s=1}^t (1 − 1/(n+1))^s, and therefore every actual instance has ratio at least S_n/n = (1/n) Σ_{s=1}^n (1 − 1/(n+1))^s = 1 − (1 − 1/(n+1))^n. As n → ∞ that's 1 − e^{−1} = 1 − 1/e ≈ 0.632. So RANKING is (1 − 1/e)-competitive.

I want to sanity-check the constant a second way, because it's the same e showing up and I want to trust it. Phrase it through the "EARLY" refusal variant — refuse to match row i if its column was already matched before it arrived — which on the complete upper-triangular matrix coincides with RANKING. There, working with w_t (the probability mass at "matched at time t") and the constraints that the m_t = Pr[some column matched at time t] satisfy m_t = 1 − (1/n)Σ_{s<t} w_s, the minimizing solution is forced to be greedy: w_t = (1 − 1/n)^{t−1}. Then summing Σ_t t·w_t with θ = 1 − 1/n, θ^T = 1 − α, the algebra gives E|M| ≥ n/2 + (n/2)(α + (1−α)ln(1−α)); comparing this lower bound with E|M| = αn forces α − 1 ≥ (1 − α)ln(1 − α), hence 1 − α ≤ 1/e and α ≥ 1 − 1/e. Same constant, same 1 − 1/e, by the discrete-to-continuous geometric decay (1 − 1/n)^t → e^{−t/n}. Good — the e is coming from the geometric thinning of "still-available probability" at rate 1/n per rank, integrated, which is exactly an exponential. The constant isn't a coincidence of one derivation.

And is 1 − 1/e the ceiling, not just what RANKING happens to get? Look at the complete upper-triangular matrix T and the algorithm RANDOM (match each arriving column to a uniformly random eligible row) — on T, RANDOM and RANKING have the same expected size. Track, as columns arrive, x(t) = columns remaining and y(t) = eligible rows; Δx = −1, and E[Δy]/E[Δx] = 1 + (y−1)/x, which in the continuum is dy/dx = 1 + (y−1)/x. Let z = (y−1)/x. Then dz/dx = 1/x, so z = ln x + C. The initial condition is x = n, y = n, hence C = (n−1)/n − ln n, and y = 1 + x((n−1)/n + ln(x/n)). Setting y = 1 gives x = n exp(−(n−1)/n) = n/e + o(n), so the matching has size n(1 − 1/e) + o(n). Then Yao's lemma closes it: the expected size of *any* randomized online algorithm against the uniform distribution over row-permutations of T is at most the best deterministic algorithm's expected size on that distribution, which (the best deterministic one is greedy w.l.o.g.) equals RANDOM's n(1 − 1/e) + o(n). So no randomized online algorithm exceeds 1 − 1/e, and RANKING attains it. RANKING is optimal.

That whole combinatorial argument *works*, but it took a perturbed permutation, a monotone-alternating-path lemma, a refusal variant, and a delicate recurrence — and the most natural version of the recurrence was outright wrong because of a hidden dependence I had to manufacture away. I have a nagging sense the truth is simpler than this. The constant 1 − 1/e is exactly the constant that shows up in the *fractional* online matching problem, where a deterministic water-filling rule — keep a level y_i on each left vertex, pour an arriving vertex into its lowest-level neighbors — gets 1 − 1/e with no randomness at all. Two different problems, two different algorithms, same magic number. That coincidence is begging for one analysis that explains both. Let me try to find it by writing the matching LP and chasing duality, the way the fractional side is analyzed.

The matching LP: maximize Σ x_ij subject to, for every vertex, the sum of its incident x's ≤ 1, and x ≥ 0. Its dual: minimize Σ_i α_i + Σ_j β_j subject to α_i + β_j ≥ 1 for every edge (i,j), and α, β ≥ 0. Weak duality says any feasible dual is an upper bound on the primal optimum OPT. So if I can produce a *feasible* dual whose value is at most (1/F)·|my matching|, then |my matching| ≥ F·(dual value) ≥ F·OPT — that would prove an F-competitive ratio. The plan, then: as I build the matching, also build a dual, keeping the dual's total value pinned at exactly (1/F) times my matching size, and arrange for the dual to be feasible. That's the standard online primal-dual recipe.

But there's a wall the moment I try to keep the dual feasible *deterministically* online. Feasibility means α_i + β_j ≥ 1 on every edge, including edges to left vertices I never matched (where I'd want α_i = 0) and arriving vertices that found all neighbors taken (where β_j = 0). If I insist on deterministic feasibility at every step, I'm essentially solving the fractional problem, and I already know the integral problem can't be that easy — deterministic gets only 1/2. So deterministic online dual feasibility for the *integral* matching is too much to ask. I need to relax it. What if the dual is only required to be feasible *in expectation*? That is, for every edge (i,j), E[α_i + β_j] ≥ 1, with the randomness being RANKING's random order. Weak duality still gives me what I want if I'm careful: if the dual value is always exactly (1/F)|matching| (deterministically) and the dual constraints hold in expectation, then taking expectations, E|matching| = F·E[dual value] ≥ F·OPT, since OPT is at most the expected value of an expectation-feasible dual (the constraints E[α_i + β_j] ≥ 1 mean the *expected* dual vector is feasible, and its value is E[dual value]). So feasibility-in-expectation is exactly the right relaxation. This is the crack that lets randomness in.

Now I need the right random dual. Switch to the threshold form of RANKING, which is equivalent: instead of a permutation, each left vertex i draws Y_i ∈ [0,1] uniformly and independently; an arriving j matches to its free neighbor with the smallest Y_i. (Sorted Y's are a uniformly random permutation, so this *is* RANKING.) The continuous Y is what lets me make the dual smooth. When i gets matched to j, I'll split a total of 1/F between α_i and β_j using a function of i's threshold. Pick a monotone non-decreasing g : [0,1] → [0,1] with g(1) = 1, and set, on the match (i,j),

  α_i = g(Y_i)/F,  β_j = (1 − g(Y_i))/F,

and α_i = β_j = 0 for everyone unmatched. Why split it this way? Because then on every match α_i + β_j = 1/F exactly, so the dual's total value is exactly (1/F)·(number of matches) = (1/F)|matching|, deterministically — the primal-to-dual ratio is locked at F no matter what the coins do. That's half of what I needed for free; the split by g is the one degree of freedom I'll tune. There's a nice reading of it: when i is matched to j, it generates value 1/F; i keeps g(Y_i)/F for itself (α_i) and *offers* (1 − g(Y_i))/F to j (β_j). Since g is increasing, a low-threshold (high-priority) i keeps less and offers more — and "j matches to the unmatched neighbor making the highest offer" is exactly "smallest Y_i," so the offer story reproduces RANKING precisely.

Now the only thing left is expected feasibility: for every edge (i,j), E_{Y_i}[α_i + β_j] ≥ 1, for every fixed setting of all the *other* thresholds. Fix all Y_{i′}, i′ ≠ i, and vary Y_i. I need a reference point. Run the algorithm on the graph with i deleted, G\{i}, with the same other thresholds. In that run j gets matched to some neighbor whose threshold I'll call y^c (the "critical value"); set y^c = 1 if j goes unmatched there. Two structural facts.

First, the dominance fact: in the real run (with i present), i gets matched whenever Y_i < y^c. Why? Suppose i is *not* matched when j arrives. Then up to that point the run with i present is identical to the run without i — i being unused means it never affected anyone — so j faces the same neighbors and would match to the same vertex with threshold y^c. But Y_i < y^c means i is a *better* (lower-threshold) free neighbor of j than that one, so j would have taken i. Contradiction. So Y_i < y^c forces i matched. Therefore E_{Y_i}[α_i] = E[g(Y_i)/F · 1{i matched}] ≥ ∫_0^{y^c} g(y) dy / F, integrating g over exactly the thresholds that guarantee a match.

Second, the monotonicity fact: β_j ≥ (1 − g(y^c))/F for *every* value of Y_i. Why? Compare the real run and the i-deleted run in lockstep. At every step, the set of free left vertices in the real run is a superset of the free set in the deleted run — proof by induction: the only way it breaks is if the real run matches some i′ that's still free in the deleted run while the deleted run instead grabs some i″ with Y_{i″} < Y_{i′}; but by the induction hypothesis i″ was also free in the real run, so the real run (which takes the smallest-threshold free neighbor) would have taken i″, not i′ — contradiction. If j is unmatched in the deleted run, then y^c = 1 and β_j ≥ 0 = (1 − g(1))/F. Otherwise, when j arrives, its free neighbors in the real run are a superset of its free neighbors in the deleted run, including the neighbor with threshold y^c. So in the real run j is matched to some neighbor with threshold ≤ y^c, and since g is increasing and β_j = (1 − g(threshold))/F, we get β_j ≥ (1 − g(y^c))/F. (This is why g must be monotone — it's what makes "matched to a no-worse threshold" turn into "β no smaller.")

Put the two together:

  E[α_i + β_j] ≥ (1/F)[ ∫_0^{y^c} g(y) dy + 1 − g(y^c) ].

So if I can choose g and F so that ∫_0^θ g(y) dy + 1 − g(θ) ≥ F for *all* θ ∈ [0,1], then E[α_i + β_j] ≥ 1 and the dual is feasible in expectation, and RANKING is F-competitive. Now I just maximize F. To make the bound tight, ask for equality: ∫_0^θ g(y) dy + 1 − g(θ) = F for all θ. Differentiate in θ: g(θ) − g′(θ) = 0, so g′ = g, so g(θ) = C e^θ. The boundary condition g(1) = 1 gives C = e^{−1}, so g(θ) = e^{θ−1}. Then ∫_0^θ e^{y−1} dy + 1 − e^{θ−1} = (e^{θ−1} − e^{−1}) + 1 − e^{θ−1} = 1 − e^{−1}. So F = 1 − 1/e, constant in θ — the inequality holds with equality everywhere, and this is the largest F for which a valid g exists. RANKING is (1 − 1/e)-competitive. There it is again, and this time the e fell straight out of g′ = g — the exponential is forced by demanding the dual split be self-consistent across all thresholds. No perturbation, no monotone-path lemma, no broken-then-fixed recurrence; just weak duality, a dominance lemma, a monotonicity lemma, and one ODE.

And now the coincidence with the fractional problem explains itself. In the deterministic fractional/water-level world, y_i is the *fraction of i's capacity used so far*, and the same dual accounting uses α_i = (1/F)∫_0^{y_i} g(y) dy and β_j = (1 − g(y_i))/F with the same condition ∫_0^θ g + 1 − g(θ) ≥ F to get 1 − 1/e — there the dual is feasible deterministically because the level moves continuously. The integral RANKING is the *same scheme* with the level replaced by a single random threshold Y_i, and the price of integrality is that feasibility now only holds in expectation. That's the unification: the fractional water-filling and the integral RANKING are one primal-dual algorithm differentiated only by whether the offline vertex's "level" is a deterministic accumulation or a one-shot random threshold, and the magic constant 1 − 1/e is the solution of the same integral equation in both.

Let me write the algorithm and a check. The core is tiny — one global random object, consulted greedily — and the simulation should show the two landmarks at once: the fixed priority rule near 1 − 1/e on the complete upper-triangular family, and deterministic first-neighbor greedy stuck at 1/2 on the standard half-trap.

```python
"""
RANKING for online bipartite matching.

Offline side L is known up front; online side R arrives one vertex at a time,
revealing its edges only on arrival. Each arriving j must be matched (or dropped)
irrevocably before the next arrival.

Two equivalent forms are implemented:
  - permutation form: fix a uniformly random total order on L, match each arriving
    j to its unmatched neighbor of smallest rank.
  - threshold form: each i in L draws Y_i ~ U[0,1] independently, match each
    arriving j to the unmatched neighbor of smallest Y_i. (This is the form that
    makes the primal-dual analysis natural; g(Y_i)=e^{Y_i-1} sets the duals.)

Both are the SAME algorithm: the sorted Y-values induce a uniformly random
permutation of L.
"""

import random
import math
import heapq


def initialize_state(L, rng):
    """Draw the persistent random priorities used for the whole run."""
    return {i: rng.random() for i in L}      # Y_i ~ U[0,1]; ties have prob 0


def choose_neighbor(avail, state):
    """Choose the available neighbor with smallest persistent priority."""
    return min(avail, key=lambda v: state[v])


def stateful_match(L, arrivals, neighbors, state):
    """Generic one-pass harness filled with RANKING's priority rule."""
    matched_to = {}          # i in L  ->  j in R
    for j in arrivals:
        # On arrival we only get to see edges of j; among its UNMATCHED neighbors,
        # take the one earliest in the fixed order. No look-ahead, no revocation.
        avail = [i for i in neighbors[j] if i not in matched_to]
        if avail:
            matched_to[choose_neighbor(avail, state)] = j
    return matched_to


def ranking_match(L, arrivals, neighbors, rank):
    """Run RANKING.

    L:         iterable of offline vertices (known in advance).
    arrivals:  list giving the online arrival order (vertices of R).
    neighbors: dict j -> iterable of its neighbors in L (revealed on arrival of j).
    rank:      dict i -> comparable key; j is matched to the unmatched neighbor
               with the SMALLEST key. A uniformly random key gives RANKING.

    Returns the matching as a dict i -> j.
    """
    return stateful_match(L, arrivals, neighbors, rank)


def random_rank(L, rng):
    """Uniformly random total order on L (the random priority permutation)."""
    return initialize_state(L, rng)


def greedy_match(L, arrivals, neighbors):
    """Deterministic first-neighbor greedy.

    Always yields a maximal matching, hence >= OPT/2; an adversary forces = OPT/2.
    """
    matched_to = {}
    for j in arrivals:
        for i in neighbors[j]:
            if i not in matched_to:
                matched_to[i] = j
                break
    return matched_to


# --- complete upper-triangular graph ------------------------------------------
# Rows (offline L) and columns (online R) are 1..n. The matrix is upper-triangular
# with ones on and above the diagonal: column j is adjacent to rows 1..j. The
# unique perfect matching sits on the diagonal (row j matched to column j).
# Columns arrive in the order n, n-1, ..., 1: the first column
# to arrive sees every row, so an unlucky ranking wastes a high-priority row that a
# later, sparser column would have needed. RANKING attains 1 - 1/e here in the
# limit, matching the general upper bound.
def upper_triangular(n):
    L = list(range(1, n + 1))
    neighbors = {j: list(range(1, j + 1)) for j in range(1, n + 1)}  # rows 1..j
    arrivals = list(range(n, 0, -1))                                 # n, n-1, ..., 1
    return L, arrivals, neighbors


def competitive_ratio_on_upper_triangular(n, trials, match_fn, init_state, seed=0):
    rng = random.Random(seed)
    L, arrivals, neighbors = upper_triangular(n)
    total = 0
    for _ in range(trials):
        state = init_state(L, rng)
        total += len(match_fn(L, arrivals, neighbors, state))
    return (total / trials) / n


def ranking_size_upper_triangular(n, rng):
    """Fast simulation of RANKING on the upper-triangular graph."""
    rank = random_rank(range(1, n + 1), rng)
    heap = [(rank[i], i) for i in range(1, n + 1)]
    heapq.heapify(heap)

    matched = 0
    for j in range(n, 0, -1):
        while heap and heap[0][1] > j:
            heapq.heappop(heap)
        if heap:
            heapq.heappop(heap)
            matched += 1
    return matched


def ranking_ratio_on_upper_triangular(n, trials, seed=0):
    """Monte-Carlo estimate of E[|RANKING|]/OPT on the n x n upper-triangular graph.
    OPT = n (the diagonal perfect matching). Should approach 1 - 1/e for large n."""
    rng = random.Random(seed)
    total = 0
    for _ in range(trials):
        total += ranking_size_upper_triangular(n, rng)
    return (total / trials) / n


def greedy_half_trap(n):
    """Instance where first-neighbor greedy gets exactly half of OPT."""
    if n % 2:
        raise ValueError("n must be even")
    half = n // 2
    L = list(range(1, n + 1))
    arrivals = list(range(1, n + 1))
    neighbors = {}
    for j in range(1, half + 1):
        neighbors[j] = L[:]
    for k in range(1, half + 1):
        neighbors[half + k] = [k]
    return L, arrivals, neighbors


def greedy_ratio_on_half_trap(n):
    """Deterministic greedy / OPT on its standard half lower-bound instance."""
    L, arrivals, neighbors = greedy_half_trap(n)
    return len(greedy_match(L, arrivals, neighbors)) / n


if __name__ == "__main__":
    target = 1 - 1 / math.e
    print("RANKING on the complete upper-triangular family")
    for n, trials in ((100, 2000), (1000, 2000), (10000, 500)):
        r = ranking_ratio_on_upper_triangular(n, trials=trials)
        print(f"n={n:5d}  E[RANKING]/OPT ~ {r:.4f}   (1-1/e = {target:.4f})")

    print("\nFirst-neighbor greedy on the deterministic half-trap")
    for n in (100, 1000, 10000):
        r = greedy_ratio_on_half_trap(n)
        print(f"n={n:5d}  GREEDY/OPT      = {r:.4f}   (1/2 = {0.5:.4f})")
```

The causal chain: greedy is maximal so it's at least half, and an adversary nails deterministic algorithms to exactly half; per-arrival coins are no better because memoryless randomness can be steered into wasting the very left vertices a later sparse arrival needs; the fix is one global random priority over the offline side, consulted greedily — RANKING — whose correlated decisions self-correct away from that waste; on the worst-case upper-triangular instance the probability that the rank-t vertex is matched obeys 1 − x_t ≤ (1/n)Σ_{s≤t} x_s once a permutation perturbation manufactures the independence the naive count assumed and lacked, and minimizing the resulting lower-bound constraints gives 1 − (1 − 1/(n+1))^n → 1 − 1/e, which Yao's lemma shows is the best any online algorithm can do; and reading the whole thing through LP duality, splitting a per-match value of 1/F into α_i = g(Y_i)/F and β_j = (1 − g(Y_i))/F with a dominance lemma and a monotonicity lemma forces ∫_0^θ g + 1 − g(θ) ≥ F, whose tight solution g(θ) = e^{θ−1}, F = 1 − 1/e re-derives the same bound from one ODE and unifies integral RANKING with fractional water-filling — feasible in expectation being the exact relaxation that lets randomness buy what determinism can't.
