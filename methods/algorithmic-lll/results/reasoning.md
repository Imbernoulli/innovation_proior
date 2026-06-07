OK, let me think this through from scratch. I have a probability space — a fair coin for each of a pile of boolean variables — and a family of bad events, each one saying "this particular clause is false." A uniform random assignment makes each k-clause false with probability exactly 2^{−k}. The Local Lemma tells me that if every clause shares variables with at most about 2^k/e others, then with positive probability *all* the clauses are satisfied at once. So a satisfying assignment exists. Fine. But I don't want the theorem; I want the assignment, and I want it fast.

The brute-force reading of the existence proof is useless and I should say exactly why, because the why is the whole problem. The Local Lemma guarantees Pr[all clauses satisfied] ≥ ∏(1−x(A)). That product is positive but it can be exponentially small — for a formula with m clauses each contributing a factor bounded away from 1, the good set has measure like c^m. So if I just draw a random assignment and check it, and redraw on failure, my expected number of draws is the reciprocal of that, which is exponential. The existence proof leans on conditioning on events of tiny probability — Pr[A_i | the other clauses are all satisfied] — and that conditioning is precisely the thing I can't simulate by sampling. The proof shows the good set is nonempty by an induction that never tells me where in the space to look.

So sampling-from-scratch is out. What's the smallest possible departure from it? Suppose I draw a random assignment and find violated clauses. It feels wasteful to throw the whole assignment away when each violation is witnessed by only k local variables. Why not just repair one local defect? Take a violated clause A. The reason it's violated is that the k variables in A happen to have fallen into the one forbidden pattern (every literal false). There are 2^k patterns for those k variables and only one is bad. So re-randomize *just those k variables* — flip a fresh coin for each of vbl(A), leave everything else exactly as it was. With probability 1−2^{−k} the new draw satisfies A. That's the move. Resample the variables of a violated event.

Let me write the loop and stare at it.

    pick an arbitrary violated clause A
    re-flip every variable of A, independently, afresh
    repeat until no clause is violated

That's it. It's almost embarrassingly simple. And immediately there's the fear that makes everybody distrust it: when I resample vbl(A), I might satisfy A but *break* some other clause B that was previously satisfied and shares a variable with A. Then I fix B and break C, and I chase my tail forever. "Fix one, break another." This is the thing I have to confront, because if it's a real obstruction the algorithm is worthless, and if it isn't, I need to understand precisely why not.

First, let me bound *what* can break. When I resample vbl(A), the only clauses whose truth value can possibly change are A itself and the clauses that share a variable with A — that's exactly Γ⁺(A), the inclusive dependency neighborhood, the clauses I assumed are few. Every clause disjoint from vbl(A) is literally untouched: same variables, same values, same status. So a resample is a *local* operation; the damage is confined to Γ⁺(A). That's the first reason locality matters: I assumed the dependency degree is small precisely so that each repair can only spill into a small neighborhood.

But "the damage is local" is not "the damage is bounded." It could still cascade: A spills into B, B's repair spills into C, on and on, the active frontier wandering around the graph indefinitely. I need an argument that the *total* number of resamples is small. And here's where I keep getting stuck if I try the usual tricks. There's no obvious monotone quantity. The number of violated clauses doesn't monotonically decrease — a resample can increase it. The "energy" of the assignment doesn't obviously go down. Any potential-function argument I reach for, the resample step can move the wrong way. So tracking some quantity that increases or decreases is the wrong frame. Let me drop it.

Let me think about *information* instead. The only thing the algorithm consumes is random bits — k fresh bits per resample. Suppose I'm worried the algorithm runs for a long time, say it performs M resample steps. Then it has consumed Mk fresh random bits, on top of the n bits of the initial assignment. Can I *reconstruct* all of those random bits from a short description? If a long run forces the random bits to be compressible, then a long run is by definition improbable, since random bits aren't compressible.

Let me set this up carefully. Imagine the random source as fixed in advance: an infinite list of independent fair bits for each variable, P^{(0)}, P^{(1)}, P^{(2)}, …, consumed in order — the first time I need a value for variable P I take P^{(0)}, the next time P^{(1)}, and so on. The initial assignment uses P^{(0)} for each variable; each resample of a variable advances to its next unused bit. After M resamples I'm sitting on some current assignment, and I've burned through n + Mk bits of this source (n for the start, k for each resample). Those bits are uniformly random; their total entropy is n + Mk.

Now I want to argue: from (the current assignment) plus (a short log of what the algorithm did), I can recover all n + Mk bits. If I can do that with a log substantially shorter than Mk bits, then I've compressed n + Mk uniform bits into fewer than n + Mk bits, which can't happen for most random sources — contradiction once M is large. So M can't be large (with high probability). That's the shape. Now I need to make the log short, and the whole game is in *how cheaply each resample can be logged*.

The cost of the log is where the whole thing is decided. To replay the algorithm I need to know, at each step, which clause I resampled. Naively that's log m bits per step (name one of m clauses) — too expensive, log m per step times M steps is huge. But I don't need to name clauses absolutely. I'm going to organize the resamples so each one is identified *relative to the previous one*. Watch: when a resample of A breaks something and I immediately go fix it, the clause I fix next is in Γ⁺(A) — one of at most d inclusive-neighborhood choices, where now d bounds |Γ⁺(A)|. So I can name it with log d ≈ k − O(1) bits, an *index into the inclusive neighborhood*, not an index into the whole formula. That's the crucial accounting: each "child" repair costs k−O(1) bits to log, while it consumes k fresh random bits. The bits I spend logging are strictly fewer than the bits the step injects. Each step runs an information deficit, and deficits accumulate, but the total information I'm carrying — the current state — is bounded. So the deficit can't grow forever. That's the engine: the slack d ≤ 2^{k−O(1)} is *exactly* the statement "naming the next repair costs fewer than k bits," which is exactly what makes the log shorter than the entropy it encodes.

Let me make the "which clause" cost concrete by structuring the repairs as a recursion. When I resample A, I look at its inclusive neighborhood, and for each clause there that is now violated, I recurse and fix it — and inside that, fix *its* newly-broken local clauses — before returning to A's neighborhood. A recursive procedure:

    fix(A):
        re-flip the variables of A
        while some clause D in Γ⁺(A) is now violated:
            fix(D)

where Γ⁺(A) = Γ(A) ∪ {A}. The outer driver just calls fix on violated clauses until none remain. Why recursion rather than a flat queue? Because the recursion gives me the cheap encoding for free: the tree of recursive calls *is* the log, and in that tree every child's label lies in the inclusive neighborhood of its parent's label, so each edge of the tree costs only an index-into-the-neighborhood, k−O(1) bits, never log m.

I should check exactly what a returned fix certifies. Claim: if fix(A) returns and τ is its complete recursion tree, then every clause that uses any variable touched by τ is satisfied at return. Suppose not; take a violated clause D that shares a variable with the touched region, and take the last recursive sub-call whose clause shares a variable with D. When that sub-call returned, D lay in its inclusive neighborhood, so D was satisfied; after that, by the choice of the last such sub-call, no later call touched a variable of D. So D is still satisfied, contradiction. Clauses outside the touched variables never changed. Thus a completed top-level call fixes its root and introduces no new violated clauses. If I always choose the lexicographically first violated root, the roots of completed top-level calls move forward in the fixed order, so there are at most m of them. The only way the procedure fails to finish quickly is that one local correction does not return quickly enough.

Back to the compression argument, now with the recursion forest as the log.

Let me fix the random source as a table A(x, i): for each variable x and each index i, A(x, i) is the i-th fresh bit I'd use for x. The initial assignment reads row i = 0; the j-th resample of x reads the next row. This indirect view is the clean way to talk about "which bit is current" without tracking time: in a recursion tree, the value of variable x that a given vertex saw is determined by how many earlier vertices in the tree also touched x. Concretely, walk the tree in its natural order (depth-first, children visited by a fixed lexicographic rule on their labels so the order is reconstructible from shape and labels alone); the occurrence index of x at a vertex v is the number of earlier vertices whose clause contains x. So given the tree, I know for every (vertex, variable) pair exactly which row of the table that vertex consulted.

For a single recursive call, the same statement is offset by the indirect assignment present when the call began. If the interrupted call is not the first top-level call, that offset matters, so I should not count the interrupted tree alone as a zero-offset object. The way to remove the offset is to keep the whole relevant connected component of the top-level recursion forest. I take the sequence of completed top-level recursion trees and the one interrupted tree; connect two trees when they share a variable; and keep the component W containing the interrupted tree. Trees outside W never change variables that W reads, so their offsets vanish on W. Ordering the trees in W by their root labels and walking each tree in its natural order gives one composite witness, and W is consistent with the original table.

Why does a *particular* composite witness W force specific table entries? Every vertex in W corresponds to a clause that was violated at the moment it was resampled. A k-clause is violated exactly when all k of its literals are false, i.e. the k relevant table entries take the one forbidden pattern. For every vertex v and every literal L in [v], consistency therefore pins down A(L, idx_W(vbl(L), v)). The occurrence-index bookkeeping makes the map (v, L) ↦ (vbl(L), idx_W(vbl(L), v)) injective, so a size-u witness has ku distinct table entries fixed to prescribed values. For a uniformly random table, the probability that all ku prescribed entries come out right is exactly 2^{−ku}. So:

    Pr[a fixed size-u composite witness is consistent with the random table] = 2^{−ku}.

Now I have to count possible size-u composite witnesses. Naively that could blow up: labelled trees over m clauses look like they have up to m^u labellings. I need the relative encoding to kill the m^u. Two facts help. First, a child label is always in Γ⁺ of the parent — at most d choices per edge, not m. Second, because the component W is connected through shared variables, I should be able to store only one absolute root label and recover the rest by local indices.

Let me build the encoding explicitly, because the constant matters and I want it tight. Take the infinite (2d)-ary tree I — every node has 2d children. Root it at the root clause R, and label its nodes by walking G: a node labelled D gets, among its 2d children, the |Γ⁺(D)| ≤ d "low" children labelled by the lexicographically i-th clause of Γ⁺(D), and another copy of the same labels on the d "high" children. Why two copies — low and high? Because I need the encoding to handle not just genuine recursion (a child fix-call, which goes on a low child) but also the act of *re-rooting*: to glue several recursion trees together into one connected object, and to encode a tree rooted anywhere, I walk *up* toward the root using high children. So low children encode "I recursed into this neighbor," high children encode "this neighbor is my parent in the original tree." 2d children give me room for both directions.

Then a composite witness W becomes: choose the root clause R (m choices); take a connected subtree T of I containing I's root; and give the edges of T a 2-coloring, marking ordinary recursion edges versus glueing edges where one recursion tree connects to the next. The map from witnesses to (R, T, coloring) is injective — I can reverse it, peeling off high-child paths to recover each original root and deleting glueing edges to split the forest back apart. So the count is: m for R, times the number of size-u subtrees of a (2d)-ary tree containing the root, times 2^u for the coloring. The number of u-vertex subtrees of I containing the root is a Catalan-type count; Knuth's exercise bounds it by (2ed)^u. Multiply by the 2^u colorings: (2ed)^u · 2^u = (4ed)^u. Now plug d = 2^{k−5}: 4ed = 4e · 2^{k−5} = e · 2^{k−3} < 2^{k−1}, since e < 4 = 2^2. So the number of size-u composite witnesses is at most m · (4ed)^u < m · (2^{k−1})^u = m · 2^{u(k−1)}. There's the bound I wanted, and the choice d = 2^{k−5} is exactly tuned so that 4ed slips under 2^{k−1} — that's where the "−5" comes from; it's the price of stuffing the 4e factor and the doubling-of-children under one clean power of two.

Now combine. Let X_u count size-u composite witnesses consistent with the random table. Then

    E[X_u] ≤ (number of size-u witnesses) · Pr[one is consistent] ≤ m · 2^{u(k−1)} · 2^{−ku} = m · 2^{−u}.

If I run the driver and any local correction is interrupted after at least log m + 2 invocations, taking log base 2, the connected component containing that interrupted tree is a large composite witness consistent with the table. The total expected number of consistent witnesses of size at least log m + 2 is

    Σ_{u ≥ log m + 2} m · 2^{−u} = m · 2^{−(log m)} · Σ_{u≥2} 2^{−u} = m · (1/m) · (1/2) = 1/2.

So with probability at least 1/2 over the choice of table, no large composite witness is consistent at all. Then no local correction is interrupted; each top-level correction uses fewer than log m + 2 resamples, and there are at most m completed top-level corrections. The whole run is O(m log m) resamples. If I'm unlucky and an interruption occurs, I restart with a fresh table; the expected number of restarts is at most 2. There's the running time, and notice what carried it: not a potential function, but the bald fact that random bits don't compress. A long run would force a large consistent witness, and the table does not have room for many of those.

What I find striking is not that a good assignment exists — the Local Lemma already says that. It's that the dumbest conceivable repair process — re-flip the variables of a violated clause — finds that assignment after only O(m log m) resamples in this SAT-only analysis. I used a fixed lexicographic discipline so the recursion forest is reconstructible, but the move itself contains no cleverness about how to flip. The proof of speed is an accounting identity (bits in ≥ bits logged) rather than anything about the structure of solutions.

Now I'm bothered by the "−5." The existential lemma allows neighborhoods up to ~2^k/e; my algorithm, as analyzed, needs them below 2^{k−5}, off by a constant factor about 32/e. Beck, Alon, Srinivasan, the earlier freeze-and-brute-force attempts all lost an *exponential* factor (2^{k/48}, 2^{k/8}, 2^{k/4}, 2^{k/2}); I've lost only a constant. But "only a constant" still isn't "nothing," and the loss came from the encoding — the 4e and the doubled children. The encoding is an artifact of *proving* the bound, not of the algorithm. Can I analyze the same algorithm without paying for an explicit injective encoding? Let me drop SAT-specific encoding and think directly in the general probability-space language, with general weights x(A) instead of the uniform 2^{−k}, and see if the constant evaporates.

Generalize the algorithm to arbitrary events: keep a current evaluation of all variables; while some event A is violated, resample vbl(A) (each variable afresh from its own distribution), repeat. Same locality, same "only Γ⁺(A) can change." I want to bound the *expected* number of times each individual event A is resampled — call it N_A — directly, and sum.

I need the log object again, but cleaner. Record C: step ↦ the event resampled at that step. From C I build, for each step t, a *witness tree* τ_C(t) that justifies why step t happened. Start with a root labelled C(t). Walk backward through the log i = t−1, t−2, …, 1: if the current tree has a vertex v whose label's variables meet C(i)'s variables — i.e. C(i) ∈ Γ⁺([v]) — attach a new child labelled C(i) to the *deepest* such v (break ties arbitrarily); if no such vertex exists, skip i. The result τ_C(t) is the tree of "everything in the causal past of step t that could have influenced it, arranged by depth."

The deepest-attachment rule isn't cosmetic; it forces a structural property I'll need. Claim: if u was added later than v in this backward construction and their labels share a variable, then u sits *deeper* than v. Reason: when u was attached I picked the deepest eligible vertex, and v (sharing a variable, hence in Γ⁺) was eligible, so u went at v's depth or below. Consequence: any two vertices at the *same* depth have labels that share no variable — every level of the tree is an independent set in G. In particular no two siblings carry the same label: the tree is *proper*. That property is exactly what makes the probability bookkeeping factorize, so it's worth the deepest-attachment rule.

Now the analogue of "a tree pins down table entries" can be done by coupling. Fix a proper witness tree τ. I claim Pr[τ occurs as some τ_C(t)] ≤ ∏_{v ∈ V(τ)} Pr[[v]]. I run a τ-check against the same random source the algorithm uses, the per-variable lists P^{(0)}, P^{(1)}, …: visit the vertices of τ in order of decreasing depth; at vertex v, draw fresh samples for vbl([v]) and test whether they violate [v]. The τ-check passes if every vertex's event comes out violated; since the draws for distinct (vertex, variable) pairs are independent fresh samples, Pr[τ-check passes] = ∏_v Pr[[v]] exactly. Now I match which sample each side reads whenever τ actually occurs in the algorithm's log. For a variable P in vbl([v]), let S(P) be the vertices deeper than v whose label also contains P. The τ-check visits in decreasing depth, and among vertices at v's own depth only v's label uses P (independent set per level!), so when the τ-check reaches v it has already drawn P exactly |S(P)| times, hence reads P^{(|S(P)|)}. In the real algorithm, at the step that resampled [v], variable P had been sampled at the start and then at exactly the deeper steps in S(P) — also leaving it at P^{(|S(P)|)}. Same value. And at that real step [v] was violated, because otherwise it would not have been resampled. So the τ-check sees the same values and also finds [v] violated. The check passes. Therefore Pr[τ occurs] ≤ Pr[τ-check passes] = ∏_v Pr[[v]]. No encoding, no constant — just the coupling.

Now I want to sum ∏_v Pr[[v]] over all proper witness trees rooted at A and show it's small. Plug the LLL hypothesis Pr[A] ≤ x(A) ∏_{B∈Γ(A)} (1−x(B)); call the right side x'(A) := x(A) ∏_{B∈Γ(A)} (1−x(B)). So Pr[[v]] ≤ x'([v]) and

    E[N_A] = Σ_{τ rooted at A} Pr[τ occurs] ≤ Σ_τ ∏_v Pr[[v]] ≤ Σ_τ ∏_v x'([v]).

I need Σ_τ ∏_v x'([v]) ≤ x(A)/(1−x(A)). The clean way: interpret ∏_v x'([v]) as a *probability* in a branching process and use that probabilities sum to ≤ 1. Define a Galton–Watson process that grows a proper witness tree rooted at A: start with the root A; in each round, for each existing vertex v and each label B ∈ Γ⁺([v]), independently add a child of v labelled B with probability x(B), or skip it with probability 1−x(B). Let p_τ be the probability this process produces *exactly* τ. Compute it: for each vertex v, the children present are some subset of Γ⁺([v]), and the absent labels are W_v ⊆ Γ⁺([v]), with each absent label B contributing a factor 1−x(B). So, accounting that the root is always born,

    p_τ = (1/x(A)) ∏_{v} ( x([v]) ∏_{B ∈ W_v} (1 − x(B)) ).

I want to turn the absent-label product into a clean per-vertex factor. For a fixed vertex v, let Ch(v) be its children. Since the present child labels and W_v partition Γ⁺([v]),

    ∏_{B ∈ W_v} (1 − x(B)) = (∏_{B ∈ Γ⁺([v])} (1 − x(B))) / (∏_{u ∈ Ch(v)} (1 − x([u]))).

Multiplying this over all v, the denominator is ∏_{u ≠ root}(1−x([u])), one factor for every non-root vertex supplied by its parent. So

    p_τ = ((1−x(A))/x(A)) ∏_{v} ( (x([v])/(1−x([v]))) ∏_{B ∈ Γ⁺([v])} (1 − x(B)) ),

where the (1−x(A))/x(A) out front fixes the root's bookkeeping: the root is born without paying x(A), and it has no parent denominator contributing 1−x(A). Now split Γ⁺([v]) = Γ([v]) ∪ {[v]}; the B = [v] term of the inner product is (1−x([v])), which cancels the (x([v])/(1−x([v]))) down to just x([v]):

    p_τ = ((1−x(A))/x(A)) ∏_{v} ( x([v]) ∏_{B ∈ Γ([v])} (1 − x(B)) ) = ((1−x(A))/x(A)) ∏_{v} x'([v]).

So ∏_v x'([v]) = (x(A)/(1−x(A))) · p_τ. Therefore

    Σ_τ ∏_v x'([v]) = (x(A)/(1−x(A))) Σ_τ p_τ ≤ x(A)/(1−x(A)),

because the Galton–Watson process makes at most one tree, so Σ_τ p_τ ≤ 1 (it might also run forever, which only loses mass). Putting it together:

    E[N_A] ≤ x(A)/(1−x(A)),  and the total expected resamples ≤ Σ_A x(A)/(1−x(A)).

And *that* condition is precisely the existential Local Lemma hypothesis — no 4e, no doubled children, no "−5." The same trivial algorithm now provably terminates under the *exact* condition that guarantees a solution exists. The constant I was annoyed about was an artifact of the explicit encoding; replacing the encoding with the coupling-plus-branching-process accounting removes it entirely. The x'(B) factor I defined wasn't pulled from a hat — it's the unique per-vertex weight that makes the witness-tree sum telescope into a branching-process probability, which is the only reason the sum is bounded by 1.

Two quick consequences fall out of the same machinery, worth noting because they cost nothing extra. The algorithm parallelizes: at each round, pick a maximal independent set of currently-violated events and resample them all simultaneously (their variable sets are disjoint, so the simultaneous resamples don't interfere). In the sequential simulation, the events resampled in parallel round j sit at depth j−1 of the witness tree, so a run of k parallel rounds forces a witness tree of depth k−1, hence of size ≥ k; the tail bound Σ_{|τ|≥k} ∏ x'([v]) ≤ (1−ε)^k Σ x(A)/(1−x(A)) (using a slightly stronger hypothesis Pr[A] ≤ (1−ε) x'(A)) makes the expected number of rounds O((1/ε) log Σ x(A)/(1−x(A))). And the whole thing derandomizes when the dependency degree is bounded: there are only polynomially many witness trees up to the relevant size (a consistent tree larger than u always contains a consistent sub-tree of size in [u,(k+1)u], so I only enumerate trees of size O(log m)), so I can use the method of conditional expectations to build, deterministically, a table against which no large tree is consistent, then run the algorithm on that fixed table with no chance of a long run.

Let me bring it back to the concrete k-SAT object I actually want to compute, because the whole point was constructing the thing, not the theorem. For CNF the events are clauses, vbl(A) is the clause's variables, "violated" is "all literals false," and resampling is re-flipping the clause's variables with fair coins. The selection rule is free — take any violated clause. The code is the loop exactly as I first wrote it.

```python
import random

def is_clause_violated(clause, assignment):
    # clause = list of (var, sign); satisfied iff some literal is true.
    # A bad event "this clause is false" = every literal false.
    for var, sign in clause:
        if assignment[var] == sign:
            return False
    return True

def violated_clauses(clauses, assignment):
    return [i for i, c in enumerate(clauses) if is_clause_violated(c, assignment)]

def random_assignment(n_vars, rng):
    # One fair coin per variable: a uniform point of the probability space.
    return [rng.random() < 0.5 for _ in range(n_vars)]

def search_good_assignment(clauses, n_vars, rng=None, max_resamples=None):
    rng = rng or random.Random()
    assignment = random_assignment(n_vars, rng)
    resamples = 0
    bad = violated_clauses(clauses, assignment)
    while bad:
        i = bad[0]  # any violated clause; the bound is selection-independent
        for var, _sign in clauses[i]:
            # Resample exactly vbl(A), leaving every other variable untouched.
            assignment[var] = rng.random() < 0.5
        resamples += 1
        if max_resamples is not None and resamples > max_resamples:
            raise RuntimeError("resample budget exhausted")
        bad = violated_clauses(clauses, assignment)
    return assignment, resamples
```

The causal chain, start to end: a good assignment exists but is exponentially rare, so sampling-from-scratch fails; the defect after one random draw is local, so repair it locally by resampling a violated event's variables; the fear that this loops forever is unfounded because each resample only touches the inclusive dependency neighborhood, and a long SAT-only correction would force a large consistent composite witness whose count is too small; sharpening that information bound into a coupling against the random source plus a Galton–Watson factorization replaces the lossy encoding and yields the exact tail E[N_A] ≤ x(A)/(1−x(A)), so the same dumb resample loop terminates fast under the very condition that the Local Lemma uses to promise the solution exists. The landing artifact is that loop, run on k-SAT.
