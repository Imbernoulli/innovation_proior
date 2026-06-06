OK, let me think this through from scratch. I have a ground set V of size m, a value function f on subsets, and a hard budget: choose S with at most k elements to make f(S) as large as possible. Maximum coverage is the picture I keep in my head -- m subsets of a universe, pick k of them to cover the most. I know that even maximum coverage is NP-hard, so exact optimization is off the table; what I actually want is something cheap that I can *prove* never does too badly. Not "it tends to work" -- an inequality that holds on every instance.

The dumb-but-natural thing to do is grow S one element at a time, each time grabbing whatever helps most right now: pick the element e maximizing f(S + e) - f(S), add it, repeat k times. Greedy. The problem is that greedy on a generic combinatorial objective has no guarantee at all -- I can usually cook up an instance where the locally-best first picks paint me into a corner and I finish arbitrarily far from OPT. So before I trust this, I need to find what is special about f here that could stop that from happening.

Let me write down what I actually know about f. It's monotone: bigger sets are worth at least as much, so I never want to *remove* anything, the only thing stopping me is the budget k. And it has diminishing returns -- the marginal value of an element drops (or stays) as the set it's added to grows: for A inside B and e not in B,

    f(A + e) - f(A) >= f(B + e) - f(B).

That's submodularity. Coverage has it: a set covers fewer *new* elements once more of the universe is already covered. Let me sit with this, because it's the only leverage I have.

Here's the worry with greedy, made concrete. Suppose OPT is some size-k set O with value f(O), and greedy after a few steps has some set S that's "wrong" -- it shares nothing with O. Why shouldn't greedy be stuck? Well, monotonicity says f(S ∪ O) >= f(O), so all of OPT's value is still *reachable* from where greedy stands -- adding O's elements on top of S can only help. The elements of O are sitting right there in V, available to greedy. So at least *some* element must still carry a decent marginal gain. The question is how to turn "OPT is still reachable" into a number.

Let me try to lower-bound the gain greedy makes in one step. Greedy takes the single best marginal. In particular it does at least as well as adding any *one* specific element of O: for every o in O,

    f(S_{i+1}) - f(S_i) >= f(S_i + o) - f(S_i),

where S_i is greedy's set at the start of round i. That's just the greedy rule -- it beat o, so it beat the best one too. Now, which o do I plug in? Picking "the best o" feels right but I have no handle on it. Let me instead not pick at all and *average* over all of O. Since the left side doesn't depend on o, averaging the right side over the |O| <= k elements only weakens it:

    f(S_{i+1}) - f(S_i) >= (1/k) * sum_{o in O} [ f(S_i + o) - f(S_i) ].

Good, now I have a sum of single-element marginals on the right. This is where I want submodularity to do its work, because the thing I'd really like on the right is the *joint* gain f(S_i ∪ O) - f(S_i) -- "what OPT would add to greedy's current set all at once" -- since monotonicity will then connect it to f(O).

So I need: sum of the individual marginals of O's elements (each added to S_i alone) is at least the marginal of adding all of O at once. Let me check that's the right direction. Add the elements of O to S_i one at a time, in any order o_1, o_2, ...; the joint gain telescopes into a sum of marginals where the j-th element is added on top of S_i *plus the earlier o's*:

    f(S_i ∪ O) - f(S_i) = sum_j [ f(S_i + o_1 + ... + o_j) - f(S_i + o_1 + ... + o_{j-1}) ].

Each term here is a marginal of o_j on top of a set *bigger than* S_i. Diminishing returns says that marginal is no larger than o_j's marginal on top of S_i alone. So term by term,

    f(S_i ∪ O) - f(S_i) <= sum_{o in O} [ f(S_i + o) - f(S_i) ].

The sum of individual marginals dominates the joint gain. Wait -- that's exactly the inequality I wanted, and it points the right way: the right side of my averaging step is >= f(S_i ∪ O) - f(S_i). Chaining,

    f(S_{i+1}) - f(S_i) >= (1/k) [ f(S_i ∪ O) - f(S_i) ].

Now monotonicity closes it: S_i ∪ O contains O, so f(S_i ∪ O) >= f(O) = f(OPT). Therefore

    f(S_{i+1}) - f(S_i) >= (1/k) [ f(OPT) - f(S_i) ].

Let me stare at this for a second, because it's the whole ballgame. It says: every greedy step closes at least a 1/k fraction of the *remaining gap* to OPT. The gap to OPT shrinks geometrically. That's a contraction, and contractions I can solve.

Let me name the gap a_i = f(OPT) - f(S_i). Rearrange the inequality:

    f(OPT) - f(S_{i+1}) <= f(OPT) - f(S_i) - (1/k)[f(OPT) - f(S_i)]
    a_{i+1} <= (1 - 1/k) a_i.

So the gap multiplies by (1 - 1/k) each round. After k rounds (greedy starts at S_1 = empty and returns S_{k+1}):

    a_{k+1} <= (1 - 1/k)^k a_1.

And a_1 = f(OPT) - f(empty) <= f(OPT), since f is non-negative. So

    f(OPT) - f(S_{k+1}) <= (1 - 1/k)^k f(OPT)
    f(S_{k+1}) >= [ 1 - (1 - 1/k)^k ] f(OPT).

Now what is (1 - 1/k)^k? Use 1 - x <= e^{-x} with x = 1/k: (1 - 1/k)^k <= (e^{-1/k})^k = e^{-1} = 1/e. So

    f(S_{k+1}) >= (1 - 1/e) f(OPT) ≈ 0.632 f(OPT).

There it is. The plain greedy algorithm -- take the maximum marginal gain, k times -- never returns less than (1 - 1/e) of the optimum, on *every* instance. No instance-dependent badness, no restarts, no luck. And nothing magical happened with e: it's purely the residue of a geometric (1 - 1/k) contraction compounded k times. The 1/k came from averaging over the k elements of OPT; the contraction came from "each step kills a 1/k share of the gap"; e is just where (1 - 1/k)^k lands.

Let me make sure I didn't quietly need k = |O| exactly. I used |O| <= k when I divided by k in the averaging step -- dividing by k when |O| could be smaller only makes the lower bound smaller, so it's safe; the inequality still holds. Good. And every step used only the two structural facts: submodularity (for the termwise domination) and monotonicity (for f(S_i ∪ O) >= f(OPT)). Drop either and the chain breaks -- which tells me precisely the function class this lives on.

Two things nag at me, and I want to resolve both before writing code.

First: is (1 - 1/e) the best constant, or an artifact of my loose analysis? Maximum coverage is itself a monotone-submodular maximization, and there's a hardness result that no polynomial algorithm beats (1 - 1/e) for max coverage unless P = NP (and in the value-oracle model, beating it needs exponentially many queries). So the bound my two-line argument produced is the *true* frontier -- the loss isn't slack in my proof, it's the wall. That's a relief: greedy isn't just good, it's optimal for this class.

Second: I divided by k because |O| <= k -- I used that the constraint is a *cardinality* bound. What if the constraint were richer, say "S must be independent in a matroid" (partition budgets, a spanning constraint)? Then the averaging-over-O step gets shakier: I can no longer just smear greedy's one move against all of O uniformly, because greedy's feasible moves and O's elements interleave under the matroid's exchange structure. Working through the same engine under a general matroid, the clean (1/k)-per-step contraction degrades, and the guarantee one can squeeze out is only 1/2, not 1 - 1/e. So the cardinality case is special -- it's the uniform matroid, the most forgiving independence system -- and that's *why* it gets the sharper constant. Good to know where the 1 - 1/e lives and where it stops.

Now, code. The algorithm is almost embarrassingly direct, but the cost is in oracle calls: naive greedy evaluates the marginal of every remaining element every round, O(k m) evaluations of f, and f can be expensive. I notice I have an unused structural fact: marginals are *monotonically non-increasing* in the set they're conditioned on -- that's literally what submodularity says. So a marginal I computed for element e at an *earlier*, smaller S is an upper bound on e's marginal now. If I keep all the gains in a max-heap, the top is a possibly-stale upper bound. Re-evaluate just that top at the current S; if its refreshed gain is still at least every other (stale, hence upper-bound) entry, then by submodularity no other element can exceed it -- it's the true argmax this round, and I never touched the rest. Lazy evaluation. Same output as naive greedy, far fewer oracle calls.

Let me write both -- the oracle, the coverage instance, naive greedy, and the lazy version.

```python
import heapq


class SubmodularFunction:
    """Monotone submodular f over a ground set of indices, as a value oracle."""
    def __init__(self, ground_set):
        self.ground_set = list(ground_set)

    def value(self, S):
        raise NotImplementedError

    def marginal(self, e, S):
        # f(S + e) - f(S); >= 0 for monotone f, non-increasing in S (submodular).
        return self.value(S | {e}) - self.value(S)


class MaxCoverage(SubmodularFunction):
    """f(S) = | union of the sets indexed by S |. Non-neg, monotone, submodular."""
    def __init__(self, sets):
        super().__init__(range(len(sets)))
        self.sets = [set(s) for s in sets]

    def value(self, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(covered)

    def marginal(self, e, S):
        covered = set()
        for i in S:
            covered |= self.sets[i]
        return len(self.sets[e] - covered)   # only newly covered elements


def greedy(f, k):
    """Take the maximum marginal gain k times. >= (1 - 1/e) f(OPT)."""
    S = set()
    for _ in range(k):
        best_e, best_gain = None, -1.0
        for e in f.ground_set:                 # the argmax-marginal step
            if e in S:
                continue
            g = f.marginal(e, S)
            if g > best_gain:
                best_e, best_gain = e, g
        if best_e is None or best_gain <= 0:   # monotone f has plateaued
            break
        S.add(best_e)
    return S


def lazy_greedy(f, k):
    """Minoux's acceleration: a stale gain is an upper bound (diminishing
    returns), so the heap top, if its fresh gain still tops the heap, is the
    true argmax. Same output as greedy(), fewer oracle calls."""
    S = set()
    heap = []                                  # [-gain, element, size_when_evaluated]
    for e in f.ground_set:
        heapq.heappush(heap, [-f.marginal(e, S), e, 0])

    while len(S) < k and heap:
        neg_gain, e, evaluated_at = heap[0]
        if e in S:
            heapq.heappop(heap)
            continue
        if evaluated_at == len(S):             # fresh and on top => true argmax
            heapq.heappop(heap)
            if -neg_gain <= 0:
                break
            S.add(e)
        else:                                  # stale upper bound: refresh, re-sift
            heap[0][0] = -f.marginal(e, S)
            heap[0][2] = len(S)
            heapq.heapreplace(heap, heap[0])
    return S
```

So the whole chain: f is monotone and submodular; greedy's one move beats adding any single element of OPT, so it beats the average over OPT; submodularity turns that average of single marginals into the joint gain f(S_i ∪ O) - f(S_i), and monotonicity lifts that to f(OPT) - f(S_i); hence every step closes a 1/k fraction of the remaining gap, the gap contracts by (1 - 1/k) per round, k rounds leave at most (1 - 1/k)^k <= 1/e of it, and greedy lands within 1 - 1/e of the optimum -- which is exactly the best any polynomial algorithm can promise here. The diminishing-returns property that powered the proof also powers the lazy implementation, because it makes every stale marginal a safe upper bound.
