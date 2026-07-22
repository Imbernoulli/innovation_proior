OK, let me think this through from scratch. I have a ground set V of size m, a value function f on subsets, and a hard budget: choose S with at most k elements to make f(S) as large as possible. Maximum coverage is the picture I keep in my head -- m subsets of a universe, pick k of them to cover the most. I know that even maximum coverage is NP-hard, so exact optimization is off the table; what I actually want is something cheap that I can *prove* never does too badly. Not "it tends to work" -- an inequality that holds on every instance.

The dumb-but-natural thing to do is grow S one element at a time, each time grabbing whatever helps most right now: pick the element e maximizing f(S + e) - f(S), add it, repeat k times. Greedy. The problem is that greedy on a generic combinatorial objective has no guarantee at all -- I can usually cook up an instance where the locally-best first picks paint me into a corner and I finish arbitrarily far from OPT. So before I trust this, I need to find what is special about f here that could stop that from happening.

Let me write down what I actually know about f. It's monotone: bigger sets are worth at least as much, so I never want to *remove* anything, the only thing stopping me is the budget k. And it has diminishing returns -- the marginal value of an element drops (or stays) as the set it's added to grows: for A inside B and e not in B,

    f(A + e) - f(A) >= f(B + e) - f(B).

That's submodularity. Coverage has it: a set covers fewer *new* elements once more of the universe is already covered. Let me sit with this, because it's the only leverage I have.

The worry with greedy is concrete. Suppose OPT is some size-k set O with value f(O), and greedy after a few steps has some set S that's "wrong" -- it shares nothing with O. Why shouldn't greedy be stuck? Well, monotonicity says f(S ∪ O) >= f(O), so all of OPT's value is still *reachable* from where greedy stands -- adding O's elements on top of S can only help. The elements of O are sitting right there in V, available to greedy. So at least *some* element must still carry a decent marginal gain. The question is how to turn "OPT is still reachable" into a number.

Let me try to lower-bound the gain greedy makes in one step. Greedy takes the single best marginal. In particular it does at least as well as adding any *one* specific element of O; if o is already in S_i, then f(S_i + o) - f(S_i) is just 0, and monotonicity keeps greedy's gain non-negative. So for every o in O,

    f(S_{i+1}) - f(S_i) >= f(S_i + o) - f(S_i),

where S_i is greedy's set at the start of round i. That's just the greedy rule. Now, which o do I plug in? Picking "the best o" feels right but I have no handle on it. Let me instead not pick at all. Since the left side doesn't depend on o, it is at least the average marginal over the |O| elements of O; because those marginals are non-negative and |O| <= k, replacing |O| by k only lowers the right side:

    f(S_{i+1}) - f(S_i) >= (1/k) * sum_{o in O} [ f(S_i + o) - f(S_i) ].

Good, now I have a sum of single-element marginals on the right. This is where I want submodularity to do its work, because the thing I'd really like on the right is the *joint* gain f(S_i ∪ O) - f(S_i) -- "what OPT would add to greedy's current set all at once" -- since monotonicity will then connect it to f(O).

So I need: sum of the individual marginals of O's elements (each added to S_i alone) is at least the marginal of adding all of O at once. Let me check that's the right direction. Add the elements of O to S_i one at a time, in any order o_1, o_2, ...; the joint gain telescopes into a sum of marginals where the j-th element is added on top of S_i *plus the earlier o's*:

    f(S_i ∪ O) - f(S_i) = sum_j [ f(S_i + o_1 + ... + o_j) - f(S_i + o_1 + ... + o_{j-1}) ].

Each term here is a marginal of o_j on top of a set *bigger than* S_i. Diminishing returns says that marginal is no larger than o_j's marginal on top of S_i alone. So term by term,

    f(S_i ∪ O) - f(S_i) <= sum_{o in O} [ f(S_i + o) - f(S_i) ].

The sum of individual marginals dominates the joint gain. Wait -- that's exactly the inequality I wanted, and it points the right way: after the 1/k averaging, I still get the 1/k-scaled joint gain. Chaining,

    f(S_{i+1}) - f(S_i) >= (1/k) [ f(S_i ∪ O) - f(S_i) ].

Now monotonicity closes it: S_i ∪ O contains O, so f(S_i ∪ O) >= f(O) = f(OPT). Therefore

    f(S_{i+1}) - f(S_i) >= (1/k) [ f(OPT) - f(S_i) ].

Let me stare at this for a second, because it's the whole ballgame. It says: every greedy step closes at least a 1/k fraction of the *remaining gap* to OPT. The gap to OPT shrinks geometrically. That's a contraction, and contractions I can solve.

Equivalently, before naming the gap, I can write the same inequality as a concave-combination lower bound:

    f(S_{i+1}) >= (1/k) f(OPT) + (1 - 1/k) f(S_i).

So the next value is at least one fresh 1/k slice of OPT plus the remaining (1 - 1/k) share of what I already have. If I unroll that form for t additions, with S_0 empty and S_t after t rounds, the coefficients stack into a finite geometric sum:

    f(S_t) >= [1 + (1 - 1/k) + ... + (1 - 1/k)^{t-1}] * f(OPT) / k
           = [1 - (1 - 1/k)^t] f(OPT).

The same thing is even cleaner as a gap recurrence. Let a_i = f(OPT) - f(S_i). Rearrange the inequality:

    f(OPT) - f(S_{i+1}) <= f(OPT) - f(S_i) - (1/k)[f(OPT) - f(S_i)]
    a_{i+1} <= (1 - 1/k) a_i.

So the gap multiplies by (1 - 1/k) each round. After k rounds (greedy starts at S_0 = empty and returns S_k):

    a_k <= (1 - 1/k)^k a_0.

And a_0 = f(OPT) - f(empty) <= f(OPT), since f is non-negative. So

    f(OPT) - f(S_k) <= (1 - 1/k)^k f(OPT)
    f(S_k) >= [ 1 - (1 - 1/k)^k ] f(OPT).

Now what is (1 - 1/k)^k? If k = 1, it is zero. For k > 1, the tangent bound for log gives log(1 - x) <= -x on 0 < x < 1, so with x = 1/k,

    k log(1 - 1/k) <= -1,

and therefore (1 - 1/k)^k <= e^{-1} = 1/e. So

    f(S_k) >= (1 - 1/e) f(OPT) ≈ 0.632 f(OPT).

There it is. The plain greedy algorithm -- take the maximum marginal gain, k times -- never returns less than (1 - 1/e) of the optimum, on *every* instance. No instance-dependent badness, no restarts, no luck. And nothing magical happened with e: it's purely the residue of a geometric (1 - 1/k) contraction compounded k times. The 1/k came from averaging over the k elements of OPT; the contraction came from "each step kills a 1/k share of the gap"; e is just where (1 - 1/k)^k lands.

Let me make sure I didn't quietly need k = |O| exactly. If O is empty, then monotonicity already makes f(empty) optimal and the guarantee is immediate. Otherwise I used |O| <= k when I divided by k in the averaging step -- dividing by k when |O| could be smaller only makes the lower bound smaller, because all marginals are non-negative, so it's safe; the inequality still holds. If k = 0 there is no averaging step, but then the only feasible set is empty and OPT is empty too, so the guarantee is trivial. Good. And every real step used only the two structural facts: submodularity (for the termwise domination) and monotonicity (for f(S_i ∪ O) >= f(OPT)). Drop either and the chain breaks -- which tells me precisely the function class this lives on.

Now I need to know whether this proof is merely conservative. The recurrence imagines a very specific failure mode: each greedy step captures exactly a 1/k fraction of the value still sitting in each optimal block, and then the same thing happens again. Coverage can realize that.

Take k optimal blocks B_1, ..., B_k, each of size N; choose N = k^k so all the slice sizes below are integers. The k sets I wish greedy would take are O_i = B_i, and together they cover kN elements. Inside each block B_i, carve disjoint slices P_{i,1}, ..., P_{i,k} with

    |P_{i,t}| = (N/k) (1 - 1/k)^{t-1},

and let R_i be the leftover part of the block, of size N(1 - 1/k)^k. Now add k tempting cross-block sets

    G_t = P_{1,t} ∪ P_{2,t} ∪ ... ∪ P_{k,t}.

At the start, every O_i has value N, and G_1 also has value k * (N/k) = N. If the tie-break chooses G_1, then each block has exactly N(1 - 1/k) elements not yet covered. In round t, after G_1, ..., G_{t-1} have been chosen, any O_i has marginal

    N(1 - 1/k)^{t-1},

because it would cover the still-uncovered future slices in that block plus R_i. The cross-block set G_t has the same marginal, since it contributes one fresh slice from each of k blocks:

    k * (N/k)(1 - 1/k)^{t-1} = N(1 - 1/k)^{t-1}.

So the tie can keep sending greedy to G_t for t = 1, ..., k. Greedy's final coverage is

    sum_{t=1}^k |G_t|
      = N [1 + (1 - 1/k) + ... + (1 - 1/k)^{k-1}]
      = kN [1 - (1 - 1/k)^k],

while OPT takes O_1, ..., O_k and covers kN. The ratio is exactly 1 - (1 - 1/k)^k, which tends to 1 - 1/e. Tiny perturbations of the slice sizes can make the ties strict while moving the ratio by an arbitrarily small amount. So the proof is not hiding a better constant for this rule; the geometric loss is a real max-coverage obstruction.

The pressure point is the constraint. I divided by k because every element of OPT was a comparable feasible single move under a cardinality budget. If the rule were richer -- "S must be independent in a matroid," say a partition budget or another exchange system -- then I cannot smear greedy's one move uniformly over all of O in the same clean way. Feasibility now depends on which elements are already in S. The matroid exchange axiom can still pair greedy's accepted elements with elements of an evolving copy of OPT, and submodularity lets those displaced optimal marginals be charged to greedy's marginals, but the charge gives a comparison of the form f(OPT) <= f(G) + f(G), not a k-step geometric contraction. That is why the same plain greedy instinct has a 1/2 guarantee under a general matroid while the cardinality case -- the uniform matroid -- gets the sharper 1 - 1/e.

The code should mirror the proof. I need a value oracle, a coverage instance to make the oracle concrete, a tiny exhaustive checker so I can compare against OPT on a toy instance, and the selection routine whose only real decision is "scan all currently available elements and take the largest marginal." The proof itself is the plain max-marginal selector, so that is the code I should make runnable.

```python
from itertools import combinations
from math import e


class SetFunction:
    """Value oracle over a ground set of element indices."""
    def __init__(self, ground_set):
        self.ground_set = list(ground_set)

    def value(self, S):
        raise NotImplementedError

    def marginal(self, e, S):
        # f(S + e) - f(S)
        return self.value(S | {e}) - self.value(S)


class Coverage(SetFunction):
    """f(S) = | union of the subsets indexed by S |."""
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


def exact_best(f, k):
    """Exhaustive optimum for tiny instances, used only as a checker."""
    best_S, best_value = set(), f.value(set())
    limit = min(k, len(f.ground_set))
    for r in range(limit + 1):
        for combo in combinations(f.ground_set, r):
            S = set(combo)
            value = f.value(S)
            if value > best_value:
                best_S, best_value = S, value
    return best_S, best_value


def select(f, k):
    """Grow S by repeatedly adding the element with maximum marginal gain."""
    if k < 0:
        raise ValueError("k must be non-negative")

    S = set()
    for _ in range(min(k, len(f.ground_set))):
        best_e, best_gain = None, 0
        for e in f.ground_set:                 # the argmax-marginal step
            if e in S:
                continue
            g = f.marginal(e, S)
            if best_e is None or g > best_gain:
                best_e, best_gain = e, g
        if best_e is None or best_gain <= 0:   # monotone f has plateaued
            break
        S.add(best_e)
    return S


if __name__ == "__main__":
    sets = [
        {1, 2, 3, 8},
        {1, 2, 3, 4, 5},
        {1, 4, 6, 7},
        {5, 6, 7, 8},
        {2, 3, 9},
    ]
    f = Coverage(sets)
    k = 2

    chosen = select(f, k)
    opt, opt_value = exact_best(f, k)
    chosen_value = f.value(chosen)
    ratio = 1.0 if opt_value == 0 else chosen_value / opt_value

    assert chosen_value >= (1 - 1 / e) * opt_value
    print("selected", sorted(chosen), "value", chosen_value)
    print("optimum ", sorted(opt), "value", opt_value)
    print("ratio   ", round(ratio, 3))
```
