I call the method Probably Approximately Correct learning, or PAC learning for short, following Valiant's original framing. It is the proposal that learning a recognizer can be made into a rigorous complexity-theoretic question by combining the statistical idea of approximation with the computational idea of polynomial-time deduction. The goal is not to recover the hidden target program exactly, but to produce a hypothesis recognizer that is, with high probability over the random sample, almost as accurate as the target under the same distribution that generated the examples.

The setting is deliberately concrete. There is a fixed number t of Boolean variables, and the concept class is a collection of Boolean recognizers over those variables. One recognizer in the class is the hidden target. The learner receives positive examples drawn independently from an unknown but fixed distribution D over the inputs that the target accepts. The learner may also consult an oracle that reports whether a specific input is positive. The learner does not know D, yet error is measured under D. This makes the guarantee distribution-free in the sense that the same deduction algorithm must work for every D, while remaining meaningful because the error is judged on the actual world from which the examples came.

The learnability criterion is the heart of the definition. A concept class is PAC learnable if there is a single deduction algorithm that, for every target in the class and every distribution D, runs in time polynomial in t, in the size of the target, in one over the accuracy parameter epsilon, and in one over the confidence parameter delta, and outputs a hypothesis h such that with probability at least one minus delta the distributional error of h is at most epsilon. For the positive results the guarantee is one-sided sound: the hypothesis never accepts an input that the target rejects, so its only mistakes are false negatives, and the total mass of falsely rejected positives under D is at most epsilon.

The conceptual move is to give up exact recovery. If a rare input determines whether a variable belongs to the target, a feasible sample may never reveal that input, so demanding certainty after polynomially many examples is impossible even for a single conjunction. Instead, PAC learning accepts that the hypothesis may err on low-probability regions. Because error is measured under the same distribution that produced the examples, those unobserved regions are precisely the ones that matter least. Confidence is also relaxed: a finite random sample can be unlucky, so the promise holds with probability at least one minus delta rather than with certainty.

The technical engine is a progress bound over random examples. Let L(h, S) be the number of independent trials needed so that, if each trial succeeds with probability at least one over h, the probability of observing fewer than S successes is below one over h. A Chernoff lower-tail argument gives L(h, S) at most two h times the quantity S plus the natural logarithm of h. This bound is useful because it turns high current error into likely progress. If the current hypothesis still has error larger than one over h, then the next random positive example falls into the mistaken region with probability greater than one over h, and that example triggers a concrete structural repair. If at most S repairs are ever possible, then L(h, S) examples are enough to make persistent high error unlikely.

The simplest illustration is conjunctions. Start with the hypothesis that is the conjunction of every possible literal, both x_i and not x_i for each variable. This hypothesis accepts essentially nothing, so it is trivially one-sided sound. Every positive example deletes every literal that the example falsifies. A literal that truly belongs to the target can never be falsified by a positive example, so target literals survive forever. A literal that does not belong to the target is falsified as soon as the sample contains a positive example that disagrees with it. There are at most two t candidate literals, so the progress bound with S equal to two t gives a polynomial sample size. Once the deletions stop, the remaining hypothesis is exactly the target up to a set of rejected positives whose D-mass is at most one over h, and rescaling h to one over epsilon finishes the argument.

The same structure lifts to bounded CNF. Fix a maximum clause width k and initialize the hypothesis as the conjunction of all clauses containing at most k literals. There are fewer than (2t) raised to the power k plus one such clauses. A positive example cannot violate any clause that actually belongs to the target, so every deletion is safe. While the current hypothesis rejects more than a one over h mass of true positives, the next example lands in that rejected region with probability greater than one over h and deletes at least one clause. Because only polynomially many deletions are possible for fixed k, the progress bound again yields a polynomial sample complexity. Conjunctions are simply the special case k equals one.

A useful cross-check comes from the finite-hypothesis argument. If the learner returns any hypothesis consistent with the observed examples from a finite class H, then a hypothesis whose true error exceeds epsilon survives m independent examples with probability at most one minus epsilon raised to m, which is at most exp minus epsilon m. A union bound over H shows that m at least one over epsilon times the natural logarithm of the size of H plus the natural logarithm of one over delta suffices. For monotone conjunctions the class has size two to the n, and for general conjunctions it has size three to the n, because each variable can appear positively, appear negatively, or be absent. This gives the same logarithmic dependence on n and confirms that the deletion algorithm achieves the expected sample complexity.

Monotone DNF shows that the progress engine can also build a recognizer upward rather than only deleting constraints. Begin with the always-false hypothesis, which again accepts nothing and is one-sided sound. When a positive example is not yet accepted, use the oracle to strip away inessential positive coordinates until the remaining partial vector is a prime implicant of the target, then add that monomial to the hypothesis. Each added monomial is sound and new, and a monotone target of degree d has at most d prime implicants. While the hypothesis misses more than a one over h mass of positives, a fresh example reveals a missing implicant with probability greater than one over h. Thus L(h, d) examples and at most d t oracle calls are enough.

The boundary of the model is just as informative as the positive results. Unrestricted DNF over partial vectors is not known to be efficiently learnable in this sense, because testing whether a partial vector forces a DNF formula includes the tautology problem. That separation is what makes PAC learning a complexity-theoretic boundary rather than a vague claim that examples help. Some Boolean classes admit polynomial sample and time bounds, while others inherit computational obstacles that prevent a generic efficient learner.

The broader lesson is that learning becomes a theorem about progress through random examples. High error is not merely a failure to generalize; it is probability mass that makes the next example useful. By counting possible repairs and bounding the trials needed to see enough of them, PAC learning turns induction into a quantitative, computational statement. The code below simulates the deletion algorithm for general conjunctions: it samples positives from a hidden target conjunction, deletes every literal falsified by those positives, and checks that the final hypothesis is approximately correct under the same distribution.

```python
import random
import math


def learn_conjunction(n, examples):
    """Delete literals falsified by any positive example."""
    # Literals 0..n-1 are x_0..x_{n-1}; n..2n-1 are not x_i.
    alive = set(range(2 * n))
    for assignment in examples:
        falsified = [idx for idx in alive if assignment[idx % n] != (idx < n)]
        alive.difference_update(falsified)

    def hypothesis(assignment):
        for idx in alive:
            if assignment[idx % n] != (idx < n):
                return False
        return True

    return hypothesis


def random_target(n):
    """Generate a satisfiable conjunction of literals."""
    target = set(range(2 * n))
    for i in range(n):
        r = random.random()
        if r < 0.3:
            # variable absent from target
            target.discard(i)
            target.discard(n + i)
        elif r < 0.65:
            target.discard(n + i)  # keep x_i
        else:
            target.discard(i)      # keep not x_i
    if not target:
        target.add(random.randrange(2 * n))

    def target_func(assignment):
        for idx in target:
            if assignment[idx % n] != (idx < n):
                return False
        return True

    return target_func, target


def sample_positives(target, n, m):
    """Rejection-sample m positive examples uniformly."""
    out = []
    while len(out) < m:
        a = [random.randint(0, 1) for _ in range(n)]
        if target(a):
            out.append(a)
    return out


def empirical_error(target, hypothesis, n, trials=20000):
    """Estimate Pr[h rejects | target accepts] under uniform distribution."""
    positives = errors = 0
    for _ in range(trials):
        a = [random.randint(0, 1) for _ in range(n)]
        if not target(a):
            continue
        positives += 1
        if not hypothesis(a):
            errors += 1
    return errors / positives if positives else 0.0


def one_trial(n, epsilon, delta):
    target, _ = random_target(n)
    # Finite-class PAC bound: m >= (1/epsilon)(ln|H| + ln(1/delta)).
    m = math.ceil((1.0 / epsilon) * (math.log(3 ** n) + math.log(1.0 / delta)))
    examples = sample_positives(target, n, m)
    hypothesis = learn_conjunction(n, examples)
    return empirical_error(target, hypothesis, n)


if __name__ == "__main__":
    n, epsilon, delta = 8, 0.05, 0.05
    errors = [one_trial(n, epsilon, delta) for _ in range(100)]
    mean_err = sum(errors) / len(errors)
    print(f"n={n}, epsilon={epsilon}, delta={delta}")
    print(f"sample size m={math.ceil((1/epsilon)*(math.log(3**n)+math.log(1/delta)))}")
    print(f"empirical error mean={mean_err:.4f}, max={max(errors):.4f}")
```
