The canonical result I am presenting is the No Free Lunch theorem for black-box search and supervised learning. It answers the question that is usually hidden inside a benchmark table: when someone reports that their optimizer wins on nine out of twelve test functions, what does that actually license about the thirteenth, unseen function? The theorem says that, without an assumption about which problems are likely, the answer is nothing at all. Averaged uniformly over all possible cost functions, every search algorithm has exactly the same expected performance, and every learning algorithm has exactly the same expected off-training-set performance. Any advantage an algorithm enjoys on some problems is paid for, by an exact counting identity, on the complementary problems. All real performance therefore comes from the match between an algorithm's inductive bias and the actual distribution over problems, not from the algorithm itself.

The setting is deliberately finite, because any computation is finite anyway. There is a search space \(\mathcal{X}\) with \(|\mathcal{X}|\) points and a finite set of cost values \(\mathcal{Y}\) with \(|\mathcal{Y}|\) values. A problem is just a function \(f : \mathcal{X} \to \mathcal{Y}\), so the space of all problems is \(\mathcal{F} = \mathcal{Y}^{\mathcal{X}}\), whose size is the gigantic but finite number \(|\mathcal{Y}|^{|\mathcal{X}|}\). An algorithm is modeled as a map from a sample of already visited points and their observed costs to a new, previously unvisited point. Counting only distinct evaluations is important because it removes the messy, algorithm-and-problem-dependent phenomenon of revisiting points, and any revisiting algorithm can be made non-wasteful by remembering where it has been. Performance after \(m\) steps is any function \(\Phi\) of the cost sequence observed so far.

The theorem itself is clean. For any fixed cost sequence \(d_m^y\) and any number of steps \(m\), the sum over all problems of the probability of observing that sequence is independent of the algorithm:

\[
\sum_{f \in \mathcal{F}} P(d_m^y \mid f, m, a) \text{ is independent of } a.
\]

Because the sum over all problems is the same for every algorithm, the uniform average of any performance measure is also the same. That is the No Free Lunch statement: averaged over all problems, no algorithm can beat any other.

The proof is a short induction on the sample size. For the first step, the algorithm chooses some point \(d_1^x\) based only on its own definition, and the observed cost must be \(f(d_1^x)\). Summing the Kronecker delta \(\delta(d_1^y, f(d_1^x))\) over all functions pins the value of \(f\) at that one point but leaves the other \(|\mathcal{X}| - 1\) points free, giving \(|\mathcal{Y}|^{|\mathcal{X}| - 1}\) functions. That number does not depend on which point the algorithm picked, so the base case is algorithm-independent. For the inductive step, assume the claim holds for samples of size \(m\). A sample of size \(m+1\) is the size-\(m\) sample plus a new pair, so the probability factors into the probability of the first \(m\) costs times the probability of the new cost. The first factor is controlled by the inductive hypothesis. The second factor pins the value of \(f\) only at the new, unvisited point chosen by the algorithm. That point was not in the previous sample, so the values of \(f\) on the remaining unseen points are still completely free. The recursion therefore multiplies the size-\(m\) sum by exactly \(1/|\mathcal{Y}|\), which is independent of the algorithm. The induction closes, and the theorem follows.

The intuition underneath the algebra is that the algorithm's entire cleverness is about where to look next given what it has already seen. But the next point is, by construction, an unseen point, and when you sum over all functions the cost at an unseen point is a flat draw over \(\mathcal{Y}\). The algorithm's knowledge of the visited part buys it nothing about the unvisited part, because the unvisited part is equally likely to be anything. Past performance, summed over all problems, has no bearing on future performance. This turns Hume's old worry about induction into a bookkeeping statement over a finite set of functions.

There is a useful geometric way to read the same fact. If you place a non-uniform prior \(P(f)\) over problems, the performance of an algorithm is an inner product between a vector \(\vec{v}\), whose components are the probabilities the algorithm assigns to outcomes on each problem, and a vector \(\vec{p}\), whose components are the prior probabilities. No Free Lunch says that every deterministic algorithm's vector \(\vec{v}\) has the same length and the same projection onto the uniform diagonal. All algorithm vectors live on a cone around the diagonal; they differ only in their tilt. With a uniform prior, every inner product is identical, so every algorithm ties. The only way to outperform another algorithm is for \(\vec{v}\) to tilt toward a non-uniform \(\vec{p}\). In other words, all advantage comes from matching the prior. Uniform \(\mathcal{F}\) is not a claim that the world is uniformly random; it is the skeleton of optimization theory before any particular prior is imposed, and the skeleton plays no favorites.

A common misreading is to think the theorem says every algorithm is identical on every single problem. It does not. It flattens only the average over all problems. Head-to-head per-function behavior can be asymmetric: there may be functions where algorithm \(a_1\) beats \(a_2\) by a large margin and no functions where it loses by that same margin, as long as the signed differences still sum to zero. The wins and losses cancel in total, but their shape need not be symmetric. In supervised learning, for example, the random learner has a flat profile over targets under homogeneous off-training-set loss, so it is head-to-head minimax-superior to every other learner, even though it is exactly tied on average. That distinction matters if you care about worst-case per-problem behavior rather than expected behavior.

The learning version of the theorem is the same idea applied to off-training-set error. Generalization is about inputs the learner has not seen, so the relevant cost is the loss averaged over \(q \notin d_X\), not over the training set or a random test set that might overlap it. With a uniform prior over targets and a zero-one loss, summing over all targets makes the probability of any off-training-set cost independent of the learning algorithm. Cross-validation, choosing the candidate with the best held-out score, does as well over all targets as anti-cross-validation, choosing the one with the worst held-out score. To prefer cross-validation you need an assumption that connects held-out behavior to unseen behavior, and that is precisely the assumption No Free Lunch says you cannot have for free. Bias-variance decomposition is still valid, but it operates inside an assumed problem distribution; it is not an assumption-free guarantee.

The practical consequence is that there is no universally best optimizer or learner. Every algorithm's edge on some problems is offset on others, and the only source of real performance is the alignment between the algorithm's built-in assumptions and the structure of the problems it will actually face. Inductive bias is not a flaw and not an optional feature; it is the entire source of generalization. The No Free Lunch theorem prices Hume's assumption about induction at exactly everything: without it, no algorithm can help, and with it, all help comes from it.

The small Python script below makes the theorem concrete for a tiny search space. It enumerates every function from three points to two cost values, runs three different deterministic algorithms for two steps, and prints the average best-so-far cost and the number of functions that produce a fixed cost sequence. Both quantities are identical across the algorithms, illustrating the counting identity.

```python
import itertools


def all_functions(X, Y):
    """Return every function X -> Y as a dict."""
    return [dict(zip(X, vals)) for vals in itertools.product(Y, repeat=len(X))]


def run(algorithm, f, m):
    """Run a deterministic algorithm for m distinct evaluations."""
    sample = []
    for _ in range(m):
        x = algorithm(sample)
        sample.append((x, f[x]))
    return [y for _, y in sample]


def avg_best_so_far(algorithm, X, Y, m):
    funcs = all_functions(X, Y)
    total = 0
    for f in funcs:
        total += min(run(algorithm, f, m))
    return total / len(funcs)


def count_sequence(algorithm, X, Y, target_seq):
    funcs = all_functions(X, Y)
    count = 0
    for f in funcs:
        if tuple(run(algorithm, f, len(target_seq))) == tuple(target_seq):
            count += 1
    return count


if __name__ == "__main__":
    X = [0, 1, 2]
    Y = [0, 1]
    m = 2

    def fixed_order(sample):
        visited = {x for x, _ in sample}
        return next(x for x in X if x not in visited)

    def reverse_order(sample):
        visited = {x for x, _ in sample}
        return next(x for x in reversed(X) if x not in visited)

    def shifted_order(sample):
        visited = {x for x, _ in sample}
        if not visited:
            return X[1]
        return next(x for x in X if x not in visited)

    for name, algo in [("fixed", fixed_order),
                       ("reverse", reverse_order),
                       ("shifted", shifted_order)]:
        avg = avg_best_so_far(algo, X, Y, m)
        cnt = count_sequence(algo, X, Y, (1, 0))
        print(f"{name:8s}: avg best-so-far = {avg:.4f}, "
              f"functions producing (1,0) = {cnt}")
```
