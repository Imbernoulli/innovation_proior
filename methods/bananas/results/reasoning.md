Let me start from what actually hurts. I want the architecture `a` in my search space with the lowest validation error `f(a)`, and the only way I can learn `f(a)` is to train that architecture — hours of GPU time per query — so a single evaluation is brutally expensive, there's no gradient of `f` with respect to a discrete graph, and the space is enormous: tens of thousands of architectures on the small benchmarks, up to `10^18` on the realistic ones. If I'm handed a budget of a few tens of evaluations, enumeration is laughable and uninformed random sampling spends the budget without ever compounding what it learns — each draw is a fresh gamble. So the whole problem reduces to a query-selection strategy: a thing that looks at the `(a, f(a))` pairs I've already paid for and uses them to pick the single most informative architecture to spend the next query on. That framing — model what you've seen, choose the next point to maximize expected progress, evaluate, update, repeat — is exactly Bayesian optimization, and BO is the right paradigm precisely because it's built for noisy, expensive, black-box objectives where each query is precious. So I'm not inventing the paradigm; I'm going to be ruthless about every component I plug into it, because the components are where the performance actually lives.

The textbook BO surrogate is a Gaussian process. I love GPs in Euclidean space: give me a kernel `kappa(x, x')`, condition on the data, and I get a closed-form posterior mean and variance, which is exactly the two numbers — predicted value `fhat` and uncertainty `sigmahat` — that every acquisition function wants. But here's where it breaks the moment I point it at architectures. A GP *is* its kernel; the kernel is a similarity function on the input domain, and my inputs are labeled DAGs, not vectors. There is no off-the-shelf kernel on the space of directed acyclic graphs with operation labels. So before I can even run GP-BO I have to invent a similarity function between two neural networks, and that's a hard modeling problem in its own right. NASBOT took exactly that route — they hand-built a distance called OTMANN, an optimal-transport distance over the "masses" of the layers, turned that into a kernel, and ran GP-BO on top with an evolutionary search to optimize the acquisition. And it works, but look at the cost: the distance metric is cumbersome to define, it has its own hyperparameters that need tuning, it's specific to a particular architecture parameterization, and underneath it all the GP still pays its `O(N^3)` inference cost — inverting a dense `N x N` covariance matrix every iteration, plus that matrix inversion step is itself a bottleneck as the evaluations pile up. So GP-BO for NAS makes me pay twice: once to design a bespoke distance function, and once more in cubic time. Wall.

Let me reconsider what the surrogate has to *be*. All I actually require from it is: consume the architectures-so-far, predict the validation error of an unseen architecture, and give me an uncertainty on that prediction. Nothing about that demands a GP. And there's prior art that already broke this open in the hyperparameter setting — Snoek's DNGO replaced the GP with a neural network, doing Bayesian linear regression on the net's last-layer features so that inference scales *linearly* in the number of observations instead of cubically; and BOHAMIANN used a Bayesian neural network sampled by Hamiltonian Monte Carlo. Their stated motivation was efficiency — escaping the GP's cubic scaling to enable massive parallelism — but for my problem a neural surrogate buys me something even more valuable than speed. A neural network consumes a *feature vector* and learns its own notion of similarity from the data. So if I use a neural predictor, the entire painful "invent a distance function between DAGs" problem evaporates: I don't define how architectures are similar, I just hand the net a featurization of each architecture and let it learn. Swapping the GP for a neural predictor resolves both problems at once: BO for NAS no longer requires a hand-crafted kernel, and the surrogate no longer has cubic GP inference. The surrogate is a neural network trained, from scratch, on the `(a, f(a))` pairs at every BO iteration, predicting `f` for unseen `a`.

But "hand the net a featurization of each architecture" just relocated the hard part. How do I turn a labeled DAG into a vector the predictor can read? The default, the thing everyone reaches for, is the adjacency-matrix encoding: fix an ordering of the nodes, put a binary feature for each possible edge `i < j`, and tack on the list of operations at each node. I keep wanting to use it because it's standard, but every time I think about what the predictor actually sees, I wince. First, the node ordering is arbitrary, so the *same* architecture maps to many different adjacency matrices depending on how I happened to label the nodes — the encoding isn't even well-defined per architecture, which is the isomorphism headache that forces people to bolt on isomorphism-removal subroutines. Second, and worse for a learner, the features are violently inter-dependent. An edge from the input into node 2 means nothing unless there's also a path from node 2 onward to the output — so the *usefulness* of that one binary feature is conditional on a bunch of other features. And if there is an edge from node 2 to the output, then that edge feature is strongly correlated with the feature describing which operation node 2 performs. Highly correlated, conditionally-meaningful, order-dependent features are exactly the kind of input a neural network has the hardest time fitting — it has to spend capacity untangling dependencies that the encoding manufactured. So the adjacency matrix is fighting my predictor. Wall.

What would a *good* encoding look like? I want features that are as close to independent as I can make them, each one carrying a self-contained piece of information about the architecture, and I want a single architecture to map to a single encoding so there's no isomorphism ambiguity. Stare at what a cell actually *does*: a data tensor enters at the input node and flows along edges through operations to the output node. Every distinct route the tensor can take from input to output is a *path*, described by the sequence of operations along it — input → `conv_1x1` → `pool_3x3` → output, say. So instead of describing the architecture by its wiring (edges between arbitrarily-numbered nodes), describe it by the set of computational paths it contains. Concretely: enumerate every possible input-to-output path expressed as an operation sequence, and give each one a binary feature that's 1 if that path is present in the architecture and 0 otherwise. To encode an architecture, walk it, list its paths, set those bits. That's the path encoding, and I can immediately see why it answers both complaints. There's no node ordering anywhere in the definition — a path is just an operation sequence — so each architecture maps to exactly one encoding; the representation is well-defined. And the features are far less entangled than adjacency bits, because each feature is a *complete* statement — "the tensor can flow input→conv→pool→output" — rather than a fragment whose meaning depends on other fragments. The features describe what the network computes, not how its nodes happen to be numbered.

There is one honest wrinkle: the path encoding is well-defined but not one-to-one — two genuinely different architectures can contain the same set of paths and so collide to the same encoding. At first that looks like information loss I should be worried about. But think about what a collision *means* for my predictor: if two architectures expose the same input-to-output operation routes, then the predictor is being asked to treat them as functionally close even if their internal wiring differs. That is a modeling assumption, not a theorem about exact equality of errors, but it is the right kind of smoothing for a low-budget search: collapse distinctions that are hard to learn and keep the computational routes that directly describe how data can flow. I'll keep the path encoding, with the collision caveat explicit.

Now I hit a scaling problem that nearly kills the idea. How long is the path-encoding vector? It's the total number of possible input-to-output paths in a cell. With `n` nodes and `r` operation choices per node, the number of paths of length `i` is `r^i` (one choice of operation at each of the `i` steps), so the full length is `sum_{i=0}^{n} r^i`, which is at least `r^n` — *exponential* in the number of nodes. Meanwhile the adjacency-matrix encoding I was complaining about scales merely *quadratically* in `n` (it's basically the `n(n-1)/2` possible edges). So on the face of it I've traded a quadratic encoding for an exponential one, which is absurd for any cell of real size. Wall. I'm not going to abandon the path encoding over this — its independence and well-definedness are too good — but I have to find a way to make it scale.

Let me look harder at *which* paths actually ever occur. NAS algorithms don't draw architectures uniformly from "all DAGs"; they sample from a constrained random procedure — pick `n` nodes, label each with one of `r` operations, and add each possible forward edge `(i,j)` (for `i<j`) independently with some small probability, subject to a cap on the expected number of edges — and then mutate those. The key intuition: a *long* path requires many specific edges to all be present simultaneously, and edges are individually rare under a sparse edge budget, so long paths are exponentially unlikely to appear. If the vast majority of the exponentially-many paths essentially never occur, then I can just *drop* them from the encoding — truncate to the paths that actually show up — and lose almost nothing. Let me make that precise instead of hand-waving, because "almost nothing" is exactly the kind of claim that needs a bound.

Set up the random architecture model carefully. Take `G_{n,k,r}`: `n` nodes labeled 1 to `n`; label each node with one of `r` operations; for every pair `i < j` add edge `(i,j)` independently with probability `2k / (n(n-1))`; and if there's no path from node 1 to node `n`, reject and resample. There are `n(n-1)/2` candidate edges, so that edge probability makes the expected number of edges exactly `k` — that's the whole reason for the `2k/(n(n-1))` factor. By "path" I mean a path from node 1 to node `n`. I want to show that when the cell is sparse — `k = n + c` for a small constant `c`, i.e. only a constant more edges than nodes — almost all the probability mass sits on a small number of *short* paths, so I can keep just those.

First, how many short paths are there? A path of length `ell` (visiting `ell` operations) has `r^ell` possible operation labelings. Let `L = floor(log_r n)`. The number of possible labeled paths of length at most `L - 1` is
`sum_{ell=0}^{L-1} r^ell = (r^L - 1)/(r - 1) <= (n - 1)/(r - 1) < n`.
Good — there are fewer than `n` paths below length `L`, so the set of the `n` shortest paths contains every path of length at most `L - 1`. If I can show paths of length at least `L` almost never appear, then "keep the short ones" means keeping a *linear* number of features, `O(n)`, not exponential. That's the prize.

Now I need the expected number of length-`ell` paths and I need to show it collapses for large `ell`. Let `a_{n,k,ell}` be the expected number of length-`ell` paths from node 1 to node `n` (in the version of the model without the rejection step, call it `G'`, which only inflates probabilities slightly). A length-`ell` path from 1 to `n` uses `ell - 1` intermediate nodes chosen from the `n - 2` non-endpoint nodes, and then needs all `ell` of its edges to be present, each independently with probability `2k/(n(n-1))`. So
`a_{n,k,ell} = C(n-2, ell-1) * (2k / (n(n-1)))^ell`.
Sanity check at `ell = 1`: `C(n-2, 0) = 1`, so `a_{n,k,1} = 2k/(n(n-1))`, and with `k = n + c` this is about `2/n`, which is `>= 1/n` for large `n`. So short paths — direct-ish routes — are not rare; their expected count is order `1/n` and up. Good, that's the denominator for the conditioning step.

Now bound `a_{n,k,ell}` from above for general `ell`. I'll use the standard binomial bounds `(n/ell)^ell <= C(n, ell) <= (en/ell)^ell`. Take the upper one:
`a_{n,k,ell} <= (e(n-2)/(ell-1))^{ell-1} * (2k/(n(n-1)))^ell`.
Pull out one factor of the edge probability and group the rest:
`a_{n,k,ell} <= (2k/(n(n-1))) * (2ek(n-2) / ((ell-1) n (n-1)))^{ell-1}`.
Since `k = n + c`, for large `n` we have `k/(n-1) <= 2`, and `(n-2)/n <= 1`, so `2ek(n-2)/(n(n-1)) <= 4e`, while the leading `2k/(n(n-1)) <= 4/n`. That gives the clean form
`a_{n,k,ell} <= (4/n) * (4e/(ell-1))^{ell-1}`.
The shape to notice: the bracket `(4e/(ell-1))^{ell-1}` is enormous when `ell` is small but plummets once `ell - 1` exceeds `4e` — superexponentially, because the base `4e/(ell-1)` drops below 1. So the expected path count is dominated by short paths and crushed for long ones. Let me sum the long tail, from `ell = L` up to `n - 1`:
`sum_{ell = L}^{n-1} a_{n,k,ell} <= sum (4/n)(4e/(ell-1))^{ell-1} <= sum (4e/(ell-1))^{ell-1}`.
Since `ell >= L` in the tail, replacing `ell - 1` by a slightly smaller threshold only changes constants, and for large `n` the ratio `4e/(L-1)` is below `1/2`. I can bound the tail by a geometric series:
`sum_{ell = L}^{n-1} a_{n,k,ell} <= 2 * (4e/(L-1))^{L-1}`.
Write the last expression as `2(4e)^{L-1}/(L-1)^{L-1}`. The numerator is only a fixed polynomial in `n`, because `L = Theta(log_r n)`: `(4e)^L = n^{log_r(4e)+o(1)}`. The denominator beats every fixed polynomial. The identity behind that is
`(log n)^{log n} = (e^{loglog n})^{log n} = (e^{log n})^{loglog n} = n^{loglog n}`;
with base-`r` logs it is the same statement up to constant factors, `(log_r n)^{log_r n} = n^{log_r log_r n}`. Since `log_r log_r n` eventually exceeds any fixed constant, for large enough `n`,
`sum_{ell = L}^{n-1} a_{n,k,ell} < 1/n^3`.
The expected number of *any* long path is below `1/n^3`. By Markov, the probability that a long path exists is also below `1/n^3` (in the no-rejection model).

Last step, handle the rejection. In the real model I condition on returning a graph that *has* some path from 1 to `n` — that conditioning is what step (4) does. The probability of having at least one path is at least the probability of the single direct edge `(1, n)` existing, which is `a_{n,k,1} >= 1/n`. So conditioning divides by something `>= 1/n`:
`P(exists a long path in G_{n,k,r}) = P(exists long path in G') / P(exists any path in G') <= (1/n^3)/(1/n) = 1/n^2`.
There it is. If I let `P'` be the `n` shortest paths and truncate the encoding to just those, the probability that a sampled architecture contains a path I *dropped* is at most `1/n^2`, vanishing as the cell grows. So I can truncate the exponential path encoding down to a *linear* number of features with a controlled probability of missing an observed path. The asymptotic theorem is a proof-of-concept — and I'll keep two honest caveats in view: it's stated for freshly-sampled architectures, not mutated ones, and the most *frequent* paths aren't guaranteed to be the most *informative* for predicting accuracy. But the structure of the argument tells me exactly which paths to keep: the shortest, most frequent operation paths. For a 7-node, 3-operation cell, the full count including the empty path is `sum_{i=0}^5 3^i = 364`; for a 4-node, 5-operation cell, the count including the empty path is `sum_{i=0}^3 5^i = 156`, while the nonempty-path code vector has `5 + 25 + 125 = 155` entries and a natural short-path cutoff such as 30. Cutting rare long-path features is the scalable version of the encoding.

So the encoding question is settled: path encoding, truncated to the frequent short paths, fed to a neural predictor. Now the next component the acquisition functions demand — uncertainty. Every acquisition function I'd want needs not just `fhat` but a `sigmahat`, a calibrated sense of how unsure the predictor is, so it can explore where it's ignorant rather than only exploit where it's confident. A single feedforward net gives me a point prediction and nothing else. How do I get uncertainty out of a neural surrogate? The Bayesian-NN route — BOHAMIANN's HMC-sampled posterior over weights — is the principled answer, and it gives a prediction and an uncertainty from one model. But it's a heavy hammer: Bayesian NNs are fiddly to implement, slow because they need long sampling chains, and the quality of the uncertainty is sensitive to the prior and to how well the posterior was approximated. I'm going to be retraining this surrogate at *every* BO iteration, so a slow, fiddly uncertainty mechanism is a real tax. Is there something cheaper that's actually as good?

There is, and it's almost embarrassingly simple: train an ensemble. Instead of one net, train `M` copies of the same architecture from different random weight initializations and different orderings of the training data, and for any input read off the *mean* of the `M` predictions as `fhat` and their *standard deviation* as `sigmahat`. The disagreement among independently-initialized nets is itself an uncertainty estimate — where they agree, the function is well-determined by the data; where they scatter, it isn't. And the evidence is that this ensemble uncertainty is as well-calibrated as a Bayesian NN, often better, even with `M` as small as 3 or 5 — the Deep Ensembles result. It's simple to implement (just train the same net a few times), trivially parallelizable (the members are independent), and needs almost no extra tuning. Given that I'm in a loop retraining the surrogate constantly, a small feedforward ensemble is the right surrogate: it has the uncertainty I need without the sampling cost and implementation friction of a Bayesian neural network. I'll set
`fhat(a) = (1/M) sum_{m=1}^M f_m(a)`, `sigmahat(a) = sqrt( sum_{m=1}^M (f_m(a) - fhat(a))^2 / (M - 1) )`,
with `M = 5`: small enough to retrain cheaply each iteration, large enough that the ensemble statistics from the Deep-Ensembles evidence are reliable. And one more thing the loop budget tells me — since the genuinely expensive operation is *evaluating* an architecture (training it), not training the tiny predictor, I can afford the predictor to be retrained from scratch with five members every single iteration; its cost is a rounding error against one real architecture evaluation.

With a mean and an uncertainty per architecture I can finally talk about the acquisition function — the rule that converts `(fhat, sigmahat)` into a single "evaluate this one next" score. I'm minimizing validation error, so let `y_min` be the best (lowest) error I've found so far, and "improvement" means landing *below* `y_min`. Let me write down the standard family and feel out which fits. Expected improvement is the expected amount by which a candidate beats `y_min`, treating the predictive density as normal `N(fhat, sigmahat^2)`:
`phi_EI(a) = E[max(0, y_min - f(a))] = integral_{-infty}^{y_min} (y_min - y) N(fhat, sigmahat^2)(y) dy`.
With `gamma = (y_min - fhat)/sigmahat`, the closed form is `sigmahat * (gamma Phi(gamma) + phi(gamma))`, so larger is better. Probability of improvement is just the chance of beating `y_min`:
`phi_PI(a) = P(f(a) < y_min) = Phi(gamma)`,
again with larger better, but it ignores *how much* you'd improve, so it tends to be overly exploitative. Upper-confidence bound is really a lower confidence bound because I'm minimizing: `phi_UCB(a) = fhat - beta * sigmahat`, an optimistic error estimate that explicitly rewards uncertainty by the knob `beta` (I'll use `beta = 0.5`), and here smaller is better. Thompson sampling samples one model from the posterior and acts greedily under it — with an ensemble, the analogue is to pick a single random ensemble member `f_mtilde` and score every candidate by *that* member, again selecting the smallest sampled error.

That last one bothers me a little. Plain Thompson sampling with an ensemble draws *one* member and uses it for *all* candidates, so all candidate scores are perfectly correlated through that single draw — the randomness is shared. I'd rather the exploration be decorrelated *across* candidates, so that two architectures the ensemble is unsure about get independent rolls of the dice rather than being judged by the same coincidentally-optimistic-or-pessimistic member. So let me sample *independently per architecture*: for each candidate `a`, draw `ftilde_a ~ N(fhat(a), sigmahat(a)^2)` from *its own* predictive normal, and score `a` by that draw — `phi_ITS(a) = ftilde_a`. Call it independent Thompson sampling. It keeps Thompson's two big advantages — it's a *stochastic* acquisition, so it slots directly into batch/parallel BO (I can ask for `k` architectures at once by taking the `k` best draws, and the stochasticity gives me diversity for free) — while decorrelating the exploration across the search space. The implementation convention can now be clean: maximize raw EI and PI, minimize UCB, TS, and ITS, or equivalently negate EI and PI so a single ascending sort chooses every acquisition. For the loop I want the stochastic, decorrelated, batch-friendly one: independent Thompson sampling, selecting the architecture(s) with the smallest independent sampled errors.

One component left, and it's the one that's easy to forget until it bites: even with a perfect acquisition function, I have to optimize it over the search space each iteration, and the space has up to `10^18` architectures — I obviously can't score every one. I need a candidate set of manageable size, say a hundred to a thousand, and then I score the acquisition over just those. The dumbest choice is to draw the candidates uniformly at random. But think about where my predictor is actually trustworthy: it was trained on the architectures I've evaluated, so it gives reliable predictions for architectures *near* that training set in the input space, and increasingly unreliable predictions far from it. A uniformly random candidate is, almost surely, far from everything I've measured — so I'd be scoring my acquisition on predictions the surrogate has no business making confidently, and chasing phantom optima. Wall. The fix is to generate candidates that are *close* to where I've already looked — specifically, take the architectures with the best validation error found so far and *mutate* them by small edits (change one operation or one edge), filling a candidate pool of about 100 with these neighbors. This does two good things at once: the candidates sit at small edit distance from evaluated architectures, exactly where the predictor is more reliable; and they cluster around the *best* region I've found, which is where the optimum is most likely to be. It's the same "mutate the best" instinct that powers regularized evolution, repurposed as acquisition optimization. So acquisition optimization = mutate the best-so-far architectures into a pool of candidates, score `phi_ITS` over the pool, pick the minimizer(s).

Now the loss function for the predictor, which I almost left as plain mean-squared or mean-absolute error. But what do I actually care about? In NAS I care intensely about getting the *good* architectures right — the ones with low validation error — and I don't care much whether the predictor nails a terrible architecture's error to the decimal. Plain MAE weights a 10-point error on a 90%-accuracy architecture and a 10-point error on a 50%-accuracy architecture identically, which is the wrong priority: it spends the net's capacity fitting the junk as carefully as the gems. I want a loss that puts *more* weight on low-error architectures. Take a fixed global lower bound `y_LB` on the best achievable validation error, and measure each prediction's error *relative to its distance above that floor*:
`L(y_pred, y_true) = (1/n) sum_i | (y_pred_i - y_LB)/(y_true_i - y_LB) - 1 |`.
This is a mean absolute *percentage* error against the shifted floor. For an architecture whose true error is just barely above `y_LB` — a near-optimal one — the denominator `y_true_i - y_LB` is small, so a given absolute prediction error becomes a large *relative* error and gets penalized hard; for a bad architecture far above the floor, the same absolute error is a small relative one and is largely forgiven. Exactly the asymmetry I want: the predictor is pushed to be accurate precisely in the low-error regime that drives the search. I'll set `y_LB` to a safe lower bound on the minimum validation error in the space (around 4.5 on these benchmarks' error scale).

Let me now assemble the full loop and see it cohere. Warm-start by drawing some number of architectures — about 10 — uniformly at random and evaluating them, just to have enough `(a, f(a))` pairs that fitting the ensemble is meaningful; with fewer than a couple of points the predictor is hopeless and I should keep sampling randomly. Then, each round: encode every evaluated architecture with the truncated path encoding; train the ensemble of `M = 5` feedforward predictors on the `(encoding, val_error)` pairs using the MAPE loss; generate a candidate pool of ~100 by mutating the best-so-far architectures; encode the candidates; predict the mean and std for each from the ensemble; compute the independent-Thompson-sampling acquisition (draw `ftilde_a ~ N(fhat_a, sigmahat_a^2)` per candidate); evaluate `f` on the `k` candidates with the smallest draws (taking `k > 1` to exploit parallel training); add them to the data; repeat until the budget is spent. Return the architecture with the lowest validation error seen. That's the algorithm. Every piece earned its place against an alternative I tried and rejected: neural surrogate over GP (no kernel needed, linear not cubic), path encoding over adjacency (independent, well-defined features), truncation justified by the long-paths-are-rare proof, ensemble over Bayesian-NN uncertainty (simple, calibrated, cheap), independent Thompson sampling over plain TS/EI/PI/UCB (stochastic and decorrelated), mutation over random candidates (predictor-reliable, near the optimum), MAPE over MAE (weight the good architectures).

Let me write it as the code I'd actually run, filling the empty selection slot in the search harness. The predictor is a small feedforward net retrained each round; I'll write it with a Keras-style sequential model and the MAPE loss, and the ensemble/acquisition logic on top.

```python
import numpy as np
from tensorflow import keras
import tensorflow as tf


# ---- the architecture featurization: the truncated path encoding ----
NUM_OPS = 5            # operations per edge in the cell
LONGEST_PATH = 3       # longest input->output path in this cell

def encode_paths(arch):
    """Binary indicator over input->output operation-paths.
    Full length = NUM_OPS + NUM_OPS**2 + NUM_OPS**3; each present path's
    index is set to 1. Path features are near-independent and node-order-free,
    unlike adjacency bits."""
    o = arch                                   # arch = list of edge op-indices
    L = sum(NUM_OPS ** i for i in range(1, LONGEST_PATH + 1))
    v = np.zeros(L, dtype=np.float32)
    # the cell's paths, as op-index sequences along input->output routes:
    # one length-1, two length-2, one length-3 (cell-specific blueprints)
    v[o[3]] = 1.0                                                  # length-1 path
    off = NUM_OPS
    v[off + o[0] * NUM_OPS + o[4]] = 1.0                           # length-2 path
    v[off + o[1] * NUM_OPS + o[5]] = 1.0                           # length-2 path
    off = NUM_OPS + NUM_OPS ** 2
    v[off + o[0] * NUM_OPS ** 2 + o[2] * NUM_OPS + o[5]] = 1.0     # length-3 path
    return v

def path_encoding(arch, cutoff=30):
    """Truncate to the cutoff most-frequent (lowest-index / shortest) paths;
    the long paths are rare under the sparse random-graph model."""
    full = encode_paths(arch)
    return full[:cutoff] if cutoff else full


# ---- the neural predictor: a small feedforward net, MAPE loss ----
def mape_loss(y_true, y_pred):
    # weight low-error (high-accuracy) architectures more: relative error
    # against a global lower bound y_LB on the best achievable val error.
    y_lb = 4.5
    frac = (y_pred - y_lb) / (y_true - y_lb)
    return tf.abs(frac - 1.0)

class Predictor:
    def fit(self, X, y, num_layers=10, width=20, epochs=150, lr=0.01):
        net = keras.models.Sequential(
            [keras.layers.Dense(width, activation='relu') for _ in range(num_layers)]
            + [keras.layers.Dense(1)])
        net.compile(optimizer=keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.99),
                    loss=mape_loss)
        net.fit(X, y, batch_size=32, epochs=epochs, verbose=0)
        self.net = net
        return self

    def predict(self, X):
        return np.squeeze(self.net.predict(X))


# ---- independent Thompson sampling over the ensemble's mean/std ----
def its_scores(ensemble_preds):
    """ensemble_preds: (M, num_candidates). For each candidate draw one
    independent sample from its own predictive normal N(fhat, sigmahat^2)."""
    preds = np.array(ensemble_preds)
    fhat = preds.mean(axis=0)                                  # ensemble mean
    sigmahat = preds.std(axis=0, ddof=1)                       # ensemble std
    return np.random.normal(fhat, sigmahat)                   # one draw per candidate


class NASOptimizer:
    """BO with path-encoded feedforward predictors and mutation candidates."""

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.warm_start = min(10, num_epochs)    # random points to seed the ensemble
        self.ensemble_size = 5                    # Deep-Ensembles: 5 is enough
        self.num_candidates = 100                 # acquisition-optimization pool
        self.num_arches_to_mutate = 1
        self.patience_factor = 5
        self.seen = {}                            # arch_tuple -> val_loss
        self.best_arch, self.best_val_loss = None, np.inf

    def _record(self, arch, val_loss):
        self.seen[tuple(arch)] = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss, self.best_arch = val_loss, list(arch)

    def _fit_ensemble(self):
        X = np.stack([path_encoding(list(a)) for a in self.seen])
        y = np.array([self.seen[a] for a in self.seen], dtype=np.float32)
        return [Predictor().fit(X, y) for _ in range(self.ensemble_size)]

    def _propose_next(self):
        ensemble = self._fit_ensemble()
        # acquisition optimization: mutate the best-so-far into a candidate pool,
        # staying edit-close to evaluated archs where the predictor is reliable.
        best = sorted(self.seen, key=lambda a: self.seen[a])[
            : self.num_arches_to_mutate * self.patience_factor
        ]
        cands = []
        while len(cands) < self.num_candidates:
            parent = list(best[np.random.randint(len(best))])
            child = mutate_architecture(parent)            # one-edge edit
            if tuple(child) not in self.seen:
                cands.append(child)
        Xc = np.stack([path_encoding(a) for a in cands])
        preds = [p.predict(Xc) for p in ensemble]          # (M, num_candidates)
        # ITS scores predicted ERROR; pick the architecture with the lowest draw
        scores = its_scores(preds)
        return cands[int(np.argmin(scores))]

    def search_step(self, epoch):
        if epoch < self.warm_start or len(self.seen) < 2:
            arch = random_architecture()
            while tuple(arch) in self.seen:
                arch = random_architecture()
        else:
            arch = self._propose_next()
        val_loss = self.api.query_val_loss(arch)
        self._record(arch, val_loss)
        return {"best_val_loss": self.best_val_loss, "queries": self.api.query_count}

    def get_best_architecture(self):
        return self.best_arch
```

The causal chain, start to finish: I needed a query-selection strategy for an expensive, gradient-free, discrete objective, which is the BO setting, but the textbook GP surrogate demanded a kernel on DAGs that doesn't exist and paid cubic cost — NASBOT's hand-built distance metric papered over the kernel but kept the cumbersomeness and the cubic time. Swapping the GP for a *neural* surrogate dissolved both problems: a net learns similarity from a feature vector and scales linearly. That relocated the difficulty to featurizing a DAG, where the default adjacency matrix gave order-dependent, correlated, isomorphic features that fight a learner — so I built the *path* encoding, one independent, well-defined feature per input-to-output operation-path, which is exponentially long until I proved that under sparse random sampling the long paths appear with probability under `1/n^2`, licensing truncation to a linear number of features and dropping rare features that are likely to be hard to estimate. For the uncertainty every acquisition function needs, a Bayesian NN was too slow and fiddly to retrain each round, so I took an *ensemble* of feedforward predictors and read mean and std off their disagreement — simple, calibrated, cheap. From the mean and std I derived the acquisition family and chose *independent* Thompson sampling, drawing a per-architecture sample so exploration is stochastic (hence batch-able for parallel training) and decorrelated across candidates. Because the search space is too large to score exhaustively, I optimized the acquisition by *mutating the best-so-far* into a small candidate pool, edit-close to evaluated architectures where the predictor is trustworthy and near the current optimum. And I trained the predictor with a MAPE loss against a lower-error floor so it spends its accuracy on the good architectures that actually matter. The whole thing is a BO loop whose surrogate, encoding, uncertainty, acquisition, and acquisition optimization were each chosen by elimination against a concrete alternative that failed.
