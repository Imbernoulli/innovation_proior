I want to take a learner that I only know to be a little better than chance and turn it into one that is essentially always right. Let me be honest about why the first thing I'd reach for doesn't work, because the obstacle is the whole story. If I have a subroutine that, on any distribution I hand it, comes back with a hypothesis of error at most a half minus some small edge gamma, the tempting move is to call it many times and take a majority vote. But every one of those calls sees the same distribution. There is nothing forcing the calls to disagree about *which* examples they get wrong. The learner can keep finding the same cheap regularity and keep falling down on the same hard pocket of the space, and a majority over a pile of correlated mistakes is still a mistake. So the problem is not "collect many weak hypotheses." It is "make their errors point in different directions."

That immediately tells me where the leverage is. The weak-learning promise is *distribution-free*: an edge on *every* distribution, not just the natural one. So I am allowed to be adversarial toward my own committee. After I've collected some hypotheses, I can construct a distribution that puts the mass exactly where the current committee is failing, and the weak learner is still obligated to return something with an edge *there*. That converts the weak learner into a probe I can aim. The thing I actually have to design is the aiming: how do I move from the current distribution to the next one, and once I have a stack of hypotheses each good on its own slice, how much do I trust each in the final combination.

Now, the existence question is already settled, and I should look hard at how, because the *way* it's settled tells me what's still clumsy. The first proof takes the weak learner and runs it three times on three manufactured distributions. The first hypothesis h1 comes off the original distribution. Then it cooks up a second distribution on which h1 is right exactly half the time — h1's advantage is deliberately neutralized — and gets h2 there. Then a third distribution concentrated on the examples where h1 and h2 disagree, giving h3. The majority of the three has error g(a) = 3a^2 - 2a^3 when each sub-hypothesis has error a. For the recursion to drive error down I need g(a) < a, and I should actually check that rather than take it on faith. Form a - g(a) = a - 3a^2 + 2a^3 and factor: a - 3a^2 + 2a^3 = a(1 - 3a + 2a^2) = a(1 - a)(1 - 2a). For a in (0, 1/2) all three factors are positive, so a - g(a) > 0, i.e. g(a) < a strictly; at a = 1/2 the (1 - 2a) factor vanishes and g = a exactly; for a > 1/2 it goes negative and the majority is worse. Let me sanity-check the boundary numerically: at a = 0.3, g = 3(0.09) - 2(0.027) = 0.27 - 0.054 = 0.216 < 0.3, and a(1-a)(1-2a) = 0.3(0.7)(0.4) = 0.084, which matches 0.3 - 0.216 = 0.084. Good — each layer strictly shrinks the error as long as the sub-calls stay below a half, and recursing drives it to zero. Beautiful as a proof that weak implies strong. But as a *machine* it bothers me. The final object is a deep recursive majority-of-three circuit, the bookkeeping charges every sub-call at one fixed worst-case error, and crucially, if one sub-call comes back far better than the worst case, the construction has no way to cash that in — it's locked to the pessimistic level.

The next attempt fixes the shape. Don't recurse into a tree; lay all the weak hypotheses out flat and take *one* majority over them. To decide the example weights it treats the run as a majority-vote game: for each example, roughly, how many more correct future votes does it still need to land on the right side of the final tally? That yields a binomial-tail weighting schedule that's provably near-optimal, and a number of rounds that's near the information-theoretic floor for a given edge. This is genuinely tighter. But the whole schedule is computed from a *fixed* edge gamma that I have to commit to before the run starts. In any real run that's a fiction: some induced distributions may be easy, others much harder, and a schedule pinned to one presumed worst-case edge throws away everything I learn round to round. And the final vote is *unweighted*, so a very strong round and a barely useful round count exactly the same when I tally the votes. That bothers me too. A hypothesis that was much better on its slice should plausibly both push the next distribution harder *and* speak louder in the final answer.

So what I keep returning to is that both predecessors treat the edge as a constant known in advance, when after the weak learner returns h_t I can just *look* at it. Under the current weighting I can measure how often it's wrong. That single number — the weighted error of this particular hypothesis on this particular distribution — is sitting right there, and the predecessors ignore it because they plan everything up front. Let me see how far I get by letting that number drive the machine. I'll ask it to do two jobs at once: set how aggressively I shift the distribution for the next round, and set how much this hypothesis counts in the final vote. A small weighted error should mean a strong round — shift hard and vote loud; a weighted error near a half should mean the round barely helped — shift almost nothing and vote almost not at all. I don't want to guess any of the constants; I want the error analysis itself to tell me what they must be, if it can.

Let me build it concretely. I keep a weight w_i^t on each training example, normalize to a distribution p_i^t = w_i^t / sum_j w_j^t, hand p^t to the weak learner, and get back h_t. I measure its weighted error. Let me work in the {0,1} convention first because it keeps the bookkeeping honest: labels and predictions live in {0,1}, and |h_t(x_i) - y_i| is 0 when h_t is right on example i and 1 when it's wrong. So

  eps_t = sum_i p_i^t |h_t(x_i) - y_i|

is exactly the weighted fraction misclassified. Now I want a multiplicative update — multiplicative because I want correct examples to decay geometrically and I want the total-weight bookkeeping to factor cleanly across rounds, which is the lesson from the weighted-majority / online-allocation template, where a weight is updated by a factor and the analysis tracks the sum of all weights. The update I'll try is

  w_i^{t+1} = w_i^t * beta_t^(1 - |h_t(x_i) - y_i|),

with some beta_t in (0,1) still to be chosen. Check the direction: if h_t is *correct* on i, the exponent 1 - 0 = 1, so the weight is multiplied by beta_t < 1 and shrinks — that example is handled, demote it. If h_t is *wrong* on i, the exponent 1 - 1 = 0, so the weight is multiplied by beta_t^0 = 1 and stays put — that example is still a problem, keep it heavy. That is exactly the pressure that forces the next weak learner onto the current failures. And the final combination should run on the same scale beta_t carries: if a tiny beta_t means I shrank the correct examples hard because the round was strong, then that round should get a large vote. The natural coefficient is log(1/beta_t), and the combined classifier predicts 1 exactly when

  sum_t log(1/beta_t) h_t(x) >= (1/2) sum_t log(1/beta_t).

Everything now hangs on beta_t, and I refuse to guess it. I'll track the total weight and see whether a bound chooses it for me.

Define W_t = sum_i w_i^t, with W_1 = 1 since I start uniform. After an update,

  W_{t+1} = sum_i w_i^t * beta_t^(1 - |h_t(x_i) - y_i|).

The exponent 1 - |h_t - y_i| lives in [0,1], and beta^x is convex on [0,1], so it lies below the chord joining its endpoints: beta^x <= 1 - (1 - beta) x for x in [0,1]. Substitute x = 1 - |h_t - y_i|:

  beta_t^(1 - |h_t - y_i|) <= 1 - (1 - beta_t)(1 - |h_t - y_i|).

Sum against w_i^t. Now sum_i w_i^t (1 - |h_t - y_i|) = W_t (1 - eps_t), because sum_i p_i^t |h_t - y_i| = eps_t by definition and W_t pulls out. So

  W_{t+1} <= W_t [1 - (1 - beta_t)(1 - eps_t)].

Chain this from W_1 = 1:

  W_{T+1} <= prod_{t=1}^{T} [1 - (1 - beta_t)(1 - eps_t)].

That's the upper handle on the surviving weight. I need a *lower* handle that ties the surviving weight to how many final mistakes I make, and then I squeeze. Take an example i that the final weighted vote gets wrong. "Wrong" means the weighted vote for its true label fell short, i.e. at least half the total coefficient mass effectively went the other way. Writing that condition multiplicatively (this is the same inequality the vote threshold encodes, just exponentiated), a final mistake forces

  prod_t beta_t^(-|h_t(x_i) - y_i|) >= (prod_t beta_t)^(-1/2).

Meanwhile the final weight of example i is just the running product of its update factors times its starting weight,

  w_i^{T+1} = D_1(i) * prod_t beta_t^(1 - |h_t(x_i) - y_i|) = D_1(i) * (prod_t beta_t) * prod_t beta_t^(-|h_t - y_i|).

For a mistaken i, plug in the inequality above for the last product:

  w_i^{T+1} >= D_1(i) * (prod_t beta_t) * (prod_t beta_t)^(-1/2) = D_1(i) * (prod_t beta_t)^(1/2).

Sum over the mistaken examples. The starting weights of the mistaken set sum to the training error rate (uniform start, so D_1(i) = 1/m and the count of mistakes over m is the error), giving

  W_{T+1} >= error * (prod_t beta_t)^(1/2).

Now squeeze the two handles on W_{T+1} together — lower bound <= upper bound:

  error * (prod_t beta_t)^(1/2) <= prod_t [1 - (1 - beta_t)(1 - eps_t)],

so

  error <= prod_t [1 - (1 - beta_t)(1 - eps_t)] / sqrt(beta_t).

And this product factors across rounds, so I can minimize each factor independently — which is where I'm hoping beta_t gets *chosen by the bound* instead of guessed. For one round, with eps = eps_t,

  f(beta) = [1 - (1 - beta)(1 - eps)] / sqrt(beta) = [eps + (1 - eps) beta] * beta^(-1/2)
          = eps * beta^(-1/2) + (1 - eps) * beta^(1/2).

Differentiate: f'(beta) = -(1/2) eps * beta^(-3/2) + (1/2)(1 - eps) * beta^(-1/2). Set to zero and multiply through by 2 beta^(3/2): -eps + (1 - eps) beta = 0, hence

  beta_t = eps_t / (1 - eps_t).

Before I trust that, let me make sure it's a minimum and not just a critical point, and that the formula does what I claimed numerically. Second derivative f''(beta) = (3/4) eps beta^(-5/2) + ... ; both terms positive for beta > 0, so f is convex and this is the minimizer. And a quick numeric pass: at eps = 0.1, the closed form gives beta = 0.1/0.9 = 0.1111; minimizing f over a grid of beta in (0,1) lands the minimizer at 0.1111 with f-min 0.6000, which equals 2*sqrt(0.1*0.9) = 2*sqrt(0.09) = 0.6. At eps = 0.3, closed form beta = 0.3/0.7 = 0.4286, grid minimizer 0.4286, f-min 0.91652 = 2*sqrt(0.21) = 0.91652. They agree to the digits I can read. There's the adaptivity, and it dropped straight out of minimizing the error bound. Read it back: if eps_t is small (a strong round), beta_t is small, so I shrink the correct examples hard and the vote coefficient log(1/beta_t) is large. If eps_t is near 1/2 (a useless round), beta_t is near 1, the distribution barely budges, and the coefficient is near 0. The round contributes in exact proportion to its measured quality, and I never had to know the edge in advance — I just read eps_t off each round and the formula does the rest. Substitute beta_t = eps/(1-eps) back into f:

  f(beta_t) = eps * sqrt((1-eps)/eps) + (1 - eps) * sqrt(eps/(1-eps)) = sqrt(eps(1-eps)) + sqrt(eps(1-eps)) = 2 sqrt(eps_t(1 - eps_t)).

So the training error of the combined classifier obeys

  error <= prod_t 2 sqrt(eps_t(1 - eps_t)).

Make the edge explicit: write eps_t = 1/2 - gamma_t, where gamma_t > 0 is how much better than chance round t was. Then 4 eps_t(1 - eps_t) = 4(1/4 - gamma_t^2) = 1 - 4 gamma_t^2, so each factor is sqrt(1 - 4 gamma_t^2). Using 1 - u <= e^{-u} with u = 4 gamma_t^2,

  sqrt(1 - 4 gamma_t^2) <= exp(-2 gamma_t^2),

and therefore

  error <= exp(-2 sum_t gamma_t^2).

This is the amplification in the form I was after. If every round clears a fixed edge gamma, then after T >= (1/(2 gamma^2)) ln(1/epsilon) rounds the training error is below epsilon — exponential decay in the number of rounds. But the algorithm never needs to *know* gamma. It just measures eps_t each round, sets beta_t from it, reweights, and votes. And the bound is better than worst-case: it accumulates the *actual* squared edges sum_t gamma_t^2, so a few unusually strong rounds buy progress that a fixed-edge schedule would have left on the table — exactly the slack the predecessors couldn't use.

Before I go further I want to actually run this machine on a tiny problem and watch it, because so far it's all algebra and I could easily have a sign or a normalization wrong. Take four points on a line, x = 1, 2, 3, 4, with alternating labels y = +1, -1, +1, -1, in the {-1,+1} convention, and let the weak learner be a decision stump: a single threshold rule sign(s(x - c)) for some c and orientation s. No single stump can get all four right — any threshold splits the line into a left block and a right block, but the labels alternate, so the best a stump can do is err on exactly one of the four. That makes it an honest weak learner with eps just under a half initially, and a sharp test of whether voting over reweighted stumps can reach zero training error where one stump cannot.

Start uniform, weights (0.25, 0.25, 0.25, 0.25). Round 0: the best stump is threshold at 1.5 predicting +1 left of it, giving predictions (+1, -1, -1, -1); it is wrong only on example 3 (x=3, y=+1), so eps_0 = 0.25. Then alpha_0 = (1/2) log(0.75/0.25) = (1/2) log 3 = 0.5493. Reweighting D_{t+1}(i) ∝ D_t(i) exp(-alpha y_i h_t(x_i)): the three correct examples each get multiplied by exp(-0.5493) and the one wrong example by exp(+0.5493), and after renormalizing the weights become (0.1667, 0.1667, 0.5, 0.1667) — example 3 has jumped from 0.25 to 0.5, exactly the "concentrate on the failure" pressure I designed. I can also check the normalizer here: Z_0 = sum of the unnormalized new weights worked out to 0.8660, and 2*sqrt(eps_0(1-eps_0)) = 2*sqrt(0.25*0.75) = 2*sqrt(0.1875) = 0.8660. They match, which is the per-round-factor identity falling out of an actual run rather than my asserting it.

Round 1: on the reweighted distribution the best stump is threshold at 3.5 predicting +1 to its left, predictions (+1, +1, +1, -1), wrong only on example 2 (x=2, y=-1) which now carries little weight — eps_1 = 0.1667, alpha_1 = (1/2) log(5) = 0.8047, Z_1 = 0.7454 = 2*sqrt(0.1667*0.8333). The running F = alpha_0 h_0 + alpha_1 h_1 still signs to (+1, -1, -1, -1) on the four points — training error 0.25, no improvement yet, two stumps aren't enough. Round 2: best stump threshold at 2.5 predicting -1 to its left, predictions (-1, -1, +1, +1), eps_2 = 0.2, alpha_2 = (1/2) log 4 = 0.6931. Now F = 0.5493(+1,-1,-1,-1) + 0.8047(+1,+1,+1,-1) + 0.6931(-1,-1,+1,+1) signs to (+1, -1, +1, -1) — every point correct, training error 0. Three stumps, each individually unable to separate the data, combine to a perfect classifier exactly as the amplification claim promised. And the bound tracks it honestly: prod 2*sqrt(eps_t(1-eps_t)) over the three rounds is 0.8660, then 0.8660*0.7454 = 0.6455, then *0.8000 = 0.5164, and the true training error (0.25, 0.25, 0.0) stays at or below the bound every round. That's the whole mechanism working end to end on numbers I can see, not just symbols.

Let me rewrite the same machine in the signed {-1,+1} convention, because it's cleaner to reason about and exposes why the constant is what it is. Put y_i in {-1,+1} and h_t(x_i) in {-1,+1}, and define the running score F(x) = sum_t alpha_t h_t(x). The reweighting becomes

  D_{t+1}(i) = D_t(i) * exp(-alpha_t y_i h_t(x_i)) / Z_t,

where Z_t normalizes D_{t+1} back to a distribution. The sign is right: if h_t is correct on i then y_i h_t(x_i) = +1 and the factor is exp(-alpha_t) < 1 (demote), if wrong then y_i h_t(x_i) = -1 and the factor is exp(+alpha_t) > 1 (promote). The final classifier is sign(F(x)). Now Z_t should write the optimal alpha_t for me. Splitting the examples into the correct ones (weight 1 - eps_t) and the wrong ones (weight eps_t) under D_t,

  Z_t = (1 - eps_t) exp(-alpha_t) + eps_t exp(alpha_t).

Minimize over alpha_t: dZ_t/dalpha_t = -(1 - eps_t) exp(-alpha_t) + eps_t exp(alpha_t) = 0, so exp(2 alpha_t) = (1 - eps_t)/eps_t, i.e.

  alpha_t = (1/2) log((1 - eps_t)/eps_t).

This had better agree with the {0,1} derivation, so let me check the minimized normalizer against it. Plugging alpha_t back, Z_t = (1-eps) sqrt(eps/(1-eps)) + eps sqrt((1-eps)/eps) = sqrt(eps(1-eps)) + sqrt(eps(1-eps)) = 2 sqrt(eps_t(1 - eps_t)) — the very same per-round factor I minimized in the {0,1} world, and the very same number my toy run printed each round. So the two conventions are the same algorithm. The factor of one half is pure notation: in the {-1,+1} world the margin y_i h_t swings across a width of 2 (from -1 to +1), whereas in the {0,1} world the indicator swings across 1, so the signed coefficient is exactly half of the {0,1} vote weight log(1/beta_t) = log((1-eps_t)/eps_t) = 2 alpha_t. Same algorithm, two dialects. (And in the toy I used alpha_t = (1/2) log((1-eps_t)/eps_t) directly and it landed at training error zero, so the convention I'll implement in is the signed one with this coefficient.)

There's a temptation to now declare that this algorithm is "really" minimizing exponential loss, and I want to be careful about the order of discovery here so I don't fool myself. If I unroll the chain of reweightings, the product of the normalizers telescopes and I'm left with (1/m) sum_i exp(-y_i F(x_i)) on one side — the average of exp(-y_i F(x_i)) over the training set — and since the 0/1 mistake indicator 1[y_i F(x_i) <= 0] is bounded above by exp(-y_i F(x_i)), the algorithm is greedily driving down an exponential upper bound on the training mistake count. Round t, having fixed h_t, picks alpha_t to minimize Z_t, which is exactly minimizing this exponential loss along one coordinate — a stagewise coordinate-descent reading. That's a genuinely useful lens. But I keep the causal order straight: the guarantee I proved came from the *weight* argument — the convexity bound, the squeeze, the per-round minimization — not from positing a loss function and descending it. The exponential-loss view is a way to *understand* the same algebra after the fact, not the reason it works; the minimization of a convex surrogate alone wouldn't explain the behavior I'm about to worry about.

Because now I should worry about test error, not just training error. The combined classifier is a thresholded linear combination of T base hypotheses. If the base class has VC dimension d, the class of T-round weighted votes over it has VC dimension that grows roughly like d T log T — concretely it is bounded by something on the order of 2(d+1)(T+1) log_2(e(T+1)). So the crude complexity story warns me about a tradeoff: training error falls with T, but the capacity term climbs with T. That is a blunt invariant, though. A thresholded vote can be barely correct or overwhelmingly correct, and the 0/1 label treats both cases as identical. The confidence quantity I should watch is the *margin*: for example i, the quantity y_i F(x_i) / sum_t |alpha_t|, the weighted fraction of votes for the correct label minus the weighted fraction against, normalized to [-1,+1]. The sign of the margin is just correctness; its magnitude is how decisive the vote was. So once examples are on the correct side of zero, the useful question is whether the small positive margins keep moving to the right. A bound that depends on the margin distribution and the base-class complexity, rather than only on the round count, is the kind of generalization statement this machine needs.

Now the part I actually need for the harness in front of me, because the targets aren't always class labels — sometimes the response is a real number and I have to predict it well in squared or absolute error. The classification machine is built on a binary right/wrong indicator. To carry the *same* adaptive-error mechanism over to regression, I need a per-example "loss" that plays the role the indicator played: it has to live in [0,1], be 0 for a perfect prediction and 1 for the worst, so that the very same beta = Lbar/(1-Lbar), the same beta^(1-loss) reweighting, and the same log(1/beta) vote all reuse intact. The raw absolute error |y'_i - y_i| won't do — it's unbounded and unit-dependent. So normalize each round by the largest absolute error that round produced, D = max_i |y'_i - y_i|. The linear choice is then

  L_i = |y'_i - y_i| / D,

which sits in [0,1] by construction, is 0 on the best-predicted example and 1 on the worst. If I want to penalize large errors more steeply I can square it, L_i = (|y'_i - y_i|/D)^2, or use a saturating form L_i = 1 - exp(-|y'_i - y_i|/D); all three stay in [0,1], and linear is the natural default. With this bounded loss everything from the classification derivation transfers symbol for symbol. The weighted average loss this round is

  Lbar = sum_i p_i L_i,

and if Lbar >= 1/2 the round is no better than a trivial predictor on the weighted problem, so I stop (the regression mirror of the eps_t >= 1/2 stop, where beta would cross 1 and flip the update). Otherwise the confidence is

  beta = Lbar / (1 - Lbar) in [0,1),

the example weights update by

  w_i <- w_i * beta^(1 - L_i),

so well-predicted examples (L_i near 0, exponent near 1) shrink by beta and badly-predicted ones (L_i near 1, exponent near 0) hold their weight — the identical demote-the-solved, keep-the-hard pressure — and the round's vote weight is log(1/beta), large for an accurate round and small for a marginal one. The only piece that genuinely *can't* be copied from classification is how to combine the outputs. A weighted *mean* of the learners is the obvious analogue of the weighted majority, but a mean is fragile: one wild learner with a large coefficient can drag the ensemble's prediction arbitrarily far. The robust order-statistic analogue of the majority vote is the weighted *median*. For an input x, take the predictions y_t(x) and their confidence weights c_t = log(1/beta_t), sort the predictions, and walk up the cumulative confidence; the ensemble output is the prediction at the point where the cumulative confidence first reaches half the total — sum_{t: y_t(x) <= y*} c_t >= (1/2) sum_t c_t. That is exactly "the weighted majority of the votes is at least this value," the median read off the confidence-weighted CDF of the predictions, and it inherits the median's resistance to a single bad learner.

Let me pin the design choices that aren't forced by the algebra but that I'm relying on. The weak learner is a shallow tree — a stump or depth-2/3 tree — for two reasons. It only has to clear better-than-chance on each reweighted problem, which a stump almost always does; and it must stay *weak*, because a fully grown tree would fit a single weighted problem nearly perfectly, drive its eps_t toward 0, take an enormous alpha, and leave essentially no edge for any later round to exploit — boosting works by having many rounds each contribute a little, so I deliberately cripple the base learner to keep the contributions spread out. Depth 3 is the natural compromise for regression: enough to capture a low-order interaction, shallow enough to remain a weak learner. And a learning-rate / shrinkage factor folded into the vote weight trades more rounds for smaller per-round steps, the standard regularization knob for additive models.

So let me write the actual loop. In the classification branch I store the full log-odds estimator weight log((1-eps)/eps), which is twice the signed alpha_t above but gives the same argmax vote and the correct multiplicative update. In the regression branch I use the bounded loss, draw the learner's training set from the current weights, and combine by the weighted median:

```python
import numpy as np


class AdaBoost:
    """Discrete classification boosting plus bounded-loss regression boosting."""

    def __init__(self, make_weak_learner, task_type="classification",
                 n_rounds=200, learning_rate=1.0, loss="linear", random_state=None):
        self.make_weak_learner = make_weak_learner
        self.task_type = task_type
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.learners_, self.estimator_weights_, self.estimator_errors_ = [], [], []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        w = np.ones(n, dtype=float) / n
        if self.task_type == "classification":
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)

        for t in range(self.n_rounds):
            learner = self.make_weak_learner()

            if self.task_type == "regression":
                p = w / w.sum()
                idx = rng.choice(np.arange(n), size=n, replace=True, p=p)
                learner.fit(X[idx], y[idx])
            else:
                learner.fit(X, y, sample_weight=w)

            pred = learner.predict(X)
            p = w / w.sum()

            if self.task_type == "classification":
                incorrect = (pred != y)
                err = float(np.average(incorrect, weights=p))
                if err <= 0:
                    self.learners_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break
                if err >= 1.0 - 1.0 / n_classes:
                    break
                learner_weight = self.learning_rate * (
                    np.log((1.0 - err) / err) + np.log(n_classes - 1.0)
                )
                if t != self.n_rounds - 1:
                    w = np.exp(np.log(w) + learner_weight * incorrect * (w > 0))
            else:
                mask = w > 0
                loss_vec = np.abs(pred[mask] - y[mask])
                loss_max = loss_vec.max()
                if loss_max != 0:
                    loss_vec = loss_vec / loss_max
                if self.loss == "square":
                    loss_vec = loss_vec ** 2
                elif self.loss == "exponential":
                    loss_vec = 1.0 - np.exp(-loss_vec)
                err = float(np.dot(p[mask], loss_vec))
                if err <= 0:
                    self.learners_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(0.0)
                    break
                if err >= 0.5:
                    break
                beta = err / (1.0 - err)
                learner_weight = self.learning_rate * np.log(1.0 / beta)
                if t != self.n_rounds - 1:
                    w[mask] *= np.power(beta, (1.0 - loss_vec) * self.learning_rate)

            w = w / w.sum()
            self.learners_.append(learner)
            self.estimator_weights_.append(float(learner_weight))
            self.estimator_errors_.append(float(err))
        return self

    def predict(self, X):
        weights = np.asarray(self.estimator_weights_, dtype=float)
        if self.task_type == "classification":
            votes = np.zeros((len(X), len(self.classes_)))
            for weight, learner in zip(weights, self.learners_):
                pred = learner.predict(X)
                for j, cls in enumerate(self.classes_):
                    votes[:, j] += weight * (pred == cls)
            return self.classes_[np.argmax(votes, axis=1)]

        preds = np.array([learner.predict(X) for learner in self.learners_]).T
        out = np.empty(preds.shape[0])
        for i in range(preds.shape[0]):
            order = np.argsort(preds[i])
            cdf = np.cumsum(weights[order])
            j = np.searchsorted(cdf, 0.5 * cdf[-1])
            out[i] = preds[i, order[min(j, len(order) - 1)]]
        return out
```

Let me trace the whole causal chain once more. I wanted to amplify a barely-better-than-chance learner into an accurate one, and the obstacle was that rerunning it on the same distribution gives correlated mistakes that voting can't fix. The distribution-free promise let me aim the learner: reweight the examples toward the committee's current failures. The first proof did this with a rigid recursive majority-of-three at a worst-case error level — and checking g(a) = 3a^2 - 2a^3 by factoring a - g(a) = a(1-a)(1-2a) confirmed it shrinks error only while a < 1/2 — and the flat-majority improvement still needed the edge gamma fixed in advance and counted every round equally. The move that broke the logjam was to *measure* each round's weighted error eps_t and let that one number set both the reweighting strength and the vote weight. Maintaining example weights, updating them multiplicatively by beta_t^(1 - indicator), and tracking the total weight through the convexity bound gave an upper handle on the surviving weight; the final-mistake condition gave a lower handle; squeezing them produced error <= prod_t [eps_t + (1-eps_t)beta_t]/sqrt(beta_t), and minimizing each factor *chose* beta_t = eps_t/(1-eps_t) for me (verified to be the minimizer both by the second-derivative sign and numerically), yielding error <= prod_t 2 sqrt(eps_t(1-eps_t)) <= exp(-2 sum_t gamma_t^2) — exponential amplification that needs no advance knowledge of the edge and banks the actual squared edges. The four-point stump run confirmed all of this concretely: three stumps none of which can separate the data combined to zero training error, each round's normalizer Z_t equaled 2 sqrt(eps_t(1-eps_t)) on the nose, and the bound stayed above the true error every round. The signed {-1,+1} form gave alpha_t = (1/2)log((1-eps_t)/eps_t) by minimizing the same normalizer, while the implementation can store the equivalent full log-odds vote weight and use the same normalized classifier. The exponential-loss / coordinate-descent reading came afterward as a way to see the same algebra. To handle real-valued targets I bounded the per-example loss to [0,1] by normalizing the absolute error by the round's maximum, which let the identical beta = Lbar/(1-Lbar), beta^(1-L) reweighting, and log(1/beta) vote carry straight over, and I combined the regressors by a robust weighted median instead of a fragile mean. Shallow trees keep the base learner weak so the contributions stay spread across rounds; a shrinkage factor trades step size for round count. The whole thing is a sequential reweight-fit-vote loop that drops into the existing harness.
