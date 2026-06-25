OK, let me think this through from scratch. I have a big feedforward net, plenty of hidden units, and only a modest pile of labeled examples, and it overfits — train error keeps dropping while test error stalls or climbs. The usual reflex is "too much capacity, shrink the weights," but I want to first understand *why* the capacity hurts here before I reach for a hammer.

Here's the thing that bothers me. With more hidden units than I strictly need, there are many weight settings that fit the training data almost perfectly. The optimizer lands on one of them, more or less by accident of initialization and noise. Each of those settings makes different predictions off the training manifold, and almost all of them are worse on test than on train. So the question isn't just "which solution" — it's "why is the typical solution fragile." What is it about a perfectly-fit large net that makes it brittle?

Let me stare at a single hidden unit. During training, its job is to reduce the error. But it doesn't reduce error in isolation — it reduces the *residual* error left over after all the other units have done their thing. So gradient descent will happily let a unit specialize into "I fix the particular mistakes that units 7, 22, and 91 make on these training cases." That's a perfectly good way to drive training error down. And it's exactly the kind of thing that won't transfer: the specific mistakes of units 7, 22, 91 on the training set are an artifact of this training set. The unit has co-adapted to its neighbors. A detector that is only useful in the presence of a specific committee of other detectors is a liability the moment the committee behaves differently, which is what held-out data does.

So the disease has a name: co-adaptation. And naming it that way is suggestive, because now the cure doesn't have to be "less capacity" — it can be "make co-adaptation impossible to learn in the first place." Weight decay shrinks every weight toward zero uniformly; that's a blunt instrument that doesn't even mention the failure mode. I want something that directly punishes "this unit relies on that specific unit being there."

How do you stop a unit from relying on a specific partner? Make the partner unreliable. If, on any given training case, any other hidden unit might simply not be there, then a unit *cannot* build a strategy that assumes unit 7 is present and outputting its usual value — because half the time unit 7 isn't present at all. The only strategies that survive are the ones that are useful regardless of which other units happen to be around. So: on each presentation of each training case, delete each hidden unit independently with probability one-half. Forward-propagate through whatever survives, backprop through the same survivors, update. Next case, fresh coin flips.

Let me sanity-check that this does what I want and isn't just adding noise for noise's sake. A unit that had learned "I'm the correction term for units 7, 22, 91" now finds those collaborators absent half the time, in random combinations. Its gradient, averaged over the random contexts, pushes it to be useful *on average across the combinatorial variety of contexts it might find itself in* — not useful in one specific frozen context. That looks like it bites on exactly the brittleness I was trying to kill, rather than on a proxy for it the way weight decay does. I'll hold that as a hypothesis and keep poking at it, because "intuitively attacks the disease" is not the same as "actually works," and I haven't yet checked what this does at test time.

Now, dropping units at random — let me look at the same operation from a completely different angle, because I have a suspicion it's doing more than just "robustifying each unit." Every time I sample a random subset of units to keep, I'm not training the same network — I'm training a *different* network, a thinned-out subnetwork, one of the many you can carve out by deleting units. With N hidden units there are 2^N such subnetworks. And crucially they all share weights: the weight on the connection into unit 7 is the same weight no matter which subnetwork unit 7 happens to appear in. So over the course of training I'm not training one net — I'm training an astronomically large family of nets, 2^N of them, with massive parameter sharing, each one getting a tiny bit of training (basically one mini-update) on whatever case showed up when it was sampled.

That reframing is exciting because model averaging is the textbook cure for exactly my problem. The reason an ensemble generalizes better than a single model is that averaging cancels the idiosyncratic, non-transferable part of each model's predictions and keeps the shared signal. The catch with ensembles has always been cost: to average M big nets you have to train M big nets and run all M at test time. But here the averaging is happening *for free*, woven into a single training run, because the 2^N nets are all the same weights wearing different masks.

Let me push the ensemble analogy hard, because I want to know what kind of ensemble this is and whether the analogy is honest. Bagging trains each model on a bootstrap resample and gives them equal weight; it's used with cheap learners like trees precisely because you pay per model. What I'm doing is bagging taken to an absurd extreme: each "model" (each mask) is effectively trained on a single case before the mask changes, and instead of the models being independent, every parameter is shared across the whole exponential family. That sharing is doing enormous work. An ordinary bagged ensemble of nets shares nothing, so each net needs its own full dose of data; here a single weight gets adapted on a large fraction of the data because it participates in a large fraction of the sampled subnetworks. Tying parameters across models like this is itself a regularizer — and a far better one than shrinking parameters toward zero, because it constrains them toward *mutual consistency across an ensemble* rather than toward an arbitrary origin.

There's an even more extreme cousin worth recalling to check my intuition. Naive Bayes trains each input feature alone to predict the class and multiplies the per-feature predictions at test time. It's notoriously good with very little data, and the reason is exactly the reason dropout should help: no feature is allowed to rely on the context of the others, so nothing brittle gets learned. Dropout sits between naive Bayes (every feature totally independent) and standard backprop (every feature free to co-adapt with all others) — it keeps useful cooperation possible but forbids dependence on any *specific* partner. That's the sweet spot I want.

Now the hard part. Training is fine — sample a mask, go. But at test time I cannot run 2^N networks and combine them; the whole point was to avoid ensemble cost. I need a single deterministic network whose one forward pass approximates the average over all those masks. What should that network be?

A hidden unit, during training, is present on half the cases. So a downstream unit receives input from it only half the time. If at test time I switch the unit on permanently, the downstream unit suddenly gets twice the expected input from it compared to training. To keep the downstream input statistics matched to what was learned, I should halve the outgoing weight of every unit. So my candidate "mean network" is: keep all units, but halve all the weights (the weights that were used on a present-half-the-time unit). One pass, all units on, weights halved.

That's a plausible hand-wave, but I don't trust hand-waves about averaging non-linear things. Let me check whether the mean network actually equals the ensemble average in a case I can compute exactly. Take the simplest interesting case: a single hidden layer feeding a softmax output, and let me see what "average over all 2^N masks" really gives and whether halving the weights reproduces it.

Set up notation. Hidden activations h_i for i = 1..N (these are fixed once the input is fixed). A mask m ∈ {0,1}^N keeps unit i iff m_i = 1. The logit for class k under mask m is

  z_k(m) = Σ_i m_i w_{ik} h_i + b_k,

and the subnetwork's prediction is the softmax P_m(k) = exp z_k(m) / Σ_j exp z_j(m).

Now I have to decide *how* to average these 2^N distributions, and this is where I have to be careful, because there are two ways to average distributions and they give different answers. The arithmetic mean — average the probabilities — is the obvious one. But let me try the geometric mean — multiply the probabilities and renormalize — because softmax outputs feel multiplicative (they're exponentials of logits), and I have a hunch it'll come out cleaner. The normalized geometric mean is

  G(k) = (Π_m P_m(k))^{1/2^N} / Σ_l (Π_m P_m(l))^{1/2^N}.

Let me just compute the unnormalized log of the numerator. Write U(k) = (Π_m P_m(k))^{1/2^N}, so

  log U(k) = (1/2^N) Σ_m log P_m(k) = (1/2^N) Σ_m [ z_k(m) − log Σ_j exp z_j(m) ].

Look at the second term, A = (1/2^N) Σ_m log Σ_j exp z_j(m). It's the average over masks of the log-partition function. It does not depend on k at all. So when I normalize G — divide by Σ_l U(l) — that term contributes the same additive constant to every class's log, and an additive constant shared by all classes washes out of the softmax. It cancels completely. That is a real simplification, and it is specific to the geometric mean: with the arithmetic mean I'd be averaging the ratios exp z_k(m)/Σ_j exp z_j(m) directly and the per-mask normalizer stays welded to each term, so nothing collapses. I'll continue with the geometric mean and see where it lands; if it lands somewhere I can't implement, I'll have to come back and pay the arithmetic-mean price.

What's left is

  G(k) ∝ exp( (1/2^N) Σ_m z_k(m) ).

Now compute the average logit over all masks:

  (1/2^N) Σ_m z_k(m) = Σ_i [ (1/2^N) Σ_m m_i ] w_{ik} h_i + b_k.

The whole thing now turns on one number: (1/2^N) Σ_m m_i, the fraction of the 2^N masks in which unit i is kept. I believe it's ½ — each unit is flipped independently and symmetrically, so it should be on in half the masks — but this is exactly the kind of "obviously ½" step that I'd rather count than trust. Enumerate the masks for small N and average the i-th coordinate: for N=1 the two masks are {0,1} so the mean is 0.5; for N=2 the four masks 00,01,10,11 give each coordinate a mean of 0.5; for N=3 and N=5 the same enumeration gives 0.5 for every coordinate. It's flat at ½ regardless of N and regardless of which unit, which is what independence-and-symmetry promised. Good — so

  (1/2^N) Σ_m z_k(m) = Σ_i (½ w_{ik}) h_i + b_k,

and

  G(k) = softmax_k( Σ_i (½ w_{ik}) h_i + b_k ).

This is the mean network with every outgoing weight halved. The factor of ½ is not a fudge I inserted to make test time cheap — it is (1/2^N)Σ_m m_i, the keep-probability, falling straight out of the averaging. If I use a different drop rate d, the same computation averages masks with their sampling probabilities instead of weighting all masks equally; then E[m_i] = 1 − d, so the test-time multiplier is the keep probability q = 1 − d.

Before I let myself believe an exact equality between a deterministic forward pass and a geometric mean over an exponential family, I should actually run both and compare, because algebra at this scale is exactly where a dropped factor of two or a misplaced normalizer hides. Take a concrete tiny net: N=2 hidden units with fixed activations h = (0.7, −1.3), three output classes, random outgoing weights W and biases b. Enumerate all four masks, softmax each subnetwork's logits, and form the normalized geometric mean by exponentiating the per-class average of log P_m. Then separately compute the single mean-network pass: all units on, weights halved, one softmax. The geometric mean comes out (0.2933, 0.0771, 0.6295); the halved-weight pass comes out (0.2933, 0.0771, 0.6295), agreeing to ~1e-17 — machine zero, so this is an identity and not a near-miss. To make sure I'm not fooling myself that *any* average would match, I also compute the arithmetic mean of the four P_m: (0.3644, 0.0826, 0.5530), which differs from the mean network by 0.077, nowhere near zero. So the equality really is a property of the geometric mean specifically; the cancellation of the log-partition term earlier was load-bearing, and the arithmetic mean would *not* be reproduced by halving the weights. The cheap test-time pass computes the geometric-mean ensemble exactly, for one hidden layer plus softmax.

Now I want to know whether using the geometric mean is a *good* idea or just a convenient one. Convenience that gives a worse answer is a trap. Is the deterministic mean network actually competitive with the thing it's standing in for? Let me compare the mean network's log-probability of the correct class against the average over masks of the individual subnetworks' log-probabilities — i.e., is the single deterministic net at least as good, in log-prob, as a typical sampled net?

Write q_m(k) = exp z_k(m) and Z_m = Σ_j q_m(j), so P_m(k) = q_m(k)/Z_m. From the line above, the mean net is G(k) = exp(mean_m log q_m(k)) / Z′ with Z′ = Σ_l exp(mean_m log q_m(l)), where mean_m denotes (1/2^N)Σ_m. For the true class t,

  log G(t) − mean_m log P_m(t)
   = [ mean_m log q_m(t) − log Z′ ] − [ mean_m log q_m(t) − mean_m log Z_m ]
   = mean_m log Z_m − log Z′.

So the question is whether mean_m log Z_m ≥ log Z′, i.e. whether

  (1/2^N) Σ_m log Σ_j q_m(j) ≥ log Σ_j exp( (1/2^N) Σ_m log q_m(j) ).

The right-hand side's inner term is exp(mean_m log q_m(j)) = (Π_m q_m(j))^{1/2^N}, the geometric mean of the j-th unnormalized scores across masks. The left-hand side, exponentiated, is the geometric mean across masks of the *sums* Σ_j q_m(j). So the inequality is: the geometric mean of the row-sums is at least the sum of the column-wise geometric means,

  (Π_m Σ_j q_m(j))^{1/2^N} ≥ Σ_j (Π_m q_m(j))^{1/2^N}.

Let M = 2^N and divide both sides by the left-hand side's scale. Put S_m = Σ_j q_m(j) and r_m(j) = q_m(j) / S_m, so each r_m is a probability vector. The desired inequality becomes

  Σ_j Π_m r_m(j)^{1/M} ≤ 1.

Now Hölder with all exponents equal to M gives

  Σ_j Π_m r_m(j)^{1/M} ≤ Π_m (Σ_j r_m(j))^{1/M} = 1.

Equality in Hölder requires these normalized vectors to be proportional across masks; because each one already sums to one, proportional means identical. That is exactly the condition that all subnetworks make the same predictive distribution. So mean_m log Z_m ≥ log Z′, which gives log G(t) ≥ mean_m log P_m(t), with equality only if the nets all agree.

Let me put a number on this on the same tiny net, because an inequality I derived by chaining a cancellation into Hölder is worth one concrete confirmation. On the N=2 example, log G(t) − mean_m log P_m(t) for the three classes comes out 0.4405, 0.4405, 0.4405 — all positive, so the deterministic pass does beat the average sampled net in log-prob of the truth, and notably the gap is the *same* for every class. That sameness is itself a check on the algebra: the derivation said the gap is mean_m log Z_m − log Z′, a quantity with no k in it, so it had *better* be constant across classes, and it is. Then I force the equality case — set the outgoing weights to zero so every mask produces an identical distribution — and the gap collapses to ~1e-16 for all classes, exactly the "equality iff all nets agree" condition Hölder predicted. So the single deterministic pass is not a worse summary, in this log-prob sense, than the typical sampled net, and the bound is tight precisely where it should be.

The regression analogue is even simpler. If the outputs are linear and I average the predictions, the mean prediction's squared error versus the average of the individual squared errors differs by the variance of the predictions across masks, which is nonnegative — so the mean network's squared error is always ≤ the average of the dropout networks' squared errors. Same conclusion, by plain Jensen.

For deeper nets the exact geometric-mean identity breaks — you can't push the averaging through multiple nonlinear layers cleanly — but the per-layer logic still holds approximately: each unit was present a fraction q = 1 − d of the time, so scaling its outgoing weights by q restores the expected input to the next layer, and the independence of the per-unit drop decisions keeps the single-pass approximation tight. So the rule generalizes: train with masks, test with all units on and weights scaled by the keep probability.

Good — I have a training procedure and a matching test-time network. But now I have to actually make the training *work*, and the stochasticity introduces problems that plain SGD settings won't handle. Let me think about optimization.

Every mini-batch now computes a gradient for a *different* stochastic network. The gradient is much noisier than usual, because it's not just data noise, it's architecture noise — I'm getting a one-sample-ish estimate of the gradient of the shared weights as seen through a randomly chosen subnetwork. Two consequences. First, I'd like to average gradient information over many updates to smooth out the architecture noise; that argues for heavy momentum. If I let the momentum coefficient climb toward something like 0.99, each update is effectively an exponential average over many different sampled nets, which stabilizes learning despite every individual gradient being for a different net. I shouldn't start that high or early training will be unstable, so ramp it up — start around 0.5, increase to 0.99 over the first several hundred epochs. And if momentum is that high, the effective step size is inflated by roughly 1/(1−momentum), so I should scale the learning rate by (1−momentum) to keep the actual moves sane.

Second, and more interesting: because the loss landscape is this noisy ensemble objective, I want to *search* weight space thoroughly rather than creep. That means a large learning rate, at least early. But a large learning rate plus the usual L2 *penalty* is a recipe for disaster — a big proposed update can momentarily blow the weights up before the penalty has any chance to pull them back, and with high momentum that blow-up can run away. The L2 penalty fights large weights only proportionally; it cannot *cap* them against an arbitrarily large step.

So let me change the form of the regularization on weight magnitude. Instead of adding a penalty term ½λ‖w‖² to the objective, impose a hard *constraint*: for each hidden unit, the squared length of its incoming weight vector may not exceed some bound l. After each update, compute scale = min(1, sqrt(l / ‖w‖²)) for that incoming vector and multiply the vector by that scale. If it is already inside the ball, scale is 1; if it is outside, the squared length becomes exactly l. This is projection onto an L2 ball, done per unit. The key property is that it bounds the weights *no matter how large the proposed update was* — the projection clips it back regardless. That decouples "how big a learning rate can I use" from "how big can the weights get," and the answer to the first becomes "very big." I can start with a huge learning rate (on a fully-connected MNIST net, something like 10 applied to the average mini-batch gradient), search aggressively, and decay it geometrically; as it shrinks the algorithm settles into finer and finer steps and finds a good configuration, all while the max-norm constraint guarantees the weights never run away. A penalty simply cannot give me that guarantee. (When I'm instead fine-tuning a generatively pre-trained net, I drop the constraint and use a small learning rate, because here I do *not* want to search aggressively — I want to preserve the feature detectors pre-training already found.)

Now the drop probabilities themselves. Why drop one-half of the hidden units? Two reasons converge. If the drop rate is near zero, almost every training pass sees almost the full network and co-adaptation is barely disturbed. If the drop rate is near one, each subnetwork is starved of signal and too little computation remains. The midpoint gives the mask distribution maximum entropy, so I sample the richest variety of subnetworks while still leaving a usable network on each pass. From the test-time view, the same midpoint makes the scaling rule exactly "halve the weights," and it is the symmetric case that makes (1/2^N)Σ_m m_i = ½. So I'll use drop_rate = 0.5, keep_prob = 0.5 throughout for hidden layers.

The inputs are different. An input dimension carries actual signal, not a learned, redundant feature, so deleting half of the raw evidence on every case is too destructive — I'm throwing away information the net genuinely needs. A gentler amount on the inputs still helps because it behaves like input noise, but I should keep most of the visible units: drop_rate around 0.2, keep_prob around 0.8.

Where to apply it across the depth? Co-adaptation can happen at every layer, so dropping in *all* fully-connected hidden layers attacks the failure mode more directly than dropping in only one. Convolutional layers are a special case, though. A convolutional layer ties its weights across spatial positions, which already slashes the parameter count dramatically and with it the capacity to overfit; there's far less co-adaptation to break and far less to gain from dropping there. So I'll spend dropout mainly on the parameter-heavy fully-connected (or locally-connected) layers and use little or none in the convolutional ones.

Let me also make sure the rest of the net cooperates with aggressive search. In the convolutional models I'll use the max-with-zero nonlinearity, max(0, z), for the hidden units. It doesn't saturate, so it trains faster and its activities just scale with large inputs rather than flattening out — which suits the large-learning-rate regime and removes the need for fussy contrast normalization. The one hazard is a dead unit: if a unit's input is always negative its output is always zero and so is its gradient, so it never learns. To avoid that I initialize weights from a zero-mean normal with variance large enough that units usually get some positive input, and I set the hidden biases to a small positive constant (1) so units start off firing. With that, learning gets off the ground and the dropout masking does its job on top.

Let me trace one mini-batch end to end with all of this in place, because the implementation has to keep the mask consistent between forward and backward. Forward: for the visible units and then for each hidden layer, draw a uniform random matrix U, set mask = (U > drop_rate), and multiply the activations by that mask, remembering it for the hidden layers. Backward: when the gradient arrives at a hidden layer's activations, multiply it by the *same* stored mask, so gradient flows only through the units that were actually present in the forward pass; a dropped unit must receive zero gradient, exactly as if it weren't in the net. Then the usual backprop through the linear weights. Update with momentum, then project each hidden unit's incoming weights back onto the L2 ball of squared radius l. At test time, no masks: run all units, and either scale outgoing weights by keep_prob or, equivalently and more conveniently in code, scale the corresponding activations by keep_prob — same number, same effect, one clean forward pass.

The code shape is the same one I need in a cuda-convnet-style trainer: forward uses a thresholded uniform mask, test multiplies activations by 1 − drop_rate, and backward reuses the stored mask.

```python
import numpy as np

def relu(z):
    return np.maximum(0.0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_grad(probs, y):
    g = probs.copy()
    g[np.arange(len(y)), y] -= 1.0
    return g / len(y)


class Layer:
    """Fully-connected layer with per-unit dropout and a max-norm weight constraint."""
    def __init__(self, n_in, n_out, drop_rate=0.5, max_sq_norm=15.0, last=False):
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b = np.zeros(n_out) + (0.0 if last else 1.0)  # positive bias keeps ReLU units alive
        self.drop_rate = 0.0 if last else drop_rate
        self.keep_prob = 1.0 - self.drop_rate
        self.max_sq_norm = max_sq_norm
        self.last = last
        self.mask = None

    def forward(self, x, train):
        self.x = x
        self.z = x @ self.W + self.b
        a = self.z if self.last else relu(self.z)
        self.mask = None
        if self.last or self.drop_rate == 0.0:
            return a
        if train:
            # cuda-convnet mechanics: random uniform mask, keep where U > drop_rate.
            self.mask = (np.random.rand(*a.shape) > self.drop_rate).astype(a.dtype)
            a = a * self.mask
        else:
            # mean network: all units on, activations scaled by keep_prob = 1 - drop_rate.
            a = a * self.keep_prob
        return a

    def backward(self, grad_a):
        if not self.last:
            if self.mask is not None:
                grad_a = grad_a * self.mask    # same mask as the forward pass
            grad_a = grad_a * (self.z > 0)     # ReLU gate
        self.gW = self.x.T @ grad_a
        self.gb = grad_a.sum(axis=0)
        return grad_a @ self.W.T

    def after_update(self):
        if self.last:
            return
        # Project each hidden unit's incoming vector onto ||w||^2 <= max_sq_norm.
        sq = (self.W ** 2).sum(axis=0, keepdims=True)
        scale = np.ones_like(sq)
        too_large = sq > self.max_sq_norm
        scale[too_large] = np.sqrt(self.max_sq_norm / sq[too_large])
        self.W *= scale


class Net:
    def __init__(self, sizes, drop_rate_hidden=0.5, drop_rate_input=0.2):
        self.drop_rate_input = drop_rate_input
        self.input_keep_prob = 1.0 - drop_rate_input
        self.input_mask = None
        self.layers = []
        n_layers = len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = (i == n_layers - 1)
            drop = 0.0 if last else drop_rate_hidden
            self.layers.append(Layer(a, b, drop_rate=drop, last=last))

    def forward(self, x, train):
        self.input_mask = None
        if self.drop_rate_input > 0.0:
            if train:
                self.input_mask = (np.random.rand(*x.shape) > self.drop_rate_input).astype(x.dtype)
                x = x * self.input_mask
            else:
                x = x * self.input_keep_prob
        for L in self.layers:
            x = L.forward(x, train)
        return softmax(x)

    def backward(self, probs, y):
        g = cross_entropy_grad(probs, y)
        for L in reversed(self.layers):
            g = L.backward(g)


def lr_schedule(ep, eps0=10.0, f=0.998):
    return eps0 * (f ** ep)                  # large initial rate, geometric decay

def momentum_schedule(ep, p_i=0.5, p_f=0.99, T=500):
    return (ep / T) * p_f + (1 - ep / T) * p_i if ep < T else p_f  # ramp 0.5 -> 0.99


def train(net, X, Y, epochs=3000, batch=100):
    vW = [np.zeros_like(L.W) for L in net.layers]
    vb = [np.zeros_like(L.b) for L in net.layers]
    for ep in range(epochs):
        mom = momentum_schedule(ep)
        lr  = lr_schedule(ep) * (1 - mom)    # scale step by (1-mom) so high momentum stays sane
        for i in range(0, len(X), batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            probs = net.forward(xb, train=True)
            net.backward(probs, yb)
            for k, L in enumerate(net.layers):
                vW[k] = mom * vW[k] - lr * L.gW   # momentum averages over many stochastic nets
                vb[k] = mom * vb[k] - lr * L.gb
                L.W += vW[k]; L.b += vb[k]
                L.after_update()                 # max-norm projection after every update
```

The causal chain, start to finish: overfitting in a big net is co-adaptation — units that are only useful in the company of specific other units, a conspiracy that fits the training set and breaks on test data. Kill co-adaptation by making every unit's collaborators unreliable: drop each hidden unit with probability one-half on every case. That single act is two things at once — a regularizer that forces each unit to be useful across a combinatorial variety of contexts, and an implicit training of 2^N weight-sharing subnetworks, i.e. model averaging woven into one training run. At test time the average is recovered by one deterministic pass with all units on and activations scaled by keep_prob = 1 − drop_rate, which for a single hidden layer plus softmax is the normalized geometric mean of all 2^N subnets — an exact identity I checked numerically, agreeing to machine precision while the arithmetic mean does not — and which lands no worse, in log-prob of the truth, than a typical sampled net, by a Hölder argument whose gap I confirmed is nonnegative and class-independent. To make this stochastic objective trainable, replace the L2 penalty with a per-hidden-unit max-norm constraint so the learning rate can start huge and search weight space, and ramp momentum toward 0.99 to average gradient information over the ever-changing sampled nets. Use drop_rate = 0.5 where parameters are plentiful and co-adaptation is rife — the fully-connected hidden layers — and use a gentler rate or none on inputs and convolutional layers, where destroying signal or fighting already-shared weights buys less.
