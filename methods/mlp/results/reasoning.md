Let me start from the thing that keeps blocking me. I have a network of simple units — each one forms a weighted sum of the values coming in on its connections and emits some function of that sum — and I want it to learn a mapping from input patterns to desired output patterns. When the units are wired straight from input to output and I use a single output unit, I know exactly how to train it: the perceptron rule, or the smoother delta rule, nudges each weight by the output error. And I know exactly what it can represent: a threshold output unit fires when w·x + b > 0, which is one side of a hyperplane in input space, so the patterns it answers 1 on must be linearly separable from the rest. The trouble is that the mappings I actually care about are not. Exclusive-or is the smallest example and it's enough to ruin me: 00 and 11 must give 0, 01 and 10 must give 1, and there is no single line in the plane with 00 and 11 on one side and 01 and 10 on the other — the two classes interleave at the corners of a square. Parity is the same disease scaled up: flipping a single input bit flips the answer, so the most similar inputs demand opposite outputs, and no hyperplane respects that. Whenever the similarity structure of the inputs and the similarity structure of the required outputs disagree, a single hyperplane can't bridge the gap. So a directly-wired net is not just inconvenient here, it's representationally incapable.

What gets me past the hyperplane ceiling? I have to change the space the decision is made in. If I insert a layer of units between the inputs and the output, the output unit no longer sees the raw inputs — it sees whatever those interior units compute. And there's an existence fact I can lean on: for XOR, suppose one interior unit fires exactly when both inputs are on, the AND of the two bits. Now describe each of the four patterns by (input1, input2, AND): 00 becomes 000, 01 becomes 010, 10 becomes 100, 11 becomes 111. In this three-coordinate recoded space, is "output 1 for 010 and 100, output 0 for 000 and 111" linearly separable? I want the weighted sum to clear a threshold for exactly the two middle patterns. Give input1 and input2 weight +1 each and the AND unit weight −2. Then 000 scores 0, 010 scores 1, 100 scores 1, 111 scores 1 + 1 − 2 = 0. Threshold at 0.5: 010 and 100 clear it, 000 and 111 don't. It separates. So the interior unit re-coded the inputs into a space where the originally-impossible mapping became a plain hyperplane decision. That's the whole idea in miniature — the interior units are free to *represent* intermediate features, and with the right features the hard mapping turns easy. And this isn't special to XOR: the catalog of single-layer limitations also says, constructively, that with enough interior units there is always some recoding of the inputs that makes any desired mapping linearly separable at the output. So the representational problem is solved in principle the moment I allow a layer of interior units.

But "in principle, with the right interior unit" is exactly the catch. I picked the AND feature by hand. For XOR I can do that; for an arbitrary mapping over many inputs I cannot guess the right interior features, and the whole point of a learning machine is that it shouldn't need me to. So the real problem isn't representation, it's learning: I need a rule that *discovers* what the interior units should compute, using only the input patterns and the desired outputs. And here's where every rule I have falls down. The delta rule changes a weight into unit j by comparing j's output to a supplied target value for j. For an output unit I have that target — it's in the training data. For an interior unit there is no target. The data never says what the AND unit should output; it only says what the *final* output should be. So I have a scalar error at the top of the network and a pile of interior weights with no local error to compare against. That's the wall: credit assignment to units that have no target.

Let me think about what I'd need to break it. To change an interior weight by gradient descent, I don't actually need a target for that unit — I need the derivative of the final error with respect to that weight. If I could compute, for every weight anywhere in the network, how the one scalar error E responds to wiggling it, I could just step every weight downhill and never need an interior target at all. So reframe: forget targets for interior units, compute dE/dw for all w. The question is whether that's cheap enough to be practical and whether it can be done from locally-available quantities.

Before I chase the derivative, I have to fix the unit, because two of my instincts about it are in direct conflict and the choice of f decides whether a derivative even exists. The perceptron's threshold unit makes the clean binary decision I like, but its output is a step: the derivative is zero everywhere except at the threshold, where it's undefined. A gradient method on step units has nothing to descend — every small weight change leaves the output, and hence the error, exactly unchanged until the sum crosses the threshold and the output jumps. So step units are out the moment I commit to gradient descent. I need a continuous, differentiable unit so that a small weight change produces a small, measurable change in the error, with a slope I can follow. So f has to be smooth and, since "more total input shouldn't decrease the output of a unit that's supposed to act like a soft threshold," nondecreasing. A differentiable, nondecreasing squashing function — that's the unit I'm forced into.

And it must be nonlinear, which I want to nail down rather than assume, because if a linear unit would do then this whole interior-layer business is pointless. Suppose every interior unit is linear, f(x) = x, and keep the bias terms instead of hiding them. With row-vector activations, layer 1 computes y¹ = y⁰W₁ + b₁, and layer 2 computes y² = y¹W₂ + b₂ = y⁰(W₁W₂) + (b₁W₂ + b₂). But W₁W₂ is just some matrix W′, and b₁W₂ + b₂ is just some bias b′, so the two-layer linear net computes y² = y⁰W′ + b′ — exactly what a single direct connection with weights W′ and bias b′ computes. Stack as many linear layers as I like and they collapse to one equivalent affine map, which has the single-hyperplane ceiling I'm trying to escape. So linear interior units add precisely zero representational power. The nonlinearity isn't a flourish; it's the only thing that makes interior layers more than decoration. The unit must be both differentiable (so I can descend) and nonlinear (so depth buys expressiveness). Those two requirements together point at a smooth saturating curve.

Let me pick the concrete one and get its slope, because I'll need the slope all over the derivative. The logistic, y = 1/(1 + e^{−x}), is the natural smooth stand-in for the step: it squashes any real total input into (0,1), it's differentiable everywhere, it's monotone, and it asymptotes to 0 and 1 like a soft threshold. Its derivative is clean — and the cleanliness is going to matter for cost, so let me actually compute it rather than wave at it. With y = (1 + e^{−x})^{−1}, dy/dx = −(1 + e^{−x})^{−2} · (−e^{−x}) = e^{−x} / (1 + e^{−x})². Now rewrite e^{−x}/(1+e^{−x}) as (1+e^{−x} − 1)/(1+e^{−x}) = 1 − 1/(1+e^{−x}) = 1 − y. So dy/dx = [1/(1+e^{−x})] · [e^{−x}/(1+e^{−x})] = y(1 − y). The slope of a logistic unit is y(1 − y) — expressible entirely in terms of the unit's own output, which I already have lying around from the forward pass. No extra evaluation of any exponential. That's a real practical gift: the backward pass can reuse the forward activations. I notice in passing that y(1−y) is largest at y = 0.5 and vanishes as y heads to 0 or 1, so a unit that has already saturated to a firm decision barely moves, while a unit sitting on the fence at 0.5 is the most adjustable — the learning naturally concentrates on units not yet committed, which feels like the right kind of stability.

Now the objective. I want something differentiable whose gradient I can actually take, and I want it to reduce to the rule I already trust in the case I already understand. Squared error does both: define

  E = (1/2) Σ_c Σ_j (y_{j,c} − d_{j,c})²,

the sum over output units j and over input-output cases c of the squared mismatch between produced output y and desired output d. The 1/2 is bookkeeping so the derivative comes out clean. Differentiate with respect to one output unit's value, dropping the case index since the cases just add:

  ∂E/∂y_j = y_j − d_j.

So the sensitivity of the error to an output unit's value is just its signed error. That's the first piece, and notice it already matches the delta rule's "compare output to target." Good sign — whatever I build should contain the rule I trust as a special case.

The weights touch a unit through its *total input* x_j, not directly through its output y_j, so the derivative I need at each unit is the error's sensitivity to x_j. The two are linked by the activation function, y_j = f(x_j), and by the chain rule

  ∂E/∂x_j = (∂E/∂y_j) · (dy_j/dx_j) = (∂E/∂y_j) · y_j(1 − y_j)

for a logistic unit. So if I know how the error depends on a unit's output, I get how it depends on the unit's total input by multiplying by the local slope y_j(1−y_j). One local multiply.

Now from ∂E/∂x_j I can read off the weight gradients immediately, because the total input is a *linear* function of the weights: x_j = Σ_i y_i w_ji. The partial of x_j with respect to one incoming weight w_ji is just y_i, the output of the unit on the other end. So

  ∂E/∂w_ji = (∂E/∂x_j) · (∂x_j/∂w_ji) = (∂E/∂x_j) · y_i.

The gradient on the weight into j from i is the error-sensitivity at j times the activation at i. That's local: it needs only a quantity computed at j and a quantity computed at i, the two endpoints of the connection. Beautiful — exactly the locality I wanted. The bias drops out of the same formula: a bias is the weight on an always-on input y_i = 1, so its gradient is just ∂E/∂x_j. No special case.

So the entire problem has reduced to one thing: getting ∂E/∂y_j for *every* unit, including interior ones. For an output unit I have it, it's y_j − d_j. For an interior unit I don't — and this is where the recursion has to come from. An interior unit i affects the error only through the units it feeds. Wiggle y_i; that changes the total input x_j of every unit j that i connects to, by exactly ∂x_j/∂y_i = w_ji (again because x_j is linear in its inputs). And I know how the error responds to each x_j — that's ∂E/∂x_j, which I just defined. So by the chain rule, summing over every unit j that i feeds,

  ∂E/∂y_i = Σ_j (∂E/∂x_j) · (∂x_j/∂y_i) = Σ_j (∂E/∂x_j) · w_ji.

There it is. The error-sensitivity at an interior unit i is a weighted sum of the error-sensitivities at the total-inputs of the units it feeds, weighted by exactly the same connection weights w_ji used in the forward direction. So I don't need a target for an interior unit at all. I compute ∂E/∂x_j at the output units first (from y_j − d_j times the slope), then push those backward: each interior unit collects Σ_j (∂E/∂x_j) w_ji to get its own ∂E/∂y_i, multiplies by its own local slope to get its ∂E/∂x_i, and passes *that* down to the layer below. The error flows backward through the very same connections, in the reverse direction, scaled by the same weights and gated by each unit's local slope. The interior unit's "target" was a phantom — what it actually needed was its share of the downstream error, and that share is computable locally from quantities the layer above already has.

Let me make sure this is genuinely cheap and not secretly recomputing everything. The naive way to get every weight's gradient would be to perturb each weight, re-run the whole forward pass, and see how E changes — that's one full network evaluation per weight, hopeless for a big net. The recursion above does something completely different: one forward sweep to fill in all the y's, then one backward sweep that visits each unit once, at each unit doing a sum over its outgoing connections (which is the same amount of arithmetic as the forward sum over its incoming connections) and one multiply by the slope. So the backward pass costs about the same as the forward pass, total, and it produces the gradient for *every* weight at once. It works because the error-sensitivities ∂E/∂x_j are shared: every weight into unit j reuses the same ∂E/∂x_j, and every unit below reuses the ∂E/∂x_j of the units above. It's the chain rule organized so that shared subexpressions are computed once and reused — a dynamic program over the network, not a separate derivative per weight.

Let me name the shared quantity to make the algorithm crisp. Write δ_j = −∂E/∂x_j, the (negative) error-sensitivity at unit j's total input. For an output unit,

  δ_j = (d_j − y_j) · y_j(1 − y_j),

the output error times the local slope. For an interior unit, the recursion is

  δ_j = y_j(1 − y_j) · Σ_k δ_k w_kj,

the local slope times the back-propagated weighted sum of the δ's of the units j feeds (I've absorbed the sign by defining δ as the negative sensitivity, so the recursion carries δ's straight). And every weight changes by

  Δw_ji = ε · δ_j · y_i,

stepping each weight down the gradient by an amount proportional to the δ at the receiving end and the activation at the sending end. This has the same shape as the delta rule — error-signal-at-j times input-from-i — except that for interior units the "error signal" δ_j is no longer a supplied target mismatch but this recursively back-propagated quantity. Whether it really is the same rule and not just a look-alike, I should check by collapsing to the case I already trust. Take a single layer of *linear* output units, f(x) = x, so the slope dy/dx = 1 and there are no interior units to recurse into. Then δ_j = (d_j − y_j)·1 = (d_j − y_j), and Δw_ji = ε(d_j − y_j)y_i. That is Widrow and Hoff's delta rule character for character. And on that single linear layer the objective is honestly convex: E = (1/2)Σ_c Σ_j (Σ_i y_{i,c} w_ji − d_{j,c})² is a sum of squares of functions that are *linear* in the weights, i.e. a quadratic with a positive-semidefinite Hessian, so it is a bowl with no spurious minima and small-step descent reaches the bottom. So the generalization does contain the trusted rule as its f-linear, no-interior-unit case — that's the reduction I wanted, and it's reassuring that the construction degrades to exactly the right thing rather than to something merely similar.

That reduction tests the output layer, but the part I'm least sure of is the *interior* recursion — the δ_j = y_j(1−y_j)Σ_k δ_k w_kj step, where I claimed an indirect chain-rule argument hands me the right gradient for a weight that has no target anywhere near it. I derived it on paper, but the derivation is exactly the kind of thing where a dropped slope factor or a transposed weight would still *look* plausible. The honest way to catch that is to pit the recursion against the definition of a derivative: perturb a single weight by a tiny ±h, recompute E by a full forward pass, and form (E(w+h) − E(w−h))/2h. If my backward recursion is right, it must reproduce that finite difference for every weight, including the interior ones. Let me actually run it on the smallest net that has an interior unit: two inputs → one logistic hidden unit → one logistic output, with weights I just write down so the numbers are reproducible — into the hidden unit, weights 0.5 and −0.4 with bias 0.1; from hidden to output, weight 0.3 with bias −0.2; feed the input (1, 0) and ask for target 0.9.

Forward: the hidden total input is 1·0.5 + 0·(−0.4) + 0.1 = 0.6, so y_h = 1/(1+e^{−0.6}) = 0.645656. The output total input is 0.645656·0.3 − 0.2 = −0.006303, so y_o = 0.498424, and E = ½(0.498424 − 0.9)² = 0.080632.

Backward by the recursion: δ_o = (0.9 − 0.498424)·0.498424·(1 − 0.498424) = 0.100393. The output-weight gradient is ∂E/∂w = −δ_o·y_h = −0.100393·0.645656 = −0.064819, and the output-bias gradient is −δ_o = −0.100393. Propagating down, δ_h = y_h(1−y_h)·(δ_o·w) = 0.645656·0.354344·(0.100393·0.3) ≈ 0.006890 (carrying full precision through the product), so the gradient on the first input weight is −δ_h·1 = −0.006890, on the second input weight −δ_h·0 = 0 (that input was off, so the weight genuinely doesn't affect E here), and on the hidden bias −δ_h = −0.006890.

Now the brute-force check. Nudging the output weight by h = 10⁻⁵ and recomputing E both ways gives (E(0.3+h) − E(0.3−h))/2h = −0.064819 — matching the recursion's −0.064819. Running the same perturbation over *every* weight, the largest discrepancy between the backward recursion and the finite differences is about 3×10⁻¹¹, which is finite-difference truncation noise, not a real disagreement. So the recursion is computing the true gradient, interior unit included — the "manufactured" error signal for the hidden unit really is its share of ∂E/∂w and not an artifact of a derivation I wanted to believe. That's the check I actually needed; the on-paper chain rule and the numerical derivative agree.

Now I should worry about what I've given up by going to interior units, because the convexity I just leaned on was a real comfort. With a single linear layer the error is a quadratic bowl — one minimum, descent always finds it. With interior units and a nonlinear f, E is a function of weights that appear inside f's composed nonlinearly, so the surface is no longer convex: there can be local minima, places where every small weight change increases the error yet the error isn't zero. Gradient descent can get stuck in one. I can't make that guarantee go away — it's the price of the expressiveness that lets me do XOR at all. What I *can* note is that the danger is about getting trapped short of a solution, and that adding spare interior units or spare connections opens up extra directions in weight space, which tends to provide paths around the barriers that would trap a leaner net; the failure mode is real but it is something to manage, not a contradiction in the method.

There's a subtler trap that isn't about the error surface's shape but about where I *start* on it, and it's lethal if I'm careless. Suppose I initialize every weight to the same value — say all zero, which looks like the neutral choice. Then on the forward pass every interior unit in a layer computes the identical total input, hence the identical output. On the backward pass each of those units feeds the output through identical weights, so each gets the identical δ, hence the identical weight change. They started identical and they update identically, so they stay identical forever — the network can only ever express what a single interior unit could, and if the task needs the interior units to specialize into *different* features (XOR needs them to detect different things), it can never learn. Worse, equal weights sit at a fixed point of the dynamics: it's a kind of symmetric stationary point, a maximum of the error along the symmetry-breaking directions, so descent has no gradient to leave it by. The cure is forced and simple: start the weights at small *random* values. Randomness breaks the symmetry so the units feel different gradients from step one and are free to differentiate; small keeps every unit in the responsive midrange of the logistic (away from the saturated 0/1 tails where the slope y(1−y) is near zero and learning crawls) and keeps the initial outputs from being prematurely committed.

A related practical point about targets. The logistic asymptotes to 0 and 1 but reaches neither at any finite total input. So if I set the desired outputs to exactly 0 and 1, the only way to drive the error to zero is to send the relevant total inputs to ±∞, i.e. the weights grow without bound chasing a value they can never attain, and the error never actually zeroes. The honest fix is to ask for targets the unit can actually hit — use 0.1 and 0.9 (or similar) as the desired values for "off" and "on." Then finite weights suffice and the learning settles instead of running the weights off to infinity.

Now batch versus per-pattern, because the derivation summed E over all cases. True gradient descent uses the gradient of the *total* error, which is the sum of the per-case gradients — so strictly I should accumulate Δw over a full sweep through all patterns and then step. But it's tempting and common to step after each pattern, using that single pattern's gradient. That isn't the exact total gradient — the weights move between patterns within a sweep, so I'm following a slightly different surface at each step — but with a small enough rate the departure is negligible and per-pattern updates closely approximate true gradient descent, while being cheaper and often faster to converge in wall-clock. Either is fine; the per-pattern version just needs a smaller, more cautious ε.

That brings me to ε and to a real problem with plain gradient descent that I should preempt. The error surface, especially with interior units, tends to have long curved ravines: directions across the ravine where the surface is sharply curved (the floor walls rise steeply) and directions along the ravine where it's nearly flat (a gently descending channel). Pure gradient descent in such a place is miserable. To avoid diverging across the steep direction I have to keep ε small; but a small ε means I crawl along the flat direction, where most of the actual progress lives. The cross-ravine gradient component flips sign step to step (overshooting back and forth across the channel) while the along-ravine component points steadily the same way. So I want something that cancels the oscillating component and accumulates the steady one — exactly what a running average of the recent weight changes does. Add a momentum term:

  Δw_ji(t) = ε · δ_j · y_i + α · Δw_ji(t − 1),

so each update is the current gradient step plus a fraction α of the previous update. The oscillating cross-ravine contributions, alternating in sign, partly cancel in the running sum; the consistent along-ravine contributions reinforce and build up speed. It filters the high-frequency wiggle out of the trajectory and lets me take effectively larger steps along the floor without the steep direction blowing up. A value of α around 0.9 keeps a memory of roughly the last several updates — enough smoothing to damp the oscillation, short enough to stay responsive. With α = 0 I recover plain descent and get the same solutions, just slower. So momentum isn't changing what's being minimized, only the path taken to the minimum, and it earns its place by curing the ravine pathology that forces ε uncomfortably small.

Let me now also be clear-eyed about the picture this all paints, because it's worth stating what the method buys. The output unit only ever sees the interior layer's outputs, and the interior weights are now trained, by exactly this back-propagated error, to make those outputs into whatever features the output unit needs — the network *invents* its own internal representation, the recoding I had to supply by hand for XOR. After learning, the interior units come to encode the structure the task actually depends on (for parity, units that effectively count how many inputs are on; for symmetry, units sensitive to mirror-image positions), not because I told them to but because those are the features that drive the output error down. The whole machine is just: forward pass through layers of smooth nonlinear units, compute the output error, sweep that error backward through the same weights gated by each unit's local slope to get a δ at every unit, and step each weight by ε δ_j y_i with a dash of momentum. Differentiable units make a slope to follow; the nonlinearity makes the interior layer worth having; the backward recursion manufactures the error signal that interior units lack; small random init breaks the symmetry that would otherwise freeze them identical.

I can write the network and training rule directly now: batch updates over a sweep (summing the per-case gradients into one step), momentum, logistic units, small random initialization, and biases as always-on inputs folded into each layer.

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))          # y = 1/(1+e^{-x}): smooth, nonlinear, monotone


class MLP:
    """A multilayer perceptron: layers of logistic units, trained by propagating
    the output error backward through the same weights (the generalized delta rule)."""

    def __init__(self, layer_sizes):
        # small RANDOM init breaks the symmetry that would freeze interior units identical,
        # and keeps units in the responsive midrange of the logistic
        self.W = [np.random.uniform(-0.3, 0.3, (a, b))
                  for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(b) for b in layer_sizes[1:]]
        self.vW = [np.zeros_like(W) for W in self.W]   # momentum buffers (last weight change)
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, Y):
        # x_j = sum_i y_i w_ji + bias ;  y_j = sigmoid(x_j).  Keep every layer's output
        # so the backward pass can reuse it (the slope is y(1-y), no recompute needed).
        activations = [Y]
        for W, b in zip(self.W, self.b):
            X = activations[-1] @ W + b
            activations.append(sigmoid(X))
        return activations

    def backward(self, activations, target):
        # delta at the OUTPUT layer: (d - y) * y(1-y)  -- output error times local slope
        Y_out = activations[-1]
        delta = (target - Y_out) * Y_out * (1.0 - Y_out)
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        for k in reversed(range(len(self.W))):
            Y_in = activations[k]                       # outputs of the layer below
            # gradient of E w.r.t. each weight: dE/dw_ji = -y_i * delta_j  (batch-summed)
            dW[k] = -(Y_in.T @ delta)
            db[k] = -np.sum(delta, axis=0)
            if k > 0:
                # propagate the error down: each interior unit collects sum_j delta_j w_kj,
                # then multiplies by its OWN local slope y(1-y)
                Y_below = activations[k]
                delta = (delta @ self.W[k].T) * Y_below * (1.0 - Y_below)
        return dW, db

    def train(self, data, lr=0.5, momentum=0.9, n_sweeps=1000):
        for _ in range(n_sweeps):
            X = np.stack([inp for inp, _ in data])      # one batch = one sweep
            T = np.stack([t for _, t in data])
            activations = self.forward(X)
            dW, db = self.backward(activations, T)
            for k in range(len(self.W)):
                # Delta w(t) = -lr * dE/dw + momentum * Delta w(t-1)
                self.vW[k] = -lr * dW[k] + momentum * self.vW[k]
                self.vb[k] = -lr * db[k] + momentum * self.vb[k]
                self.W[k] += self.vW[k]
                self.b[k] += self.vb[k]
```

The causal chain, start to finish. A directly-wired net can only carve input space with one hyperplane, so non-linearly-separable mappings like XOR and parity are out of reach — the inputs' and outputs' similarity structures disagree and no single line bridges them. Inserting a layer of interior units fixes the *representation* (a hand-built AND feature makes XOR separable, and in general enough interior units can recode any mapping into a separable one), but not the *learning*, because the delta rule needs a target for every trainable unit and interior units have none. Reframing learning as "compute dE/dw for every weight" removes the need for interior targets, but demands a differentiable unit — so the step unit goes, replaced by the smooth logistic, whose nonlinearity is also exactly what keeps stacked layers from collapsing to a single linear map, and whose slope conveniently equals y(1−y). Squared error gives ∂E/∂y = y − d at the output; the chain rule turns that into ∂E/∂x via the local slope and into ∂E/∂w via the upstream activation; and the same chain rule, run backward unit by unit, manufactures the missing interior error signal as δ_j = y_j(1−y_j) Σ_k δ_k w_kj — the error flowing back through the same weights it flowed forward through. One backward sweep, as cheap as the forward one, yields every weight's gradient — and the finite-difference check on the smallest interior-unit net (recursion vs. perturbed forward passes, agreeing to ~10⁻¹¹) is what convinced me the manufactured interior error signal really is the gradient and not a plausible-looking impostor. Small random init breaks the symmetry that would otherwise lock interior units into identical features; momentum cancels cross-ravine oscillation so the step size needn't be crippled; targets of 0.1/0.9 keep weights finite against the logistic's asymptotes. Collapsing f to linear with no interior units reproduces the trusted Widrow–Hoff delta rule character for character, which is the reduction that pins this down as the right generalization rather than a near miss. And it all drops into the layered forward/backward/update loop as two reused sweeps plus a local update per weight.
