We have a set of input patterns, each paired with a desired output, and we want a network of simple neuron-like units that produces the right output on every training case and generalizes sensibly to inputs it has not seen. Each unit forms a weighted sum $x_j = \sum_i y_i w_{ji}$ of the values on its incoming connections and emits some function $y_j = f(x_j)$; the weights are the only adjustable quantities, and a bias is folded in as the weight on an always-on input. When such units are wired straight from input to output with a single threshold output unit, we know exactly how to train them — the perceptron rule, or the smoother Widrow–Hoff delta rule, nudges each weight by the output error — and we know exactly what they can represent: a threshold unit answers $1$ on the points where $w\cdot x + b > 0$, which is one side of a hyperplane, so the patterns it maps to $1$ must be linearly separable from the rest. The mappings we actually care about are not. Exclusive-or is the smallest example and it is enough to ruin a direct net: $00$ and $11$ must give $0$, $01$ and $10$ must give $1$, and no single line separates $\{01,10\}$ from $\{00,11\}$. Parity is the same disease scaled up — flip one input bit and the answer flips, so the most similar inputs demand opposite outputs. Whenever the similarity structure of the inputs and the similarity structure of the required outputs disagree, a single hyperplane cannot bridge the gap, and a directly-wired net is not merely inconvenient but representationally incapable.

The way past the hyperplane ceiling is to change the space in which the decision is made. Insert a layer of interior (hidden) units between input and output, and the output unit no longer sees the raw inputs — it sees whatever those interior units compute. There is an existence fact to lean on: for XOR, let one hidden unit fire exactly when both inputs are on (the AND), and describe each pattern by $(x_1, x_2, \text{AND})$, so $00\to000$, $01\to010$, $10\to100$, $11\to111$. Give $x_1$ and $x_2$ weight $+1$ and the AND unit weight $-2$; then $000$ scores $0$, $010$ and $100$ each score $1$, and $111$ scores $1+1-2=0$. A threshold at $0.5$ separates the two middle patterns from the rest, so the hidden unit has recoded the inputs into a space where the impossible mapping became a plain hyperplane decision. This is not special to XOR: with enough hidden units, some recoding always exists that makes any desired mapping separable at the output. But "with the right hidden unit" is exactly the catch — for XOR I chose the AND feature by hand, and for an arbitrary mapping I cannot guess the right features, which defeats the purpose of a learning machine. The real obstacle is therefore not representation but learning: the delta rule changes a weight into unit $j$ by comparing $j$'s output to a supplied target, and a hidden unit has no target. The data says only what the final output should be, not what an interior unit should compute. We have a scalar error at the top and a pile of interior weights with no local error to compare against. Unsupervised recoding of the hidden units does not tie the features to the particular mapping asked for; freezing hand-chosen features sacrifices the network's ability to discover representations; the Boltzmann-machine line learns hidden units but demands stochastic units, two relaxation-to-equilibrium phases per update, and symmetric connections. None of these is the lightweight, deterministic, feedforward rule we want.

I propose the multilayer perceptron, trained by the generalized delta rule — error backpropagation. The reframing that breaks the wall is this: to change a hidden weight by gradient descent I do not need a target for that unit, I only need $\partial E/\partial w$, the derivative of the final error with respect to that weight. If I can compute, for every weight anywhere in the network, how the one scalar error responds to wiggling it, I can step every weight downhill and never need a hidden target at all. Two requirements on the unit follow before any derivative can exist. First it must be differentiable: a step unit's derivative is zero almost everywhere and undefined at the threshold, so a small weight change leaves the error exactly unchanged until the sum crosses the threshold and the output jumps — there is nothing to descend. Second it must be nonlinear, and this is not a flourish but a necessity, because with linear hidden units the layers collapse: with row-vector activations, $(xW_1 + b_1)W_2 + b_2 = x(W_1W_2) + (b_1W_2 + b_2)$, a single equivalent affine map with exactly the one-layer ceiling we are trying to escape. The logistic $y = 1/(1+e^{-x})$ satisfies both — it is a smooth, monotone, bounded soft threshold — and its slope is unusually convenient. Writing $y = (1+e^{-x})^{-1}$, its derivative is $e^{-x}/(1+e^{-x})^2$, and since $e^{-x}/(1+e^{-x}) = 1 - y$, this collapses to
$$\frac{dy}{dx} = y(1-y),$$
expressible entirely from the unit's own output, which is already stored from the forward pass — no extra exponential in the backward pass. The slope peaks at $y=0.5$ and vanishes as $y\to0$ or $y\to1$, so committed (saturated) units barely move while units sitting on the fence are the most adjustable, and learning naturally concentrates where it is needed.

For the objective I take squared error, both because it is differentiable and because it reduces to the rule I already trust. Define $E = \tfrac12 \sum_c \sum_j (y_{j,c} - d_{j,c})^2$, where the $\tfrac12$ makes the derivative clean. Differentiating with respect to an output unit's value (the cases simply add, so I drop the case index) gives $\partial E/\partial y_j = y_j - d_j$ — the sensitivity of the error to an output's value is just its signed error, which already matches the delta rule's "compare output to target." Weights touch a unit through its total input $x_j$, not directly through $y_j$, so by the chain rule $\partial E/\partial x_j = (\partial E/\partial y_j)\, y_j(1-y_j)$: one local multiply by the slope. And because $x_j = \sum_i y_i w_{ji}$ is linear in the weights, $\partial x_j/\partial w_{ji} = y_i$, giving
$$\frac{\partial E}{\partial w_{ji}} = \frac{\partial E}{\partial x_j}\, y_i,$$
the error-sensitivity at the receiving end times the activation at the sending end — local to the connection's two endpoints, with the bias falling out as the special case $y_i = 1$. The whole problem has reduced to getting $\partial E/\partial y_j$ for every unit, including hidden ones. A hidden unit $i$ affects the error only through the units it feeds; wiggling $y_i$ changes each downstream total input $x_j$ by $\partial x_j/\partial y_i = w_{ji}$ (again because $x_j$ is linear in its inputs), so by the chain rule, summing over every unit $j$ that $i$ feeds,
$$\frac{\partial E}{\partial y_i} = \sum_j \frac{\partial E}{\partial x_j}\, w_{ji}.$$
The error-sensitivity at a hidden unit is a weighted sum of the sensitivities at the units it feeds, through the very same connection weights used in the forward direction. The phantom "target" was never needed — what the hidden unit needed was its share of the downstream error, and that share is computable locally.

Naming the shared quantity makes the algorithm crisp. Let $\delta_j = -\partial E/\partial x_j$ be the error signal at unit $j$. For an output unit $\delta_j = (d_j - y_j)\, y_j(1-y_j)$, the output error times the local slope; for a hidden unit
$$\delta_j = y_j(1-y_j) \sum_k \delta_k w_{kj},$$
the local slope times the back-propagated weighted sum of the $\delta$'s of the units $j$ feeds. Every weight then changes by
$$\Delta w_{ji} = \varepsilon\, \delta_j\, y_i,$$
which is exactly the form of the delta rule — error signal at $j$ times input from $i$ — except that for a hidden unit $\delta_j$ is the recursively back-propagated quantity rather than a supplied target mismatch. The single-layer, linear-output, no-hidden-unit case has slope $1$ and recovers $\Delta w_{ji} = \varepsilon(d_j - y_j)y_i$ precisely, the Widrow–Hoff rule on a convex bowl with a guaranteed minimum, which confirms this is the right generalization and not a different animal. What makes it cheap is that the naive alternative — perturb each weight and re-run the forward pass — costs one full network evaluation per weight, whereas this recursion does one forward sweep to fill in all the $y$'s and one backward sweep that visits each unit once (a sum over its outgoing connections plus one slope multiply, the same arithmetic as the forward sum over incoming connections), producing the gradient for every weight at once. The $\partial E/\partial x_j$ are shared subexpressions computed once and reused — the chain rule organized as a dynamic program over the network.

Several design choices keep this working in practice. The weights must start at small random values: if all weights were equal, every hidden unit in a layer would compute the same total input, receive the same $\delta$, and update identically, so they would stay identical forever and the network could only ever express what one hidden unit can — equal weights sit at a symmetric stationary point with no gradient to leave by, so randomness is forced to break the symmetry and let units differentiate, while smallness keeps units in the responsive midrange of the logistic away from the saturated tails where $y(1-y)$ is near zero. The desired outputs should be $0.1$ and $0.9$ rather than exactly $0$ and $1$, because the logistic reaches its asymptotes only at infinite total input, so exact $0/1$ targets would drive the weights to $\pm\infty$ chasing values they can never attain and the error would never zero. And because the error surface, especially with hidden units, tends to have long curved ravines — sharply curved across, nearly flat along — plain descent must keep $\varepsilon$ small to avoid diverging across the steep direction, which crawls along the flat direction where the real progress lives. The cross-ravine gradient flips sign step to step while the along-ravine component points steadily, so a running average of recent updates cancels the oscillation and accumulates the steady descent. I add a momentum term
$$\Delta w_{ji}(t) = \varepsilon\, \delta_j\, y_i + \alpha\, \Delta w_{ji}(t-1),$$
with $\alpha \approx 0.9$ keeping a memory of roughly the last several updates; it does not change what is being minimized, only the path, and $\alpha = 0$ recovers plain descent at lower speed. One honest caveat remains: with hidden units and a nonlinearity the error surface is no longer convex, so descent can stall in a local minimum — the price of the expressiveness that makes XOR possible at all — though spare hidden units and connections add weight-space directions that tend to route around such barriers. The whole machine is then: a forward pass through layers of smooth logistic units, the output error, a backward sweep through the same weights gated by each unit's local slope to get a $\delta$ everywhere, and a step of $\varepsilon\,\delta_j\,y_i$ with momentum — and the hidden units come to invent, on their own, the recoding I once had to supply by hand.

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))            # y = 1/(1+e^{-x}); slope is y(1-y)


class MLP:
    def __init__(self, layer_sizes):
        # small RANDOM init: break symmetry, stay in the logistic's responsive midrange
        self.W = [np.random.uniform(-0.3, 0.3, (a, b))
                  for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(b) for b in layer_sizes[1:]]
        self.vW = [np.zeros_like(W) for W in self.W]   # momentum buffer = last weight change
        self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, Y):
        activations = [Y]
        for W, b in zip(self.W, self.b):
            activations.append(sigmoid(activations[-1] @ W + b))   # x = y@W+b ; y = f(x)
        return activations

    def backward(self, activations, target):
        Y_out = activations[-1]
        delta = (target - Y_out) * Y_out * (1.0 - Y_out)          # output: (d-y)*y(1-y)
        dW, db = [None] * len(self.W), [None] * len(self.b)
        for k in reversed(range(len(self.W))):
            dW[k] = -(activations[k].T @ delta)                   # dE/dw_ji = -y_i * delta_j
            db[k] = -np.sum(delta, axis=0)
            if k > 0:
                Y_below = activations[k]
                # hidden delta: local slope * back-propagated sum_j delta_j w_kj
                delta = (delta @ self.W[k].T) * Y_below * (1.0 - Y_below)
        return dW, db

    def train(self, data, lr=0.5, momentum=0.9, n_sweeps=1000):
        X = np.stack([inp for inp, _ in data])
        T = np.stack([t for _, t in data])
        for _ in range(n_sweeps):
            activations = self.forward(X)
            dW, db = self.backward(activations, T)
            for k in range(len(self.W)):
                self.vW[k] = -lr * dW[k] + momentum * self.vW[k]   # Δw(t) = -ε·∂E/∂w + α·Δw(t-1)
                self.vb[k] = -lr * db[k] + momentum * self.vb[k]
                self.W[k] += self.vW[k]
                self.b[k] += self.vb[k]
```
