Let me be clear about what I'm trying to learn, because it's not the usual thing. I don't want to fit a function from a vector to a number. I want to learn an *operator* G: u ↦ G(u) — a map that eats a whole function u and returns a whole function G(u). The solution map of a differential equation is exactly this: hand it a forcing signal or an initial condition, get back the solution trajectory. An integral operator is this. A derivative operator is this. So the object I'm regressing is a function-to-function map, learned from examples, and I want it to work for non-local operators like integrals as well as differential ones, without forcing everything onto a grid.

First, can a neural network even represent such a thing? The universal approximation theorem I usually invoke is about functions — one hidden layer, enough width, approximate any continuous function. But there's a stronger result I can lean on: a single-hidden-layer network can approximate any nonlinear continuous *operator* to arbitrary accuracy. That's the existence result I need, and — more usefully — it comes with an explicit *form*, which is the kind of thing I can turn into an architecture. Let me write down what that form actually says, because the structure is the whole point.

Set up the inputs honestly. G(u) is a function, so to get a real number out I evaluate it at a query point y: G(u)(y) ∈ ℝ. The network has to take *two* things: the input function u and the query location y, and output the scalar G(u)(y). But u is a function — I can't feed a function to a network. The simplest faithful encoding is to sample u at a fixed, finite set of locations {x₁, …, x_m} — call them sensors — and feed the vector of values [u(x₁), …, u(x_m)]. I'll require the *same* sensors for every input function (they don't have to lie on a lattice), and I'll put no constraint at all on the query points y. So a data point is a triplet (u, y, G(u)(y)), and one function u generates many data points, one per y.

Now the operator theorem, in its explicit form, says: for any tolerance ε there are integers n, p, m and constants such that

G(u)(y) ≈ Σ_{k=1}^{p} [ Σ_{i=1}^{n} c_iᵏ σ( Σ_{j=1}^{m} ξ_{ij}ᵏ u(x_j) + θ_iᵏ ) ] · [ σ(w_k·y + ζ_k) ].

Stare at the two bracketed factors. The first depends *only* on the sampled input values u(x_j) — it's a shallow one-hidden-layer network reading [u(x₁), …, u(x_m)] and producing a scalar. The second depends *only* on the query point y — it's a single neuron σ(w_k·y + ζ_k). And the operator value is a sum over k = 1, …, p of the *product* of these two scalars. So the theorem isn't just telling me a good approximator exists; it's handing me a factorized structure: p terms, each a (function-of-u) times a (function-of-y).

The temptation is to ignore all that and just do the obvious thing: I have two inputs, [u(x₁), …, u(x_m)] and y, so concatenate them into one vector [u(x₁), …, u(x_m), y] and run a plain fully-connected network. The function-approximation theorem says that's a universal approximator too, so it *can* represent G. Why not stop here? Two reasons it's the wrong move. First, the concatenated vector has no structure — no locality, no sequence — so there's no reason to use anything fancier than an FNN, and an FNN treats every input coordinate symmetrically. But u(x_j) and y are *not* the same kind of thing playing the same role: the u-values describe *which* input function I'm in, and y describes *where* I'm querying its image. In a d-dimensional problem y is a d-vector whose dimension doesn't even match the m sensor values, so lumping them together is plainly wrong. Second — and this is the empirical tell — when I actually train that concatenated FNN and grow it, the training error keeps dropping but the generalization gap (test minus train) keeps *growing*. That's the signature of a model with too little inductive bias: it has the capacity to fit but no structural prior to generalize. The lesson from images is exactly this — a plain FNN can represent the ideal classifier, but a CNN, with its structural prior, generalizes far better. So I shouldn't fight the theorem's structure; I should bake it into the architecture.

The structure tells me to use *two* sub-networks, because u and y enter through separate factors. One sub-network takes the sensor values [u(x₁), …, u(x_m)] and the other takes the query y, and I combine their outputs by the sum-of-products the theorem prescribes. Let me name them by what they do: the one reading the input-function values is the *branch*, the one reading the query location is the *trunk*. The trunk takes y and outputs a vector [t₁, …, t_p] ∈ ℝᵖ; the branch takes [u(x₁), …, u(x_m)] and outputs a vector [b₁, …, b_p] ∈ ℝᵖ; and I merge them by

G(u)(y) ≈ Σ_{k=1}^{p} b_k t_k,

a dot product over the p latent coordinates. This is exactly the theorem's Σ_k (branch_k)(trunk_k). One reading I like: the trunk produces p basis functions t_k(y) of the query location, and the branch produces, from the input function, the *coefficients* b_k(u) in that basis — so G(u)(·) is being expanded in a learned, query-side basis whose coefficients are read off the input function. Equivalently, it's a trunk network whose last-layer weights, instead of being free scalars, are each supplied by a branch network — the operator's dependence on u is pushed entirely into those coefficients. And since the theorem's trunk factor is σ(w_k·y + ζ_k), the trunk should apply its activation in the *last* layer too, so the t_k really are basis-function outputs.

Two deviations from the literal theorem, each for a reason. The theorem only guarantees a *shallow* one-hidden-layer net for each factor. I'll make both sub-networks *deep*, because depth buys expressivity and I want the approximation error small without absurd width — the theorem's blessing of the shallow form doesn't forbid deepening it. And the theorem's branch factor has no bias in its final aggregation. Bias isn't needed for the approximation guarantee, but adding a bias in the branch's last layer and an overall bias b₀ at the end,

G(u)(y) ≈ Σ_{k=1}^{p} b_k t_k + b₀,

gives the optimizer an easier target and — this is the part I care about — reduces the generalization error and the run-to-run variance from random initialization. So I add it; it costs almost nothing and the theorem doesn't object.

Now a practical issue with realizing the sum over k. The theorem has p separate branch terms — literally p independent branch networks, each producing one scalar b_k, sharing a single trunk. That's the direct transcription, and I'll call it the *stacked* form: one trunk, p branch nets stacked in parallel. But p is at least of order ten in anything realistic, and running ten-plus independent branch networks is expensive in compute and memory. There's no reason the p coefficients have to come from p disjoint networks, though — I can let a *single* branch network output the whole vector [b₁, …, b_p] ∈ ℝᵖ at once, sharing all its hidden features across the p coefficients. Call this the *unstacked* form: one trunk, one branch producing all p coefficients. It has far fewer parameters, trains faster, uses less memory.

Does collapsing to one branch hurt? I'd worry it reduces capacity, and indeed the unstacked form shows a somewhat *larger training* error than the stacked one. But the quantity I actually care about is test error, and there the unstacked form is *smaller* — its generalization gap is tighter (train and test MSE track each other almost linearly), which is consistent with the general lesson that the leaner, more-shared model generalizes better. So the parameter sharing isn't just a memory saving; it's additional regularization. Unstacked, with bias, is the configuration to use.

Step back and ask *why* this whole construction generalizes when the concatenated FNN didn't, because that's the real justification. G(u)(y) has two genuinely independent inputs — the function u and the location y — and the branch/trunk split encodes that independence structurally: the branch is responsible for "which function," the trunk for "where." That's a strong inductive bias, the operator-learning analogue of convolution's prior for images or recurrence's prior for sequences, and it's why even with plain FNN sub-networks the architecture generalizes well. Another way to see it: G(u)(y) is a function of y *conditioned on* u, and the branch is the mechanism that injects the conditioning into the trunk's basis. The architecture is high-level — it says nothing about what the sub-networks must be — so the branch and trunk can be FNNs (the simplest choice, which I'll use), or CNNs if the sensors happen to lie on a grid, or attention-based for general settings.

One more thing the theorem leaves me to pin down: how many sensors m do I need? Intuitively, m must be large enough that the vector [u(x₁), …, u(x_m)] determines u finely enough to hit accuracy ε. I can make this rigorous for, say, an ODE operator. Let ℒ_m be the operator that reconstructs a function from its m sensor values (an interpolation), and consider the set of reconstructed functions U_m = {ℒ_m(u) : u ∈ V}. Because V is compact and a continuous operator preserves compactness, U_m is compact; the union W_m = V ∪ U_m is compact, and the union over all m, W = ∪_i W_i, is still compact. Since G is continuous, G(W) is compact in the output space. Working over this compact set, the operator approximation result applies, so there exist a sensor count m and a (shallow) network 𝒲₂·σ(𝒲₁·[u(x₀) … u(x_m)]ᵀ + b₁) + b₂ such that ‖(Gu)(d) − that net‖₂ < ε for *all* u. So a finite m suffices, and the right m depends on how rich the input-function space is. I'd expect the error to fall fast as I add sensors until u is resolved, then plateau once more sensors add no information — which is what I'd want to verify, and it tells me there's a natural sensor count past which spending more is wasted.

Let me write the architecture. Branch and trunk are both FNNs; the trunk activates its last layer; both output ℝᵖ; I merge by a dot product over the p coordinates and add a scalar bias. For a batch where each input function u (its m sensor values) is paired with its own query y, that's an element-wise dot product per sample.

```python
import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(), last_activation=False):
        super().__init__()
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        )
        self.activation = activation
        self.last_activation = last_activation

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1 or self.last_activation:
                x = self.activation(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, m, p, branch_layers, trunk_layers, activation=nn.Tanh()):
        super().__init__()
        # branch: [u(x_1),...,u(x_m)] -> [b_1,...,b_p] in R^p
        self.branch = FNN([m] + branch_layers + [p], activation)
        # trunk: y -> [t_1,...,t_p] in R^p, WITH activation on the last layer (t_k are basis functions)
        self.trunk = FNN(trunk_layers + [p], activation, last_activation=True)
        # overall bias b_0 (reduces generalization error; not required by the theorem)
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors, y):
        # u_sensors: [batch, m] ;  y: [batch, d]
        b = self.branch(u_sensors)           # [batch, p]  coefficients b_k(u)
        t = self.trunk(y)                    # [batch, p]  basis functions t_k(y)
        # merge: G(u)(y) = sum_k b_k t_k + b_0   (dot product over the p latent dimension)
        out = torch.einsum("bi,bi->b", b, t)
        out = out + self.b0
        return out                            # [batch]
```

(For evaluating one input function at many query points on a grid, the same merge becomes an outer product over the latent dimension, `einsum("bi,ni->bn", b, t)`, giving every (u, y) pair — the unstacked, single-branch form makes this cheap.)

Recapping the chain: I want to learn a function-to-function operator, so I encode the input function by its values at fixed sensors and take the query location as a second input; the operator universal-approximation result gives an explicit factorized form, a sum over p of (a network of the sensor values) times (a neuron of the query), which immediately argues against concatenating the two inputs into one FNN (no inductive bias, generalization gap grows) and for two sub-networks — a branch reading the input function and a trunk reading the query — merged by the dot product Σ_k b_k t_k that the theorem dictates; I deepen both sub-networks for expressivity, add a bias to lower generalization error, and replace the theorem's p separate branch networks with a single branch that emits all p coefficients (the unstacked form), which is cheaper and generalizes better; the branch/trunk split is the operator-learning inductive bias (which function vs. where) that makes even plain FNN sub-networks generalize, and a compactness argument shows a finite sensor count m suffices for any target accuracy.
