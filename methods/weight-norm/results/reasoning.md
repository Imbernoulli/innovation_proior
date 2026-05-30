OK, let me start from the thing that actually bites when I train deep nets, which is that first-order SGD lives and dies by the curvature of the loss. When the Hessian at the solution is ill-conditioned — a big spread between its largest and smallest eigenvalues — the gradient points mostly along the steep directions and the iterate bounces back and forth across a narrow valley while crawling along the flat floor of it. Martens and the Sutskever-Martens-Dahl-Hinton line on initialization and momentum hammered this: pathological curvature is what stalls deep nets, far more than anything to do with the choice of nonlinearity. So if I want faster training, the lever is conditioning.

Here is the fact I keep underusing, though. The curvature the optimizer sees is not a property of the function I'm fitting — it's a property of the *coordinates* I write the parameters in. Amari said this cleanly years ago: the same model can be parameterized many equivalent ways, and the geometry of the loss in parameter space, the thing that determines how a gradient step behaves, changes when I change coordinates. Two parameterizations computing the identical input-output map can have wildly different condition numbers. So "find better coordinates for the parameters" is a real and largely free lever, completely separate from "find a better optimizer." That reframes the whole problem: I don't have to estimate curvature and fight it; I might be able to *reparameterize* it away.

What's the principled target if I did want to fight curvature head-on? The natural gradient. Precondition the gradient by the inverse Fisher information matrix and the update becomes invariant to reparameterization — it whitens the step, so a unit of progress is a unit of progress in every direction. The catch is purely practical: the Fisher is enormous and you have to estimate it and invert it. The whole recent crop of methods is approximations to exactly this. KFAC writes the Fisher as a Kronecker product of two small factors and inverts those. FANG keeps a sparse approximate Cholesky of the inverse Fisher. PRONG whitens the input to each layer so that the plain gradient comes out approximately natural. They work, but every one of them pays in computation, memory, and bookkeeping to track curvature, and they bolt onto the optimizer rather than improving the model.

So I'd rather take the other road, the one Raiko, Valpola and LeCun pointed at: leave the optimizer as vanilla SGD, and instead change the parameterization of the model itself so that the *ordinary* gradient already looks whitened. They transformed each neuron's output to have, on average, zero value and zero slope, and showed that this approximately diagonalizes the Fisher — which is exactly making the ordinary gradient look natural. That's the template I want to follow: a reparameterization, not a preconditioner. The question is which reparameterization.

Now the obvious incumbent in this space is batch normalization. For each neuron it takes the pre-activation t = v·x and standardizes it over the minibatch: t' = (t − μ[t])/σ[t], with μ[t], σ[t] the mean and standard deviation of t across the examples in the batch, and then a learnable scale and shift go back on. It accelerates training, it lets you crank the learning rate, and the stated reason is that it reduces the drift in each neuron's input distribution and pushes the Fisher toward the identity. So in spirit it is doing the same thing I want — improving conditioning by controlling the distribution at each neuron.

But let me stare at *how* it does it, because the mechanism is where the trouble is. The μ[t] and σ[t] are computed across the examples in the minibatch. That single choice — statistics estimated over the batch — drags in a cluster of problems, and they're the reason I don't just use it and go home. First, the output for one example now depends on the other examples that happened to share its batch; the examples are coupled, which is a strange thing for a per-example prediction to be. Second, μ[t] and σ[t] are *estimates*, so the layer is injecting stochastic noise into the gradient, and when the minibatch is small that noise has high variance. Third, train and test compute different functions: at test time there's no representative batch, so you freeze running averages and substitute them, and the train-time and test-time maps don't match. Fourth, recurrence breaks it — in an LSTM the same weights are reused at every timestep and the cell state is supposed to carry information across many steps; standardizing the cell states each step bleeds that information away, and it's not even clear what "the batch statistic" should mean across timesteps. Fifth, it costs real time and memory. The noise alone makes it a bad fit for reinforcement learning, where I've watched DQN destabilize, and for generative models, which are sensitive to exactly this kind of injected noise.

So I want the conditioning benefit of batch normalization without the batch. Let me localize what's actually doing the work in t' = (t − μ[t])/σ[t]. The subtraction of μ[t] centers; the division by σ[t] rescales. The rescaling is the interesting one for conditioning. σ[t] is, up to the data, the scale of the pre-activation, and the pre-activation scale is set by the magnitude of the weight vector v. Dividing by it is, roughly, making the neuron's output insensitive to the length of v. And here's the thing I keep circling back to: the length of the weight vector and its direction are two very different objects geometrically, but in the plain parameterization they're tangled together in one vector v, and the gradient has to move both at once. That tangling smells like a source of bad conditioning all on its own.

Let me make that concrete by writing the special case where I can compute everything. Take a single-layer net, and suppose the inputs x are whitened — independent, zero mean, unit variance. Then the pre-activation is t = v·x, its mean is μ[t] = v·E[x] = 0, and its variance is Var(t) = Var(Σ_i v_i x_i) = Σ_i v_i² Var(x_i) = Σ_i v_i² = ‖v‖². So σ[t] = ‖v‖. Now substitute back into batch norm: t' = (t − μ[t])/σ[t] = (v·x)/‖v‖. The minibatch statistics have completely dissolved into a deterministic quantity: the whole effect of the normalization, in this case, is just to divide the pre-activation by the norm of the weight vector. There's no batch left in it at all.

That's the crack I want to pry open. In this clean case, normalizing the *activations* over the batch is exactly the same as normalizing the *weight vector* by its own norm — a quantity that has nothing to do with the batch, nothing stochastic, computable from the parameters alone. So instead of standardizing t by an estimated σ[t], what if I just build the (v·x)/‖v‖ behaviour directly into the parameterization of the weight? Let the weight that actually multiplies x be

    w = (g / ‖v‖) · v,

where v is the trainable direction-carrying vector, g is a separate trainable scalar, and I do SGD directly in v and g (and the bias) rather than in w. Then ‖w‖ = ‖(g/‖v‖)v‖ = g·‖v‖/‖v‖ = g, exactly, independent of v. The length of the effective weight is owned solely by g; the direction of the effective weight, w/‖w‖ = v/‖v‖, is owned solely by v. I've torn the magnitude and the direction apart into two parameters that no longer fight over the same coordinate. The pre-activation becomes w·x = g·(v·x)/‖v‖, which is exactly g times the deterministic normalized quantity from the whitened-input calculation. So I get the rescaling effect of batch norm, deterministically, per example, with no batch coupling and no injected noise — precisely the five problems gone — as long as this actually improves conditioning. That last clause is the thing I have to check, not assume.

A note before I go on, because someone has normalized weight norms before. The max-norm idea (Srebro & Shraibman) constrains ‖w‖, but it keeps optimizing in w and just projects the norm back after each SGD step. That is a completely different animal: the optimizer still sees the plain gradient in w; the projection is an afterthought that doesn't change the geometry the step is taken in. What I'm proposing is to *reparameterize the model* and take the gradient steps in v and g themselves. The geometry the optimizer experiences is the new one. So I need to actually differentiate through w = (g/‖v‖)v and see what gradient the optimizer gets — that's where any conditioning benefit has to show up, and it's where I'll find out if this was a good idea.

Let me differentiate. Write r = ‖v‖ for brevity, so w_i = g·v_i / r and r = (Σ_j v_j²)^{1/2}, giving ∂r/∂v_j = v_j / r. The chain rule sends any loss L through w.

For g: w_i = g·v_i/r, so ∂w_i/∂g = v_i/r, and

    ∇_g L = Σ_i (∂L/∂w_i)(∂w_i/∂g) = Σ_i (∂L/∂w_i)(v_i/r) = (∇_w L · v)/‖v‖.

Clean: the gradient on the scalar g is the projection of the ordinary weight-gradient onto the current direction v, divided by ‖v‖. It measures how much the ordinary gradient wants to *lengthen or shorten* the weight along its own direction.

For v it's more involved because v appears both directly and through r. Differentiate w_i = g·v_i·r^{−1}:

    ∂w_i/∂v_j = g[ δ_ij · r^{−1} + v_i · (−r^{−2}) · ∂r/∂v_j ]
              = g[ δ_ij/r − v_i v_j / r³ ],

using ∂r/∂v_j = v_j/r. Then

    ∇_v L|_j = Σ_i (∂L/∂w_i) · g[ δ_ij/r − v_i v_j/r³ ]
             = (g/r)·(∂L/∂w_j) − (g/r³)·v_j·Σ_i (∂L/∂w_i) v_i
             = (g/‖v‖)·∇_w L|_j − (g/‖v‖³)·(∇_w L · v)·v_j.

Now I recognize the inner product: ∇_w L · v = ‖v‖·∇_g L from the g-gradient above. Substitute it into the second term: (g/‖v‖³)(∇_w L·v) = (g/‖v‖³)·‖v‖·∇_g L = (g·∇_g L)/‖v‖². So in vector form

    ∇_v L = (g/‖v‖)·∇_w L − (g·∇_g L/‖v‖²)·v.

Let me read this. The first term is just the ordinary weight gradient, scaled by g/‖v‖. The second term subtracts off a piece proportional to v — proportional, in fact, to ∇_g L, which was exactly the component of ∇_w L along v. So the v-gradient is the ordinary gradient with its along-v component removed, then scaled. Let me confirm that's a clean projection by checking the geometry. Pull the g/‖v‖ out front and rewrite the bracket: the second term's coefficient on v is (∇_g L/‖v‖²)·g = (g/‖v‖)·(∇_g L/‖v‖) = (g/‖v‖)·(∇_w L·v)/‖v‖². So

    ∇_v L = (g/‖v‖)·[ ∇_w L − ((∇_w L · v)/‖v‖²)·v ].

The bracket is ∇_w L minus its orthogonal projection onto v — i.e. the component of ∇_w L *perpendicular* to v. And since v points along w (they share a direction), v/‖v‖ = w/‖w‖, so I can write the projection with w just as well:

    ∇_v L = (g/‖v‖) · M_w · ∇_w L,   with   M_w = I − (w w')/‖w‖².

M_w is the projector onto the complement of w. So differentiating through the reparameterization does two specific things to the ordinary gradient: it *scales* it by g/‖v‖, and it *projects* it onto the subspace orthogonal to the weight vector. Those two operations are the whole content of weight normalization on the backward pass.

Now, does that buy conditioning? Think about what the covariance of the gradient does under these two operations. If the ordinary weight-gradient has covariance C across the data, then after scaling by g/‖v‖ and projecting by M_w (which is symmetric and idempotent), the covariance of the v-gradient is

    D = (g²/‖v‖²) · M_w C M_w.

The projection kills the component of the gradient along w. Empirically — and this is the load-bearing observation — w tends to sit close to a dominant eigenvector of C: the direction in which the gradient varies most across examples is, often, roughly the direction of the weight itself. So M_w is removing the single biggest, most variable eigen-direction of the gradient noise, and what's left, D, is flatter — closer to a scaled identity — than C was. A flatter gradient covariance is precisely better conditioning, the same whitening that the natural-gradient methods buy by inverting the Fisher, except here it falls out for free from the parameterization. So the reparameterization is doing the Raiko-style trick: making the plain gradient look whitened.

And there's a second, sneakier benefit hiding in the projection that I didn't go looking for. Because ∇_v L is always orthogonal to v — M_w v = 0 since v ∥ w, so ∇_v L · v = 0 identically — a steepest-descent update Δv ∝ ∇_v L is always perpendicular to the current v. Take a plain gradient step without momentum: v' = v + Δv with Δv ⟂ v. Then by Pythagoras

    ‖v'‖ = √(‖v‖² + ‖Δv‖²) = √(‖v‖² + c²‖v‖²) = √(1 + c²)·‖v‖,   where c = ‖Δv‖/‖v‖.

Since √(1+c²) ≥ 1, the norm of v grows monotonically with every update. That's not a bug — watch what it does to the effective scale. The factor multiplying the direction in the forward pass is g/‖v‖. As ‖v‖ grows, that factor shrinks, which means the effective learning rate on the direction shrinks. And the growth rate of ‖v‖ is governed by c, the relative size of the update, which is large exactly when the gradients are noisy. So: noisy gradients → big c → ‖v‖ shoots up → effective step on the direction is throttled down. Quiet gradients → c ≈ 0 → √(1+c²) ≈ 1 → ‖v‖ stops growing → the step stabilizes. The parameterization self-regulates its own effective learning rate, automatically damping when things are noisy. That's why I'd expect a network in these coordinates to tolerate a wide range of learning rates: pick one too large and ‖v‖ just inflates until the effective rate is sane again, then holds there. Interesting that batch-normalized nets are also famously robust to learning rate — this analysis says it's the same mechanism, a norm that grows to soak up an oversized step. Strictly the monotonic-growth argument is for plain steepest descent; with momentum or a per-parameter optimizer like Adam (which I'll use), ‖v‖ can grow or shrink, so it doesn't hold exactly, but qualitatively the same self-stabilizing pull is there.

So the core reparameterization checks out: w = (g/‖v‖)v, SGD in (g, v, b), decoupling length from direction, with a scale-and-project gradient that whitens the noise and a self-stabilizing effective rate. Let me also sanity-check that g doesn't need to be anything fancier than a scalar. I could parameterize it on a log scale, g = e^s, learning s instead — that's more natural in that it lets g roam over many orders of magnitude with bounded steps. Worth a thought, but I don't see it changing conditioning; it just reshapes how g moves, and I'd expect it to wash out. (If I try it, I'd bet test performance is a wash and optimization maybe a touch slower, so I'll keep the plain g.)

Now, what did I lose relative to batch norm that I have to put back by hand? Batch norm, beyond the conditioning effect, also pins the *scale* of every layer's features to roughly unit variance, which is what makes it forgiving of bad weight initializations — if one layer's outputs come out ten times too big, batch norm divides it back down before it compounds through the next layer. My reparameterization does not do that. ‖w‖ = g, and g is just initialized to some value; nothing guarantees that the pre-activations of layer five aren't exploding because the scales compounded. The analytic init schemes (Glorot; He) try to set scales right under distributional assumptions, but those assumptions only hold at init and drift immediately. So I need to fix the initial scales empirically, and I have all the pieces to do it cheaply.

Here's the move. Initialize the direction vectors v from a simple fixed-scale distribution — say each entry drawn from a normal with mean 0 and standard deviation 0.05 — and don't worry about getting their scale exactly right, because g and b are going to absorb it. Then, before training, push *one* minibatch X through the network. At each neuron compute the normalized pre-activation t = (v·x)/‖v‖ on this batch, and measure its mean μ[t] and standard deviation σ[t] over the examples. Now I want the neuron's output to start out standardized, y = φ((t − μ[t])/σ[t]). But (t − μ[t])/σ[t] = (g·t + b) exactly when I pick

    g ← 1/σ[t],   b ← −μ[t]/σ[t],

because then g·t + b = t/σ[t] − μ[t]/σ[t] = (t − μ[t])/σ[t]. So a single feed-forward pass, computing the running mean and std of each neuron's normalized pre-activation and setting that neuron's g and b from them, makes every pre-activation in the network start at zero mean and unit variance — exactly the good starting point batch norm would give, but baked into the initial g and b and then left to train freely. It only holds for the minibatch I used, and later batches drift, but it puts every layer on the same footing at step zero, which is the thing that was missing. This is the same data-dependent initialization idea that LSUV (Mishkin & Matas) and Krähenbühl et al. arrived at around the same time, just applied to the (v, g) coordinates. One caveat: this init needs a feed-forward pass with a meaningful batch, so it doesn't transfer to recurrent nets, where I'll fall back to standard initialization — but for those the whole appeal was that the reparameterization is trivial to apply anyway.

There's still one asymmetry nagging me. The reparameterization fixes the *scale* of activations to be roughly independent of v — that's the σ side of batch norm. But the *mean* of the activations still depends on v, because nothing in w = (g/‖v‖)v centers the pre-activation; I only controlled its length. Batch norm subtracts μ[t] every step; I don't. So if I want to recover the mean-centering benefit too, but without dragging the noisy variance estimate back in, I can do a stripped-down batch norm that subtracts the minibatch mean and *nothing else*:

    t = w·x,   t̃ = t − μ[t] + b,   y = φ(t̃),

with μ[t] the minibatch mean of the pre-activation, running-averaged during training so I can substitute it at test time. Call it mean-only batch normalization. It centers but never divides by an estimated standard deviation. What does it cost on the backward pass? Differentiate: t̃ = t − μ[t] + b, and μ[t] = mean over the batch of t, so for a single example's t the derivative of its own t̃ picks up a 1 from the t term and a −1/m shared piece from the mean over the m batch examples; aggregating over the batch, the gradient that flows back to t is

    ∇_t L = ∇_{t̃} L − μ[∇_{t̃} L],

i.e. the backpropagated gradient gets its minibatch mean subtracted off — it's *centered*. That's the whole effect: forward it centers the activations, backward it centers the gradients, and that's cheap. It does still inject some noise, since μ[t] is estimated — but only the *mean* is estimated now, and by the central limit theorem a minibatch mean is approximately Gaussian, so the injected noise is light-tailed. Compare that to full batch norm, whose noise comes from the minibatch *variance* estimate, which is highly kurtotic — heavy-tailed, occasional big spikes. So mean-only batch norm gives me a gentle, near-Gaussian regularizing noise on top of the clean conditioning from the weight reparameterization, instead of the harsh noise of the full thing. I'd expect the combination — reparameterized weights for conditioning, mean-only batch norm for a touch of benign regularizing noise — to be the sweet spot, and to beat mean-only batch norm bolted onto the plain weight parameterization, because that version has the noise but not the conditioning. That's the hypothesis I'd want to validate.

Let me also pin down the cheapness claim, since it's part of why this beats full batch norm in practice. A convolutional layer has far fewer weights than it has pre-activations — one filter is reused across every spatial location and every example. Full batch norm has to compute statistics over all those pre-activations every step; my reparameterization only touches the weights, computing ‖v‖ once per filter. And ‖v‖ is non-stochastic, a deterministic function of the parameters, so there's no variance in it at all, versus μ[t] and σ[t] which carry the full variance of a minibatch estimate. So weight normalization is a cheaper, deterministic, lower-variance stand-in for batch normalization — exactly equivalent in the single-whitened-layer case I worked out, an approximation in deep nets, but one that keeps much of the speed-up while dropping the batch coupling, the inference-time discrepancy, the recurrent incompatibility, and the heavy gradient noise.

Now let me land this in code. The forward side is a hook that, before each forward pass, rebuilds the effective weight from its two parts. Concretely I keep the direction vector as a parameter (call it the weight's "v" part) and the scalar magnitudes as a per-output-channel parameter (the "g" part), and recompute w = g · v / ‖v‖ where the norm is taken over every dimension except the output-channel dimension (each output neuron has its own weight vector and its own g). Then the layer's forward uses that w as usual. The data-dependent init is a separate one-shot pass: run a batch, for each parameterized layer measure the mean and the standard deviation of its pre-activations over the batch and the spatial axes, set g to one over that std and b to minus mean over that std. Here it is, mirroring how a standard library wires it up — the magnitude/direction split implemented as a pre-forward hook on an ordinary layer, plus the init routine.

```python
import torch
import torch.nn as nn


# --- the reparameterization: w = g * v / ||v||, taken over all dims but the
#     output-channel dim. Implemented as a pre-forward hook that rebuilds the
#     layer's `weight` from a direction `weight_v` and a magnitude `weight_g`
#     before every forward. SGD then runs in (weight_v, weight_g). ---

def _norm_except_dim(v, dim):
    # Euclidean norm of v over every dimension except `dim` (one norm per
    # output channel). Keepdim so it broadcasts back against v.
    if dim == -1:
        return v.norm()
    perm = [dim] + [d for d in range(v.dim()) if d != dim]
    flat = v.permute(*perm).reshape(v.size(dim), -1)
    n = flat.norm(dim=1)
    shape = [1] * v.dim()
    shape[dim] = v.size(dim)
    return n.reshape(shape)


def _compute_weight(module, name, dim):
    g = getattr(module, name + "_g")           # scalar magnitude per output channel
    v = getattr(module, name + "_v")           # direction vector
    return g * (v / _norm_except_dim(v, dim))  # w = g * v / ||v||


class _WeightNorm:
    def __init__(self, name, dim):
        self.name, self.dim = name, dim

    def __call__(self, module, _inputs):       # runs right before forward()
        setattr(module, self.name, _compute_weight(module, self.name, self.dim))


def weight_norm(module, name="weight", dim=0):
    """Reparameterize module.<name> as magnitude (<name>_g) x direction (<name>_v).
    dim=0 keeps a separate magnitude per output channel."""
    w = getattr(module, name)
    del module._parameters[name]               # the raw weight is no longer a parameter
    g = _norm_except_dim(w, dim).data          # initialize g = ||w||  (so w is unchanged)
    v = w.data                                 # initialize v = w
    module.register_parameter(name + "_g", nn.Parameter(g))
    module.register_parameter(name + "_v", nn.Parameter(v))
    setattr(module, name, _compute_weight(module, name, dim))
    module.register_forward_pre_hook(_WeightNorm(name, dim))
    return module


# --- data-dependent initialization: one batch sets g and b so that every
#     pre-activation starts at zero mean / unit variance.  g <- 1/std,
#     b <- -mean/std, measured on the normalized pre-activation t = (v.x)/||v||. ---

@torch.no_grad()
def data_dependent_init(layer, x, dim=0):
    # forward with the current (unit-magnitude) direction, g = 1, b = 0
    g = getattr(layer, "weight_g"); g.fill_(1.0)
    if layer.bias is not None:
        layer.bias.zero_()
    t = layer(x)                               # pre-activation on this minibatch
    reduce = [d for d in range(t.dim()) if d != 1]   # over batch (+ spatial), per channel
    mean = t.mean(dim=reduce)
    std = t.std(dim=reduce) + 1e-10
    g.copy_((1.0 / std).reshape(g.shape))      # g <- 1/std
    if layer.bias is not None:
        layer.bias.copy_(-mean / std)          # b <- -mean/std
    return layer


# --- mean-only batch normalization: subtract the minibatch mean of the
#     pre-activation (running-averaged for test), no variance division. The
#     backward pass then centers the gradient: grad_t = grad_tt - mean(grad_tt). ---

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.momentum = momentum

    def forward(self, t):                      # t = w . x, weight-normalized weight
        reduce = [d for d in range(t.dim()) if d != 1]
        if self.training:
            mu = t.mean(dim=reduce)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.detach())
        else:
            mu = self.running_mean
        shape = [1, -1] + [1] * (t.dim() - 2)
        return t - mu.reshape(shape) + self.bias.reshape(shape)


# --- using it: a weight-normalized conv layer ---
conv = weight_norm(nn.Conv2d(3, 96, 3, padding=1), name="weight", dim=0)
# data_dependent_init(conv, first_minibatch)   # one-shot, before training
# then train with Adam in the usual way; the hook rebuilds `weight` each forward.
```

To recap the chain that got me here: first-order training is bottlenecked by curvature, and curvature is a property of the parameterization, so I can attack it by choosing better coordinates instead of preconditioning. Batch normalization improves conditioning but does it by standardizing activations over the minibatch, which couples examples and injects high-variance noise. In the clean single-layer whitened-input case the standard deviation it divides by is exactly ‖v‖, so the same effect can be had by dividing the weight by its own norm — a deterministic, batch-free quantity. That motivates reparameterizing w = (g/‖v‖)v and doing SGD in (g, v): differentiating gives ∇_g L = (∇_w L·v)/‖v‖ and ∇_v L = (g/‖v‖)∇_w L − (g∇_g L/‖v‖²)v = (g/‖v‖)M_w∇_w L, which scales the gradient and projects it orthogonal to w, removing the dominant noise direction and whitening the gradient covariance; the orthogonality also makes ‖v‖ grow to self-stabilize the effective learning rate. Decoupling magnitude from direction loses batch norm's automatic per-layer scaling, so a one-batch data-dependent initialization sets g ← 1/σ[t] and b ← −μ[t]/σ[t] to start every pre-activation standardized; and since only the scale is fixed by the reparameterization, mean-only batch normalization optionally adds back gentle, near-Gaussian mean-centering noise without the heavy-tailed variance noise of the full version.
