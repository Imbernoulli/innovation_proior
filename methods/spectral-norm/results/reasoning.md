OK, let me think this through from scratch. The thing I keep running into when I train GANs is that the discriminator is the whole game and it is the part that misbehaves. The generator never touches the data; it learns only through whatever gradient D hands back. So if D is badly conditioned, G is doomed, and everything I see — the collapse, the oscillation, the dead training — traces back to D.

Let me pin down exactly what goes wrong with D, because "it's unstable" is too vague to act on. Two distinct failures. One: in high dimensions the density-ratio quantity D is implicitly estimating is just inaccurate and jumpy, and G never gets a coherent target, so it misses whole modes. Two, and this is the sharp one: think about the supports. Real images live near some low-dimensional manifold in a huge ambient space; G's samples live near some other low-dimensional manifold. Generically those two manifolds are essentially disjoint. And if the supports are disjoint, there exists a discriminator that separates them perfectly — it can be 1 on the data manifold and 0 on the generator manifold with arbitrarily sharp transition between. The optimizer is happy to find it. But then look at what that perfect D does to the generator: away from the data manifold D is flat at 0, so ∇_x D = 0 almost everywhere, so the gradient that flows back to G is zero. The discriminator wins so completely that it stops teaching. That is the failure mode I have to prevent.

So the disease isn't that D is too weak; it's that D is allowed to become too sharp. The fix can't be "make D better at classification" — that makes it worse. The fix has to be a restriction on D. I want to draw D not from all functions but from some restricted, well-behaved family, and search

    argmax_{f in restricted family} V(G, D),

so that even the best D in the family stays smooth enough to keep handing G a usable gradient.

What restriction, though? Let me look at the optimal discriminator to see what quantity is actually blowing up. For the standard objective the optimum is D* = q/(q+p) = sigmoid(f*), with f* = log q − log p. Differentiate: ∇_x f* = ∇q/q − ∇p/p. Those terms are not bounded — near the edges of either support the density goes small and the ratio explodes. So the dangerous quantity is the *derivative* of D with respect to its input. D's sensitivity to x. That's what I want to cap. Cap how fast D can change as you move the input, and both failures soften at once: the density-ratio statistics stay finite, and the perfect-separator D with its near-infinite slope at the boundary is no longer in the family, so the gradient to G can't collapse to zero.

"How fast can the function change as you move the input" — that's exactly the Lipschitz constant. ‖f‖_Lip is the smallest M with ‖f(x)−f(x')‖ ≤ M‖x−x'‖ for all x, x'. So the restricted family I want is the K-Lipschitz functions, and the game becomes

    argmax_{‖f‖_Lip ≤ K} V(G, D).

Good. This is also where the existing crowd is converging — bounding the discriminator's Lipschitz constant is the thing that keeps the statistics bounded. So the question is no longer "what to control" — it's Lipschitz — but "how to enforce it on a deep net cheaply."

Now, how do the methods I know enforce it, and why am I not satisfied? Let me actually go through them, because the gap one of them leaves is going to point me at the answer.

WGAN's weight clipping. Wasserstein GAN reframes the objective as the Wasserstein-1 distance, whose dual form needs the critic to be 1-Lipschitz, and it enforces that by clipping every weight entry into a box [−c, c] after each update. Crude but it does bound the Lipschitz constant. But watch what the box does to the matrix. The discriminator wants ‖Wx‖ to be large to stay sensitive. Under a hard box on each entry, the way to make ‖Wx‖ largest is to line all the entries up — and that drives W toward rank one. So the layer ends up using essentially one direction. They even named it: capacity underuse. And it trains slowly. So clipping enforces Lipschitz but at the price of crushing the rank.

WGAN-GP, the gradient penalty. This drops the box and adds a soft penalty pushing the input-gradient norm to 1: λ E[(‖∇_x̂ D(x̂)‖_2 − 1)^2], evaluated at points x̂ = εx + (1−ε)x̃ interpolated between real and generated samples. It avoids the rank collapse and it does train strong critics. But two things bug me. First, it only pins the gradient where samples currently are — on the support of the *current* generator. It says nothing about the rest of input space, and the generator's support drifts during training, so the regularization target is a moving thing; I'd expect it to get shaky, and indeed aggressive learning rates wreck it. Second, the cost: to put ‖∇_x̂ D‖ into the loss and then backprop through it I need a gradient of a gradient — an extra forward–backward round every update. That's expensive.

Weight normalization. Normalize each row of W to unit norm, w̄_i = w_i/‖w_i‖. Let me check what that secretly does to the spectrum. The sum of squared singular values is the trace of W̄W̄^T, and with every row unit-norm that's Σ_i (w_i/‖w_i‖)·(w_i^T/‖w_i‖) = d_o. So Σ_t σ_t(W̄)^2 = d_o, a fixed budget. Now I want ‖W̄h‖ large for sensitivity. Under a fixed sum of squared singular values, ‖W̄h‖ for a unit h peaks when one singular value swallows the entire budget — σ_1 = √d_o, all the rest zero. Rank one again. Same disease as clipping, just arrived at differently. Frobenius normalization W/‖W‖_F is the same story with the budget fixed to 1.

I'm seeing the shape of the problem now. Three different methods, three different mechanisms, and all three end up secretly favoring low-rank weights, because they all constrain the *whole* spectrum — they fix Σσ_t^2 or bound entries — and then the optimizer, pushing for sensitivity, spends that whole budget on one singular direction. And if I stare at the singular values of a discriminator trained with any of them, that's exactly what I see: the squared singular values pile up on one or two components, the matrices go nearly rank one in the lower layers. A rank-one layer is a discriminator looking at the world through one feature. That's the impoverished D that makes a sloppy G.

And there's the over-correction too: orthonormal regularization adds ‖W^TW − I‖_F^2, which sets *all* singular values to 1. That fixes the collapse — but it goes too far the other way: it wipes out the spectrum entirely and forces the layer to weight every direction equally, including the directions that ought to be thrown away. So either the spectrum collapses to one value (clipping/WN) or it's flattened to all-ones (orthonormal). Neither is what I want.

So what *do* I want? Let me restate the goal precisely. I want to bound the Lipschitz constant — and nothing more. Look back at what the Lipschitz constant of a linear layer actually is. The Lipschitz constant of a differentiable map g is sup_h σ(∇g(h)), the largest singular value of its Jacobian. For a linear layer g(h) = Wh the Jacobian is just W everywhere, so ‖g‖_Lip = σ(W) — the largest singular value of W, the spectral norm. So the Lipschitz constant of a layer is *only* its top singular value. It depends on σ_1 alone. Nothing about σ_2, σ_3, …, nothing about the rank.

That's the whole thing. The other methods are over-constraining because they put a budget on the entire spectrum. But the Lipschitz constant only cares about the top of the spectrum. If all I need is to bound ‖f‖_Lip, then all I need is to bound σ_1 of each layer — and I should leave σ_2, σ_3, … completely free, so the layer can use as many features as it likes. That's the freedom clipping and weight-norm threw away for no reason.

Let me push this from one layer to the whole network. f is W^{L+1} a_L(W^L(… a_1(W^1 x)…)) — linear maps interleaved with activations. The Lipschitz constant is sub-multiplicative under composition: ‖g_1 ∘ g_2‖_Lip ≤ ‖g_1‖_Lip · ‖g_2‖_Lip. And the activations — ReLU, leaky ReLU — are 1-Lipschitz, so they contribute a factor of 1 and drop out. So

    ‖f‖_Lip ≤ Π_l ‖h ↦ W^l h‖_Lip · Π_l ‖a_l‖_Lip = Π_l σ(W^l).

The network's Lipschitz constant is bounded by the product of the layers' spectral norms. So if I force σ(W^l) = 1 at every layer, the product is 1 and ‖f‖_Lip ≤ 1. Clean.

And forcing σ(W) = 1 — how? Not with a penalty, not with a box. Just divide the matrix by its own spectral norm:

    W̄_SN = W / σ(W).

Then σ(W̄_SN) = σ(W)/σ(W) = 1, exactly, by construction. Every layer is exactly 1-Lipschitz, the product bound gives ‖f‖_Lip ≤ 1, and crucially I touched *only* the scale of the top singular value — the *shape* of the spectrum, the ratios σ_2/σ_1, σ_3/σ_1, …, is untouched. The layer keeps full freedom in how many features to use. This is the restriction the Lipschitz constant actually demands, and not one bit more. It also has just one knob — the target Lipschitz constant — and I can fold that into a single global scale, so there's essentially nothing to tune.

Let me contrast this once more against the spectral-norm *penalty* idea — adding a term that pushes σ(W) down in the loss. That's different in kind: a penalty nudges σ toward small values as a soft data-independent regularizer, like L2, and it never actually sets σ to a designated value; the constraint isn't enforced, it's bargained against the main loss. Dividing by σ is a hard normalization — σ is *set* to 1 every step, baked into the weight the forward pass uses. That's what I want.

Now the catch, and it's a real one. W̄_SN = W/σ(W) needs σ(W) = the largest singular value of W, at every layer, every single update. Computing that exactly means a singular-value decomposition per layer per step. That is far too expensive to sit inside a GAN training loop. So either this idea is dead, or I find a cheap estimate of just the top singular value.

I don't need the whole SVD — I need only σ_1, the largest one. That's exactly what power iteration gives. Write W in its SVD as W = Σ_t σ_t u_t v_t^T. Start from a random vector u and iterate

    v ← W^T u / ‖W^T u‖,   u ← W v / ‖W v‖.

Why does this find the top singular pair? Composing the two steps, u gets hit by W W^T each round (and v by W^T W). In the eigenbasis, W W^T has eigenvalues σ_t^2 with eigenvectors u_t; applying it k times scales the component along u_t by σ_t^{2k}. The largest σ_1^2 dominates exponentially, so u → u_1 and v → v_1, as long as the random start isn't exactly orthogonal to u_1 — which happens with probability zero, so I'm safe. Once u ≈ u_1 and v ≈ v_1, read off the singular value:

    u^T W v = u_1^T (Σ_t σ_t u_t v_t^T) v_1 = σ_1,

since u_1^T u_t = δ and v_t^T v_1 = δ. So σ(W) ≈ u^T W v, and each iteration is just two matrix–vector products — cheap.

But honestly, even one power-iteration sweep per layer per step, fully restarted from random each time, might not converge tight enough, and running many sweeps to convergence would eat the savings. So here's the move that makes it practically free. Across consecutive SGD updates W barely changes — it's one small gradient step — so its top singular vector barely moves too. So I don't restart the iteration from scratch each step. I keep a persistent u for each layer, stored as a buffer, and at the next step I run the power iteration starting from last step's u. It's already almost aligned with u_1; one more iteration nudges it back onto the slowly-drifting true direction. So one power-iteration step per update, reusing the carried-over u, tracks σ(W) closely enough. The cost is two mat-vecs per layer — negligible next to the forward and backward passes through the conv stack. That's much cheaper than WGAN-GP's extra gradient-of-gradient round, and the regularization lives in the operator (the weight) itself, not at sample points, so it doesn't care about the generator's drifting support and doesn't get shaky at high learning rates.

For convolutions the weight is a 4-tensor W ∈ R^{d_out × d_in × kh × kw}. I'll just reshape it to a 2-D matrix of shape d_out × (d_in·kh·kw) and take the spectral norm of that. It's not the exact operator norm of the convolution (that depends on stride and padding), but it's off only by a fixed constant, which folds into the single Lipschitz knob — fine.

Let me now make sure I understand what this normalization does to the gradient, because that tells me whether training will behave and whether there are hidden dynamics. The weight the layer uses is W̄_SN = W/σ(W), but the parameter I update is the raw W, so I need ∂W̄_SN/∂W. First I need ∂σ(W)/∂W. The derivative of the largest singular value with respect to the matrix is the outer product of its singular vectors: ∂σ(W)/∂W_{ij} = [u_1 v_1^T]_{ij}. (Almost surely σ_1 is simple, so this is a genuine gradient, not a subgradient.) Then, treating W̄_SN entrywise,

    ∂W̄_SN/∂W_{ij} = (1/σ) E_{ij} − (1/σ^2)(∂σ/∂W_{ij}) W
                   = (1/σ) E_{ij} − (1/σ^2)[u_1 v_1^T]_{ij} W
                   = (1/σ)( E_{ij} − [u_1 v_1^T]_{ij} W̄_SN ),

where E_{ij} is the single-one matrix. Now push the loss through. Let h be the layer's input and δ := (∂V/∂(W̄_SN h))^T be the backprop signal arriving at the layer's output. Summing the chain rule over the mini-batch, the gradient with respect to the raw weight is

    ∂V/∂W = (1/σ(W)) ( Ê[δ h^T] − λ u_1 v_1^T ),   with   λ := Ê[δ^T W̄_SN h],

where Ê is the empirical average over the batch. Let me read that. The first term, Ê[δh^T], is exactly the ordinary unnormalized weight gradient. The second term is a correction: it subtracts off a multiple of u_1 v_1^T — the top singular direction — with an *adaptive* coefficient λ. And λ = Ê[δ^T W̄_SN h] is positive precisely when the backprop signal δ and the layer's own output W̄_SN h point the same way, i.e. when the layer is being pushed to lean even harder on its dominant direction. In that case the second term pushes back, penalizing further growth along u_1 v_1^T. So spectral normalization isn't just a static rescale — its gradient automatically discourages the column space of W from piling up into one direction, which is exactly the rank-collapse pathology that sank weight clipping and weight normalization. It stops any single direction from running away. The fixed point ∂V/∂W = 0 needs Ê[δh^T] = k u_1 v_1^T for some scalar k, which only happens when the raw gradient is itself aligned with the top singular direction — a much milder condition than forcing the whole matrix to rank one.

This generalizes, which is a nice sanity check. For any scalar normalizer N(W), with W̄ = W/N(W), the same chain rule gives ∂V/∂W = (1/N)( ∇_{W̄}V − trace((∇_{W̄}V)^T W̄) ∇_W N ). For Frobenius, N = ‖W‖_F and ∇_W N = W̄, so the penalized direction is W̄ itself. For spectral, N = σ and ∇_W N = u_1 v_1^T, so the penalized direction is the top singular pair. Same skeleton, but the spectral version touches only σ_1 while Frobenius touches the whole matrix — same reason spectral keeps the rank and Frobenius doesn't.

One more idea falls out of having σ in hand. Once I can normalize a layer to be exactly 1-Lipschitz, I can also *deliberately* relax that by reintroducing a single learned scale: W̃ = γ W̄_SN, with γ a learnable scalar. This gives up the strict 1-Lipschitz guarantee at that layer but hands the model one extra degree of freedom — useful when I'm pairing this with another Lipschitz control like a gradient penalty rather than relying on the normalization alone. Worth keeping in the back pocket; it's the same reparametrization idea weight normalization used, but built on the spectral norm instead of the row norm.

Let me put it together as code. The clean way is a hook on each weight-bearing layer that, on every forward pass, recomputes the normalized weight: run one power-iteration step on persistent u, v buffers, estimate σ = u^T W v, and hand the layer W/σ. The power iteration runs under no-grad — the buffers are bookkeeping, not parameters; the gradient flows through the σ in the division, which is what produces the adaptive-penalty term I just derived.

```python
import torch
import torch.nn as nn
from torch.nn.functional import normalize

class SpectralNorm:
    # On every forward pass we replace `weight` by weight / sigma(weight),
    # with sigma estimated by ONE power-iteration step that reuses the
    # singular vectors carried over from the previous step.
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim                      # which axis is "output" (0 for Linear/Conv)
        self.n_power_iterations = n_power_iterations   # one step is enough (warm start)
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        # conv weight (d_out, d_in, kh, kw) -> 2-D matrix (d_out, d_in*kh*kw)
        w = weight
        if self.dim != 0:
            w = w.permute(self.dim, *[d for d in range(w.dim()) if d != self.dim])
        return w.reshape(w.size(0), -1)

    def compute_weight(self, module, do_power_iteration):
        W      = getattr(module, self.name + '_orig')   # the raw trainable weight
        u      = getattr(module, self.name + '_u')       # persistent left vector (buffer)
        v      = getattr(module, self.name + '_v')       # persistent right vector (buffer)
        W_mat  = self.reshape_weight_to_matrix(W)

        if do_power_iteration:
            with torch.no_grad():               # the vectors are bookkeeping, not parameters
                for _ in range(self.n_power_iterations):
                    # v <- W^T u / ||W^T u|| , u <- W v / ||W v||  -> (u, v) drift toward (u_1, v_1)
                    v = normalize(torch.mv(W_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(W_mat, v),     dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)   # so two D-passes can backprop
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(W_mat, v))   # sigma(W) ~= u^T W v
        return W / sigma                            # W_SN ; grad through sigma gives the adaptive term

    def __call__(self, module, inputs):
        # forward pre-hook: only iterate while training
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    @staticmethod
    def apply(module, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        with torch.no_grad():
            W_mat = fn.reshape_weight_to_matrix(weight)
            h, w = W_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)  # random start, not _|_ u_1
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, name)
        module.register_parameter(name + '_orig', weight)   # keep raw W as the parameter we update
        setattr(module, name, weight.data)
        module.register_buffer(name + '_u', u)              # u, v persist across steps -> warm start
        module.register_buffer(name + '_v', v)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, dim=None, eps=1e-12):
    if dim is None:
        dim = 0   # output axis for Linear/Conv2d
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module
```

And wiring it into the discriminator is just wrapping every weight-bearing layer; the generator is an ordinary stack.

```python
def D_conv(in_ch, out_ch, k, s, p):
    return spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p))

class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            D_conv(3,   ch,   3,1,1), nn.LeakyReLU(0.1),   # leaky ReLU is 1-Lipschitz
            D_conv(ch,  ch*2, 4,2,1), nn.LeakyReLU(0.1),
            D_conv(ch*2,ch*4, 4,2,1), nn.LeakyReLU(0.1),
            D_conv(ch*4,ch*8, 4,2,1), nn.LeakyReLU(0.1),
        )
        self.fc = spectral_norm(nn.Linear(ch*8*4*4, 1))   # final scalar, also normalized
    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc(h)
```

So the causal chain, start to finish: the discriminator is the only teacher and it destabilizes by becoming too sharp — in the disjoint-support limit a perfect D has zero input-gradient and kills G's learning signal; the cure is to draw D from the K-Lipschitz family; the Lipschitz constant of a linear layer is exactly its largest singular value σ(W), and by sub-multiplicativity the network's Lipschitz constant is bounded by the product of the layers' spectral norms; so I enforce σ = 1 per layer by dividing the weight by its own spectral norm, W̄_SN = W/σ(W) — which, unlike clipping or weight/Frobenius normalization, constrains only the top of the spectrum and so never forces the weight toward rank one; computing σ exactly per step is too costly, so I estimate just the top singular value by power iteration, made nearly free by carrying the singular vector across steps and running a single iteration per update; and the gradient of the normalized weight turns out to carry a built-in adaptive penalty against any one direction dominating, which is precisely the rank-collapse it was designed to avoid.
