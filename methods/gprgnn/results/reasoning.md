Let me start from what actually breaks when I try to classify nodes on a real graph. I have features `X` and a graph, some labels, and I want to predict the rest. Everyone's recipe is the same: aggregate each node's neighborhood with the normalized adjacency `Â = D̃^{-1/2} Ã D̃^{-1/2}` (self-loops folded in), transform with a weight matrix, repeat. And on Cora or CiteSeer it works, because those graphs are homophilic — neighbors mostly share my label, so averaging my neighborhood denoises my own label estimate. But hand me a graph where neighbors tend to *disagree* — a webpage graph where a student page links to a course page links to a faculty page, or any disassortative structure — and the same averaging is poison. It smears together exactly the things I need to keep apart. The signal that says "I am different from my neighbors" is precisely what neighborhood averaging annihilates. So the first thing I have to admit is that there is no single propagation rule that is right for both kinds of graph; the operation I want is graph-dependent, and the standard stack has baked one choice — averaging — into its bones.

And there's a second wound that's tangled up with the first. Suppose the label information I need is several hops away. I'd like to propagate far. But I know what happens when I push `Â` to a high power. `Â` is symmetric, so write `Â = U Λ U^T`. The Laplacian `L̃_sym = I − Â` is PSD — for any `u`, set `f = D̃^{-1/2} u` and expand, `u^T L̃_sym u = u^T u − u^T D̃^{-1/2} Ã D̃^{-1/2} u = Σ_i D̃_ii f_i^2 − Σ_{i,j} f_i f_j Ã_{ij}`, and symmetrizing the middle term, `= (1/2) Σ_{i,j} Ã_{ij}(f_i − f_j)^2 ≥ 0`. So every eigenvalue of `Â` is `≤ 1`. For a connected graph, `0` is a simple eigenvalue of `L̃_sym`: the vector `π` with `π_i = √D̃_ii / √(Σ_v D̃_vv)` satisfies `L̃_sym π = π − D̃^{-1/2} Ã D̃^{-1/2} (D̃^{1/2} 𝟙)/√(ΣD̃) = π − D̃^{-1/2} Ã 𝟙 /√(ΣD̃) = π − D̃^{-1/2} D̃ 𝟙/√(ΣD̃) = π − π = 0`, using `Ã 𝟙 = D̃ 𝟙` (row sums are degrees). So `Â π = π`, `λ_1 = 1`. And the same quadratic form is bounded above: `(1/2)Σ Ã_{ij}(f_i − f_j)^2 ≤ Σ Ã_{ij}(f_i^2 + f_j^2) = 2 Σ_i (u_i^2/D̃_ii) Σ_j Ã_{ij} = 2 Σ_i u_i^2 = 2`, so the largest Laplacian eigenvalue is `≤ 2`, with equality only for a bipartite graph — and self-loops make `G` non-bipartite, so strictly `< 2`, i.e. `λ_n > −1`. Putting it together, `1 = λ_1 > λ_2 ≥ … ≥ λ_n > −1`, the top eigenvalue is the unique one of magnitude `1`. That single fact is the whole over-smoothing story: `Â^k = U Λ^k U^T`, and every `|λ_i|^k` with `i ≥ 2` decays geometrically while `λ_1^k = 1` survives, so `Â^k → π π^T`, rank one. Concretely `Â^k H^(0) = π β^T + o_k(1)` with `β^T = π^T H^(0)`: after enough hops every node's representation is the *same* vector `β`, scaled only by its degree factor `π_i`. The features have lost all the information that distinguished the nodes. So I cannot just stack more `Â`'s to reach far — depth and discrimination are at war.

Now, one idea on the table already half-solves the depth problem, and I want to steal its good part before I fix its bad part. The decoupling idea: instead of interleaving transform and propagation layer by layer, do all the transforming first — `H^(0) = f_θ(X)` with a plain MLP, per node, no graph — and only then propagate. The reason this helps is exactly the eigenvalue fact above: if I propagate with personalized PageRank, the operator is `α(I − (1−α)Â)^{-1}`, and its limit still depends on `H^(0)` (it's not the bare `Â^k → ππ^T`), so I get a large effective neighborhood without the transform itself ever over-smoothing. Let me make sure I understand *why* PPR escapes. Power-iterate `Z^{(k+1)} = (1−α)Â Z^{(k)} + α H`, starting `Z^{(0)} = H`. Unroll: `Z^{(K)} = (1−α)^K Â^K H + α Σ_{k=0}^{K-1} (1−α)^k Â^k H`, and as `K → ∞` with `α ∈ (0,1)`, `(1−α)^K Â^K → 0` and the sum becomes the geometric series `α Σ_{k=0}^∞ (1−α)^k Â^k = α(I − (1−α)Â)^{-1}`. So PPR is, exactly, a *fixed-weight combination of all hop powers* `Σ_k γ_k Â^k` with `γ_k = α(1−α)^k`. It never collapses to rank one because it keeps a positive weight `α` on hop 0 (the un-propagated `H`) plus geometrically fading weights on the deeper hops; the early hops anchor it to the features.

So decoupling is clearly part of the answer, and PPR's "keep all hops, weighted" structure is suggestive. But stare at those weights `γ_k = α(1−α)^k`. They are *fixed* — chosen by one scalar `α` — and they are all positive and monotonically decreasing. Two complaints. First, fixed means I'm asserting a priori how much each hop should count, the same for every graph; that's the same sin as hard-wiring averaging. Second, all-positive-decreasing — I have a nagging feeling that shape is *intrinsically* a low-pass filter, the very thing that fails on heterophily. SGC has the same disease in cartoon form: it's `γ_k = δ_{kK}`, all weight on a single deep hop, which for large `K` is basically the over-smoothing operator itself. Both of these are points in a space of "weighted sums of hop powers," and both points happen to be bad for heterophily. That reframes the whole problem: don't pick a point in that space — *learn* it. Keep the decoupling, `H^(0) = f_θ(X)`, then propagate as `Z = Σ_{k=0}^K γ_k H^(k)`, `H^(k) = Â H^(k-1)`, but let the `γ_k` be free parameters trained end-to-end with `θ` against the classification loss.

Why is this the right object and not just a third arbitrary choice? Because `Σ_{k=0}^K γ_k Â^k = U (Σ_k γ_k Λ^k) U^T` is a polynomial graph filter, with frequency response `g_{γ,K}(λ) = Σ_k γ_k λ^k` applied to each eigenvalue. Learning the `γ_k` is *exactly* learning the filter's response curve, and polynomials of `Â` can approximate any graph filter, with larger `K` giving a finer approximation. So this isn't a third heuristic — it's the universal family that contains GCN-ish low-pass, SGC, and APPNP as special cases, and lets the data pick the response. APPNP falls out at `γ_k = α(1−α)^k`; SGC at `γ_k = δ_{kK}`. Good. The decoupled prediction-then-propagate stays, but the propagation is now a learnable degree-`K` polynomial in `Â`.

Now heterophily forces the sign question. APPNP's `γ_k = α(1−α)^k > 0` and SGC's `γ_k = δ_{kK} ≥ 0` are both *nonnegative*. I want to let `γ_k` go *negative*. Is that actually meaningful, or am I just adding parameters? Let me prove it's the difference between low-pass and high-pass. Take the response `g_{γ,K}(λ) = Σ_k γ_k λ^k`. The low-frequency component is at `λ_1 = 1`, the high-frequency ones at the small/negative `λ_i`. A filter is "low-pass" if it amplifies `λ_1` relative to the rest, i.e. `|g(λ_i)/g(λ_1)| < 1` for all `i ≥ 2`; "high-pass" if that ratio exceeds `1`. Suppose `γ_k ≥ 0`, `Σ_k γ_k = 1`, and at least one `γ_{k'} > 0` with `k' > 0`. Then `g(λ_1) = Σ_k γ_k 1^k = 1`, and for `i ≥ 2`, since `|λ_i| < 1`,
```
|g(λ_i)| ≤ Σ_k γ_k |λ_i|^k ≤ Σ_k γ_k = 1,
```
and the second inequality is *strict*: equality would need `|λ_i|^k = 1` wherever `γ_k > 0`, but `|λ_i|^k < 1` for every `k ≥ 1`, and by assumption some `γ_{k'} > 0` with `k' ≥ 1`. So `|g(λ_i)| < 1 = |g(λ_1)|` for all `i ≥ 2` — nonnegative weights *force* a low-pass filter. That's not a tuning artifact; it's a theorem. APPNP and SGC, being nonnegative, can never be anything but low-pass, which is exactly why they drown on heterophilic graphs. Now let the signs alternate — take `γ_k = (−α)^k`, `α ∈ (0,1)`, and `K → ∞`. Then `g(λ) = Σ_k (−αλ)^k = 1/(1+αλ)` (converges since `|αλ| < 1`). Compare frequencies: `|g(λ_i)/g(λ_1)| = |(1+α)/(1+αλ_i)|`, and since `λ_i < 1`, the denominator `1 + αλ_i < 1 + α`, so the ratio is `> 1` for every `i ≥ 2` — high-pass, and the gain is largest as `λ → −1` (the most oscillatory, near-bipartite component). So signed coefficients are what unlock high-pass behavior, and high-pass is what heterophily needs. The freedom to be negative is not cosmetic; it is the entire mechanism by which one model spans homophily and heterophily. I'll let `γ_k` be unconstrained reals.

But now I've re-opened the depth wound from a new angle. I argued I want large `K`. With learnable signed `γ_k`, what stops the deep hops — the ones near the over-smoothing limit — from contributing garbage `π β^T` terms that swamp the prediction? With APPNP the answer was structural: the weights fade geometrically so deep hops are pre-suppressed. I threw that away by making the weights free. So I need a *different* guarantee, and the natural place to look is the gradient: if a deep hop is over-smoothing and therefore unhelpful, does the loss itself push its `γ_k` toward zero? If so, the model self-heals — it can afford large `K` because the optimizer mutes the harmful hops, and crucially it does so guided by the *labels*, unlike APPNP whose suppression is label-blind.

Let me actually compute that gradient. Suppose hop `k` is in the over-smoothing regime, so `H^(k) = Â^k H^(0) = π β^T + o_k(1)`, where `β^T = π^T H^(0)` is the same row for every node up to the scalar `π_i`. To make the argmax behavior tractable I'll write the prediction with a temperature, `P̂ = softmax_η(Z)`, `softmax_η(z)_i = e^{η z_i}/Σ_j e^{η z_j}` (`η = 1` is the usual softmax; I'll take `η` large at the end), and cross-entropy loss `L = Σ_{i∈T} −log⟨P̂_{i:}, Y_{i:}⟩` over the training set `T`, one-hot labels `Y`. First the generic derivative. `L = Σ_i (log Σ_m e^{η Z_{im}} − η⟨Z_{i:}, Y_{i:}⟩)`, and `∂Z_{im}/∂γ_k = H^(k)_{im}`, so
```
∂L/∂γ_k = Σ_{i∈T} η ( Σ_m softmax_η(Z)_{im} H^(k)_{im} − ⟨H^(k)_{i:}, Y_{i:}⟩ )
        = Σ_{i∈T} η ⟨ P̂_{i:} − Y_{i:}, H^(k)_{i:} ⟩.
```
Now substitute the over-smoothed `H^(k)_{i:} = π_i β + o_k(1)`. The `π_i` is a positive scalar (every node has a self-loop so `D̃_ii > 0`, hence `π_i > 0`), pull it out:
```
∂L/∂γ_k = Σ_{i∈T} η π_i ⟨ P̂_{i:} − Y_{i:}, β ⟩ + o_k(1).
```
And what is `P̂_{i:}` in this regime? `Z_{i:} = Σ_k γ_k H^(k)_{i:}`, and if the over-smoothing hops dominate `Z` with, say, the dominant `γ_k > 0`, then by definition of over-smoothing `Z_{i:} = c_0 π_i β` for some `c_0 > 0` (a positive scaling of the common direction `β`). So `P̂_{i:} = softmax_η(c_0 π_i β)`. Take `η` large: `softmax_η(v) → 𝟙[v]`, the indicator of the argmax of `v` (I should check this — for `v` with max `v̂`, `softmax_η(v)_j = e^{−η(v̂−v_j)}/Σ_m e^{−η(v̂−v_m)}`; as `η→∞` every non-max term `→ 0` and the `p` maxima each `→ 1/p`, so it converges to the uniform indicator of the argmax set, `softmax_η(v) = 𝟙[v] + o_η(1)`). Since `c_0 π_i > 0`, `argmax_j (c_0 π_i β_j) = argmax_j β_j`, so `P̂_{i:} = 𝟙[β] + o_η(1)` — *the same prediction for every node*, namely the class that maximizes `β`. Then
```
∂L/∂γ_k = Σ_{i∈T} η π_i ⟨ 𝟙[β] − Y_{i:}, β ⟩ + o_k(1) + o_η(1)
        = Σ_{i∈T} η π_i ( max_j β_j − β_{ℓ(i)} ) + o_k(1) + o_η(1),
```
where `ℓ(i)` is node `i`'s true class, so `⟨𝟙[β], β⟩ = max_j β_j` and `⟨Y_{i:}, β⟩ = β_{ℓ(i)}`. Now read the sign. `max_j β_j ≥ β_{ℓ(i)}` always, so this gradient is `≥ 0`. Apart from degenerate ties that make every class share the same collapsed score, a training set containing every class makes the inequality strict: one collapsed argmax cannot match all labels. So when `γ_k > 0` and hop `k` is over-smoothing, gradient descent decreases `γ_k`. Symmetrically, if the dominant over-smoothing `γ_k < 0`, then `Z_{i:} = −c_0 π_i β`, the argmax flips to `argmin_j β_j`, and
```
∂L/∂γ_k = Σ_{i∈T} η π_i ( min_j β_j − β_{ℓ(i)} ) + o_k(1) + o_η(1) ≤ 0,
```
strictly negative under the same nondegenerate all-classes condition, so descent increases `γ_k` toward zero. In the sign regime that matters, `∂L/∂γ_k` has the same sign as `γ_k`: the loss pushes an over-smoothing hop's weight toward `0`. The features that have collapsed to the degree profile are recognized by the loss as unhelpful — predicting one class for everyone cannot be right once the training labels span all classes — and their hops are muted. This is the depth guarantee I needed, and it's *label-driven*: the model escapes over-smoothing because the labels disagree with the smoothed-out prediction. APPNP escapes by a fixed `α`; here the escape is steered by which hops actually help on this graph. So I can set `K` large (say 10) and let the optimizer suppress whatever hops over-smooth.

I should pin down the two lemmas I leaned on, because the whole over-smoothing-escape argument rides on them. The `Â^k H^(0) = π β^T + o_k(1)` claim is just the spectral collapse I derived above: `Â^k → π π^T` because `λ_1 = 1` is the unique unit-magnitude eigenvalue, so `Â^k H^(0) → π π^T H^(0) = π β^T`. The softmax-to-argmax claim I verified inline. The gradient formula `∂L/∂γ_k = Σ_T η π_i ⟨P̂_{i:} − Y_{i:}, β⟩` is the chain rule through the temperatured softmax cross-entropy with `∂Z/∂γ_k = H^(k)` plus the substitution `H^(k) = π β^T`. Nothing exotic; the work is in noticing that the over-smoothed representation makes `P̂` node-independent so the inner product reduces to a max/min gap.

Now, do I constrain `γ`? I considered the nonnegativity-plus-sum-to-one assumption while proving the low-pass theorem, but that was a *hypothesis to characterize what nonnegative weights do*, not a constraint I want — imposing it would lock me into low-pass. So no nonnegativity, no normalization on `γ`; it's a free real `(K+1)`-vector. I do want to think about how to *initialize* it, though, because that's a real degree of freedom and it interacts with how much label information I have. The clean default is uniform, `γ_k = 1/(K+1)` — an equal-weight average over all hops. I should not pretend that this initial filter is neutral in frequency: by the low-pass result I just proved, nonnegative uniform weights with mass beyond hop 0 are low-pass. What uniform does buy is a simple equal-hop starting point instead of a decaying PPR shape or a single SGC spike. Because the parameters are unconstrained, the optimizer can still move away from this low-pass start and cross into alternating signs if the labels demand high-pass structure. There are other sensible inits — a PPR shape `α(1−α)^k` if I want to start near APPNP, a `δ_{kK}` SGC shape, a normalized random vector — and which one I pick matters as an *implicit prior* when labels are scarce (sparse splits), because then the data cannot fully overrule the starting shape. But under a dense split the shape is genuinely learned and the init mostly washes out; that's the whole premise — the `γ_k` are learned, not imposed. For a simple default I'll take uniform.

I want to make sure I'm honest about the cost of this particular parameterization. I'm building the filter as `Σ_k γ_k Â^k` in the *power* basis `{Â, Â^2, …, Â^K}`. Those powers are not orthogonal — as `K` grows the high powers all look increasingly like the same rank-one `π π^T`, so the basis becomes nearly collinear and the coefficient-to-response map gets ill-conditioned. Chebyshev-style bases are better conditioned. So why accept the monomial basis? Because it *is* Generalized PageRank: each `γ_k` is literally the weight on `k`-hop propagation, which makes the learned filter directly readable — plot `γ_k` versus `k` and you can see whether the graph wanted low-pass (positive, decaying or growing), high-pass (alternating signs), or feature-dominated (weight on the first few hops). That interpretability is a feature I care about, and the conditioning cost is tolerable at moderate `K` like 10. It's a deliberate trade, not an oversight.

Let me also be clear about why the GCN normalization with self-loops is the right operator to take powers of, since the whole spectral argument depended on it. The self-loops are what guarantee `|λ_n| < 1` strictly (no bipartite `λ = −1` to make `Â^k` oscillate forever instead of converging), so the over-smoothing limit `π π^T` is well-defined and the low-pass/high-pass dichotomy at `λ = 1` vs `|λ| < 1` is clean. The symmetric normalization `D̃^{-1/2} Ã D̃^{-1/2}` is what makes `Â` symmetric (real spectrum, the spectral story) and keeps its top eigenvalue exactly `1` with the degree-profile eigenvector `π`. So `gcn_norm` it is.

Now let me turn the math into the actual propagation module, and I want it sparse and `O(K · |E|)` — never form `Â^k` as a dense matrix. The structure is: compute `H^(0)`, then iterate `H^(k) = Â H^(k-1)` one sparse step at a time, accumulating `Z = Σ_k γ_k H^(k)` as I go. One sparse `Â·x` is exactly a message-passing step with edge weights `norm` from `gcn_norm` and sum aggregation; the message is `norm · x_j`. So the propagation layer holds the `(K+1)` learnable weights, normalizes the adjacency once, seeds the accumulator with `γ_0 · x`, and folds in `γ_{k+1} · (Â^{k+1} x)` after each propagation step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GPR_prop(MessagePassing):
    """Learnable polynomial filter in the monomial basis.

    Z = sum_{k=0}^{K} gamma_k * Â^k x,   Â = D̃^{-1/2} Ã D̃^{-1/2}.
    The gamma_k are free reals (may be negative) so the response
    g(λ)=sum_k gamma_k λ^k can be low-pass (homophily) or high-pass
    (heterophily), learned end-to-end from the labels.
    """

    def __init__(self, K, alpha=0.1, Gamma=None, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.alpha = alpha                       # kept for compatibility with common training args
        self.Gamma = Gamma
        if Gamma is None:
            temp = torch.ones(K + 1, dtype=torch.float) / (K + 1)
        else:
            temp = torch.as_tensor(Gamma, dtype=torch.float)
        self.temp = Parameter(temp)               # the GPR weights gamma_0..gamma_K

    def reset_parameters(self):
        # Uniform init 1/(K+1): an equal-hop low-pass start. The weights remain
        # unconstrained, so training can move them to alternating signs.
        if self.Gamma is None:
            nn.init.constant_(self.temp, 1.0 / (self.K + 1))
        else:
            gamma = torch.as_tensor(
                self.Gamma, dtype=self.temp.dtype, device=self.temp.device)
            self.temp.data.copy_(gamma)

    def forward(self, x, edge_index, edge_weight=None):
        # symmetric GCN-normalized adjacency Â (self-loops -> |λ_n|<1, λ_1=1)
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        hidden = x * self.temp[0]                # gamma_0 * Â^0 x
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)   # one step: x <- Â x
            hidden = hidden + self.temp[k + 1] * x           # + gamma_{k+1} * Â^{k+1} x
        return hidden                            # Z = sum_k gamma_k Â^k H^(0)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j            # sparse mat-vec (Â @ x)


class GPRGNN(torch.nn.Module):
    """Decoupled model: MLP transform f_theta first, then learnable propagation."""

    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = GPR_prop(args.K, args.alpha, getattr(args, "Gamma", None))
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))                 # H^(0) = f_theta(X), per node, no graph
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)        # propagate with learned gamma_k
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
```

The `temp` vector is the filter; I train it in the same Adam run as the MLP but put the propagation parameters in their own optimizer group with zero weight decay — the `γ_k` are a tiny `(K+1)`-vector that *is* the contribution, and decaying it toward zero would fight the filter I'm trying to learn — while keeping ordinary weight decay on the MLP weights. `K = 10` gives ten hops, matching the depth PPR-style propagation uses, and the over-smoothing-escape argument means the extra hops can be attempted: any that over-smooth are pushed back toward zero by the gradient.

Let me trace the whole causal chain back. I started stuck: neighborhood averaging is right for homophily and ruinous for heterophily, and stacking it to reach far hops over-smooths features into the degree profile `π β^T`. The decoupling idea — transform once with an MLP, then propagate — frees the representation from the over-smoothing trap and showed me, via PPR's geometric-series expansion, that a propagation rule is really a *weighted sum of hop powers* `Σ_k γ_k Â^k`. APPNP and SGC are fixed, nonnegative points in that space, and I proved nonnegative weights are *provably low-pass* (`|g(λ_i)| < 1 = g(λ_1)`), which is exactly why they fail on heterophily; allowing `γ_k = (−α)^k` gives `g(λ) = 1/(1+αλ)`, provably high-pass, so signed coefficients are the heterophily mechanism. Making the `γ_k` free and learned end-to-end turns the propagation into a learnable polynomial graph filter `U g_{γ,K}(Λ) U^T`, the universal family containing the baselines as special cases, with the response shape read off the labels. The depth worry I reopened by un-fixing the weights I close by computing `∂L/∂γ_k` in the over-smoothing regime: it equals `Σ_T η π_i (max_j β_j − β_{ℓ(i)}) ≥ 0` for positive `γ_k` and the `min` analogue `≤ 0` for negative `γ_k`, so gradient descent pushes over-smoothing weights toward zero in the nondegenerate all-classes case — a label-guided escape that lets me try large `K`. Uniform initialization is only an equal-hop low-pass start, not a frequency-neutral promise; the important part is that the weights are unconstrained after initialization. The monomial basis costs some conditioning but buys interpretability (each `γ_k` is the weight on `k`-hop propagation); and the GCN-normalized self-looped `Â` is what makes the spectrum clean (`λ_1 = 1`, `|λ_{others}| < 1`) so all of this holds. The result drops into the MLP-then-propagate pipeline as a `(K+1)`-vector of filter weights and `K` sparse message-passing steps.
