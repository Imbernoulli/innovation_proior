Let me start from what actually goes wrong when I try to propagate a signal across a graph with the filters I have. Everything I use is some polynomial of the normalized Laplacian: I form `sum_{k=0}^K w_k L^k x`, and in the eigenbasis `L = U Lambda U^T` that is the same as applying a scalar response `g(lambda) = sum_k w_k lambda^k` to every frequency, `U diag[g(lambda_i)] U^T x`. The spectrum of `L` lives in `[0, 2]`; near `lambda = 0` is the smooth end, near `lambda = 2` the oscillatory end. So the whole design problem collapses to one curve: what shape do I want `g` to have over `[0, 2]`, and how do I get my coefficients to produce that shape.

And the shapes I need are not all alike. On a homophilic graph — a citation network where neighbors share labels — the useful signal is smooth, so I want a low-pass curve: keep small `lambda`, kill large `lambda`. That is what GCN does, and it works there. But on a heterophilic graph — a web-page graph where linked pages often differ — the discriminative content sits at *high* frequency, and the existing benchmark behavior already warns that a filter pinned to low-pass can lose to a plain MLP that ignores the graph. So the one curve I am designing has to be able to be low-pass on one dataset and high-pass, or band-shaped, on another. I cannot hard-wire the band.

Take stock of what each existing filter gives me, because the gap has to be precise. GCN is `z = P x = (I - L) x`, response `g(lambda) = 1 - lambda`. That is a fixed line sloping down — a low-pass, yes, but look at it past `lambda = 1`: `1 - lambda` goes *negative*. Half my spectrum gets a sign-flipped weight. What does a negative response even mean as a propagation? Hold that thought; it is going to matter. APPNP is cleaner: it propagates with truncated Personalized PageRank, `z = sum_k alpha(1-alpha)^k P^k x`, whose response is the monotone-decreasing `alpha / (alpha + (1-alpha) lambda)`-shaped curve, always positive, but pinned to low-pass — there is no `alpha` that bends it into a high-pass. So the fixed filters are stuck on one shape.

The learned filters loosen that, but at a cost. ChebNet writes `g = sum_k theta_k T_k(tilde L)` in the Chebyshev basis, with `tilde L = 2L/lambda_max - I` rescaled into `[-1, 1]` and the stable recurrence `T_k(y) = 2 y T_{k-1}(y) - T_{k-2}(y)`, and learns the `theta_k` freely. Free coefficients mean it can in principle bend into many shapes. But "free" is exactly the problem in two ways. First, a free `theta_k` can drive `g(lambda)` negative anywhere — nothing stops it. Second, a Chebyshev coefficient is an abstract projection onto `T_k`; if I hand you the learned `theta_k` vector you cannot tell me what response the filter applies at, say, `lambda = 1`. It is opaque.

GPR-GNN makes the opacity sharper and teaches me the key constraint. It learns the filter in the *monomial* basis, `h(P) = sum_k gamma_k P^k`, training the `gamma_k` by gradient descent. Why does it let the `gamma_k` go negative? Because in this basis non-negativity is not neutral. Its low-pass theorem says that if the weights are normalized, non-negative, and nontrivial, the resulting filter is provably low-pass. So to get a high-pass response — to serve heterophily at all — this basis has to allow negative weight somewhere. That is a real theorem and it tells me something I should not forget: in the monomial basis used for PageRank-style propagation, "all coefficients non-negative" collapses the design to low-pass only. If I want the full range of shapes I cannot simply demand non-negative monomial coefficients.

So I have a tension. Expressiveness seems to demand unconstrained (and sometimes negative) coefficients. But unconstrained coefficients are exactly what makes ChebNet and GPR-GNN able to produce a filter that dips negative, and they are exactly what makes the learned filter unreadable. I keep circling the word "negative" — negative responses, negative coefficients — and I should figure out whether a negative *response* is actually a problem or just an aesthetic complaint, because if it is harmless I should stop worrying about it and just learn freely.

Let me chase down what a "valid" filter even is, from first principles, rather than asserting it. The cleanest handle I have on propagation is the energy-minimization view: a lot of propagations are the minimizer of
```
f(z) = (1 - alpha) z^T gamma(L) z + alpha || z - x ||_2^2,
```
a smoothness penalty `gamma(L)` plus a stay-close-to-input term. Set `gamma(L) = L` and minimize — take the gradient `2(1-alpha) L z + 2 alpha (z - x) = 0` and solve — and I get `z* = alpha (alpha I + (1-alpha) L)^{-1} x`. Expand that inverse as a Neumann series in `P = I - L`: `z* = sum_k alpha(1-alpha)^k P^k x`, which is exactly PPR / APPNP. Good, so propagation = solving a regularized smoothing problem, and the response is dictated by the choice of `gamma`. Now I can ask the validity question properly. When is this `f(z)` even a sensible problem? It needs a minimum. If `gamma(L)` is *not* positive semidefinite — if some `gamma(lambda) < 0` — then along the eigenvector for that eigenvalue I can send `z` to a multiple of it and drive `z^T gamma(L) z` to `-infinity`, so `f` has no minimum and my "solution" of `df/dz = 0` is a saddle point, not a minimizer. So the minimum requirement is `gamma(lambda) >= 0` on `[0, 2]`.

Now push that through to the response. With `gamma(L)` PSD the problem is convex, the optimum is
```
z* = alpha (alpha I + (1-alpha) gamma(L))^{-1} x
   = U diag[ alpha / (alpha + (1-alpha) gamma(lambda_i)) ] U^T x,
```
so the response is `h(lambda) = alpha / (alpha + (1-alpha) gamma(lambda))`. Since `gamma(lambda) >= 0`, the denominator is at least `alpha`, so `h(lambda) <= alpha/alpha = 1`; and the denominator is finite and positive, so `h(lambda) > 0`. The response of any valid smoothing of this form lives in `(0, 1]`. *There* is the answer to whether negative responses matter: a polynomial filter that dips below 0 (or above 1) is not approximating any valid smoothing problem at all — it is approximating something whose energy is unbounded below. GCN's `1 - lambda` going negative past `lambda = 1` is precisely this pathology, and the renormalization trick only shrinks the spectrum to soften it — the top eigenvalue of the renormalized `tilde L` still exceeds 1, so the response can still cross zero. The validity constraint is not aesthetics. It is the condition for the filter to *be* a filter.

So I can write the design target sharply now. I want a polynomial `g(lambda) = sum_k w_k lambda^k` with
```
0 <= g(lambda) <= 1   for all lambda in [0, 2].
```
That is the whole specification. Hit it, and every filter I can express is automatically valid; and if I also want full expressiveness, the family of polynomials satisfying this should be rich enough to bend into low-pass, high-pass, band, comb — whatever shape the data wants.

Let me try the obvious thing first and watch it fail, because I want to feel exactly where the wall is. The upper bound `g(lambda) <= 1` is easy: whatever polynomial I have, I can divide all coefficients by `sum_k |w_k| 2^k`, which caps `|g|` at 1 on `[0,2]`; a rescale, no loss of generality. So the real fight is `g(lambda) >= 0` on the whole interval. My first instinct is to copy the simplest constraint I know: keep monomial weights non-negative. But the GPR-GNN theorem is exactly the warning sign. In the PageRank/adjacency monomial basis, non-negative normalized coefficients force a low-pass response, so the constraint that makes coefficients easy also kills the high-pass and band-pass shapes I need. In the Laplacian monomial basis, non-negative coefficients give `g >= 0`, but they do not give the upper bound or a flexible local handle on the curve; they are just powers stacked at one endpoint. Either way, monomial non-negativity is the wrong tool. Wall.

The lesson is that the issue is the *basis*. In the monomial basis, "coefficients non-negative" and "polynomial non-negative on the interval" are wildly different conditions — the first is a crude sufficient condition, not the shape-preserving validity condition I want. What I actually want is a basis in which non-negative coefficients track non-negativity on the interval after choosing a sufficient degree. If such a basis exists, then I can parameterize by non-negative coefficients — cheap to enforce, just clamp them — and keep the hard lower-bound side of validity while still spanning the valid shapes through degree elevation, because non-negative combinations would not be tied to one frequency end.

Does such a basis exist? This is now a pure question about non-negative polynomials on an interval, divorced from graphs. Let me think about what a basis of "bumps" would do. Suppose I had basis functions `b_0, ..., b_K` that are each individually non-negative on a normalized coordinate `t`, and suppose they form a partition of unity, `sum_k b_k(t) = 1`. Then any combination `sum_k theta_k b_k(t)` is bounded between `min_k theta_k` and `max_k theta_k`, because the basis values are non-negative weights summing to one. Non-negative `theta_k` gives the lower bound, and `theta_k <= 1` gives the upper bound; if a learned curve overshoots, the earlier rescaling argument handles that scale. That is a useful property and it tells me the kind of basis I want: non-negative, partition of unity, and ideally each `b_k` concentrated as a bump at a distinct location so the coefficients have local meaning.

The basis that does this is the Bernstein basis. On `[0, 1]`, define
```
b_k^K(t) = C(K, k) (1 - t)^{K-k} t^k,    k = 0, ..., K.
```
Let me verify it has the properties I just wished for, by hand, because I am going to lean on them. Non-negativity: for `t in [0,1]`, every factor `C(K,k) >= 0`, `(1-t)^{K-k} >= 0`, `t^k >= 0`, so `b_k^K(t) >= 0`. Partition of unity: `sum_{k=0}^K C(K,k)(1-t)^{K-k} t^k` is exactly the binomial expansion of `((1-t) + t)^K = 1^K = 1`. So they sum to 1 identically. And the bump: differentiating the log for `0 < k < K` gives `k/t - (K-k)/(1-t) = 0`, hence `t = k/K`; the endpoint cases peak at `0` and `1`. So `b_k^K` is concentrated around `k/K`, and the `K+1` humps tile the interval at `0, 1/K, 2/K, ..., 1`. This is the partition-of-unity-of-bumps I wanted.

But two things have to hold for this to actually solve my problem, and I should not assume them. First, the *converse* I really need: can a polynomial that is non-negative on the interval be put into this basis with non-negative coefficients? Non-negative combination implies non-negative polynomial is trivial (I just showed each `b_k >= 0`). The hard direction is the other way. The classical result says yes after choosing a sufficiently high Bernstein degree: a polynomial positive on the interval has positive Bernstein coefficients after degree elevation, and the non-negative case follows by the limiting version; Powers and Reznick make the degree requirement explicit. So I do not have to give up valid shapes merely because I insist on non-negative coefficients. That kills the fear that constraining coefficients to be non-negative throws away expressiveness the way it did in the monomial basis.

Second, completeness of the *approximation*: as `K -> infinity`, can I hit any target response, not just polynomials? The Bernstein operator `B_K(f)(t) = sum_k f(k/K) b_k^K(t)` converges uniformly to any continuous `f` on `[0,1]` (this is Bernstein's constructive proof of the Weierstrass theorem). So if I just *set* `theta_k = f(k/K)`, I get a polynomial that converges to `f` as `K` grows. That means I can approximate an arbitrary continuous response to any precision by raising `K`; discontinuous impulse ideals are not uniform-convergence targets, but individual Bernstein bumps still give the exact localized polynomial forms I can use as low-, high-, and band-pass approximants. Expressiveness, validity, and now arbitrary-shape approximation, all from the same basis. The pieces are fitting.

Now I have to mount this onto the graph spectrum, which is `[0, 2]`, not `[0, 1]`. Reparameterize: set `t = lambda / 2`, so `t in [0, 1]` as `lambda` ranges over `[0, 2]`. Then `1 - t = (2 - lambda)/2` and `t = lambda/2`, and the basis becomes
```
b_k^K(lambda/2) = C(K, k) (1 - lambda/2)^{K-k} (lambda/2)^k
               = (1 / 2^K) C(K, k) (2 - lambda)^{K-k} lambda^k.
```
The `1/2^K` is just the two halves of the rescaling `(2-lambda)/2` and `lambda/2` collecting their denominators: `(1/2)^{K-k} (1/2)^k = (1/2)^K`. So my filter response is
```
p_K(lambda/2) = sum_{k=0}^K theta_k (1 / 2^K) C(K, k) (2 - lambda)^{K-k} lambda^k,   theta_k >= 0.
```
And — this is the payoff of choosing `t = lambda/2` rather than some other affine map — the bump of `b_k^K` that sat at `t = k/K` now sits at `lambda = 2k/K`. So `theta_k` controls the response near frequency `2k/K`. If I want to pass a band around some `lambda*`, I raise the `theta_k` with `2k/K` near `lambda*`; to reject it I lower that `theta_k`. The coefficient vector *is* the filter, sampled at `K+1` uniform points across `[0, 2]`. That is the interpretability I could never get from Chebyshev or monomial coefficients — and it falls straight out of the bump structure, I did not have to engineer it.

Lift the scalar response to the operator by `lambda -> L`. The convolution becomes
```
z = sum_{k=0}^K theta_k (1 / 2^K) C(K, k) (2I - L)^{K-k} L^k x,   theta_k >= 0.
```
Let me sanity-check that this reproduces filters I trust, because if it cannot even recover the easy ones it is wrong. All-pass: I want `g(lambda) = 1`. Set every `theta_k = 1`; then by the partition of unity `p_K(lambda/2) = sum_k b_k^K(lambda/2) = 1`, so `z = I x = x`. The operator collapses to the identity exactly — not approximately, exactly. Linear low-pass `g(lambda) = 1 - lambda/2`: set `theta_k = 1 - k/K`. Since `theta_k = (1 - k/K)` are the values of the line `1 - t` sampled at `t = k/K`, and the Bernstein operator reproduces linear functions exactly (it preserves affine functions — `B_K` of a line is that same line), I get `p_K(lambda/2) = 1 - lambda/2` exactly, and the operator is `I - L/2 = (I + P)/2`, which is the pre-renormalization GCN operator. Linear high-pass `g = lambda/2`: `theta_k = k/K` gives `L/2`. For impulse-style filters I should be precise: the ideal impulse is discontinuous, so what I get is the exact polynomial operator for a single Bernstein bump. A low-frequency bump sets `theta_0 = 1`, all other `theta_k = 0`; then `p_K = b_0^K = (1 - lambda/2)^K`, and the operator is `(2I - L)^K / 2^K`, a `K`-fold linear low-pass. A high-frequency bump sets `theta_K = 1` and gives `L^K / 2^K`. A middle bump, for even `K`, sets `theta_{K/2} = 1` and gives `C(K, K/2)(2I - L)^{K/2} L^{K/2} / 2^K`, a low-pass and a high-pass stacked into a localized band operator. The exact all-pass and linear filters, plus these exact bump operators, confirm the construction.

I should pin down why this basis and not the alternatives, because that is the real decision. Against the monomial basis: non-negative monomial coefficients in the PageRank basis give low-pass only (the wall I already hit) and the basis is numerically delicate on a wide spectrum. Against Chebyshev: Chebyshev is orthogonal and stable and only `O(K)` to apply, which is genuinely nice, but its coefficients are unconstrained, so the learned filter can be ill-posed and the coefficients carry no per-frequency meaning. The Bernstein choice is a deliberate trade: I will pay more in compute (I will count exactly how much in a moment) to *buy* the two things nobody else has — a cheap non-negative-response constraint, with the `[0,1]` upper bound handled by scale, and coefficients that are literally the sampled filter values. For a method whose whole pitch is "design and learn arbitrary valid, interpretable filters," that is the right trade.

Now, learning. I do not want to set `theta_k = h(2k/K)` by hand in general; I want to learn them from data — from `(x, z)` pairs, or end-to-end from node labels — by gradient descent, exactly as GPR-GNN learns its `gamma_k`. The extra requirement over GPR-GNN is the lower-bound side of validity, `theta_k >= 0`; the upper-bound side is a scale condition from the graph-optimization derivation, handled when I normalize, rescale, or keep the learned coefficients within the intended range. How do I enforce the hard non-negativity cheaply and differentiably in the layer? I keep a free parameter vector `temp` and pass it through `ReLU`: `theta = ReLU(temp)`. ReLU clamps negatives to zero, is differentiable almost everywhere, and lets gradients flow on the active coordinates, so the learned response never goes negative. A softplus would also keep it positive and smoother, but ReLU is the cleaner clamp and matches the "a coefficient is a filter value, and a filter value can be exactly zero (a full rejection)" reading — softplus never reaches zero, so it could never express a hard rejection band, whereas ReLU can drive a `theta_k` to exactly 0.

What should the initial `temp` be? I want the model to start unbiased — not pre-committed to low- or high-pass, so the data decides. The all-pass filter is the unbiased starting point: it passes everything, `z = x`, and imposes no frequency preference. From the all-pass calculation that is `theta_k = 1` for all `k`. So initialize `temp` to all ones; then `ReLU(temp) = 1`, the filter starts as the identity, and gradient descent moves it toward whatever shape the loss prefers. Starting from a low-pass (like PPR initialization) would bias a heterophilic dataset wrongly; the all-pass start is neutral.

Now the implementation, and here I have to be careful about cost. The naive way to evaluate `z = sum_k theta_k (1/2^K) C(K,k) (2I - L)^{K-k} L^k x` is to compute, for each `k`, the matrix `(2I - L)^{K-k} L^k` applied to `x`. If I form each term independently that is a lot of repeated propagation. Let me find the structure that reuses work. Notice the `k`-th term needs `(2I - L)^{K-k} L^k x`. I will build the two operator families separately. First precompute the chain of `(2I - L)` powers applied to `x`: let `tmp[0] = x`, and `tmp[i] = (2I - L) tmp[i-1] = (2I - L)^i x` for `i = 1, ..., K`. That is `K` sparse propagations. The `k = 0` term needs `(2I - L)^K x = tmp[K]` (no `L` factor), weighted by `theta_0 C(K,0)/2^K`. For `k = i+1` (`i = 0, ..., K-1`), the term is `theta_{i+1} C(K,i+1)/2^K * L^{i+1} (2I - L)^{K-(i+1)} x = L^{i+1} (2I - L)^{K-i-1} x`, and `(2I-L)^{K-i-1} x` is just `tmp[K-i-1]`, already computed. So I take `tmp[K-i-1]` and apply `L` to it `i+1` times. That is the second family of propagations.

Let me operationalize the two operators in message-passing terms, because I have to get the edge weights right. `L = I - D^{-1/2} A D^{-1/2}` is the symmetric normalized Laplacian — `get_laplacian(..., normalization='sym')` hands me its edge index and edge weights `norm1`. Applying `L` to `x` is one propagation step with those weights. For `2I - L`: I take the Laplacian's edges with *negated* weights (`-norm1`, that is `-L` off the self-loop) and add self-loops with `fill_value = 2.0`, which puts `+2` on the diagonal — so the resulting operator is `2I - L`, edge weights `norm2`. One propagation with `norm2` applies `2I - L`. With those two operators wired up, the loop is exactly the build I described.

Count the cost: the first loop is `K` propagations for the `tmp` chain. The second loop, for `i = 0, ..., K-1`, applies `L` a total of `i+1` times, so `sum_{i=0}^{K-1} (i+1) = K(K+1)/2` propagations. That is the `O(K^2)` cost — quadratic in the order, where ChebNet and GPR-GNN are linear. I know where it comes from (the inner re-application of `L` per term) and I am accepting it as the price of validity and interpretability. There is a linear-time corner-cutting evaluation for Bernstein polynomials in the geometric-modeling literature, but it evaluates the polynomial at points, and I need to multiply through by the signal `x` term by term, so it does not transfer directly; making this linear is a genuine open question, not a free lunch I am leaving on the table.

Let me write the propagation layer to fill the empty slot in the pipeline.

```python
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, add_self_loops
from scipy.special import comb


class Bern_prop(MessagePassing):
    """Propagation as a non-negative Bernstein-basis polynomial of L:
       z = sum_k theta_k * C(K,k)/2^K * (2I - L)^{K-k} L^k x,  theta_k = ReLU(temp).
       Non-negative coefficients keep the response non-negative, and each theta_k is
       the response near frequency 2k/K under the Bernstein sampling view."""

    def __init__(self, K, bias=True, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))   # free params; theta = ReLU(temp)
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)                     # all-pass start (theta_k=1 => identity)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)                       # enforce theta_k >= 0

        # L = I - D^{-1/2} A D^{-1/2}   (symmetric normalized Laplacian)
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim))
        # 2I - L : negate L's off-diagonal, put +2 on the diagonal
        edge_index2, norm2 = add_self_loops(
            edge_index1, -norm1, fill_value=2.0, num_nodes=x.size(self.node_dim))

        # tmp[i] = (2I - L)^i x  for i = 0..K   (K propagations, reused below)
        tmp = [x]
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        # k = 0 term: theta_0 * C(K,0)/2^K * (2I - L)^K x
        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        # k = i+1 terms: theta_{i+1} * C(K,i+1)/2^K * L^{i+1} (2I - L)^{K-i-1} x
        for i in range(self.K):
            x = tmp[self.K - i - 1]                    # (2I - L)^{K-i-1} x  (cached)
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)   # apply L once
            for j in range(i):                         # apply L i more times => L^{i+1}
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j                  # one sparse mat-vec: operator @ x
```

And the full model just drops this propagation into the existing transform-then-propagate pipeline: an MLP transforms the features, dropout, then the Bernstein propagation, then `log_softmax`. The propagation gets its own dropout rate `dprate` because — as APPNP and GPR-GNN already found — the feature transform and the propagation want different regularization and learning rates; the filter coefficients are a handful of numbers and over-regularizing them with the dense-layer dropout would smear the very bumps I am trying to learn.

```python
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class BernNet(nn.Module):
    """MLP feature transform, then Bernstein-polynomial propagation."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = Bern_prop(K)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
```

Let me retrace the chain so I am sure it closes. I started stuck: every graph filter is a polynomial response `g(lambda)` on `[0,2]`, and I need one that can be low-pass *or* high-pass *or* band-shaped depending on the data, while the filters I have are either fixed-low-pass (GCN, APPNP) or expressive-but-ill-posed-and-opaque (ChebNet, GPR-GNN). Chasing "what is a valid filter" through the energy-minimization view forced the condition: for the smoothing to have a minimum, `gamma(L)` must be PSD, and then the response is squeezed into `[0,1]` — so a valid polynomial target has `0 <= g(lambda) <= 1` on `[0,2]`, and a negative response is the signature of an unbounded-below energy, not a harmless quirk. The upper bound is a scale condition; the lower bound `g >= 0` is the fight. Forcing monomial coefficients non-negative fails — that is GPR-GNN's low-pass-only theorem in the PageRank basis — so the problem is the basis. The Bernstein basis gives what I need: each `b_k^K` is non-negative, the set forms a partition of unity, sufficiently high Bernstein degree represents non-negative interval polynomials with non-negative coefficients, and `B_K(f)` converges to any continuous target as `K` grows. Mapping `[0,2]` to `[0,1]` by `t = lambda/2` lands the `k`-th bump at frequency `2k/K`, which makes each coefficient the sampled filter value — the interpretability the others lack. Lifting to the operator gives `z = sum_k theta_k (1/2^K) C(K,k)(2I-L)^{K-k} L^k x`, with all-pass and linear low/high-pass exact and impulse-style low/high/band operators obtained by putting all coefficient mass on the corresponding bump. Learn the coefficients by gradient descent like GPR-GNN, keep them non-negative with `ReLU`, start from all-ones (all-pass, unbiased), and evaluate with the cached `(2I-L)`-chain plus per-term `L`-powers — `K + K(K+1)/2` propagations, the `O(K^2)` price paid for readable coefficients — dropped into the same MLP-then-propagate pipeline as the baselines.
