Let me start from the thing that actually bothers me, which is the tension in the numbers I keep seeing. A graph filter is a scalar function `h` of frequency applied through the Laplacian, `y = U h(Λ) U^T x`, and since I can't afford the eigendecomposition I write `h` as a degree-`K` polynomial and evaluate `sum_k w_k L^k x` with sparse mat-vecs. The question of *which* polynomial — which basis, which coefficients — is the whole game, because the filter's shape decides everything downstream. I can convince myself of that with a tiny picture: take a ring graph and a one-hot signal. If I apply an impulse low-pass response, picking out only `λ=0`, the output is the constant eigenvector, the signal smears evenly over the ring, and that's exactly the separation I want when every node shares a label — homophily. If instead I apply an impulse high-pass, picking out `λ=2`, I get the most oscillatory eigenvector, alternating signs node to node, which is exactly right when adjacent nodes have opposite labels — heterophily. Band-pass for a mixture. So the homophilic and heterophilic regimes don't just want *more or less* propagation; they want *opposite* frequency responses, and a method that can only do one of them is stuck on half the graphs. I need a filter I can *learn* into any of these shapes.

Now the first puzzle. The original learnable spectral net, ChebNet, builds exactly this kind of polynomial in the Chebyshev basis, `y ≈ sum_k w_k T_k(L_hat) x` with `L_hat = 2L/λ_max - I` to fold the spectrum into `[-1,1]` where the Chebyshev polynomials live, and in theory as `K` grows it represents arbitrary filters. But on the citation benchmarks it loses to GCN — and GCN is literally *ChebNet truncated to first order* with `K=1`, `w_0=-w_1`. The more expressive model loses to its own special case. And it gets *worse* as I raise `K`: Cora around 80.5 at `K=2` but 74.9 at `K=10`, while GCN sits at 81.3. More capacity, more flexibility, worse accuracy. That's the first thing I have to explain.

The second thing is sharper. Two newer methods, GPR-GNN and BernNet, take the same decoupled idea — run an MLP `f_θ(X)` first, then propagate with a learnable polynomial, `Y = sum_k w_k B_k(L) f_θ(X)` — and just change the *basis*. GPR-GNN uses the monomial basis `B_k = P̃^k`, so `h(λ̃) = sum_k γ_k λ̃^k`. BernNet uses the Bernstein basis, `h(λ) = sum_k θ_k (1/2^K) C(K,k)(2-λ)^{K-k} λ^k`. Both beat ChebNet handily. So out of curiosity I drop the *Chebyshev* basis into that same harness — call it ChebBase, `Y = sum_k w_k T_k(L_hat) f_θ(X)`, coefficients learned freely — and it comes in *last*: Cora 79.3, against GPR-GNN's 84.0 and BernNet's 83.2. That is genuinely strange, because in approximation theory the Chebyshev basis is the *good* one. The truncated Chebyshev expansion is near-minimax: its worst-case uniform error is within a small factor of the best possible degree-`K` polynomial. Monomials, by contrast, are a numerically terrible basis — the powers `λ^k` become nearly collinear as `k` grows, the conditioning is awful, that's the whole reason numerical analysts switched to orthogonal polynomials a century ago. So how is the near-minimax basis the worst performer and the ill-conditioned monomial basis near the top? Either approximation theory is lying to me or I'm using Chebyshev wrong. I'll bet on the latter.

Let me actually think about what "learning `w_k` freely by gradient descent" does, because both ChebNet and ChebBase do exactly that — fit the Chebyshev coefficients with no constraint. What kind of filter does an unconstrained set of `w_k` describe? Any continuous `f` on `[-1,1]` has a Chebyshev expansion `f(x) = sum_k w_k T_k(x)`, and the `T_k` with large `k` are the high-frequency modes — `T_k(x) = cos(k arccos x)`, literally a cosine of increasing frequency. So the large-`k` coefficients control how much high-frequency wiggle the filter has. Now here is the fact I keep underusing: there's a constraint on those coefficients that *any well-behaved filter must obey*. If `f` is analytic on the open interval — locally a convergent power series, the kind of smooth response a sensible filter should be — and at worst weakly singular at the endpoints, then its Chebyshev coefficients must decay, asymptotically like `1/k^q` for some positive `q` as `k → ∞`. The high-frequency coefficients of a smooth filter are *forced* to die off. I can sanity-check this against a real analytic filter: `exp(λ_hat)`, the response diffusion methods use; its Chebyshev coefficients are visibly convergent, plotted out they march toward zero.

So now flip it. ChebNet and ChebBase learn `w_k` with *nothing* enforcing this decay. Gradient descent, chasing training accuracy, is free to pile weight onto the large-`k` coefficients — to fit a response whose high-frequency coefficients *don't* decay. But a function with non-decaying high-`k` Chebyshev coefficients is, by the contrapositive of that theorem, *not* analytic — it's a wild, oscillatory, hard-to-approximate response. That's a textbook overfitting machine: enormous capacity in exactly the high-frequency modes, used to memorize the training labels through a jagged filter that generalizes terribly. And it explains *both* puzzles at once. Against GCN: GCN's first-order filter is structurally smooth, it physically *cannot* express high-frequency garbage, so on these mostly-homophilic citation graphs — where the true filter really is a smooth low-pass — GCN's inflexibility is a *feature*, while ChebNet at `K=10` has ten extra high-frequency knobs it overfits with, which is why the gap widens with `K`. And against the other bases: ChebBase isn't losing because Chebyshev is a bad approximator, it's losing because *unconstrained* Chebyshev coefficients describe an illegal, non-analytic filter. The theory was never wrong; I was handing the basis to an optimizer with no leash.

Let me test that diagnosis directly instead of just asserting it. If the disease is non-decaying coefficients, the cure is to *force* decay, and the cheapest way is to bake a `1/k` factor into the reparameterization: learn `w_k` but propagate with `Y = sum_k (w_k/k) T_k(L_hat) f_θ(X)`, with the convention `w_0/0 = w_0` for the constant term. That literally divides every coefficient down by its frequency index, pushing the spectrum toward the `1/k^q` shape the theorem demands. Call it ChebBase/k. And it works — it jumps past ChebNet, past ChebBase, *and* past GCN (Cora about 82.7 versus ChebBase's 79.3 and GCN's 81.3), and when I look at the coefficients it learns, they decay much more readily than the unconstrained ones. So the diagnosis holds: the problem really was the missing coefficient constraint, and even a crude penalty fixes it.

But I don't want to stop here, because ChebBase/k is a hack and I can feel two things wrong with it. First, it's not honest mathematically: the decay theorem is a *necessary* condition on the coefficients of an analytic filter, not a sufficient one. Multiplying by `1/k` enforces a *particular* decay rate that happens to be in the legal family, but there's nothing principled about `1/k` specifically — why not `1/k^2`, why not some other profile? I'm enforcing a symptom of legality, not legality itself. Second, and this is the one that really blocks me: I can't extend it. Suppose I want a *non-negative* filter — a response that stays `≥ 0` across the whole spectrum, which BernNet argues some valid filters require. With the form `sum_k (w_k/k) T_k(L_hat)`, how do I constrain the *function* `sum_k (w_k/k) T_k(λ̂)` to be non-negative everywhere on `[-1,1]`? The `w_k` are abstract expansion coefficients; there's no transparent relationship between a sign or magnitude of `w_k` and the *value* of the filter at any frequency. Constraining the coefficients to keep the function positive is a hard global condition, not a box constraint I can ReLU. So the penalty route is a dead end for anything beyond the one decay constraint. I need a parameterization where the *constraints I care about live directly on the parameters*.

That last sentence is the thing to chase. What would make filter-value constraints turn into parameter constraints? If my trainable numbers *were* the filter's values themselves, then "the filter is non-negative" would just be "these numbers are non-negative" — a ReLU and I'm done. So instead of parameterizing the *coefficients* of the polynomial, parameterize the polynomial by the *values* it takes at some fixed set of points, and let the coefficients be *derived* from those values. That's not a basis change; that's switching from expansion to interpolation. Given a continuous filter `h` on `[-1,1]` and its values at `K+1` distinct points `λ̂_0 < ... < λ̂_K`, there's a unique degree-`≤K` polynomial `P_K` with `P_K(λ̂_k) = h(λ̂_k)`. I can write the coefficients explicitly — they solve the Vandermonde system

```
[1  λ̂_0  λ̂_0^2  ...  λ̂_0^K]   [a_0]   [h(λ̂_0)]
[1  λ̂_1  λ̂_1^2  ...  λ̂_1^K] · [a_1] = [h(λ̂_1)]
[          ...           ]   [...]   [  ...  ]
[1  λ̂_K  λ̂_K^2  ...  λ̂_K^K]   [a_K]   [h(λ̂_K)]
```

— or equivalently in Lagrange form `P_K(λ̂) = sum_k h(λ̂_k) L_k(λ̂)` with the cardinal functions `L_k(λ̂) = prod_{j≠k}(λ̂ - λ̂_j) / prod_{j≠k}(λ̂_k - λ̂_j)`. The point isn't the closed form; the point is conceptual: now the free parameters *are* `h(λ̂_k)`, the filter's values, and a constraint on the response at the sample points is a constraint on the parameters directly. This changes where the coefficient constraint lives. If the sampled values come from a smooth response, the recovered coefficient vector is the Chebyshev-interpolation coefficient vector of that response, not an arbitrary high-order vector chosen directly by gradient descent; and the constraint issue becomes local and explicit because `γ_j ≥ 0` is exactly a non-negative response at interpolation node `j`.

Except — interpolation has its own famous trap, and if I walk into it I've traded one disease for a worse one. The trap is that more points do not reliably mean a better fit. If I sample at *equispaced* points and crank `K` up on a Runge-type target, the interpolant oscillates harder and harder near the ends of the interval, and the high-degree error can diverge. Let me see exactly where it comes from so I know what to do about it. The interpolation error at any point is

```
R_K(λ̂) = h(λ̂) - P_K(λ̂) = h^{(K+1)}(ζ) / (K+1)! · π_{K+1}(λ̂),
```

where `ζ` depends on `λ̂` and `π_{K+1}(λ̂) = prod_{k=0}^K (λ̂ - λ̂_k)` is the *nodal polynomial* — the monic degree-`K+1` polynomial whose roots are precisely my chosen interpolation points. I can't touch the `h^{(K+1)}/(K+1)!` factor, that's the function's business. But the nodal polynomial `π_{K+1}` is *entirely mine* — it's determined by where I put the points. And with equispaced points `π_{K+1}` is small in the middle of the interval and swings enormously near `±1`, so the error near the boundaries explodes. So the whole Runge problem reduces to a clean question I actually control: *where do I place `K+1` points so that the nodal polynomial `prod_k(λ̂ - λ̂_k)` stays as small as possible, uniformly, across `[-1,1]`?*

That's a minimization over the placement of `K+1` points of `max_{[-1,1]} |prod_k(x - x_k)|`, the uniform norm of a monic degree-`K+1` polynomial. And I know the answer to *that* — it's the classical extremal property of Chebyshev polynomials. Among all monic polynomials of degree `K+1` on `[-1,1]`, the one with the smallest possible maximum absolute value is the scaled Chebyshev polynomial `2^{-K} T_{K+1}(x)`, and its uniform norm is exactly `2^{-K}`. So if I choose my nodal polynomial to *be* that minimizer — i.e. choose my interpolation points to be the *roots of `T_{K+1}`* — then `||π_{K+1}|| = 2^{-K}`, the smallest it can possibly be, and it shrinks geometrically as `K` grows. The nodal-polynomial source of the Runge blowup is minimized and made geometrically small. The roots of `T_{K+1}(x) = cos((K+1) arccos x)` are where the cosine is zero, `(K+1) arccos x_j = (j + 1/2)π`, giving the **Chebyshev nodes**

```
x_j = cos((j + 1/2) π / (K + 1)),   j = 0, 1, ..., K.
```

These cluster near the endpoints `±1` — denser exactly where equispaced points let the error run wild — which is the geometric reason they tame it. That tells me to interpolate at the Chebyshev nodes. Now I want to be sure this also delivers the *approximation* quality I was promised, not just a smaller nodal polynomial. The relevant measure is the Lebesgue constant `ρ`, defined by `||f - P_K|| ≤ (1+ρ)||f - P_best||`: it's how much worse interpolation can be than the best possible degree-`K` polynomial. For *equispaced* interpolation `ρ ~ 2^K` — exponential, catastrophic, the formal statement of Runge. For *Chebyshev-node* interpolation `ρ ~ log(K)` — it grows only logarithmically, so a Chebyshev interpolant is always within a `~log K` factor of the minimax polynomial. That's the near-minimax guarantee, and it confirms the basis was never the problem: it's interpolation at the *right nodes* that's near-best.

Now let me get the coefficients of the Chebyshev-node interpolant in closed form, because I'll need them in the model and the structure is going to be the payoff. I want to write `P_K(λ̂) = (c_0/2)T_0(λ̂) + sum_{k=1}^K c_k T_k(λ̂)` and solve for the `c_k` from the values `h(x_j)` at the Chebyshev nodes. The reason there's a clean formula is a discrete orthogonality among the Chebyshev polynomials evaluated at their own nodes. Write the nodes as `x_j = cos θ_j` with `θ_j = (j + 1/2)π/(K+1)`, so `T_k(x_j) = cos(k θ_j)`. Consider the sum over the `K+1` nodes `sum_{j=0}^K T_m(x_j) T_l(x_j) = sum_j cos(m θ_j) cos(l θ_j)`. Using `cos a cos b = (cos(a-b) + cos(a+b))/2`, this is `(1/2) sum_j cos((m-l)θ_j) + (1/2) sum_j cos((m+l)θ_j)`. The half-integer offset in `θ_j` is exactly what makes these sums collapse: `sum_{j=0}^K cos(r(j+1/2)π/(K+1))` is `K+1` when `r=0`, and `0` for any integer `r` with `0 < |r| < 2(K+1)` — it's a shifted geometric series of unit roots that cancels. So for `m, l` in `0..K`, the cross term `m+l` (which ranges up to `2K < 2(K+1)`) vanishes unless `m=l=0`, and the difference term `m-l` gives `K+1` only when `m=l`. Putting it together: the sum is `0` for `m≠l`, `(K+1)/2` for `m=l≠0`, and `K+1` for `m=l=0`. There's the discrete orthogonality, and the `m=l=0` case being *twice* the others is exactly why I wrote the expansion with `c_0/2` from the start.

Armed with that, invert. Plug the interpolation conditions `P_K(x_j) = h(x_j)` into the halved-constant expansion, multiply both sides by `T_l(x_j)` and sum over the nodes `j`:

```
sum_{j=0}^K h(x_j) T_l(x_j)
  = (c_0/2) sum_j T_0(x_j)T_l(x_j)
    + sum_{k=1}^K c_k [ sum_j T_k(x_j)T_l(x_j) ].
```

By the discrete orthogonality, if `l=0` only the halved constant term survives and gives `(K+1)c_0/2`; if `l>0`, only the matching `k=l` term survives and gives `(K+1)c_l/2`. So the same normalization works for every coefficient:

```
c_k = (2/(K+1)) sum_{j=0}^K h(x_j) T_k(x_j),   k = 0, ..., K.
```

The `k=0` term is *halved* when I evaluate the polynomial — `P_K(λ̂) = (c_0/2) T_0(λ̂) + sum_{k=1}^K c_k T_k(λ̂)` — exactly compensating that `T_0` had double the discrete norm. (This is precisely the discrete cosine transform of the sampled values, and the `c_0/2` halving is the standard DCT convention; it's not a fudge, it's the inverse of the orthogonality relation I just derived.) So the recipe is: sample the filter at the `K+1` Chebyshev nodes, take this near-DCT of the values, and those are the Chebyshev coefficients of the unique near-minimax interpolant.

I don't *know* the filter `h` — I want to *learn* it. So I treat the filter values at the Chebyshev nodes as the trainable parameters: let `γ_j` stand for `h(x_j)`, `j=0..K`, free parameters. Then the interpolation coefficients are `c_k = (2/(K+1)) sum_{j=0}^K γ_j T_k(x_j)`, and the filtering operation is

```
y ≈ (c_0/2)T_0(L_hat)x + sum_{k=1}^K c_k T_k(L_hat)x
  = (2/(K+1)) [ (1/2)sum_{j=0}^K γ_j T_0(x_j)T_0(L_hat)x
      + sum_{k=1}^K sum_{j=0}^K γ_j T_k(x_j)T_k(L_hat)x ],
```

and the same expression drops into the decoupled harness by replacing `x` with `f_θ(X)`. Compared to ChebNet's
`sum_k w_k T_k(L_hat)` this is *one* change — I reparameterize the coefficient as
`c_k = (2/(K+1)) sum_j γ_j T_k(x_j)` instead of learning `w_k` directly, and then use `c_0/2` for the constant term. Because `γ_j` *is* the filter value at interpolation node `x_j`, the learned response is the Chebyshev-node interpolant of sampled values, so the coefficient vector is tied to a stable interpolation problem instead of being a free high-order vector. If the sampled values describe a smooth response, the coefficient decay comes from that response rather than from a heuristic `1/k` penalty. And the side constraints I couldn't impose before now become direct constraints at the control nodes: `γ_j ≥ 0` enforces non-negativity at every Chebyshev node. That is not a Bernstein-style certificate of positivity between interpolation nodes, but it is the simple parameter-level constraint the implementation can actually optimize. The abstract-coefficient wall is gone; the parameters are filter values, and constraints on the sampled filter are box constraints on the parameters.

Let me settle the rescaling, because Chebyshev polynomials only live on `[-1,1]` and the normalized-Laplacian spectrum is contained in `[0,2]`. ChebNet wrote `L_hat = 2L/λ_max - I`, which needs `λ_max`, an extra eigen-computation. But I already know `λ_max ≤ 2` for the normalized Laplacian a priori, and the safe, computation-free choice is to just *use* `λ_max = 2`. Then `L_hat = 2L/2 - I = L - I`. That maps the guaranteed ambient range `[0,2]` to `[-1,1]` exactly, while the actual graph spectrum may occupy only a subset, and it costs nothing to compute. Concretely: build `L = I - D^{-1/2} A D^{-1/2}` once, then shift to `L_hat = L - I` by adding self-loops with weight `-1`.

Then the propagation is the Chebyshev three-term recurrence on the operator, same as ChebNet, which is the numerically stable way to apply `T_k(L_hat)` without ever forming a power explicitly. `T_0(L_hat)x = x`; `T_1(L_hat)x = L_hat x`, one sparse mat-vec; and `T_{k}(L_hat)x = 2 L_hat T_{k-1}(L_hat)x - T_{k-2}(L_hat)x`. So I keep two running vectors, push each through one propagation step, accumulate `out += c_k · T_k(L_hat)x`, with `out` started at `(c_0/2) T_0 + c_1 T_1` to honor the halved constant term.

Before I write it I want to know what this costs and whether it's actually better than the alternatives I'm reacting to, not just different. The forward pass has two parts. Forming the `K+1` coefficients `c_k = (2/(K+1)) sum_j γ_j T_k(x_j)` is a `(K+1)×(K+1)` matrix of scalar Chebyshev values `T_k(x_j)` times the parameter vector — `O(K^2)` scalar work, and the `T_k(x_j)` are fixed because the nodes do not move, so they can be precomputed even if the simple implementation recomputes them inside `forward`. The propagation is `K` sparse mat-vecs of a graph with `m` edges over `d` hidden channels, `O(K m d)`. Total `O(K^2 + K m d)`, and since `K` is small while the graph mat-vecs dominate, the real scaling is `O(K m d)` — *linear* in `K`. Compare BernNet: its Bernstein form `(2I-L)^{K-k} L^k` re-walks the graph for every term, costing `O(K^2 m d)`, *quadratic* in `K`, with no linear-time fix on offer. So I get the near-minimax basis *and* I'm as cheap as the monomial GPR-GNN and the original ChebNet, while strictly cheaper than BernNet. That matters: the difference between linear and quadratic in `K` is the difference between running on a billion-edge graph and not.

And I should pin down the convergence claim against BernNet, since "near-minimax" and "linear in K" would be hollow if Bernstein actually approximated better per degree. For a function with modulus of continuity `ω`, the Bernstein approximation error behaves like `E(K) ~ (1 + (2K)^{-2}) ω(K^{-1/2})` — the resolution it buys grows like `K^{-1/2}`. The Chebyshev interpolation error behaves like `E(K) ~ C ω(K^{-1}) log(K)` — resolution `K^{-1}`, only a `log K` penalty. Since `ω(K^{-1})` is smaller than `ω(K^{-1/2})` for any nontrivial modulus, Chebyshev interpolation converges *faster*: for the same fidelity it needs a smaller `K`. So it's both cheaper per `K` and needs fewer of them. The picture is fully consistent now — Chebyshev was always the right basis; the prior work just used it without the right node placement and without tying the parameters to filter values.

One initialization choice to reason through. The parameters are filter *values* `γ_j = h(x_j)`, so what's the neutral starting filter? If I set every `γ_j = 1`, the interpolated filter is the constant `h ≡ 1`, which is the *all-pass* / identity response — it passes every frequency unchanged and imposes no a-priori low- or high-pass bias, leaving gradient descent free to discover whichever shape the graph needs. That's the right default precisely because I don't want to bake in the homophily assumption that crippled GCN on heterophilic graphs; I want to start neutral and let the labels pull the filter low- or high-pass. So initialize `temp` (the `γ`) to all ones. If I wanted to seed a specific shape — say a particular bump — I'd init `γ_j` to that filter's values at the interpolation nodes, and the implementation keeps an optional `x_j^2` initialization path for that case. And the ReLU on `γ` before forming coefficients enforces the non-negative interpolation-node-value constraint while I'm at it — the constraint that was impossible to impose directly in the coefficient parameterization becomes an ordinary box constraint at the interpolation nodes.

The harness around the propagation is the decoupled APPNP-style one the other learnable filters use, and for good reason: separating the MLP transform from the propagation keeps the parameter count tied to the feature dimension and the hidden width, *independent* of `K` — which is why these decoupled models sit around 92k parameters on Cora while ChebNet at `K=10`, carrying a full weight matrix `W_k` per order, balloons to 230k. ChebNet entangled the filter coefficients inside per-order weight matrices, which both blows up the parameters and hides the filter; here the filter is a single scalar function with `K+1` parameters, applied identically to every channel, and the MLP does all the feature mixing up front. So: dropout, `lin1`, ReLU, dropout, `lin2` to get the transformed signal, an optional propagation-specific dropout (`dprate`) on that signal, then the Chebyshev-interpolation propagation, then log-softmax.

Let me write the propagation layer, filling the empty `CustomProp` slot. The two nested loops over `i` (the polynomial order in the output) and `j` (the Chebyshev node) compute `c_i = (2/(K+1)) sum_j relu(γ_j) T_i(x_j)`; the node enumeration `x_j = cos((K-j+0.5)π/(K+1))` runs over the same `K+1` Chebyshev nodes as `cos((j+0.5)π/(K+1))`, just in reverse order — the sum over all nodes doesn't care about the order. Then build `L`, shift to `L_hat = L - I`, and run the three-term recurrence, accumulating `out` with the halved constant term:

```python
import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from utils import cheby  # cheby(i, x) = T_i(x), the three-term recurrence at a scalar node


class CustomProp(MessagePassing):
    """Chebyshev-interpolation graph filter.

    The K+1 learnable parameters `temp` are the filter values gamma_j = h(x_j) at the
    Chebyshev nodes x_j (zeros of T_{K+1}). They are converted to coefficients
    c_k = (2/(K+1)) sum_j relu(gamma_j) T_k(x_j) -- the discrete cosine transform of the
    sampled values. ReLU enforces non-negative sampled values. The applied filter is
    h(L_hat) = c_0/2 T_0(L_hat) + sum_{k=1}^K c_k T_k(L_hat), with L_hat = L - I.
    """

    def __init__(self, K, alpha=0.1, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.K = K
        self.temp = Parameter(torch.Tensor(K + 1))   # gamma_j = filter value at node x_j
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)                    # all-ones -> constant (all-pass) start

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)                  # gamma_j >= 0 at the interpolation nodes
        coe = coe_tmp.clone()

        # c_i = (2/(K+1)) sum_j gamma_j T_i(x_j)  -- DCT of the sampled filter values.
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))   # Chebyshev node
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # L = I - D^{-1/2} A D^{-1/2}
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight, normalization="sym",
            dtype=x.dtype, num_nodes=x.size(self.node_dim))
        # L_hat = L - I  (spectrum [0,2] -> [-1,1] using lambda_max = 2, no eigs needed)
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0, num_nodes=x.size(self.node_dim))

        # Chebyshev three-term recurrence on the operator: T_0=x, T_1=L_hat x,
        # T_k = 2 L_hat T_{k-1} - T_{k-2}; halve the constant term (c_0/2).
        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1
        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CustomFilter(torch.nn.Module):
    """Decoupled harness: MLP transform first, then the Chebyshev-interpolation filter."""

    def __init__(self, num_features, num_classes, hidden=64, K=10,
                 alpha=0.1, dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(num_features, hidden)
        self.lin2 = torch.nn.Linear(hidden, num_classes)
        self.prop = CustomProp(K)
        self.dropout = dropout
        self.dprate = dprate

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))                      # MLP transform (decoupled from propagation)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate == 0.0:
            x = self.prop(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)   # propagation-specific dropout
            x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
```

So the causal chain closes. I started with one puzzle — the near-minimax Chebyshev basis losing to ill-conditioned monomials, and the more-expressive ChebNet losing to its own first-order GCN special case, both getting worse as `K` rises. Reading the analytic-filter theorem told me a well-behaved filter's high-frequency Chebyshev coefficients must decay like `1/k^q`, so the failure is overfitting through *unconstrained* coefficients that learn a non-analytic, high-frequency-heavy response; a crude `1/k` penalty (ChebBase/k) confirmed the diagnosis by beating everything, but it was a non-extensible hack that couldn't impose constraints like non-negativity. Wanting constraints on the filter to become constraints on the parameters pushed me from expansion to *interpolation* — parameterize the polynomial by its values, not its coefficients — which dragged in the Runge phenomenon, which I traced to the nodal polynomial, which I minimized by placing the points at the roots of `T_{K+1}`, the Chebyshev nodes (nodal norm `2^{-K}`, Lebesgue constant `~log K` instead of `~2^K`, so near-minimax and resistant to the equispaced Runge failure). Inverting the discrete orthogonality of `T_k` at those nodes gave the coefficient transform `c_k = (2/(K+1)) sum_j γ_j T_k(x_j)` with the constant term halved; letting the interpolation-node values `γ_j` be the trainable parameters ties the coefficients to a stable Chebyshev-node interpolant, induces the expected decay when the sampled response is smooth, and turns constraints on sampled values (`γ_j ≥ 0` through ReLU) into box constraints. Rescaling with the a-priori `λ_max = 2` gives the free shift `L_hat = L - I`; the Chebyshev recurrence applies the filter in `O(Kmd)` — linear in `K`, faster than BernNet's quadratic and converging faster per degree — and it all drops into the decoupled MLP-then-propagate harness as a `K+1`-parameter filter that starts at the neutral all-pass identity.
