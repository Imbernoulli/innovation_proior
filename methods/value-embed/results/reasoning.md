Let me start from the thing that actually bothers me. I keep stacking more layers on these decoder-only Transformers because the scaling laws tell me depth buys capability, and yet past some point the deeper model is not better, sometimes it is worse — a 32-layer vision Transformer losing to a 24-layer one, language gains flattening as I add blocks. That is strange, because depth is supposed to be nearly free: every block has a residual connection, `H_n = H_{n-1} + f(H_{n-1})`, so the original token embedding `H_0` is always sitting one identity path away from every layer. ResNet taught me that identity shortcuts fix the gradient flow and let arbitrarily deep nets train; DenseNet pushed it further, letting every layer read all the earlier features. So in principle the initial information is preserved. But "preserved in the residual stream" and "usable by layer 24" are not the same claim, and I should pull them apart.

Here is the issue with the residual stream as a carrier. It is *one* stream, and *every* block writes into it by addition. `H_0` is in there, but so is the sum of twenty-three blocks' worth of updates layered on top. The raw token content is not gone, but it is buried under everything written since. And there is a specific reason to think the deep layers genuinely cannot recover it: attention is a smoothing operation. Each attention layer replaces a token's representation with a convex combination of all tokens' value vectors — the softmax row sums to one — which is exactly an averaging, a low-pass filter over the sequence. Iterate an averaging operation through depth and the representations should drift toward each other. That is the over-smoothing story as a slogan; before I trust it enough to build on, I want to watch it happen on something concrete, because "averaging contracts" is the kind of claim that is either obviously true or hiding a counterexample.

So let me just run it. Take three tokens in two dimensions, deliberately spread apart, and a single fixed row-stochastic "attention" matrix `A` (I row-normalize a positive affinity so each row sums to one), and apply `x ← A x` repeatedly. To measure how spread out the tokens are I use the sum of pairwise squared distances — which, I will note in a moment, is exactly the functional I am about to write down. Starting from rows `(3,0), (0,3), (-3,1)`, the spread reads:

```
step 0:  68.00
step 1:  10.77
step 2:   1.90
step 3:   0.345
step 4:   0.063
step 5:   0.012
...
limit:   rows -> (0.008, 1.408), (0.000, 1.409), (-0.008, 1.410)
```

It collapses roughly an order of magnitude per step and the three rows converge to a single shared vector — every token equal. So a single fixed averaging step really is a contraction toward uniformity, and iterating it drives the spread to zero. That settles that the mechanism is real and not a slogan. But it also tells me the over-smoothing is not an artifact of bad weights I could train away: the fixed point of repeated averaging is "all tokens identical," full stop. I want to understand *why* that is the fixed point sharply enough to know where to intervene, and for that the slogan is not enough — I need the objective the operation is descending.

There is a clean way to get it. Treat the sequence of token vectors as a function `u` sampled on a grid, and ask what objective a single self-attention update is descending. Write the nonlocal smoothing functional

  J(u) = (1/2) ∬ ||u(x) - u(y)||² k(x, y) dx dy,

where `k(x, y)` is a nonnegative affinity between positions. This penalizes any *difference* between token representations, weighted by their affinity — minimizing it means making tokens agree, and on my three-token grid it is exactly the pairwise-distance spread I just watched fall to zero. Take its first variation. Perturb `u_j -> u_j + τ h_j` and differentiate at `τ = 0`:

  (d/dτ) J|_{τ=0} = ∬ (u_j(x) - u_j(y))(h_j(x) - h_j(y)) k(x, y) dx dy.

Split the two terms and apply the change of variables `(x, y) -> (y, x)` to the second; the cross terms combine, and the functional derivative comes out

  ∂J/∂u_j(x) = ∫ (u_j(x) - u_j(y)) (k(x, y) + k(y, x)) dy.

Before I lean on this, I should check the algebra rather than trust it — the symmetrization step where the second term folds into the first is exactly the kind of move I get wrong by a factor of two. So I pin it down numerically: take four tokens in three dimensions, a *random, non-symmetric* affinity `k`, and compare this closed form against a finite-difference gradient of `J`. The max discrepancy comes out `4.3e-9` — finite-difference noise, nothing structural. Good: the gradient really is the `(k+k^T)`-weighted pull of each token toward the others, and crucially the non-symmetry of `k` is absorbed into the symmetric combination `k+k^T`, which is what lets a non-symmetric attention matrix still descend a symmetric functional.

So the gradient flow `du/dt = -∇J(u)` moves each `u(x)` toward a `(k+k^T)`-weighted average of the other positions. Now Euler-discretize with one step of size `Δt(x) = 1 / ∫ (k(x,y)+k(y,x)) dy`, initialize `u(x, 0) = v(x)` at the value vectors, choose the symmetric kernel `K(x,y) = k(x,y)+k(y,x) = exp(k(x)^T k(y)/√d)`, and the single update is

  u(x, Δt) = ∫ [K(x,y) / ∫ K(x,y') dy'] v(y) dy = Σ_j softmax(k_x^T k_j/√d) v(j),

which, once I break the symmetry by letting the query play the role of one of the keys, is exactly self-attention `u(i) = Σ_j softmax(q_i^T k_j/√d) v(j)`. So self-attention is one gradient-descent step on the smoothing functional `J`. And now the numerics from the start click into place: minimizing `J` converges to a constant function — that was the limit I watched my three tokens reach — so stacking attention layers iterates a contraction toward uniformity. Over-smoothing is not a bug in the implementation; it is the fixed point of the objective attention is descending.

That diagnosis tells me what to try. If the problem is that I am descending `J` alone, whose attractor is "all tokens identical," then I should not descend `J` alone. Add a second term that *opposes* collapse. The natural choice from the variational picture is a convex fidelity term that anchors `u` to some reference signal `f`:

  E(u, f) = J(u) + (λ/2) ∫ ||u(x) - f(x)||² dx.

The fidelity term pulls `u` back toward `f` and penalizes drifting away from it — it is exactly the regularizer used in image denoising to keep a smoothed image from washing out to gray. Now take the gradient flow of `E`:

  du/dt = -∇J(u) - λ(u - f).

The sign matters. The new term in the flow is `-λ(u-f)`, so an Euler step contributes `-λΔt(u(x,0)-f(x))`. With the same initialization as attention, `u(x,0)=v(x)`, and with the usual scaling choice `λ = λ̃/Δt(x)`, that contribution is `+λ̃(f(x)-v(x))`. Working it through, the per-token update becomes

  u(i) = Σ_j softmax(k_i^T k_j/√d) v(j) + λ̃ (f(i) - v(i)),

the ordinary attention output plus a pull `λ̃(f - v)` toward the reference. The only question left is what `f` should be. I want `f` to be a representation that has *not* been smoothed — something that still holds the per-token, localized information that the deep layers have lost. The cleanest such signal in the network is the *first* layer's value vectors `v^0 = V_1`: these values are computed directly from `H_0` before the first attention operation smooths anything. Set `f = V_1` and the layer-`n` update is

  U_n = Attn(Q_n, K_n, V_n) + λ̃ (V_1 - V_n),

with a default `λ` around 0.4. That is a principled fix on its terms: re-supply the un-smoothed first-layer value at every layer to counteract the diffusion. But before I commit to it I want to look hard at *where* and *how* this term enters, because something about the form is nagging me and I would rather find the problem now than after a training run.

Look at the term `λ(V_1 - V_n)` and where it lands: it is added to the attention *output* `U_n`. That means `V_1` is dropped onto the result *raw* — it never passes through the attention matrix `A_n` of this layer. But `V_1` is a sequence of per-token vectors; surely which positions a given query wants to read `V_1` from is itself information. By bolting `V_1` onto the output, I aggregate the *current* values `V_n` with the learned attention weights but inject the early value through an identity mapping at the same position, with no learned cross-token routing. That feels wasteful. The second thing nagging me is the *signed difference*: `V_1 - V_n` doesn't only add `V_1`, it also *subtracts* the current value `V_n`. So the strength of the injection is entangled with a simultaneous suppression of the layer's own value.

Let me make that second worry quantitative instead of leaving it as a feeling, because "entangled" is vague. Since attention `A` is linear, the output-side correction can be pushed back through it: `A V_n + λ(A V_1 - A V_n) = A((1-λ) V_n + λ V_1)`. I should confirm that rearrangement holds rather than assume it — on a random 5-token instance the two sides agree to `1.1e-16`, machine zero, so the output-side signed injection is *identically* a before-attention mix with coefficients `(1-λ, λ)`. That equivalence is the whole story: the coefficient riding on the layer's own value `V_n` is `(1-λ)`. So as I turn `λ` up, I am not just adding more `V_1` — I am turning *down* `V_n`, and at `λ=1` the layer's own value drops out completely, and beyond `λ=1` it flips sign and the layer *subtracts* its own value. Let me see what that does to the output norm across a `λ` sweep, against the alternative where I keep the `V_n` coefficient pinned at 1 and only add `V_1` on top:

```
 λ   | signed  A((1-λ)V_n + λV_1) : coef on V_n  ||out||  | positive  A(V_n + λV_1) : coef on V_n  ||out||
0.0  |        +1.0                      2.876     |           +1.0                       2.876
0.4  |        +0.6                      2.670     |           +1.0                       3.655
1.0  |         0.0                      4.140     |           +1.0                       5.593
1.5  |        -0.5                      6.020     |           +1.0                       7.459
2.0  |        -1.0                      8.069     |           +1.0                       9.411
```

There it is, concretely: in the signed form the `V_n` coefficient marches from `+1` through `0` to `-1` as `λ` goes `0 → 1 → 2`, so the meaning of the layer's output changes qualitatively across the sweep — at small `λ` it is mostly the current value, at `λ=1` the current value is gone, past that it is being actively cancelled. The net effect depends delicately on `λ`, and I would expect only a narrow window of `λ` to be any good, because too much `λ` is both too much `V_1` *and* a cancellation of `V_n`. The positive form keeps the `V_n` coefficient fixed at `+1` for every `λ`; raising `λ` only adds early value, it never erodes the current one. That is the design I want — and now I can name precisely why: not "it feels cleaner" but "the signed form couples two effects through one knob, and the equivalence `A((1-λ)V_n + λV_1)` shows exactly how."

So let me redesign the injection from that. First, get rid of the raw, attention-free addition: I want `V_1` to be aggregated by the *same* attention weights that aggregate `V_n`, so a query reads early-token information from the positions it actually cares about. The way to make `V_1` share the attention matrix is to mix it into the value *before* the attention operation rather than adding it after. So instead of `Attn(Q_n,K_n,V_n) + λ(V_1 - V_n)`, build a new value

  V_n' = λ_{n,1} V_1 + λ_{n,2} V_n,   with V_1 = H_0 W^V_1,  V_n = H_{n-1} W^V_n,

and run the *ordinary* attention on it: `U_n = A_n V_n' = Σ_j A_n(i,j) (λ_{n,1} V_1(j) + λ_{n,2} V_n(j))`. Now `V_1` and `V_n` are carried by the identical learned weights `A_n`; the early information is read from the right places. And the second complaint is now structurally absent: this is a weighted sum of two values with both coefficients free and positive — with `λ_{n,2}` decoupled from `λ_{n,1}`, raising the early-value weight never forces a subtraction of the layer's own value, which is exactly the `+1`-pinned column I just watched stay well-behaved. With `λ_{n,1}=λ_{n,2}=0.5` it is the plain average of "raw early value" and "this layer's value," the simplest identity-style choice; with `λ_{n,1}=2, λ_{n,2}=1` it can weight the early value more heavily without ever turning the current value negative.

But I should check I am not breaking the thing I was careful to protect. Why might it be safe to mess with the *value* path and not the query or key path? The deep layers have learned a specific *attention distribution* `A_n` for abstract, sequence-level mixing — that distribution is the valuable thing the depth bought me. If I add `V_1` (or `H_0`) into the query or key, I change `Q_n` or `K_n`, which changes `A_n` itself — I corrupt the learned attention pattern. If I add it into the post-softmax matrix `A_n` directly, same problem, more blatant. But the value path is different: `A_n` is computed from `Q_n, K_n` and is left completely untouched; modifying `V` only changes *what content* gets aggregated under the existing weights, not *how* tokens attend. So among the four places I could inject — Q, K, A, V — the value path is the only one that leaves `A_n` invariant. This also explains why DenseFormer's averaging of whole hidden states `H_i` is clumsier: an `H_i` that gets summed into the stream feeds *all three* projections Q, K, V of the next layer, so it perturbs the attention distribution; the value-only injection is the surgical version. I should be honest that this is an argument from structure, not a measurement — it predicts that residuals to Q, K, or the attention matrix should be *worse* than residuals to V, because only V preserves `A_n`, and that is a sanity check I would want to run before believing the value path is special rather than just convenient.

Now, the source. I chose `V_1`, the first layer's value, by the over-smoothing argument: it is the least-smoothed value. But why specifically the *first* layer and not, say, the second? Let me think about what is already reachable. The ordinary hidden residual already carries `H_1` forward — `H_1` is in the residual stream and feeds `V_2 = H_1 W^V_2` and everything after. So if I were to inject `V_2` as the early signal, I would mostly be re-supplying information (`H_1`) that the standard residual *already* delivers; it should be close to redundant. `V_1 = H_0 W^V_1`, by contrast, is a linear map of the *raw* token embedding `H_0`, and although `H_0` is also nominally in the residual stream, the over-smoothing argument says it is the most diluted thing there. So `V_1` carries the information that is both most valuable (purest token-level) and least redundant with the existing residual. I would predict, then, that a `V_1` source helps a lot and a `V_2` source barely helps — and, as a sharper test of the redundancy story, that if I *restarted* the hidden residual at `H_2` (so `H_1` is no longer freely propagated), then suddenly a `V_2` injection *would* start to help, because now `V_2` is no longer redundant. That is a clean, falsifiable consequence of the "redundant with the hidden residual" explanation, and I would want to run it before claiming the redundancy story is the right one.

What about dense — re-supplying *all* previous values `Σ_{i<n} V_i`, not just `V_1`? The trouble is dilution: most of those `V_i` for `i ≥ 2` are themselves partly-smoothed and partly-redundant, so averaging them in waters down the one clean signal, `V_1`. I expect the pure `V_1` connection to beat the dense mixture, though I would not be shocked if a *learned* dense mixture matched it by simply zeroing the redundant terms. (The general dense form `V_n' = λ_{n,n}V_n + Σ_{i<n} λ_{n,i} V_i` is worth keeping in the back pocket as the most flexible variant, but my prior is the sparse `V_1`-only version is what carries the gain.)

Let me also weigh the "share the attention matrix" decision against the obvious alternative of giving `V_1` its *own* attention. I could, for each layer, recompute a fresh attention matrix for `V_1` from `K_1` and the current `Q_n` — cross-layer attention, `softmax(Q_n Concat(K_n, K_1)^T) Concat(V_n, V_1)`. That is strictly more expressive, but it costs a second attention computation per layer, which is the expensive part, and it reintroduces a *new* learned distribution over `V_1` that could itself over-smooth. The cheap option — reuse `A_n` — assumes the positions a query wants its current value from are also the positions it wants its early value from, which is a reasonable prior and costs nothing. And the degenerate option — add `V_1` to `U_n` with no attention at all, i.e. an identity mapping for `V_1` — is the NeuTRENO-style raw injection I already argued against. Between the three I will take the middle one, reuse the current attention matrix, but I am holding the expressiveness/cost tradeoff loosely; the right call is whichever wins at matched loss, and I have only argued, not measured, that the cheap reuse is enough.

Now the coefficients. The clean default is `λ_{n,1} = λ_{n,2} = 0.5` everywhere, a fixed identity-style average; this needs zero new parameters beyond the existing `W^V` and is the obvious first thing to try. But I have a structural reason to think the *right* mix is not uniform across depth: the over-smoothing is worst in the deep layers — that is what the iterated-contraction picture says, the spread is smallest after many steps — and DenseFormer's own learned coefficients report that deeper layers want *more* of the initial signal. So make `λ` learnable, one pair per layer, initialized at 0.5, and let training decide. My prediction is that the learned `λ_{n,1}` (the weight on `V_1`) grows with depth — the later layers reach for more early information — which would both fit the over-smoothing-is-worse-deep picture and mean the model is, on its own, discovering a sparse pattern where the value residual concentrates in the late layers. If that is what learnable `λ` finds, then a hand-built sparse version — apply the `V_1` residual only in the last few layers, zero elsewhere — should preserve the mechanism while touching fewer layers. And for a fixed-constant choice, I would not assume 0.5 is optimal; sweeping a constant `λ` (the same value at every layer, of the form `λ V_1 + V_n`) I would expect a fairly wide robust plateau — wide *because* there is no `-V_n` subtraction, exactly the `+1`-pinned column from my norm sweep — and I would include values above 1, so the clean early value can be stronger than the current value. If the plateau turns out broad, that is the experimental tell that the positive mix is better-behaved than the signed difference; if it turns out narrow, my whole "decoupling helps" argument is wrong and I should know it.

I should pause on a skeptical worry: am I sure this is a *representational* improvement and not just an *optimization* speedup — a shortcut that, like any skip connection, makes gradients flow better and training faster, with no real change in what the converged model can represent? The gradient-flow story predicts some rerouting: with the value residual, a chunk of the gradient that used to reach `V_1` only through the long path via `H_1` now reaches `W^V_1` *directly* through the residual, so I would expect the first layer's `W^V` to see a larger gradient norm and its `W^O` a smaller one. But that alone would be a boring explanation. If the gain were only that gradient rerouting, then I could mimic it on a vanilla Transformer by simply boosting the learning rate on the first layer, or specifically on `W^V_1`. So the decisive test is whether those learning-rate hacks reproduce the same final loss. If they cannot, then the value residual is changing what information the model can use at depth, not just how fast the first layer trains — and if they *can*, I have to retract the representational claim. I genuinely do not know which way that one goes without running it.

There is a deeper candidate *why* I can articulate, which also predicts a battery of side effects I could check. The attention-sink pathology — deep layers dumping huge attention mass on a low-semantic token, usually the first — travels together with "value-state drains": those same sink tokens carry abnormally large value-state norms, and abnormally large hidden-state norms (the massive-activation / residual-state-peak phenomenon). There is a mutual-reinforcement loop: a token with a giant value norm is dangerous to attend to (since the output is `A_n V_n`, attending even a little to a huge-norm value swamps the result — which is the same norm-amplification I watched in the sweep when the coefficients grew), so to keep the loss low the model learns to either attend to it heavily in a controlled way or route around it, and the dynamics settle into sink-plus-drain. Now ask: does the *first layer's* value have this drain? It should not — the drain is a learned, deep-layer phenomenon, and `V_1` is computed straight from the token embedding before any of that dynamics has had a chance to form. So when I inject (presumably) drain-free `V_1` into the deep value `V_n'`, the deep value should no longer have to carry the pathological large norm on the sink token — the early, well-behaved value dominates there — which would break the value-drain side of the loop, which in turn would remove the model's reason to concentrate attention on that token. If that mechanism is right, I would predict the value residual *flattens* the attention-sink: lower token-importance entropy concentration in deep layers, smaller value-state norms and hidden-state norms on the first token, a more uniform distribution of token importance. A few more downstream predictions fall out: each deep layer, now handed a good baseline value `V_1`, would only need to learn a small *correction* `ΔV` on top of it, so the learned per-layer values (before the residual) should become more similar to each other as depth grows, with the corrections shrinking in later layers. The hidden states should also carry *more* information (higher PCA rank for fixed explained variance), and removing attention layers should hurt about as much as removing MLP layers — because the attention, now fed clean values, is contributing real content rather than just smoothing. None of these are confirmed; they are the falsifiable consequences of the single hypothesized mechanism — re-inject the un-smoothed, drain-free early value into the value path — and they are exactly the diagnostics I would point at it to find out whether the mechanism is the real reason or a just-so story.

Let me write this mechanism as concrete code, filling the value-construction slot in the attention block. The only state I need beyond a standard block is a cache of `V_1` from layer 0 and a pair of `λ`s per layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, scale):
    A = torch.softmax((q @ k.transpose(-2, -1)) * scale + causal_mask(q, k), dim=-1)
    return A @ v                                  # value residual rides the SAME A as V_n


class ResBlock(nn.Module):
    """Pre-norm decoder block with a value residual to the first layer's value.
    layer_idx == 0 produces and caches V_1; later layers consume it."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.scale = (config.n_embd // config.n_head) ** -0.5
        # learnable mix; init 0.5/0.5 == the fixed-average default. (lambda1=0 disables it.)
        if layer_idx > 0:
            self.lam1 = nn.Parameter(torch.tensor(0.5))   # weight on V_1 (early, un-smoothed)
            self.lam2 = nn.Parameter(torch.tensor(0.5))   # weight on this layer's own value

    def forward(self, x, v_first):
        h = self.ln1(x)
        q, k = self.Wq(h), self.Wk(h)
        v = self.Wv(h)                                     # V_n = H_{n-1} W^V_n
        if self.layer_idx == 0:
            v_first = v                                    # cache V_1 = H_0 W^V_1 for later layers
        else:
            v = self.lam1 * v_first + self.lam2 * v        # V_n' = lam1*V_1 + lam2*V_n, BEFORE attn
        u = attention(q, k, v, self.scale)                # A_n aggregates the mixed value
        x = x + self.Wo(u)
        x = x + self.mlp(self.ln2(x))
        return x, v_first
```

That is the value-residual block. The driving loop just threads `v_first` through and the layers self-organize where the early value matters; freezing `lam1=lam2=0.5` gives the fixed-average identity version, a fixed positive constant gives the constant version, and zeroing `lam1` outside the last few layers gives the sparse version.

Now there is one more move I want to make, because reusing `V_1` ties the injected signal to whatever `W^V_1` happens to learn — `V_1` has a day job (it is layer 1's own value), and asking it to *also* be the canonical "early token value to re-inject everywhere" is two jobs for one tensor. Step back and ask what `V_1` actually *is*: `V_1 = H_0 W^V_1`, a linear map of the token embedding — so it is, functionally, just a *token-indexed lookup that produces a value-space vector*. Nothing about the mechanism requires that this lookup be physically the first layer's value projection. I can replace it with a *dedicated* embedding table `E_v` that maps token id straight to a value-space residual, with its own free parameters, gated and added into the value path exactly as before:

  V_n' = V_n + λ_n · E_{v,n}(token).

This is the same mechanism — a per-token, un-smoothed, drain-free signal injected into the value path under the layer's own attention matrix — but the injected signal is now a free parameter that can specialize purely to "what early information should this layer's value carry," decoupled from layer 1's responsibilities. A few practical choices follow from what I already worked out. The injection should be *gated*: a learnable `λ` per table initialized at 0.5, so each layer dials in how much it wants — the same logic that made `λ` learnable for the value mix. The tables should be initialized *small* (std ≈ 0.01) so the residual starts as a gentle perturbation and the model is not shocked at step 0; it can grow the gate if it helps. And I should not pay for one table per layer. A handful of full-rank partitions is enough to express distinct token-to-value residuals, and the layer choice should mirror the mechanism: put two tables early, where raw token-value signals are still close to the input, and put three in the last layers, where over-smoothing is most acute. So the embedding specialization becomes five gated token-to-value tables, injected at layers `1`, `2`, `N-3`, `N-2`, and `N-1`, with each selected layer reading its own partition.

Let me write that form, since it is the one that drops cleanly into a fixed pretraining harness as a self-contained embedding module — the value-construction lives in the embedding object, which hands each chosen layer a residual through a `get_value_embed(layer_idx)` hook, and the attention block just adds it to `v`.

```python
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token + position embedding, plus value embeddings injected into the value
    path of selected layers. get_value_embed(i) returns lambda_i * E_v_i(token)
    for the chosen layers, else None; the attention block does v = v + that."""

    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer

        # A few dedicated token -> value-space tables. One partitioned table holds
        # all of them; small init so the injection starts as a gentle perturbation.
        self.n_ve = 5
        self.vte = nn.Embedding(config.vocab_size * self.n_ve, config.n_embd)
        nn.init.normal_(self.vte.weight, mean=0.0, std=0.01)
        # Learnable per-table gate (lambda): how much early value to inject.
        self.ve_lambda = nn.Parameter(torch.full((self.n_ve,), 0.5))
        # Use the selected layers from the value-embedding edit: layer 1, layer 2,
        # and the last three layers.
        self._ve_layers = None
        self._cached_ve = None

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        if self._ve_layers is None:
            self._ve_layers = [1, 2, self.n_layer - 3, self.n_layer - 2, self.n_layer - 1]
        # one lookup per table, each into its own partition of the joint table
        vs = self.vocab_size
        self._cached_ve = {}
        for i, layer_idx in enumerate(self._ve_layers):
            offset_idx = idx + i * vs                      # partition i of the joint table
            self._cached_ve[layer_idx] = self.vte(offset_idx)
        return self.drop(tok_emb + pos_emb)                # token+pos stream is unchanged

    def get_value_embed(self, layer_idx):
        """Gated value-space residual for this layer, or None: lambda * E_v(token)."""
        if self._cached_ve is None or layer_idx not in self._cached_ve:
            return None
        ve_idx = self._ve_layers.index(layer_idx)
        lamb = self.ve_lambda[ve_idx]
        return lamb * self._cached_ve[layer_idx]           # added to v INSIDE attention

    def get_lm_head_weight(self):
        return self.wte.weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
```

and inside the attention block, the single line that consumes the tensor returned by `get_value_embed(layer_idx)` — added to the value *before* attention so it shares the attention matrix, the whole point:

```python
        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
        if value_embed is not None:                        # lambda * E_v(token), or None
            v = v + value_embed                            # value residual: rides A as v does
        u = attention(q, k, v, self.scale)
```
