Let me start from the thing that actually bothers me. I keep stacking more layers on these decoder-only Transformers because the scaling laws tell me depth buys capability, and yet past some point the deeper model is not better, sometimes it is worse — a 32-layer vision Transformer losing to a 24-layer one, language gains flattening as I add blocks. That is strange, because depth is supposed to be nearly free: every block has a residual connection, `H_n = H_{n-1} + f(H_{n-1})`, so the original token embedding `H_0` is always sitting one identity path away from every layer. ResNet taught me that identity shortcuts fix the gradient flow and let arbitrarily deep nets train; DenseNet pushed it further, letting every layer read all the earlier features. So in principle the initial information is preserved. But "preserved in the residual stream" and "usable by layer 24" are not the same claim, and I should pull them apart.

Here is the issue with the residual stream as a carrier. It is *one* stream, and *every* block writes into it by addition. `H_0` is in there, but so is the sum of twenty-three blocks' worth of updates layered on top. The raw token content is not gone, but it is buried under everything written since. And there is a specific reason to think the deep layers genuinely cannot recover it: attention is a smoothing operation. Each attention layer replaces a token's representation with a convex combination of all tokens' value vectors — the softmax row sums to one — which is exactly an averaging, a low-pass filter over the sequence. Iterate an averaging operation through depth and the representations drift toward each other. This is the over-smoothing story, and people have measured it: in deep layers, token representations become increasingly similar, sequence-level features dominate, and the per-token, localized information from `H_0` is washed out. The hidden residual carries `H_0` forward, but the very operation the deep layers run on it is one that destroys exactly the locality I want to preserve.

I want this sharper than "attention smooths," because a slogan will not tell me where to intervene. There is a clean way to see it. Treat the sequence of token vectors as a function `u` sampled on a grid, and ask what objective a single self-attention update is descending. Write the nonlocal smoothing functional

  J(u) = (1/2) ∬ ||u(x) - u(y)||² k(x, y) dx dy,

where `k(x, y)` is a nonnegative affinity between positions. This penalizes any *difference* between token representations, weighted by their affinity — minimizing it means making tokens agree. Take its first variation. Perturb `u_j -> u_j + τ h_j` and differentiate at `τ = 0`:

  (d/dτ) J|_{τ=0} = ∬ (u_j(x) - u_j(y))(h_j(x) - h_j(y)) k(x, y) dx dy.

Split the two terms and apply the change of variables `(x, y) -> (y, x)` to the second; by symmetry of the construction the cross terms combine, and the functional derivative comes out

  ∂J/∂u_j(x) = ∫ (u_j(x) - u_j(y)) (k(x, y) + k(y, x)) dy.

So the gradient flow `du/dt = -∇J(u)` moves each `u(x)` toward a `(k+k^T)`-weighted average of the other positions. Now Euler-discretize with one step of size `Δt(x) = 1 / ∫ (k(x,y)+k(y,x)) dy`, initialize `u(x, 0) = v(x)` at the value vectors, choose the symmetric kernel `K(x,y) = k(x,y)+k(y,x) = exp(k(x)^T k(y)/√d)`, and the single update is

  u(x, Δt) = ∫ [K(x,y) / ∫ K(x,y') dy'] v(y) dy = Σ_j softmax(k_x^T k_j/√d) v(j),

which, once I break the symmetry by letting the query play the role of one of the keys, is exactly self-attention `u(i) = Σ_j softmax(q_i^T k_j/√d) v(j)`. So self-attention literally *is* one gradient-descent step on the smoothing functional `J`. And what does minimizing `J` converge to? Its minimizer is a constant function — every token equal. Read as a diffusion or a random walk, the dynamics have a stationary distribution and `u^{(k)}` converges to a single constant vector as the number of steps grows. Over-smoothing is not a bug in the implementation; it is the fixed point of the objective attention is descending. Stacking attention layers is iterating a contraction toward uniformity.

That diagnosis tells me precisely how to fight it. If the problem is that I am descending `J` alone, whose attractor is "all tokens identical," then I should not descend `J` alone. Add a second term that *opposes* collapse. The natural choice from the variational picture is a convex fidelity term that anchors `u` to some reference signal `f`:

  E(u, f) = J(u) + (λ/2) ∫ ||u(x) - f(x)||² dx.

The fidelity term pulls `u` back toward `f` and penalizes drifting away from it — it is exactly the regularizer used in image denoising to keep a smoothed image from washing out to gray. Now take the gradient flow of `E`:

  du/dt = -∇J(u) - λ(u - f).

The sign matters. The new term in the flow is `-λ(u-f)`, so an Euler step contributes `-λΔt(u(x,0)-f(x))`. With the same initialization as attention, `u(x,0)=v(x)`, and with the usual scaling choice `λ = λ̃/Δt(x)`, that contribution is `+λ̃(f(x)-v(x))`. Working it through, the per-token update becomes

  u(i) = Σ_j softmax(k_i^T k_j/√d) v(j) + λ̃ (f(i) - v(i)),

the ordinary attention output plus a pull `λ̃(f - v)` toward the reference. The only question left is what `f` should be. I want `f` to be a representation that has *not* been smoothed — something that still holds the per-token, localized information that the deep layers have lost. The cleanest such signal in the network is the *first* layer's value vectors `v^0 = V_1`: these values are computed directly from `H_0` before the first attention operation smooths anything. Set `f = V_1` and the layer-`n` update is

  U_n = Attn(Q_n, K_n, V_n) + λ̃ (V_1 - V_n),

with a default `λ` around 0.4. This is a real, principled fix: re-supply the un-smoothed first-layer value at every layer to counteract the diffusion. Good — but let me not stop here, because something about *where* and *how* this term enters is leaving value on the table, and I can feel it.

Look at the term `λ(V_1 - V_n)` and where it lands: it is added to the attention *output* `U_n`. That means `V_1` is dropped onto the result *raw* — it never passes through the attention matrix `A_n` of this layer. But `V_1` is a sequence of per-token vectors; surely which positions a given query wants to read `V_1` from is itself information. By bolting `V_1` onto the output, I aggregate the *current* values `V_n` with the learned attention weights but inject the early value through an identity mapping at the same position, with no learned cross-token routing. That feels wasteful. The second thing nagging me is the *signed difference*: `V_1 - V_n` doesn't only add `V_1`, it also *subtracts* the current value `V_n`. So the strength of the injection is entangled with a simultaneous suppression of the layer's own value, and the net effect depends delicately on `λ`. If I sweep `λ`, I would expect a narrow good window, because too much `λ` is both too much `V_1` and too much `-V_n` at once. I want the benefit — re-supplying early token information — without those two encumbrances.

Let me redesign the injection from that complaint. First, get rid of the raw, attention-free addition: I want `V_1` to be aggregated by the *same* attention weights that aggregate `V_n`, so a query reads early-token information from the positions it actually cares about. The way to make `V_1` share the attention matrix is to mix it into the value *before* the attention operation rather than adding it after. So instead of `Attn(Q_n,K_n,V_n) + λ(V_1 - V_n)`, build a new value

  V_n' = λ_{n,1} V_1 + λ_{n,2} V_n,   with V_1 = H_0 W^V_1,  V_n = H_{n-1} W^V_n,

and run the *ordinary* attention on it: `U_n = A_n V_n' = Σ_j A_n(i,j) (λ_{n,1} V_1(j) + λ_{n,2} V_n(j))`. Now `V_1` and `V_n` are carried by the identical learned weights `A_n`; the early information is read from the right places. And notice the second complaint dissolves automatically: this is a positive weighted sum `λ_{n,1}V_1 + λ_{n,2}V_n`, not a difference — no subtraction of the layer's own value. With `λ_{n,1}=λ_{n,2}=0.5` it is the plain average of "raw early value" and "this layer's value," the simplest identity-style choice; with `λ_{n,1}=2, λ_{n,2}=1` it can weight the early value more heavily without ever turning the current value negative. I would bet this is far more robust to `λ` than the signed-difference form, precisely because nothing is being subtracted.

But wait — I should check I am not breaking the thing I was careful to protect. Why is it safe to mess with the *value* path and would it be unsafe to do the same to the query or key path? The deep layers have learned a specific *attention distribution* `A_n` for abstract, sequence-level mixing — that distribution is the valuable thing the depth bought me. If I add `V_1` (or `H_0`) into the query or key, I change `Q_n` or `K_n`, which changes `A_n` itself — I corrupt the learned attention pattern. If I add it into the post-softmax matrix `A_n` directly, same problem, more blatant. But the value path is different: `A_n` is computed from `Q_n, K_n` and is left completely untouched; modifying `V` only changes *what content* gets aggregated under the existing weights, not *how* tokens attend. So the value path is the one safe channel — it lets token-level information ride along without disturbing the abstract computation. This also explains why DenseFormer's averaging of whole hidden states `H_i` is clumsier: an `H_i` that gets summed into the stream feeds *all three* projections Q, K, V of the next layer, so it perturbs the attention distribution; the value-only injection is the surgical version. If this reasoning is right, the sanity check is clear: residuals to Q, K, or the attention matrix should be worse than residuals to V, because only V preserves `A_n`.

Now, the source. I chose `V_1`, the first layer's value, by the over-smoothing argument: it is the least-smoothed value. But why specifically the *first* layer and not, say, the second? Let me think about what is already reachable. The ordinary hidden residual already carries `H_1` forward — `H_1` is in the residual stream and feeds `V_2 = H_1 W^V_2` and everything after. So if I were to inject `V_2` as the early signal, I would mostly be re-supplying information (`H_1`) that the standard residual *already* delivers; it should be close to redundant. `V_1 = H_0 W^V_1`, by contrast, is a linear map of the *raw* token embedding `H_0`, and although `H_0` is also nominally in the residual stream, the over-smoothing argument says it is the most diluted thing there. So `V_1` carries the information that is both most valuable (purest token-level) and least redundant with the existing residual. I would predict, then, that a `V_1` source helps a lot and a `V_2` source barely helps — and, as a sanity check on the redundancy story, that if I *restarted* the hidden residual at `H_2` (so `H_1` is no longer freely propagated), then suddenly a `V_2` injection *would* start to help, because now `V_2` is no longer redundant. That is a clean, falsifiable consequence of the "redundant with the hidden residual" explanation.

What about dense — re-supplying *all* previous values `Σ_{i<n} V_i`, not just `V_1`? The trouble is dilution: most of those `V_i` for `i ≥ 2` are themselves partly-smoothed and partly-redundant, so averaging them in waters down the one clean signal, `V_1`. I expect the pure `V_1` connection to beat the dense mixture. (The general dense form `V_n' = λ_{n,n}V_n + Σ_{i<n} λ_{n,i} V_i` is worth keeping in the back pocket as the most flexible variant, but my prior is the sparse `V_1`-only version is what carries the gain.)

Let me also double-check the "share the attention matrix" decision against the obvious alternative of giving `V_1` its *own* attention. I could, for each layer, recompute a fresh attention matrix for `V_1` from `K_1` and the current `Q_n` — cross-layer attention, `softmax(Q_n Concat(K_n, K_1)^T) Concat(V_n, V_1)`. That is strictly more expressive, but it costs a second attention computation per layer, which is the expensive part, and it reintroduces a *new* learned distribution over `V_1` that could itself over-smooth. The cheap option — reuse `A_n` — assumes the positions a query wants its current value from are also the positions it wants its early value from, which is a reasonable prior and costs nothing. And the degenerate option — add `V_1` to `U_n` with no attention at all, i.e. an identity mapping for `V_1` — is the NeuTRENO-style raw injection I already argued against; I expect it to be clearly worst. So: reuse the current attention matrix.

Now the coefficients. The clean default is `λ_{n,1} = λ_{n,2} = 0.5` everywhere, a fixed identity-style average; this needs zero new parameters beyond the existing `W^V` and is the obvious first thing to try. But I have a structural reason to think the *right* mix is not uniform across depth: the over-smoothing is worst in the deep layers, and DenseFormer's own learned coefficients say deeper layers want *more* of the initial signal. So make `λ` learnable, one pair per layer, initialized at 0.5, and let training decide. My prediction is that the learned `λ_{n,1}` (the weight on `V_1`) grows with depth — the later layers reach for more early information — which would both confirm the over-smoothing-is-worse-deep picture and mean the model is, on its own, discovering a sparse pattern where the value residual concentrates in the late layers. If that is what learnable `λ` finds, then a hand-built sparse version — apply the `V_1` residual only in the last few layers, zero elsewhere — should preserve the mechanism while touching fewer layers. And for a fixed-constant choice, I would not assume 0.5 is optimal; sweeping a constant `λ` (the same value at every layer, of the form `λ V_1 + V_n`) I would expect a fairly wide robust plateau — wide *because* there is no `-V_n` subtraction — and I would include values above 1, so the clean early value can be stronger than the current value. The robustness over a broad `λ` range would be the tell that this positive value mix is better-behaved than the signed difference.

I should pause on a skeptical worry: am I sure this is a *representational* improvement and not just an *optimization* speedup — a shortcut that, like any skip connection, makes gradients flow better and training faster, with no real change in what the converged model can represent? The gradient-flow story predicts some rerouting: with the value residual, a chunk of the gradient that used to reach `V_1` only through the long path via `H_1` now reaches `W^V_1` *directly* through the residual, so I would expect the first layer's `W^V` to see a larger gradient norm and its `W^O` a smaller one. But that alone would be a boring explanation. If the gain were only that gradient rerouting, then I could mimic it on a vanilla Transformer by simply boosting the learning rate on the first layer, or specifically on `W^V_1`. So the decisive test is whether those learning-rate hacks reproduce the same behavior. If they cannot, then the value residual is changing what information the model can use at depth, not just how fast the first layer trains.

There is a deeper *why* I can now articulate, which also predicts a battery of side effects. The attention-sink pathology — deep layers dumping huge attention mass on a low-semantic token, usually the first — travels together with "value-state drains": those same sink tokens carry abnormally large value-state norms, and abnormally large hidden-state norms (the massive-activation / residual-state-peak phenomenon). There is a mutual-reinforcement loop: a token with a giant value norm is dangerous to attend to (since the output is `A_n V_n`, attending even a little to a huge-norm value swamps the result), so to keep the loss low the model learns to either attend to it heavily in a controlled way or route around it, and the dynamics settle into sink-plus-drain. Now ask: does the *first layer's* value have this drain? No — the drain is a learned, deep-layer phenomenon; `V_1` is computed straight from the token embedding and has no abnormal sink-token norm. So when I inject drain-free `V_1` into the deep value `V_n'`, the deep value no longer has to carry the pathological large norm on the sink token — the early, well-behaved value dominates there — which breaks the value-drain side of the loop, which in turn removes the model's reason to concentrate attention on that token. I would therefore predict the value residual *flattens* the attention-sink: lower token-importance entropy concentration in deep layers, smaller value-state norms and hidden-state norms on the first token, a more uniform distribution of token importance. And a few more downstream predictions fall out: each deep layer, now handed a good baseline value `V_1`, only needs to learn a small *correction* `ΔV` on top of it, so the learned per-layer values (before the residual) should become more similar to each other as depth grows, with the corrections shrinking in later layers — a phenomenon that should be unique to this architecture. The hidden states should also carry *more* information (higher PCA rank for fixed explained variance), and removing attention layers should hurt about as much as removing MLP layers — because the attention, now fed clean values, is contributing real content rather than just smoothing. All of these are things I would want to verify, and all of them follow from the single mechanism: re-inject the un-smoothed, drain-free early value into the value path.

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

So the causal chain, start to finish. Deeper Transformers stopped paying off, and the reason is over-smoothing: I showed self-attention is one gradient step on a nonlocal smoothing functional whose minimizer is a constant, so stacking attention diffuses token representations toward uniformity and washes out the localized token-level information the deep layers need. The hidden residual nominally carries the initial embedding forward, but it carries the already-smoothed hidden state and feeds all three of Q, K, V, so re-injecting whole hidden states disturbs the very attention distribution depth bought me. The variational fix is to descend a regularized functional with a convex fidelity term anchoring the output to an un-smoothed reference; the Euler step contributes `+λ̃(f-v)`, and with `f=V_1` that becomes the output-side `λ(V_1 - V_n)` correction. But adding it to the attention output injects `V_1` through an identity mapping and the signed difference entangles the injection with subtracting the layer's own value. I fixed both by mixing the early value into the value before attention as `V_n' = λ_{n,1}V_1 + λ_{n,2}V_n`, so `V_1` is aggregated by the same learned attention matrix `A_n` and nothing is subtracted — and I argued the value path is the unique safe channel, since touching Q, K, or A would corrupt the learned attention distribution while touching V only changes the content aggregated under it. The source is the first layer's value because it is the least-smoothed and least-redundant signal; later values are reachable through the ordinary residual. The mechanism is that `V_1` has no value-state drain, so injecting it should break the value-drain / attention-sink loop and let each deep layer learn a smaller `ΔV`. Finally, since `V_1 = H_0 W^V_1` is functionally just a token-indexed value-space lookup, I freed it into a dedicated, small-initialized, learnably-gated value-embedding table placed at layers `1`, `2`, `N-3`, `N-2`, and `N-1` — the same mechanism with a specialized, decoupled signal — which lands as a modular embedding intervention that fills the value path of selected layers and otherwise leaves the standard Transformer untouched.
