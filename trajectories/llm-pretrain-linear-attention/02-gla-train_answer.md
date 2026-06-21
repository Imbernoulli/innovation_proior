RetNet trained stably and landed in range — validation loss $2.4795$, a real language model — but it landed *weak*, and weak in exactly the place I predicted: the perplexities are loose (WikiText-2 $66.67$, LAMBADA $82.36$) and the downstream accuracies modest (HellaSwag $31.12$, barely above chance for a four-way task). This is not an optimization failure — the loss curve was clean — it is a *memory* failure. RetNet forgets at a rate it chose before it ever saw a token: every head has a single fixed $\gamma$, and within a head that decay is the same for every channel, context, and word, so it cannot look at a token and decide "this is a key fact, hold it" versus "this is filler, let it decay fast." The 1-D RNN lesson is unambiguous — the forget gate carries most of a gated cell's capacity and must be *data-dependent* to do its job — and a fixed scalar $\gamma$ is precisely a data-independent gate. So the one move is to make the decay a function of the input. The whole risk, the thing RetNet's fixed scalar was *protecting*, is that a data-dependent gate normally destroys the matmul chunkwise form. Getting through that wall without re-breaking what made retention trainable is this rung.

I propose **gated linear attention (GLA)**: a **per-key-channel, data-dependent forget gate**,

$$S_t = \mathrm{Diag}(\alpha_t)\,S_{t-1} + k_t^\top v_t, \qquad o_t = \tfrac{1}{\sqrt{d_k}}\,q_t\,S_t,$$

with $\alpha_t\in(0,1)^{d_k}$ computed from $x_t$ *alone*. The input-only choice (following Martin and Cundy; HGRN) is load-bearing: a classic RNN forget gate depends on the previous hidden state, and that dependence is exactly what serializes the recurrence and kills parallel training. Making the gate depend only on the current input keeps the recurrence linear in the state — the gate is a sequence of input-determined coefficients — and a linear recurrence with input-determined coefficients still has the cumulative-product structure I can exploit.

The real design decision is the gate's *shape*, a three-way trade between parameters, expressivity, and training. The most general gate is a full matrix $G_t\in(0,1)^{d_k\times d_v}$ applied Hadamard, $S_t = G_t\odot S_{t-1} + k_t^\top v_t$ — maximally expressive, but mapping $x_t\to G_t$ needs a projection of size $d\cdot d_k\cdot d_v$, absurd in parameters. RetNet's scalar is the other extreme: cheap, trivially parallel, one number for the whole state, and $2.4795$ is what that costs. The middle is a low-rank outer-product gate $G_t = \alpha_t^\top\beta_t$ — only $d\cdot d_k + d\cdot d_v$ parameters, a genuine 2-D gate. The reason I cannot naively take the full matrix is training: prior matrix-gate algorithms materialize the full $d_k\times d_v$ state for *every* time step in slow memory and their gated update does not reduce to tensor-core matmuls; Mamba is the cautionary tale, its full-rank selective update cannot be a matrix multiply, so it must cap the state expansion (~16) to keep per-step states in SRAM, which shows up as weak recall. So I need a gate expressive enough to fix the fixed-decay failure but structured enough that the matmul form survives.

Crucially, the outer-product gate *does* survive — I check it rather than assume it. With $G_t = \alpha_t^\top\beta_t$, the state is $S_t = \sum_{i\le t}\big((\prod_{j=i+1}^t G_j)\odot k_i^\top v_i\big)$. The cumulative product $\prod G_j$ is the thing I feared, but it is an outer product of cumulative products: define $b_t = \prod_{j\le t}\alpha_j$ and $d_t = \prod_{j\le t}\beta_j$, then by the mixed-product property $\prod_{j=i+1}^t \alpha_j^\top\beta_j = (b_t/b_i)^\top(d_t/d_i)$ — the telescoping collapses the product-of-outer-products into a single outer product of ratios. So $(\prod G_j)\odot(k_i^\top v_i) = ((b_t/b_i)\odot k_i)^\top((d_t/d_i)\odot v_i)$, still one outer product, and the whole layer is exactly a linear-attention parallel form on *preconditioned* tensors $\tilde Q = Q\odot B$, $\tilde K = K/B$, $\tilde V = V/D$: compute $\tilde O = ((\tilde Q\tilde K^\top)\odot M)\tilde V$, read off $O = \tilde O\odot D$. The data-dependent matrix gate did not destroy the matmul structure — it just preconditions $Q,K,V$ by cumulative-product factors.

Do I need $\beta$ too? The value-side $d_t,1/d_i$ are extra cumulative products, extra log-space bookkeeping, extra failure modes. Fixing $\beta=1$ gives $G_t=\alpha_t^\top\mathbf{1}$ — every row of the state shares the same per-key-channel decay $\alpha_t$, broadcast across the value dimension — and the value transforms vanish ($\tilde V=V$, $O=\tilde O$). The model becomes $S_t = \mathrm{Diag}(\alpha_t)S_{t-1} + k_t^\top v_t$, a **per-key-channel data-dependent forget gate** — exactly the fine-grained, content-chosen forgetting RetNet's single $\gamma$ could not express — at half the gating machinery. The value-side gate would add a second decay axis, but that is the expensive part of the bookkeeping and it overlaps with the value projection and the output gate I add anyway. So I take $G_t=\alpha_t^\top\mathbf{1}$: strictly more expressive than RetNet's scalar, far cheaper than the full matrix, keeps the chunkwise form, avoids the value-side cumulative products that make stability harder.

The clean parallel form is dead on arrival numerically. Each $\alpha_j<1$, so $b_t=\prod_{j\le t}\alpha_j$ decays toward zero fast, and $K/B$ divides by something tiny and *explodes* — overflow even in fp32. The fix is to compute scores in log space: $P_{ij} = \sum_k Q_{ik}K_{jk}\exp(\log B_{ik}-\log B_{jk})$ for $i\ge j$, where the cumulative product becomes a cumulative *sum* of log-gates $\log b_t = \sum_{j\le t}\log\alpha_j$ (a sum of negative numbers, no underflow), and $\log B_{ik}-\log B_{jk}$ is the accumulated log-decay between key $j$ and query $i$ — a data-dependent relative-position factor, a learned content-dependent ALiBi, the spiritual successor to RetNet's fixed $\gamma^{n-m}$. But that $\exp$ of an $(i,k)$-dependent difference sits between $Q$ and $K$, so the score is no longer a single matmul and the tensor cores are gone. The escape is that the underflow is a *long-range* phenomenon: within a chunk of length $C$ the cumulative product runs over at most $C$ steps and stays bounded. So the chunking measures cumulative gates *relative to chunk boundaries* — every cumulative product spans at most one chunk — and a second level of tiling pushes the full-precision log-space work down to the small diagonal sub-blocks while every off-diagonal sub-block and the inter-chunk recurrence run as half-precision matmuls. The FlashLinearAttention I/O tricks (materialization for sequence-level parallelism at small batch, recomputation in the backward, a closed-form log-space gate gradient) make it fast in wall-clock, not just FLOPs.

The remaining choices each earn their place. The gate $\alpha_t\in(0,1)^{d_k}$ is computed **low-rank** — $x_t\to W_\alpha^1\to W_\alpha^2\to$ sigmoid with a rank-16 bottleneck — nearly free in parameters and enough to choose a per-channel forget rate from content, holding the same $\sim 4d^2$ budget as softmax and the RetNet rung. A subtlety that matters: a fresh sigmoid gate sits near $0.5$, meaning the state *halves* every step, far too aggressive — long-range capacity dead before training starts. I bias it toward 1 with a temperature, $\alpha_t = \sigma(\text{logits})^{1/\tau}$ with $\tau=16$; since $\sigma<1$ the $1/16$ power pushes toward 1 ($0.5^{1/16}\approx 0.96$), so slow forgetting is the default and the model must actively decide to forget — and in log space this is $\log\alpha_t = (1/16)\,\mathrm{logsigmoid}(\text{logits})$, exactly the small, well-conditioned cumulative-sum quantity the stable form wants. Queries are scaled by $1/\sqrt{d_k}$. For dimensions I depart from RetNet's symmetric width on purpose: the state is $d_k\times d_v$ and its size is memory capacity, so I want the value full-width $d_v=d$ ($\texttt{expand\_v}=1.0$) but do not need the key as large, $d_k=d/2$ ($\texttt{expand\_k}=0.5$) — controlling parameters and state size while still leaving a sizeable $d/2\times d$ state and landing the layer back at $\sim 4d^2$. Multi-head split, per-head RMSNorm (no softmax normalizes the heads here), then a Swish output gate $r=\mathrm{swish}(xW_r)$ applied multiplicatively before the projection — the same gated-nonlinearity recipe as RetNet.

In this task's edit surface, FLA ships `GatedLinearAttention` with the chunk kernel, so the edit imports it with $\texttt{mode}=\texttt{'chunk'}$, $\texttt{hidden\_size}=1024$, $\texttt{num\_heads}=16$, $\texttt{expand\_k}=0.5$, $\texttt{expand\_v}=1.0$, $\texttt{use\_output\_gate}=\texttt{True}$, $\texttt{gate\_fn}=\texttt{'swish'}$; the log-space low-rank gate, rank-16 bottleneck, per-head RMSNorm, and secondary-tiling kernel are internal. Since `torch.compile` is disabled for this task, I wrap the FLA call in a `@torch.compiler.disable`-decorated helper so the chunk kernel is never traced. I set `self.use_pos_emb = False` for the same reason as RetNet — the cumulative data-dependent decay *is* the relative position signal. The `Block` stays the scaffold default, so the only difference from the RetNet rung is the gate.

The falsifiable claim is that GLA beats RetNet on val_loss, on both perplexities, and on the downstream suite — content-chosen forgetting holds the *right* facts at the *right* rate instead of an exponential average at a fixed rate, which is what cross-entropy rewards, and the loosest metrics (WikiText-2, LAMBADA, HellaSwag) should tighten most. But I hold one doubt the next rung may have to answer: GLA's gate is still a *diagonal, multiplicative* decay. It forgets globally or per-channel, but it does not remove the *specific stored association* that a new key collides with. If the residual failure after GLA is recall — fetching a particular value stored earlier, keyed on a token now seen again — then the gate, however data-dependent, is structurally the wrong tool, and the diagnosis would point at changing the *write rule* itself.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — GLA (gated linear attention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.attn = GatedLinearAttention(
            mode='chunk',
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=0.5,                 # d_k = d/2
            expand_v=1.0,                 # d_v = d  (full-width state = memory capacity)
            use_output_gate=True,         # swish output gate
            gate_fn='swish',
        )
        self.use_pos_emb = False          # cumulative data-dependent decay encodes relative position

    @torch.compiler.disable
    def _attn_forward(self, x):
        return self.attn(x)

    def forward(self, x):
        o, _, _ = self._attn_forward(x)
        return o


# EDITABLE region 2 of nanoGPT/custom_pretrain.py (lines 88-100) — standard pre-norm block (unchanged)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
