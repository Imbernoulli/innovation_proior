GLA landed where I bet it would on the bulk metric and failed where I flagged the risk, and the split between the two is the whole story. The data-dependent diagonal gate dropped validation loss from RetNet's $2.4795$ to $2.4482$, tightened WikiText-2 from $66.67$ to $64.32$, and raised ARC-Easy from $51.47$ to $53.11$ — exactly the local language-modeling gain I predicted. But the two metrics I told myself to watch dissociated cleanly from the rest: LAMBADA perplexity got *worse*, $84.73$ versus RetNet's $82.36$, and WinoGrande *fell*, $52.01\to 49.88$, while HellaSwag sat flat. LAMBADA is the canonical long-range completion task — give a passage, ask for the final word, recoverable only by *retrieving* a specific entity named earlier — and WinoGrande turns on resolving a reference to the right antecedent. The metrics that improved reward smooth local modeling; the ones that stalled or regressed reward *associative recall*. So the diagnosis is the doubt I closed GLA with, now confirmed: GLA's gate, however data-dependent, is *diagonal and multiplicative*. It can decide how fast each channel forgets, but it cannot reach into the state and remove the *specific stored association* that collides with a new key. The failure is no longer in the decay — it is in the **write rule**.

Look at why a multiplicative gate cannot do targeted removal. RetNet, GLA, and the whole gated-linear family share one template: $S_t = S_{t-1}\odot M_t + k_t^\top v_t$, a decay times the old state plus an additive outer-product write. The write is *Hebbian* — token $t$ stamps $k_t^\top v_t$ into the state — and the decay is the only thing that ever shrinks anything. Probe the store $S=\sum_i v_i k_i^\top$ with a stored key $k_j$:

$$S k_j = v_j (k_j^\top k_j) + \sum_{i\neq j}(k_i^\top k_j)\,v_i.$$

The first term is what I wanted; the second is cross-talk from every key not orthogonal to $k_j$. In $d$ dimensions there are at most $d$ mutually orthogonal vectors, so once a sequence is longer than the key dimension — which, at block size 1024 and head dims in the dozens, is *always* — the keys cannot all be orthogonal, the store is overcapacity, and retrieval is contaminated by interference that only grows as I write more. A multiplicative gate scales the *whole* state before each write; it cannot say "the association I stored for *this* key now collides with the new key — erase *that one* and leave everything else." LAMBADA going backward under GLA is that wrongness made visible.

I propose **DeltaNet**: replace the additive Hebbian write with an *error-correcting* one, the classical delta rule of Widrow and Hoff (least-mean-squares). Treat $S$ as a little regressor meant to map $k_t$ to $v_t$, and instead of blindly adding, take one SGD step on the squared prediction error $L_t(S)=\tfrac12\|S k_t - v_t\|^2$, whose gradient is $(S k_t - v_t)k_t^\top$ — the outer product of the *residual* with the key. With rate $\beta_t$,

$$S_t = S_{t-1} - \beta_t(S_{t-1}k_t - v_t)\,k_t^\top = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top,\qquad o_t = S_t q_t.$$

Read it as a value swap: retrieve the old value $v_t^{\text{old}} = S_{t-1}k_t$, blend $v_t^{\text{new}} = \beta_t v_t + (1-\beta_t)v_t^{\text{old}}$, and replace, $S_t = S_{t-1} - v_t^{\text{old}}k_t^\top + v_t^{\text{new}}k_t^\top$ — removing the old association for *this key* and writing the new one. The write is proportional to the error $v_t - v_t^{\text{old}}$: if $S_{t-1}$ already maps $k_t$ near $v_t$, almost nothing happens; if it maps $k_t$ to a stale value, the correction is strong. Contrast the additive rule's implicit loss, the *linear* $-\langle S k_t, v_t\rangle$, whose gradient $-v_t k_t^\top$ is constant regardless of how wrong the prediction is — the no-error-correction behavior that drove the cross-talk. The quadratic loss gives gradients that scale with the error, so the rule self-corrects, and the delta-rule fast weight has long been known to have higher capacity than the Hebbian one. The scalar $\beta_t = \sigma(W_\beta x_t)\in(0,1)$ is a learned *writing strength*: at $\beta_t=1$ the old value is fully overwritten, at $\beta_t=0$ memory is untouched.

The catch is the training wall, which is why this rule has not just been trained at scale. For additive linear attention the value written at step $t$ is just $v_t$, independent of the running state, so the whole output is one masked matmul $O=(QK^\top\odot M)V$ over precomputed $V$ — matmul-rich, tensor-core-bound. The delta rule breaks that: $v_t^{\text{new}}$ is tangled with $v_t^{\text{old}}=S_{t-1}k_t$, which depends on the running state, so I cannot stack the writes ahead of time. The naive computation rolls the recurrence forward, materializing the $d\times d$ state at every step — strictly sequential, elementwise, never touching a tensor core. So I get the better write rule only if I can break the state-dependence and recover a matmul form.

Two moves do it. First, keep $S$ in vanilla linear attention's additive shape so all its machinery is reusable: claim $S_t = \sum_{i\le t} u_i k_i^\top$ for *pseudo-values* $u_i$. By induction ($u_1=\beta_1 v_1$, then expanding $S_t = S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top$ and collecting the $k_t^\top$ term),

$$u_t = \beta_t\Big(v_t - \sum_{i<t} u_i (k_i^\top k_t)\Big),$$

which matches the value-blend reading ($u_t = v_t^{\text{new}}-v_t^{\text{old}}$). So DeltaNet is vanilla linear attention with $v_i$ replaced by $u_i$, and the per-token matrix state never has to be materialized — the problem reduces to computing the $u_i$. The chunkwise form also needs the product of transition matrices $P_n=\prod_{t\le n}(I-\beta_t k_t k_t^\top)$, and the same induction gives the WY representation $P_n = I - \sum_{t\le n} w_t k_t^\top$ with $w_t = \beta_t(k_t - \sum_{i<n} w_i(k_i^\top k_n))$ — the *exact same* recurrence as $u_t$ with $k_t$ in place of $v_t$. Second, both $u_t$ and $w_t$ are still sequential within a chunk, so read the recurrence as a linear system, because it is one: the stacked rows satisfy $(I+L)W = BK$ with $B=\mathrm{diag}(\beta)$ and $L=\mathrm{tril}(\mathrm{diag}(\beta)KK^\top,-1)$, so $W = (I+L)^{-1}BK = TK$ and identically $U=TV$, with the *same*

$$T = \big(I + \mathrm{tril}(\mathrm{diag}(\beta)KK^\top,\,-1)\big)^{-1}\mathrm{diag}(\beta).$$

The sequential dependency is absorbed into one matrix $T$, and $I+L$ is unit lower-triangular so its inverse is cheap and matmul-friendly via forward substitution (the UT transform for accumulating Householder products). Now every part is a matmul — $T$ by forward substitution, $W=TK$, $U=TV$, the masked intra-chunk products, the chunk-state update — with only $L/C$ sequential steps between chunks, cost $O(LCd + Ld^2)$, recomputing chunk states in the backward to save memory: the same asymptotics and hardware profile as the GLA chunk kernel.

Stability has to be derived. The transition $M_t = I - \beta_t k_t k_t^\top$ is the identity on everything orthogonal to $k_t$ (eigenvalue 1) and scales the $k_t$ direction by $1-\beta_t\|k_t\|^2$, so I need $0\le\beta_t\|k_t\|^2\le 2$, which I get exactly by L2-normalizing the keys: with $\|k_t\|_2=1$ the contractive eigenvalue is $1-\beta_t\in[0,1]$ for $\beta_t\in(0,1)$, always stable. And at $\beta_t=1$, $M_t = I - k_t k_t^\top$ with unit $k_t$ is an orthogonal *projection*: it annihilates exactly the one-dimensional subspace spanned by $k_t$ and leaves the other $d-1$ dimensions untouched. That is targeted forgetting made literal — the content-addressed deallocation the diagonal gate could not localize, and that LAMBADA was punishing GLA for missing. So L2 normalization is not a hack; it is what makes the erase surgical. I L2-normalize $q$ and $k$, apply SiLU before the normalization (keeps sign, smooth, no hard zeroing), use $\beta_t=\sigma(W_\beta x_t)$ (one sigmoid scalar per head, negligible parameters), and add a lightweight depthwise **short convolution** (kernel 4) on the $q,k,v$ projections before the recurrence — it generalizes the shift operator and lets the layer do precise local token comparisons that pure content-addressing is bad at, cheap and empirically important. An output RMSNorm per head before the projection rounds it out.

In this task's edit surface, FLA ships `DeltaNet` with the UT-transform chunk kernel, so the edit imports it with $\texttt{hidden\_size}=1024$, $\texttt{num\_heads}=16$, $\texttt{use\_beta}=\texttt{True}$ (the learned $\beta_t$, the heart of the rule), $\texttt{use\_short\_conv}=\texttt{True}$, $\texttt{conv\_size}=4$, $\texttt{qk\_activation}=\texttt{'silu'}$ and $\texttt{qk\_norm}=\texttt{'l2'}$ (the SiLU-then-L2 that makes the transition an exact projection at $\beta_t=1$). I take the default $\texttt{expand\_k}=1.0$, $\texttt{expand\_v}=1.0$ — symmetric width, state $d\times d$, budget matched to softmax and the RetNet rung — and the default $\texttt{use\_gate}=\texttt{False}$: DeltaNet does *not* add a swish output gate here, just the per-head output RMSNorm, a real and deliberate difference from RetNet and GLA, because the error-correcting write is already the expressivity I am buying and I want this rung to isolate the *write rule* change, not confound it with the output gate. I set `self.use_pos_emb = False` — the recurrence and short conv handle ordering. The `Block` stays the scaffold default.

So the falsifiable claim. Validation loss should drop below $2.4482$, since higher capacity and cleaner retrieval help the bulk objective too. But the real test is the recall metrics GLA stalled or regressed on: **LAMBADA must come down hard** — $84.73$ under GLA, $82.36$ under RetNet, and if the error-correcting write does what I derived it should fall well below both. WikiText-2 should tighten below $64.32$, and HellaSwag (stuck at $31.1$ across both prior rungs) and ARC-Easy should rise, since both reward holding specific earlier content. The whole construction predicts DeltaNet is the strongest of the three rungs precisely on the recall axis the diagonal gate could never address.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — DeltaNet (delta-rule linear attention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import DeltaNet
        self.attn = DeltaNet(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            use_beta=True,                # learned writing strength beta_t (the delta rule)
            use_short_conv=True,
            conv_size=4,                  # depthwise short conv for local token comparison
            qk_activation='silu',
            qk_norm='l2',                 # L2-normalized keys -> transition is a projection at beta=1
        )
        self.use_pos_emb = False          # DeltaNet handles sequence ordering internally

    def forward(self, x):
        o, _, _ = self.attn(x)
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
