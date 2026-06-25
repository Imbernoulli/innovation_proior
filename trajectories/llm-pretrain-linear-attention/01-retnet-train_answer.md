The baseline I am replacing is standard softmax attention, and the thing that hurts about it is concrete. Forming $QK^\top$ over a length-$L$ sequence builds an $L\times L$ score matrix, so both compute and memory grow like $L^2$, and at decode I must keep every past key and value around — a cache that grows linearly with how far I have generated, with per-step cost that grows with context. I want the mirror-image profile of an RNN: a fixed-size state, $O(1)$ per step at inference, constant memory, no growing cache — without giving up language-model quality against a strong softmax Transformer trained on the same data. That second clause is the entire problem, and it dictates where the ladder must start: at the *cheapest credible* subquadratic mixer, the one whose residual failure will diagnose the next rung. Plain linear attention, $S_t = S_{t-1} + k_t^\top v_t$, $o_t = q_t S_t$, is too cheap to be credible — its additive write never forgets, the state just accumulates every key it has ever seen, and that is exactly why it loses to softmax. So the floor must already carry the one fix the 1-D RNN literature calls non-negotiable: a forget mechanism, in its simplest possible form.

I propose **multi-scale retention (RetNet)**: linear attention with a single *fixed scalar* decay $\gamma\in(0,1)$ per head, $S_t = \gamma S_{t-1} + k_t^\top v_t$. The reason to use one fixed scalar rather than something richer is not laziness — it is that a scalar pulls cleanly out of the cumulative product, which keeps the entire attention-style parallel and chunkwise training machinery intact. To see that this is the right operator and not an arbitrary add-on, I build it from the recurrence, because the recurrence is where the $O(1)$ inference lives. Start fully general with a state *matrix* and a transition $A$: $s_n = A s_{n-1} + k_n^\top v_n$, $o_n = q_n s_n$. Unrolling gives $s_n = \sum_{m\le n} A^{n-m} k_m^\top v_m$, so $o_n = \sum_{m\le n} q_n A^{n-m} k_m^\top v_m$. That is the bridge: a linear recurrence, unrolled, is *already* a causal weighted sum over the whole past, every term weighted by $q_n A^{n-m} k_m^\top$ — exactly the shape of causal attention. The whole game is now in the matrix power $A^{n-m}$, which must be content-aware and cheap to compute in parallel.

Content-awareness comes from making $Q=XW_Q$, $K=XW_K$ depend on the input. For the matrix power, diagonalize $A = \Lambda\,\mathrm{diag}(\gamma e^{i\theta})\,\Lambda^{-1}$ with complex eigenvalues in polar form — a per-dimension magnitude $\gamma$ and phase $\theta$. Then $A^{n-m} = \Lambda\,\mathrm{diag}(\gamma e^{i\theta})^{n-m}\,\Lambda^{-1}$, and since $\Lambda,\Lambda^{-1}$ are learnable anyway I absorb $\Lambda$ into $W_Q$ and $\Lambda^{-1}$ into $W_K$ — the change of basis is free. Splitting the relative exponent across the two positions, $(\gamma e^{i\theta})^{n-m} = (\gamma e^{i\theta})^n (\gamma e^{i\theta})^{-m}$, and attaching each piece to its own factor gives $o_n = \sum_{m\le n} \big(q_n (\gamma e^{i\theta})^n\big)\big(k_m (\gamma e^{i\theta})^{-m}\big)^\top v_m$. The phase part is exactly rotary position embedding — $q$ gets $e^{in\theta}$, $k$ gets $e^{-im\theta}$, their product depends on $n-m$ — and with the magnitude layered on it is precisely the xPos form: a relative position encoding *with a decay*. The position encoding I would otherwise bolt on by hand falls out of the state matrix.

A per-dimension magnitude $\gamma_i$ is more bookkeeping than I want, and $\gamma^{-m}$ on the key grows unbounded as $m$ shrinks — numerically ugly — so I collapse $\gamma$ from a per-dimension vector to a single scalar per head. Then $\gamma^{n-m}$ pulls out of the per-coordinate structure entirely and only the phase rotation stays inside the factors: $o_n = \sum_{m\le n} \gamma^{n-m}(q_n e^{in\theta})(k_m e^{im\theta})^\dagger v_m$. Call this operator **retention**, and it has three equivalent faces that compute the same function. The *parallel* face packs the rotation into the projections and the decay-and-causality into one matrix $D_{nm}=\gamma^{n-m}$ for $n\ge m$ and $0$ otherwise — which performs the causal mask and the exponential decay at once — giving

$$\mathrm{Retention}(X) = (QK^\top \odot D)\,V,$$

the GPU-friendly shape with softmax simply deleted. The *recurrent* face is $S_n = \gamma S_{n-1} + K_n^\top V_n$, $o_n = Q_n S_n$, a fixed-size $d_k\times d_v$ state with $O(1)$ per step; since $S_n$ unrolls to $\sum_{m\le n}\gamma^{n-m}K_m^\top V_m$, the readout $Q_n S_n$ is exactly row $n$ of $(QK^\top\odot D)V$ — the causal mask in $D$ is the same statement as "the state only accumulates the past." Train with the matmul, infer with the recurrence, no approximation. A third *chunkwise* face runs the parallel form inside chunks and carries the state recurrently across them in linear time, which is the long-sequence training mode the FLA kernels actually implement.

A single scalar $\gamma$ fixes one decay rate, one memory timescale, but different parts of language want different horizons. With softmax I get diversity from heads in different subspaces; here I have a *second* axis, the decay rate itself. So I use $h$ heads each with its own $\gamma$, geometrically spanning the range $\gamma = 1 - 2^{-5-\mathrm{arange}(h)}$ from fast forgetting to almost none, and concatenate — multi-scale retention. This creates one wrinkle: heads with different $\gamma$ produce outputs of different magnitude (a near-1 $\gamma$ sums many terms and grows large), so I normalize each head *separately* (group norm, one group per head) before mixing, or the high-variance heads swamp the rest. And deleting softmax cost me a nonlinearity — softmax was normalizing *and* injecting nonlinearity — so I restore it with a content-dependent **output gate**: $\mathrm{MSR}(X) = (\mathrm{swish}(XW_G)\odot Y)\,W_O$, the missing expressiveness back without the $O(n)$ softmax.

In this task's edit surface I do not hand-roll the chunk kernel — FLA ships `MultiScaleRetention`, which implements exactly the three faces above — so the edit imports it and wires it into the scaffold. The live design choices are the expansion ratios: $d_k = \texttt{expand\_k}\cdot d$ and $d_v = \texttt{expand\_v}\cdot d$, and the state is $d_k\times d_v$, so these set the memory capacity. The canonical retention widens the value to $2d$ and shrinks the FFN to $2d$ to keep parameters matched to a Transformer — but the scaffold's `Block` and `MLP` are fixed at the standard $4d$ GELU FFN, so I cannot do the FFN-shrink trick, and widening the value on top of an unshrunk $4d$ FFN would *inflate* the layer past the softmax budget. The honest, parameter-conservative choice is symmetric: $\texttt{expand\_k}=1.0$, $\texttt{expand\_v}=1.0$, so $d_k=d_v=d$ and the attention block lands at roughly the softmax $4d^2$ allocation. This unwidened variant is the right floor — it isolates "does multi-scale decayed retention match softmax at matched width" without confounding it with a larger state. I keep `use_output_gate=True` and `gate_fn='swish'` (the gate is the nonlinearity), and I set `self.use_pos_emb = False`: retention's $\gamma^{n-m}$ decay plus rotary phase *is* the relative position signal, so the loop must skip its learned `wpe`, which would otherwise double-encode position and fight the decay. The `Block` stays the scaffold default — only the mixer is swapped, so any quality difference is the mixer, not the wrapper.

What makes this the floor and not the finale is that retention's decay is *fixed and data-independent*: every head has one $\gamma$, chosen a priori, and the same decay applies to every channel, context, and word. It cannot look at a token and decide "this is a key fact, hold it" versus "this is filler, let it decay." It is the cheapest credible forget gate — strictly better than no decay, which is why it is a real contender — but it is exactly the data-independent gate the gated-RNN literature warns against. So my falsifiable expectation is that retention trains stably and lands in the credible range, yet is the **weakest** rung, with the gap showing most sharply on the perplexity benchmarks and the recall-flavored downstream tasks, where holding the *right* facts beats a fixed-rate exponential average of all of them. If that is what the numbers show, the diagnosis for the next rung is already written: the decay must become *data-dependent*, without destroying the matmul chunkwise form that the fixed scalar $\gamma$ protected.

```python
# EDITABLE region 1 of nanoGPT/custom_pretrain.py (lines 33-70) — RetNet (multi-scale retention)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import MultiScaleRetention
        self.attn = MultiScaleRetention(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=1.0,                 # d_k = d  (matched to softmax width)
            expand_v=1.0,                 # d_v = d  (state d x d; no value widening)
            use_output_gate=True,         # swish output gate = the restored nonlinearity
            gate_fn='swish',
        )
        self.use_pos_emb = False          # gamma^{n-m} decay + rotary phase encode position

    def forward(self, x):
        o, _, _ = self.attn(x)            # FLA returns (output, attn_weights, past_kv)
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
