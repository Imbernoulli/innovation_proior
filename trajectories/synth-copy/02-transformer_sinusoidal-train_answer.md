The recurrent baseline came back with exactly the shape the floor predicted: in-distribution perfect on every variant (`exact_match_id = 1.0`, `token_acc_id = 1.0`), but the OOD split a wall — `exact_match_ood = 0.0` on all three, with OOD token accuracy low and decaying with difficulty ($0.371$ on `delim`, $0.207$ on `repeat`, $0.180$ on `reverse`). The decoder state is seeded from a single masked-mean summary and accumulates drift over 30–40 unanchored output steps, so one symbol slips and the sequence-level match fails even while most tokens are still right. And it is slow — 400–500 s per variant — because the recurrence is sequential and the per-step alignment is an extra matmul at every position. The diagnosis is sharp: recurrence carries order *implicitly*, and implicit order does not survive being stretched past the training length. The cure has to be an *explicit* position signal, on the parallel decoder-only Transformer the rest of the ladder shares — which also kills the recurrence cost, since attention parallelizes over positions.

But the Transformer has its own problem before it can even start, and it is the entire reason a positional scheme exists. Self-attention is a function of a *set* of key/value vectors: the output at one position is $\operatorname{softmax}(q\cdot k_j/\sqrt{d})$ weights times the values $v_j$, summed over $j$. The query is a linear function of the token at my position; each key and value is a linear function of the token at position $j$. Nowhere does the index $j$ appear — only the contents of the two token vectors. Permute the input and every $k_j, v_j$ is just relabelled, the set is identical, the weighted sum is the same number; the feed-forward sublayer acts per position and cannot see neighbors either. So a stack of these reads its input as a *bag* of token vectors — `a b` and `b a` are the same object. (The causal mask does break some of this, which a later rung exploits, but the default move, and the one I take now, is to put order back in by hand.) The recurrent baseline never needed this because the recurrence supplied order; moving to attention, I have to manufacture a position signal explicitly.

I propose **sinusoidal absolute positional encoding** (Vaswani et al. 2017). The number "this token is at position $t$" has to enter the computation in a form the existing machinery — linear layers, dot products, softmax — can consume, and all of that eats $d_{\text{model}}$-dimensional vectors, so I manufacture for each position $t$ a vector $p_t \in \mathbb{R}^{d_{\text{model}}}$ that *is* the position and fuse it with the token embedding at the bottom of the stack. I *add* it rather than concatenate: the first linear layer on a concatenation $[\text{embed};\,p]$ is $W_a\,\text{embed} + W_b\,p$, which is exactly applying one matrix to the embedding and another to the position and summing — so concatenation just buys position a private projection at the cost of extra width everywhere above, whereas plain addition $\text{embed} + p$ still lets the first projection read a linear function $W p$ of the position and widens nothing. The model carves out a subspace of the $d_{\text{model}}$ dimensions to hold position and reads it off.

What should $p_t$ be? The raw integer $t$ is unbounded — train to length 20, run at 40, and the downstream linears are fed position values they never saw, pushing activations into uncalibrated regimes. Normalizing to $t/L$ bounds it but breaks the meaning of a step: one token back is a delta of $0.1$ in a length-10 sequence and $0.01$ in a length-100 one, so "look one position back" is no longer a fixed quantity the model can learn a single rule for. I want something bounded *and* shift-consistent — where a fixed positional step always advances the code by the same amount, regardless of absolute position or length. A periodic function gives both: $\sin(\omega t)$ lives in $[-1,1]$ for all $t$, and a step of $\Delta t$ always advances the phase by $\omega\,\Delta t$. But one sinusoid aliases — a single scalar cannot separate hundreds of positions — so I use a whole vector of sinusoids at geometrically spaced frequencies, the continuous analogue of a binary counter: low bits flip fast (fine local position), high bits flip slowly (coarse global position), and the *combination* separates an enormous range while each digit stays bounded. The frequencies are $\omega_i = 10000^{-2i/d}$ for $i = 0, \dots, d/2-1$, the fastest with wavelength $2\pi$ and the slowest near $10000\cdot 2\pi$; geometric spacing tiles the scale axis evenly in log-space, giving roughly equal resolution from local to global.

One more property makes this more than a bounded counter: a fixed relative shift should be a fixed linear map. If I stored only $\sin(\omega t)$, I could not recover $\sin(\omega(t+k))$ from it — the angle-addition formula needs $\cos(\omega t)$, and at a given value of $\sin$ I do not even know whether the phase is rising or falling. So I keep both $\sin(\omega t)$ and $\cos(\omega t)$ per frequency, storing the full phase as a point on the unit circle. Then the shift $t \to t+k$ is, for each frequency, the orthogonal rotation by $\omega k$, and stacking those rotations block-diagonally gives $p_{t+k} = M_k\, p_t$ — a single linear transformation that depends only on the offset $k$ and is identical at every absolute $t$. That is why the layout interleaves sine and cosine per frequency:
$$\text{PE}(t, 2i) = \sin(t\,\omega_i), \qquad \text{PE}(t, 2i+1) = \cos(t\,\omega_i).$$
And it is why I can hope for extrapolation at all: $\sin$ and $\cos$ are defined for every real $t$, so position 30 is not a missing table slot the way the scaffold default's learned table is — it is the next point along the same curves, and the rotation relation still holds there with the same $M_k$. That is the direct answer to rung one's failure mode: where recurrence had no anchor and the learned table had no entry, the sinusoid has a closed-form value everywhere.

I have to be honest about the catch, because it is precisely what this rung exists to test. "Defined everywhere" is *not* "usable everywhere." During training the attention learns to interpret the particular phase combinations that occur over $[0,20)$; past 20, the joint pattern of all those sinusoids across the dimensions is a configuration the model never encountered. The encoding function does not die, but the model's *learned interpretation* of it only covers the training range. So this is a diagnostic rung: if even a closed-form, extrapolation-friendly absolute code fails OOD, that is strong evidence the problem is *absoluteness itself*, not the particular code.

Concretely I fill `build_positional_scheme` to build the $[\text{max\_total\_len}, d_{\text{model}}]$ table in closed form — `pe[:, 0::2] = sin(positions·div_term)`, `pe[:, 1::2] = cos(positions·div_term)`, with `div_term = exp(arange(0,d,2)·(−ln 10000 / d))` computed in log space rather than by repeated `pow` — wrap it in a *frozen* `nn.Embedding.from_pretrained(pe, freeze=True)` so it carries no gradient and no learnable parameter, register it through `scheme.extra_modules` so it moves to the GPU, and return only the `token_embedding_extra` hook with `attn_bias` and `rotary` left `None`. `build_model` returns the plain `SeqModel(use_lstm=False)`, so the decoder-only backbone is back and the recurrence is gone. Two harness-specific differences from the textbook recipe matter. First, there is **no $\sqrt{d_{\text{model}}}$ embedding scaling**: the canonical scheme multiplies the token lookup by $\sqrt{d_{\text{model}}}$ to put content and position on comparable scales before the sum, but this scaffold's `SeqModel` adds `token_embedding_extra(positions)` straight onto the raw `token_embed` output, so I do not control that scaling and do not introduce it — the learned token embedding simply has to grow on its own to balance the $O(1)$ sinusoids. Second, there is **no dropout** on the sum (the task fixes `dropout = 0.0`). Positions index the *whole* stream $[0,T)$, not a separate source/target indexing, with a `clamp` to the table's last row as a safety bound; the table is sized to `max_total_len = 256`, comfortably past $2\cdot L_{\text{train}}$, so the clamp never actually fires on this task.

My falsifiable expectations against the rung-one numbers are precise. In-distribution should stay perfect on all three variants, and elapsed should drop sharply from 400–500 s to roughly the other Transformer rungs' ~130 s. The bet is OOD, and the prediction is specific: if absoluteness is the disease, sinusoidal should *not* rescue OOD exact match — I expect `exact_match_ood` at or near $0.0$ on all three, possibly with OOD token accuracy even *worse* than the LSTM's $0.37/0.21/0.18$, because the out-of-range phase patterns actively mislead the attention rather than merely under-anchoring it. Perfect ID, zero OOD, OOD token accuracy at or below rung one would make the lesson unambiguous: the failure is not "which absolute code," it is *absolute position at all* — and the move is to stop prescribing an absolute code and let the model build a relative one.

```python
# EDITABLE region of custom_strategy.py (lines 301-332) -- step 2: sinusoidal absolute PE
def build_positional_scheme(config: TaskConfig) -> PositionalScheme:
    """Sinusoidal absolute positional encoding (Vaswani et al., 2017)."""
    max_len = config.max_total_len
    d_model = config.d_model
    pe = torch.zeros(max_len, d_model)
    positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)

    extras = nn.ModuleList()
    table = nn.Embedding.from_pretrained(pe, freeze=True)
    extras.append(table)

    def token_embedding_extra(positions: torch.Tensor) -> torch.Tensor:
        return table(positions.clamp(max=max_len - 1))

    return PositionalScheme(
        name="sinusoidal",
        token_embedding_extra=token_embedding_extra,
        attn_bias=None,
        rotary=None,
        extra_modules=extras,
    )


def build_model(config: TaskConfig) -> nn.Module:
    """Decoder-only Transformer with sinusoidal APE."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```
