A causal decoder-only Transformer trained on copy-family tasks — emit the input after a delimiter, repeat it twice, or reverse it — fits the training lengths to near-perfect exact match and then, the moment I push the input past the lengths it saw in training (content sampled from $[1, 20]$, evaluated out to $2L = 40$), either keeps working or collapses depending on a single architectural choice: how I tell the model about token order. The whole difficulty of length generalization here funnels into the positional encoding, and the trouble is that every standard option fails extrapolation in a different way. Sinusoidal absolute encoding $p_j = [\sin(\omega_1 j), \cos(\omega_1 j), \dots]$ with $\omega_i = 1/10000^{2i/d}$ is closed-form and therefore *defined* at position 35 even when training stopped at 20, but being defined is not the same as working: attention learns to read the $p_j$ for $j \le 20$, the vectors for $j > 20$ are points the model never had to interpret, and the periodic code lets a far position alias a near one along some frequency, so it latches onto absolute indices it saw and stumbles past them. Learned absolute embeddings are strictly worse — there is no vector for position 21 at all, so the window is simply capped. Relative schemes are more promising because the rule for copying ("attend $k$ steps back") is length-independent, but each is a fixed prescription with its own failure shape: T5's bucketed bias $b_{\text{bucket}(t-i)}$ generalizes precisely because the log-bucketing folds unseen large distances onto a trained parameter, yet it adds a learned term to *every* attention score — roughly doubling training and inference time versus plain absolute PE — and freezes a hand-designed distance table into the architecture; rotary rotations make the dot product depend only on $i-t$ but fix the rotation frequencies in advance; and ALiBi's $q_t^\top k_i - (t-i)\,m_h$ extrapolates language-model perplexity because the linear penalty is defined at any distance, but it hard-codes a monotone recency bias, exactly the wrong shape for a task that must sometimes attend from the *end* of the output back to the *start* of the input. The deeper problem is that in-distribution accuracy cannot adjudicate between any of these — at training length they all reach near-perfect scores, and perplexity is known not to track downstream performance — so the only signal lives in the longer-than-trained regime, and there each prescribed inductive bias caps how position can be used wherever its fixed shape mismatches the task.

The premise worth questioning is that a Transformer with no position signal cannot tell $a\,b$ from $b\,a$. That is true for a bidirectional *encoder*, whose unmasked self-attention is genuinely permutation-invariant and collapses to a bag of words, but my model is a *decoder* with a causal mask, and a causal mask is not a neutral implementation detail. The query at position $t$ attends to exactly the $t$ positions $\le t$: position 1 sees a window of size 1, position 2 a window of size 2, so the *size of the visible set is the position itself*. Masking injects order. This reframes the whole search: rather than hunt for a better prescription, remove the prescription. I propose NoPE — a decoder-only Transformer trained with no explicit positional encoding of any kind: no absolute embedding added to the input, no additive relative bias on the scores, no rotation of queries and keys. The attention score is exactly $q_t^\top k_i$ under the causal mask, and the order signal comes entirely from the mask. This removes the fixed inductive bias that was the source of every failure mode above, it is free because the score carries no extra term to add or rotate, and it is the minimal edit to the architecture — delete the $+\,p_j$ at the input and add nothing to the scores.

What makes this more than a hope is that the architecture can be shown, by explicit hand-built weights, to *represent* position with nothing but the mask, so gradient descent has a real target to find. Take absolute position first, in layer 1, where the hidden state is just $H^{(0)} = W_E X$ with no position added. Reserve three coordinates of the residual stream: design the embedding so coordinate 1 is $1$ for every token, coordinate 2 is $1$ if and only if the token is $\texttt{<bos>}$ (which sits at position 1), and coordinate 3 is $0$. Let one head's key projection read coordinate 1, which is $1$ for all tokens, so every key is identical; then for a query at position $t$ every visible logit $\langle q_t, k_i\rangle$ is the same value, and softmax over identical scores on a causally-masked window of size $t$ is uniform, $\hat\alpha_i = 1/t$ for each $i \le t$. The mask did the counting, the softmax turned the count into a number. Let the value projection read coordinate 2, which is $1$ only at $\texttt{<bos>}$, so the attention output is $\sum_{i \le t} \hat\alpha_i v_i = (1/t)\,e_1$ — only $\texttt{<bos>}$ contributes, weighted by $1/t$ — and let $W_O$ copy that into coordinate 3. After layer 1, $h_t^{(1)}$ carries exactly $1/t$: the $\texttt{<bos>}$ anchor fixes the numerator, the causal window size fixes the denominator. It is $1/t$ rather than $t$, but that is a faithful injective code (strictly decreasing in $t$), and the GELU MLP that follows is a universal approximator that can map $1/t \mapsto t$; layer norm is bypassed by the standard residual-stream argument since only three carried coordinates are needed.

That handles absolute position, but absolute indices are exactly what fail to extrapolate; the signal that generalizes is relative distance, so the real question is whether a later layer can convert the absolute index now sitting in coordinate 3 into a score that depends on $t - i$. With coordinate 1 $=1$, coordinate 2 the $\texttt{<bos>}$ indicator, and coordinate 3 the position $t$, all preserved across the residual stream, choose $W_Q$ and $W_K$ in a layer $l \ge 2$ so that the query reads out $[1, -t, \dots]$ (row 1 reads coordinate 1, giving $1$; row 2 reads coordinate 3 with coefficient $-1$, giving $-t$) and the key reads out $[i, 1, \dots]$ (row 1 reads coordinate 3, giving $i$; row 2 reads coordinate 1, giving $1$), with the remaining content rows reading only the ordinary content coordinates so the leftover term is genuinely content-only. Then the logit factors cleanly:

$$\langle q_t, k_i\rangle = 1\cdot i + (-t)\cdot 1 + \sum_{j\ge 3} q_{t,j}\,k_{i,j} = f_{\text{cnt}}(q_t, k_i) - (t - i).$$

The score splits into a content interaction plus a pure function of the relative offset, $f_{\text{rel}}(t-i) = -(t-i)$. The sign is right for a causal decoder: since $t \ge i$ the positional term is non-positive and grows more negative as the key recedes into the past. This proves the reachable case $f_{\text{rel}}(d) = -d$; richer relative functions would need additional position features and matching query-key weights, not one scalar coordinate, so I do not claim every bias table is expressible — only that the architecture can synthesize both an absolute code in layer 1 and a relative score in later layers. Because it can *represent* both without my hand-designing either, SGD is free to learn whichever encoding the task rewards. There is no free lunch in the IID regime — which is exactly why IID metrics cannot separate the schemes — but in the OOD regime, removing a mismatched prior is what helps. Comparing learned attention distributions through the average Jensen–Shannon divergence of per-position distributions, $D_{\text{AT}}(P, Q) = \frac{1}{T}\sum_t D_{\text{JSD}}(P_t \,\|\, Q_t)$, with a per-layer model distance taken as the minimum over head pairs, the no-PE model's attention sits closest to the T5 relative-bias scheme and shows bimodal short-and-long-range attention — attending both to the local decoding context and far back into the input — which is precisely what copy and reverse need, and unlike ALiBi it is not forced into monotone recency. And it pays zero extra attention compute: no bias matrix, no rotation, strictly less work than T5 or rotary.

So the method lands as a plain decoder-only Transformer with the positional encoding deleted. In the harness the positional scheme supplies none of its hooks and the model is the bare causal backbone — token embedding, multi-head causal attention with score $q_t^\top k_i$ and the causal mask, residual connections, layer norm, a GELU MLP with $4\times$ expansion, and the autoregressive LM loss over the target positions. The causal mask carries the order; the network builds whatever position representation the task demands.

```python
import torch.nn as nn


def build_positional_scheme(config) -> PositionalScheme:
    """No positional encoding: none of the three hooks is supplied.

    The decoder's causal mask carries order on its own. Layer 1 can recover
    absolute position (uniform attention over t identical keys -> 1/t,
    anchored at <bos>); later layers can express a relative score depending
    on (t - i). SGD learns which signal to use; no positional params are added.
    """
    return PositionalScheme(
        name="nope",
        token_embedding_extra=None,  # no absolute position added to embeddings
        attn_bias=None,              # no T5/ALiBi additive score bias
        rotary=None,                 # no rotation of q, k
        extra_modules=nn.ModuleList(),
    )


def build_model(config) -> nn.Module:
    """Decoder-only Transformer; the score is exactly q_t^T k_i under the mask."""
    scheme = build_positional_scheme(config)
    return SeqModel(config, scheme, use_lstm=False)
```

Equivalently, in a standard HuggingFace-style decoder-only attention block, the positional branch is empty — the score is computed and only the causal mask is added before softmax:

```python
# inside a causal self-attention head, NoPE branch
scores = torch.matmul(query_states, key_states.transpose(-1, -2))  # q_t^T k_i
if attention_mask is not None:
    scores = scores + attention_mask  # causal mask only; no positional term
attn = torch.softmax(scores.float(), dim=-1).type_as(scores)
out = torch.matmul(attn, value_states)
```
