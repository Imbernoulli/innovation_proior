I start from the language-modeling objective, not from a new architecture. I have a token stream and I factorize it as $P(\mathbf{x})=\prod_t P(x_t\mid x_{<t})$. The only representation that matters for the next-token distribution is the representation of the left context. If the information that fixes the next token appeared hundreds of tokens ago, then the model either carries that information forward or it guesses from a local surrogate.

The recurrent solution looks attractive because a state can, in principle, cross every boundary in the stream. But I cannot ignore the training dynamics. Backpropagation through a long recurrent chain keeps multiplying Jacobians, and the useful signal either shrinks away or becomes unstable. LSTM gates make this much less brittle than a plain RNN, and clipping controls explosions, but the empirical picture is still that a strong LSTM language model uses only about two hundred tokens of context on average. That is enough to show that the architectural promise and the effective behavior are different things.

Self-attention fixes the gradient-path problem in a different way. If a token at position $i$ attends directly to a token at position $j$, the path between them is one attention edge, not $|i-j|$ recurrent transitions. A stack of masked attention layers should therefore be a good language model for long dependencies, provided the token at $i$ is actually allowed to see far enough back.

The usual training regime quietly removes that advantage. I take a long corpus, cut it into manageable length-$L$ segments, and train the decoder independently inside each segment. Now no forward information crosses a segment boundary, and no backward information crosses either. The maximum dependency length is therefore bounded by $L$, regardless of how good self-attention would be if the earlier tokens were visible. The first few tokens of every segment have another problem: they are predicted with little or no left context because the true history sits just outside the artificial cut. This is context fragmentation, and it is not a property of language; it is a property of the batching scheme.

Evaluation exposes the same waste from another angle. If I want each prediction from a fixed-segment model to use a full training-length context, I slide the window by one token and re-encode the whole window from scratch. Almost all hidden states I compute at one step are recomputed at the next step. So the fixed segment creates three failures at once: a hard dependency cap, fragmented predictions near each boundary, and expensive repeated evaluation.

The natural question is whether I can reuse what I have already computed. When I process segment $\tau$, every layer produces a sequence of hidden states. Those states are an encoded representation of the past segment. When I process segment $\tau+1$, I can prepend the previous segment's hidden sequence to the current hidden sequence before forming keys and values. The current segment still supplies the queries, because I only need outputs for current positions, but the key/value set now contains the cached history as well as the current tokens.

I have to stop the gradient through the cached states. If I let gradients flow from the current segment into the previous segment, and then into the segment before that, I have recreated unbounded backpropagation through time inside a Transformer. The useful bargain is the truncated-BPTT bargain: the forward pass sees history, while the backward pass stays local. For two consecutive length-$L$ segments, with layer-$n$ states $\mathbf{h}_\tau^n$, the recurrence is
$$
\widetilde{\mathbf{h}}_{\tau+1}^{n-1}
= [\,\mathrm{SG}(\mathbf{h}_\tau^{n-1}) \circ \mathbf{h}_{\tau+1}^{n-1}\,],
$$
$$
\mathbf{q}_{\tau+1}^n=\mathbf{h}_{\tau+1}^{n-1}{\mathbf W_q^n}^{\top},\qquad
\mathbf{k}_{\tau+1}^n=\widetilde{\mathbf{h}}_{\tau+1}^{n-1}{\mathbf W_k^n}^{\top},\qquad
\mathbf{v}_{\tau+1}^n=\widetilde{\mathbf{h}}_{\tau+1}^{n-1}{\mathbf W_v^n}^{\top}.
$$
The query is current-only; the keys and values are extended-context objects. This is richer than recurrent truncated BPTT because I am not passing one state vector forward. I am passing a sequence of states, and attention can address the specific old position it needs.

The dependency path is not exactly the same as an RNN path. A layer-$n$ state in segment $\tau+1$ depends on a layer-$(n-1)$ state from segment $\tau$. Each segment step backward also moves one layer downward. With $N$ layers and memory on the scale of one segment, the largest reachable dependency grows on the order of $N L$. If I cache $M$ old states rather than exactly one segment, the visible key/value span at a layer is $M+L$, and the same layer-down recurrence determines how far information can propagate through repeated segments.

This recurrence solves the obvious memory problem, but it creates a positional problem. The standard Transformer adds an absolute positional vector $\mathbf U_i$ to the input at position $i$. If every length-$L$ segment is encoded with $\mathbf U_1,\dots,\mathbf U_L$, then token $j$ in the previous segment and token $j$ in the current segment carry the same positional marker even though they are $L$ stream positions apart. Once I concatenate cached and current states, the model needs to know that one key is older than the other. Absolute segment-local positions make those keys collide.

So the position signal has to be relative to the query. For a current query at stream coordinate $i$ and a key at stream coordinate $j$, the causal distance is $i-j\ge 0$. If the key matrix is stored as memory columns followed by current columns, the raw column index is not the stream coordinate; a memory length $M$ means key column $j$ corresponds to stream offset $j-M$, so the actual distance is $M+i-j$. I can write the same idea compactly as $\mathbf R_{i-j}$ after indexing old positions on the same time axis as the current segment. The sign is important: in causal language modeling the distance counts how far back the key is, and future keys are masked.

I do not want to add relative position only once at the input. The decision "where should this query attend" is made in every attention layer, so the relative-distance signal belongs in the attention score. To derive the right score, I take the absolute-position score apart. With input embeddings $\mathbf E_{x_i}+\mathbf U_i$ and $\mathbf E_{x_j}+\mathbf U_j$,
$$
A^{\mathrm{abs}}_{i,j}
= \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf E_{x_j}
+ \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf U_j
+ \mathbf U_i^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf E_{x_j}
+ \mathbf U_i^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf U_j .
$$
These four terms say content-to-content, content-to-key-position, query-position-to-content, and query-position-to-key-position. The key-side absolute position $\mathbf U_j$ has a clean replacement: it becomes a relative sinusoid $\mathbf R_{i-j}$, or $\mathbf R_{M+i-j}$ when I write indices as current row and extended-context column. This replacement appears in the second and fourth terms.

The query-side absolute position is trickier. Under a relative scheme, the query is the origin from which distances are measured. There is no useful absolute query id left in the score. The old projected query-position vector $\mathbf U_i^\top\mathbf W_q^\top$ should therefore be replaced by a learned position-independent vector. It plays two roles, so I use two vectors: $u$ for the content-key term and $v$ for the location-key term. I also separate the content key projection from the relative-location projection because hidden states and sinusoids are different inputs. That gives
$$
A^{\mathrm{rel}}_{i,j}
= \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_{k,E}\mathbf E_{x_j}
+ \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_{k,R}\mathbf R_{i-j}
+ u^{\top}\mathbf W_{k,E}\mathbf E_{x_j}
+ v^{\top}\mathbf W_{k,R}\mathbf R_{i-j}.
$$
The first term still addresses content. The second says that the query content can prefer some distances. The third is a global content prior. The fourth is a global distance prior. Keeping the relative vector $\mathbf R$ as a fixed sinusoid matters because the model can then form encodings for distances longer than those used during training; if I collapse $\mathbf W_{k,R}\mathbf R$ into a learned table by distance, unseen distances have no structured representation.

In an actual layer I should no longer write the score in terms of raw word embeddings, because higher layers attend with hidden states. With
$$
\mathbf q_i^n=\mathbf h_i^{n-1}{\mathbf W_q^n}^{\top},\qquad
\mathbf k_j^n=\widetilde{\mathbf h}_j^{n-1}{\mathbf W_{k,E}^n}^{\top},\qquad
\mathbf r_{\delta}^n=\mathbf W_{k,R}^n\mathbf R_\delta,
$$
the score is
$$
A^n_{i,j}=(\mathbf q_i^n+u)^\top\mathbf k_j^n+(\mathbf q_i^n+v)^\top\mathbf r_\delta^n,
$$
where $\delta$ is the nonnegative causal distance from query to key. Then I scale by $1/\sqrt{d_{\text{head}}}$, apply the causal mask, softmax over keys, take the weighted sum of values, project the heads, add the residual, normalize, and pass through the feed-forward block. The scale is the standard dot-product-attention scale: without it, the variance of a head dot product grows with the head dimension and makes the softmax too sharp.

There is one more implementation wall. If I explicitly compute $\mathbf W_{k,R}\mathbf R_\delta$ for every query-key pair, I repeat the same projection many times. There are only $M+L$ possible causal distances for one segment with memory, so I should project the $M+L$ relative sinusoids once. Stack them in descending distance order, multiply by $\mathbf W_{k,R}^{\top}$, compute the dense query-by-relative matrix, then shift each row so relative-distance columns line up with key columns. This does not remove the ordinary $L(M+L)$ attention-score matrix; it removes the per-pair relative projection and the quadratic intermediate relative-key tensor. The reshape-and-shift trick in the reference implementation is exactly this alignment operation.

The final design is therefore forced by the failures of fixed segments. I reuse cached hidden states so the forward pass has history and evaluation can reuse representations. I detach those cached states so training stays a truncated problem. I abandon segment-local absolute positions because they collide under reuse. I replace them with relative sinusoidal distances inside the attention score, with two learned global biases and separate content/location key projections. The method is not just "add memory" and not just "use relative position"; the recurrence only becomes coherent once the score is expressed in relative coordinates, and the relative score is useful for long evaluation memory because its distance representation is fixed and extrapolatable.
