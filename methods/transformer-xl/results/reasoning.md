I start from the language-modeling objective, not from a new architecture. I have a token stream and I factorize it as $P(\mathbf{x})=\prod_t P(x_t\mid x_{<t})$. The only representation that matters for the next-token distribution is the representation of the left context. If the information that fixes the next token appeared hundreds of tokens ago, then the model either carries that information forward or it guesses from a local surrogate.

The recurrent solution looks attractive because a state can, in principle, cross every boundary in the stream. But I cannot ignore the training dynamics. Backpropagation through a long recurrent chain keeps multiplying Jacobians, and the useful signal either shrinks away or becomes unstable. LSTM gates make this much less brittle than a plain RNN, and clipping controls explosions, but the empirical picture is still that a strong LSTM language model uses only about two hundred tokens of context on average. That is enough to show that the architectural promise and the effective behavior are different things.

Self-attention fixes the gradient-path problem in a different way. If a token at position $i$ attends directly to a token at position $j$, the path between them is one attention edge, not $|i-j|$ recurrent transitions. A stack of masked attention layers should therefore be a good language model for long dependencies, provided the token at $i$ is actually allowed to see far enough back.

The usual training regime quietly removes that advantage. I take a long corpus, cut it into manageable length-$L$ segments, and train the decoder independently inside each segment. Now no forward information crosses a segment boundary, and no backward information crosses either. The maximum dependency length is therefore bounded by $L$, regardless of how good self-attention would be if the earlier tokens were visible. The first few tokens of every segment have another problem: they are predicted with little or no left context because the true history sits just outside the artificial cut. This is context fragmentation, and it is not a property of language; it is a property of the batching scheme.

Evaluation exposes the same waste from another angle. If I want each prediction from a fixed-segment model to use a full training-length context, I slide the window by one token and re-encode the whole window from scratch. Almost all hidden states I compute at one step are recomputed at the next step. So the fixed segment creates three failures at once: a hard dependency cap, fragmented predictions near each boundary, and expensive repeated evaluation.

The natural question is whether I can reuse what I have already computed. When I process segment $\tau$, every layer produces a sequence of hidden states. Those states are an encoded representation of the past segment. When I process segment $\tau+1$, I can prepend the previous segment's hidden sequence to the current hidden sequence before forming keys and values. The current segment still supplies the queries, because I only need outputs for current positions, but the key/value set now contains the cached history as well as the current tokens. This is one step beyond the recurrent truncated-BPTT trick, where the previous segment hands forward a single summary vector: here I hand forward an entire sequence of states, and attention can address whichever old position it actually needs rather than reading one collapsed summary.

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
The query is current-only; the keys and values are extended-context objects.

I should pin down how far back this actually reaches, because the recurrence is between adjacent segments only, and it is not obvious that adjacent-only links buy me a long horizon. The dependency path is not an RNN path. A layer-$n$ state in segment $\tau+1$ reads keys built from the layer-$(n-1)$ states of segment $\tau$. So one cross-segment hop costs one layer: to reach one segment further back, I must descend one layer. Let me just trace it. Start at the top-layer output, layer $N$, in the current segment, and walk the cheapest path backward. Each hop consumes a layer and steps one segment earlier; I run out of layers after $N$ hops. With $N=6$ layers and segment-scale memory $L=4$, starting from segment $10$ I reach segments $9,8,7,6,5,4$ — six segments, exactly $N$, before layer $0$ stops me. The earliest token reachable is therefore about $N$ segments back, i.e. on the order of $N\cdot L = 6\cdot 4 = 24$ tokens, not $L=4$. So the per-segment recurrence does compound across the depth of the stack: the reachable dependency grows as $O(NL)$, not $O(L)$. If I cache $M$ old states rather than exactly one segment, the visible key/value span at a layer is $M+L$, and the same layer-down recurrence sets the multi-segment horizon.

This recurrence solves the memory problem, but I notice it breaks the position signal. The standard Transformer adds an absolute positional vector $\mathbf U_i$ to the input at position $i$, with $i$ running $1,\dots,L$ inside each segment. So token $j$ of the previous segment was encoded with $\mathbf U_j$, and token $j$ of the current segment was *also* encoded with $\mathbf U_j$ — the same vector — even though they are $L$ stream positions apart. The moment I concatenate cached keys with current keys, a query sees two keys carrying the identical positional marker and has no way to tell which is older. Absolute segment-local positions collide under reuse. This is not a side issue I can patch later; it is created by the very caching that the method depends on.

So the position signal has to be measured relative to the query. For a current query at stream coordinate $i$ and a key at stream coordinate $j$, the causal distance is $i-j\ge 0$. In the cached layout the raw column index is not the stream coordinate: if the key matrix is memory columns followed by current columns and the memory length is $M$, then key column $j$ sits at stream offset $j-M$, so the actual distance is $M+i-j$. After indexing old positions on the same time axis as the current segment I can write this compactly as $\mathbf R_{i-j}$. The sign matters: the distance counts how far back the key is, and future keys are masked.

I do not want to add relative position only once at the input. The decision "where should this query attend" is made in every attention layer, so the relative-distance signal belongs in the score itself. To find the right relative score, I take the absolute-position score apart and see which pieces survive. With input embeddings $\mathbf E_{x_i}+\mathbf U_i$ and $\mathbf E_{x_j}+\mathbf U_j$, the score $A_{i,j}=(\mathbf E_{x_i}+\mathbf U_i)^\top \mathbf W_q^\top \mathbf W_k (\mathbf E_{x_j}+\mathbf U_j)$ is bilinear, so it expands into four terms with nothing left over:
$$
A^{\mathrm{abs}}_{i,j}
= \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf E_{x_j}
+ \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf U_j
+ \mathbf U_i^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf E_{x_j}
+ \mathbf U_i^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf U_j .
$$
I checked this is an identity and not a convenient approximation by plugging in random $d=4$ vectors and matrices: the full score and the sum of the four terms agree to floating point ($4.0663$ both ways). The four terms read as content-to-content, content-to-key-position, query-position-to-content, and query-position-to-key-position.

Now I edit the terms. The key-side absolute position $\mathbf U_j$ has a clean replacement: it becomes a relative sinusoid $\mathbf R_{i-j}$, or $\mathbf R_{M+i-j}$ in the extended-context indexing. That substitution touches the second and fourth terms.

The query-side absolute position $\mathbf U_i$ is the awkward one. Under a relative scheme the query is the origin from which distances are measured, so there is no absolute query identity left to encode. The projected query-position vector $\mathbf U_i^\top\mathbf W_q^\top$ that appears in the third and fourth terms therefore has nothing to depend on, and I replace it with a learned position-independent vector. It plays two roles — one against a content key, one against a location key — so I use two vectors, $u$ for the content-key term and $v$ for the location-key term. I also split the key projection: hidden states and sinusoids are different kinds of input, so the content key gets $\mathbf W_{k,E}$ and the relative location gets $\mathbf W_{k,R}$. That gives
$$
A^{\mathrm{rel}}_{i,j}
= \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_{k,E}\mathbf E_{x_j}
+ \mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_{k,R}\mathbf R_{i-j}
+ u^{\top}\mathbf W_{k,E}\mathbf E_{x_j}
+ v^{\top}\mathbf W_{k,R}\mathbf R_{i-j}.
$$
The first term still addresses content. The second lets the query content prefer some distances. The third is a global content prior. The fourth is a global distance prior. One choice I want to be careful about: I keep $\mathbf R$ a fixed sinusoid and do not fold $\mathbf W_{k,R}\mathbf R$ into a single learned per-distance table. The reason is extrapolation — a sinusoid has a defined value at any distance, including distances longer than any seen in training, whereas a learned table has no row for an unseen distance. Since the whole point of caching is to let evaluation memory grow past the training segment, the distance representation has to be defined out there, so the fixed sinusoid is not a cosmetic choice.

In an actual layer I should write the score with hidden states rather than raw word embeddings, since higher layers attend with $\mathbf h^{n-1}$. With
$$
\mathbf q_i^n=\mathbf h_i^{n-1}{\mathbf W_q^n}^{\top},\qquad
\mathbf k_j^n=\widetilde{\mathbf h}_j^{n-1}{\mathbf W_{k,E}^n}^{\top},\qquad
\mathbf r_{\delta}^n=\mathbf W_{k,R}^n\mathbf R_\delta,
$$
the four terms collapse to
$$
A^n_{i,j}=(\mathbf q_i^n+u)^\top\mathbf k_j^n+(\mathbf q_i^n+v)^\top\mathbf r_\delta^n,
$$
where $\delta$ is the nonnegative causal distance from query to key. Then I scale by $1/\sqrt{d_{\text{head}}}$, apply the causal mask, softmax over keys, take the weighted sum of values, project the heads, add the residual, normalize, and pass through the feed-forward block. The scale is the standard dot-product scale: without it the variance of a head dot product grows with the head dimension and the softmax gets too sharp.

There is one implementation issue left. The location term needs $\mathbf r_\delta^n=\mathbf W_{k,R}^n\mathbf R_\delta$ for every query-key pair, but there are only $M+L$ distinct causal distances in a segment with memory, so I should project the $M+L$ sinusoids once and then place each into its $(i,j)$ slot. The placement is the part I do not trust until I trace it, because the projected stack is indexed by distance and the score matrix is indexed by key column, and those indices do not line up. The standard implementation handles this with a pad-reshape-slice that goes by the name `_rel_shift`, and I want to confirm it actually maps distances to the right key columns rather than just looking plausible.

So I trace it on a small case: a current segment of $L=3$ queries with memory $M=2$, so $klen=5$ keys (columns $j=0,1$ are memory, $j=2,3,4$ are current). The relative sinusoids are built in *descending* distance order, $\mathrm{pos\_seq}=[4,3,2,1,0]$, so column $d$ of the projected stack carries distance $4-d$. I label each entry of the query-by-relative matrix $\mathbf{bd}$ by which stack column $d$ it holds; before the shift every query row is identical, $[0,1,2,3,4]$. After `_rel_shift` the rows come out
$$
\begin{pmatrix}2&3&4&0&0\\1&2&3&4&0\\0&1&2&3&4\end{pmatrix}.
$$
Now I check the entries against what they *should* be. Query row $i$ (a current position, stream coordinate $M+i$) attending to key column $j$ wants causal distance $\delta=(M+i)-j$. Take $i=1$ (stream coord $3$): the row reads $[1,2,3,4,0]$, i.e. stack columns carrying distances $[3,2,1,0,4]$. The desired distances are $(M+i)-j = 3,2,1,0,-1$ for $j=0\ldots4$. Entries $j=0,1,2,3$ match exactly: $3,2,1,0$. Entry $j=4$ has desired distance $-1$, a future key — and it landed on the stale column carrying distance $4$, which is fine because the causal mask sets that slot to $-\infty$ before the softmax. I ran the same check across all three rows: every slot with $\delta\ge0$ carries precisely the right distance's sinusoid, and every $\delta<0$ slot lands on a column that the mask kills. So `_rel_shift` is the correct alignment, not just a convenient reshape. It removes the per-pair relative projection and the quadratic relative-key tensor while leaving the ordinary $L(M+L)$ score matrix intact.

Stepping back, each part of the design earned its place by closing a specific failure. Caching hidden states and detaching them turns the forward pass into one that sees history while keeping the backward pass a truncated, local problem — and the layer-down trace shows this reaches $O(NL)$ rather than $O(L)$. The relative score was not an independent stylistic preference: the caching itself made segment-local absolute positions collide, and only re-expressing the score in query-relative distances resolves the collision. The fixed sinusoid, in turn, is what lets the relative score keep working when evaluation memory runs past the training length. The three pieces are coupled — the recurrence is incoherent without the relative score, and the relative score is most valuable precisely because the recurrence wants a long, extrapolatable memory.
