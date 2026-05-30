OK, let me think this through from scratch. I want a language model that is genuinely good at long-range dependency — the kind of prediction where the thing that determines the next token showed up hundreds or thousands of tokens ago. I'm factorizing $P(\mathbf{x})=\prod_t P(x_t\mid x_{<t})$ and modeling each conditional with a neural net that encodes the left context into a hidden state, multiplies by the embeddings, softmaxes. So the whole game is: how good is my encoding of $x_{<t}$, and in particular how far back does it actually reach.

The recurrent route is the obvious one — an LSTM carries a state forward step by step, and in principle the state at time $t$ has seen everything before it, so there's no architectural cap on dependency length. But I know how this goes in practice. Backprop through time multiplies a long chain of Jacobians; the product vanishes (or explodes), so the gradient that would carry credit from a token now back to its cause a thousand steps ago is essentially zero. Gating helps the forward signal survive and clipping stops the explosions, but the decay is still there. And there's a hard empirical fact I can't argue with: a trained LSTM-LM uses on average only about 200 words of context — feed it more history and the predictions barely move. So the standard recurrent model is, measured directly, a short-context model. There's a lot of room above 200.

Self-attention is the structural fix for *that* particular problem. In an attention layer the representation of position $i$ is a weighted sum over all positions $j$, weight $\propto q_i\cdot k_j$. Any two positions are one hop apart. So the gradient path between a token and its distant cause is constant length, not growing with the gap — the vanishing-gradient pressure that caps the LSTM is just gone. A deep stack of masked attention layers should, on paper, reach much farther back than 200 tokens.

So why doesn't it, in the way people actually train it? Here's the regime. The corpus is enormous; I can't run attention over the whole thing — attention is quadratic in length and I have finite memory. So the standard move is to cut the corpus into separate fixed-length segments of a few hundred tokens and train the model independently inside each segment, with nothing flowing across the boundaries — not in the forward pass, not in the backward pass. Each segment is its own little universe.

Stare at this for a second, because it quietly throws away the entire reason I switched to attention. The longest dependency the model can possibly represent is the segment length. The constant-path-length advantage is real but the model never gets to use it, because it never sees past the wall at the segment boundary. I bought a car that can go 200mph and I'm driving it in a parking lot.

And it's worse than just a cap. The segments are cut as consecutive chunks of a fixed number of tokens, with no respect for sentence or paragraph boundaries — you just slice every $L$ tokens. So the first token of every segment has *zero* context. The second has one token of context. The first handful of every segment are being predicted from almost nothing, by construction, even though their real context sits right there on the other side of the boundary I just drew. That's not a modeling failure, it's a bookkeeping artifact, and it both hurts those predictions and makes optimization inefficient because a big fraction of my training positions are starved. Call it context fragmentation.

There's a third thing, and it bites at evaluation. If I want each test prediction to enjoy a full segment of context (and to dodge the fragmentation problem), I have to slide the window forward by one token at a time and re-encode the whole segment from scratch at every step, so the token I'm scoring always sits at the end of a full window. That means I recompute almost the identical segment over and over, shifted by one. Evaluation is brutally expensive — quadratic work per token, repeated for every token.

So three coupled problems, all from the same root: the segment is an island. Cap on dependency length, fragmentation at the seams, and ruinous recomputation at test time. The root is that I throw away everything I computed for the previous segment and start the next one cold.

What if I don't throw it away? When I move from segment $\tau$ to segment $\tau+1$, I already computed, at every layer, the hidden states for segment $\tau$. Those states *are* an encoding of that history. What if the new segment is allowed to attend into them? Concretely: when I form the keys and values for segment $\tau+1$ at layer $n$, prepend the layer-$(n-1)$ hidden states I cached from segment $\tau$, and let the current queries attend over the concatenation. Then information from the previous segment flows into the current one through the attention, and a token near the start of segment $\tau+1$ is no longer starved — it can look left across the boundary into real cached history. Fragmentation gone, and the reachable context now extends past a single segment.

I do have to be careful about gradients. If I backprop through the cached states into segment $\tau$'s computation, and segment $\tau$'s states depend on $\tau-1$'s, I'm right back to backpropagating through an unbounded chain — the cost and the instabilities both blow up. So I'll cache the previous segment's states as a *fixed* input: stop-gradient on them. The forward pass carries history across the boundary; the backward pass stays inside the current segment. This is exactly the bargain truncated backpropagation through time strikes for recurrent LMs — pass the previous segment's state forward as a constant, keep the gradient local — except a recurrent model can only hand forward a single summary vector, whereas I get to hand forward the *whole sequence* of cached states and let attention address any position in it. That's a much richer channel.

Let me write the recurrence down. Two consecutive segments $\mathbf{s}_\tau=[x_{\tau,1},\dots,x_{\tau,L}]$ and $\mathbf{s}_{\tau+1}$ of length $L$. Let $\mathbf{h}_\tau^n\in\mathbb{R}^{L\times d}$ be the layer-$n$ hidden states for segment $\tau$. Then for the next segment, at layer $n$,
$$
\widetilde{\mathbf{h}}_{\tau+1}^{n-1}=\big[\,\mathrm{SG}(\mathbf{h}_{\tau}^{n-1})\circ \mathbf{h}_{\tau+1}^{n-1}\,\big],
$$
$$
\mathbf{q}_{\tau+1}^{n}=\mathbf{h}_{\tau+1}^{n-1}\mathbf{W}_q^\top,\quad
\mathbf{k}_{\tau+1}^{n}=\widetilde{\mathbf{h}}_{\tau+1}^{n-1}\mathbf{W}_k^\top,\quad
\mathbf{v}_{\tau+1}^{n}=\widetilde{\mathbf{h}}_{\tau+1}^{n-1}\mathbf{W}_v^\top,
$$
$$
\mathbf{h}_{\tau+1}^{n}=\text{Layer}(\mathbf{q}_{\tau+1}^{n},\mathbf{k}_{\tau+1}^{n},\mathbf{v}_{\tau+1}^{n}),
$$
where $\mathrm{SG}$ is stop-gradient and $\circ$ is concatenation along the length axis. The crucial detail: the *query* is computed from the current segment alone, but the *keys and values* are computed from the extended context that includes the cached previous-segment states. So a current token queries against both its own segment and the carried history.

One thing to notice about the shape of this recurrence. $\mathbf{h}_{\tau+1}^{n}$ depends on $\mathbf{h}_{\tau}^{n-1}$ — one layer *down*. So as I go back a segment I also drop a layer. After $N$ layers the information can have come from $N$ segments back. The reachable dependency length grows like $O(N\times L)$ — linear in depth as well as in segment length. This isn't the same-layer recurrence of an RNN (where the state at a layer feeds the same layer at the next step); it's a diagonal flow down through the stack, but it buys me exactly what I wanted: context far longer than one segment.

And the evaluation problem dissolves on its own. At test time I don't re-encode a sliding window from scratch; I keep the cached states from the previous step and just push the new segment through, reusing all the old representations. No more recomputing the same window one shift apart.

There's no reason to cache only one previous segment, either. I can keep the last $M$ hidden states — possibly spanning several previous segments — and treat them as a memory $\mathbf{m}_\tau^n\in\mathbb{R}^{M\times d}$ that the current queries attend into. Cache as much as the memory budget allows. A natural choice is $M=L$ during training, and then crank $M$ up at evaluation to read even longer context essentially for free.

I'm pleased with this — and then I hit a wall, and it's about position.

In the standard model, order is injected by adding absolute positional encodings $\mathbf{U}\in\mathbb{R}^{L_\text{max}\times d}$ to the word embeddings at the input, where row $\mathbf{U}_i$ encodes absolute position $i$. Fine inside one self-contained segment. But watch what happens under my recurrence. Each segment's input is its word embeddings plus $\mathbf{U}_{1:L}$ — positions $1\ldots L$. So:
$$
\mathbf{h}_{\tau}=f(\mathbf{h}_{\tau-1},\,\mathbf{E}_{\mathbf{s}_\tau}+\mathbf{U}_{1:L}),\qquad
\mathbf{h}_{\tau+1}=f(\mathbf{h}_{\tau},\,\mathbf{E}_{\mathbf{s}_{\tau+1}}+\mathbf{U}_{1:L}).
$$
Both segments get the *same* $\mathbf{U}_{1:L}$. Token $x_{\tau,j}$ and token $x_{\tau+1,j}$ — same offset $j$ inside their respective segments, but actually $L$ positions apart in the real stream — carry the *identical* positional encoding. When the current query attends into the cached memory, it literally cannot tell whether a key came from the current segment or the previous one; position $j$ of "now" and position $j$ of "before" look the same. The encoding that's supposed to disambiguate order is the very thing that's now ambiguous. So the whole recurrence collapses into temporal confusion — I'd be mixing the memory and the current segment with no way to order them. That's a real performance killer, and it's not a tuning issue, it's structural: absolute positions repeat every segment.

So I need positions that don't repeat. The clue is in what the query actually needs. When query $q_{\tau,i}$ attends to key $k_{\tau,j}$, does it really need to know that one is at absolute position $i$ and the other at $j$? No. To gather information coherently it needs to know how far apart they are — the relative distance $i-j$. "The word three tokens before me," "the word a hundred tokens before me" — that's the temporal clue I want to give it, and it's well-defined no matter which segment we're in, because $i-j$ doesn't reset at a boundary. (And I lose nothing: absolute position is recoverable from relative distances anyway.) So I want to encode the *relative* distance, and I want to inject it not statically at the input — where it gets added once and then washed through the layers — but into the attention score at every layer, since the score is exactly where "where should I attend" gets decided, and each layer re-decides it.

Now, how exactly to put a relative distance into the score. Let me not guess; let me take the existing absolute score apart and see what each piece is doing, then surgically relativize it. In the standard model the query at position $i$ is built from $\mathbf{E}_{x_i}+\mathbf{U}_i$ and the key at $j$ from $\mathbf{E}_{x_j}+\mathbf{U}_j$, and the score is $q_i^\top k_j=(\mathbf{E}_{x_i}+\mathbf{U}_i)^\top\mathbf{W}_q^\top\mathbf{W}_k(\mathbf{E}_{x_j}+\mathbf{U}_j)$. Expand the product into four terms:
$$
A^{\text{abs}}_{i,j}=
\underbrace{\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_k\,\mathbf{E}_{x_j}}_{(a)}
+\underbrace{\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_k\,\mathbf{U}_{j}}_{(b)}
+\underbrace{\mathbf{U}_{i}^\top\mathbf{W}_q^\top\mathbf{W}_k\,\mathbf{E}_{x_j}}_{(c)}
+\underbrace{\mathbf{U}_{i}^\top\mathbf{W}_q^\top\mathbf{W}_k\,\mathbf{U}_{j}}_{(d)}.
$$
Read them. $(a)$ is content-against-content: does the word I am care about the word at $j$, ignoring position entirely. $(b)$ is content-of-query against position-of-key: given what I am, do I care about *something at the key's location*. $(c)$ is position-of-query against content-of-key: given where I sit, do I care about that word. $(d)$ is position against position: pure where-against-where.

Every place an *absolute* position appears, it's either $\mathbf{U}_j$ (the key's absolute position, in $(b)$ and $(d)$) or $\mathbf{U}_i$ (the query's absolute position, in $(c)$ and $(d)$). Those are exactly the things that repeat across segments and have to go.

Take the key side first. $\mathbf{U}_j$ is the key's absolute position; I want it to be the relative distance instead. So replace every $\mathbf{U}_j$ with a relative encoding $\mathbf{R}_{i-j}$ — a vector that depends only on the gap $i-j$. In $(b)$ that turns $\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_k\,\mathbf{U}_j$ into $\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_k\,\mathbf{R}_{i-j}$, and likewise in $(d)$. Now $(b)$ reads "given what I am, how much do I care about something $i-j$ tokens back" — a content-dependent positional bias that no longer cares which segment we're in. Good.

Now the query side: the $\mathbf{U}_i$ in $(c)$ and $(d)$. Here's the thing — under relative encoding there is no meaningful "absolute position of the query." The query sits at *now*; everything is measured relative to it; "now" has no number. So the term $\mathbf{U}_i^\top\mathbf{W}_q^\top$, which was the query's absolute position projected through the query map, has lost its referent. And notice what it would have to be if it had one: the same for every query position, because every query is at its own "now." If the query-position contribution should be identical regardless of where the query sits, then I should just replace that whole vector $\mathbf{U}_i^\top\mathbf{W}_q^\top$ with a single learnable global vector. In $(c)$ call it $u^\top$, in $(d)$ call it $v^\top$. Two different roles, so two different vectors: $u$ pairs against key *content*, $v$ pairs against key *position*.

So the re-parameterized score is
$$
A^{\text{rel}}_{i,j}=
\underbrace{\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_{k,E}\,\mathbf{E}_{x_j}}_{(a)}
+\underbrace{\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_{k,R}\,\mathbf{R}_{i-j}}_{(b)}
+\underbrace{u^\top\mathbf{W}_{k,E}\,\mathbf{E}_{x_j}}_{(c)}
+\underbrace{v^\top\mathbf{W}_{k,R}\,\mathbf{R}_{i-j}}_{(d)}.
$$
And there's a third change I just slipped in that I should make explicit, because it isn't cosmetic. I split the single key projection $\mathbf{W}_k$ into two: $\mathbf{W}_{k,E}$ for keys built from *content* (embeddings) and $\mathbf{W}_{k,R}$ for keys built from *position* (the relative sinusoid $\mathbf{R}$). The reason: content and position are now genuinely different kinds of input — one is a learned word embedding, the other a fixed sinusoid of a distance — and there's no reason a single projection should serve both. Giving position its own projection lets the model shape how distances enter the score independently of how words enter it.

Each term now has a clean reading: $(a)$ content-based addressing (which word do I want), $(b)$ a content-dependent positional bias (given my content, which distances do I favor), $(c)$ a global content bias (a position-independent prior on which words matter at all), $(d)$ a global positional bias (a position-independent prior on which distances matter — e.g. "usually attend nearby").

One decision inside the relative encoding I want to get right rather than default. What is $\mathbf{R}$? I'll make it the *sinusoid* encoding — same construction as the absolute one, $\text{inv\_freq}_k=1/10000^{2k/d}$, $\sin$ in the even dims and $\cos$ in the odd — but now indexed by the relative distance, $\mathbf{R}_{i-j}$, with *no learnable parameters in $\mathbf{R}$ itself*. Why insist on the structured sinusoid rather than a free learned vector per distance? Because the whole point of this architecture is to read context far longer than what I trained on — I train with memory length $M$ but want to set $M$ several times larger at evaluation, which means querying distances I never saw in training. A free per-distance parameter has nothing to say about a distance it never observed. The sinusoid does: distances compose through fixed linear maps, so the encoding extrapolates smoothly to gaps beyond the training range. If instead I folded $\mathbf{W}_{k,R}\mathbf{R}$ into one trainable matrix indexed by distance — a per-distance learned key, which is the natural shortcut and is what a relative scheme for fixed-length translation would do — I'd throw away exactly that inductive bias and lose the length generalization that justifies the recurrence in the first place. So $\mathbf{R}$ stays a parameter-free sinusoid and only $\mathbf{W}_{k,R}$ is learned. (That same fixed-length-translation relative scheme also keeps only $(a)$ and $(b)$ and drops the two global-bias terms $(c)$ and $(d)$ — but those terms are cheap and meaningful, a word prior and a distance prior, so I'll keep them.)

Let me assemble the full layer with the memory folded in. Per layer $n$, with memory $\mathbf{m}^{n-1}$:
$$
\widetilde{\mathbf{h}}^{n-1}=\big[\mathrm{SG}(\mathbf{m}^{n-1})\circ \mathbf{h}^{n-1}\big],
$$
$$
\mathbf{q}^{n}=\mathbf{h}^{n-1}{\mathbf{W}_q^{n}}^\top,\quad
\mathbf{k}^{n}=\widetilde{\mathbf{h}}^{n-1}{\mathbf{W}_{k,E}^{n}}^\top,\quad
\mathbf{v}^{n}=\widetilde{\mathbf{h}}^{n-1}{\mathbf{W}_v^{n}}^\top,
$$
$$
A^{n}_{i,j}={\mathbf{q}^n_i}^\top\mathbf{k}^n_j+{\mathbf{q}^n_i}^\top\mathbf{W}_{k,R}^{n}\mathbf{R}_{i-j}+u^\top\mathbf{k}^n_j+v^\top\mathbf{W}_{k,R}^{n}\mathbf{R}_{i-j},
$$
then a causal-masked softmax over the keys, the value sum, the output projection added back as a residual with a layer-norm, and the position-wise feed-forward. The content key uses $\mathbf{W}_{k,E}$, so the first and third terms share the same $\mathbf{k}^n_j=\mathbf{W}_{k,E}^n\widetilde{\mathbf{h}}_j$ and group as $(\mathbf{q}^n_i+u)^\top\mathbf{k}^n_j$; the second and fourth share the location key $\mathbf{W}_{k,R}^n\mathbf{R}_{i-j}$ and group as $(\mathbf{q}^n_i+v)^\top(\mathbf{W}_{k,R}^n\mathbf{R}_{i-j})$. That grouping is going to matter for the implementation — adding $u$ and $v$ to the query is cheaper than carrying them separately. The score is scaled by $1/\sqrt{d}$ for the usual reason: the dot product of two $d$-dimensional vectors with unit-scale entries has variance $\sim d$, and feeding raw scores of that magnitude into the softmax saturates it into near-one-hot, killing gradients; dividing by $\sqrt{d}$ keeps the score variance order one. And $\mathbf{h}^0=\mathbf{E}_{\mathbf{s}}$, the word embeddings — no positional vector added at the input anymore, since position now lives entirely in the per-layer score.

Now the one efficiency worry. Term $(b)$ (and $(d)$) wants $\mathbf{W}_{k,R}\mathbf{R}_{i-j}$ for every query-key pair $(i,j)$. If I compute that product separately for each pair, that's $O(L\cdot(M+L))$ distinct products — quadratic, and I went to all this trouble partly to make things cheaper. But notice the relative distance $i-j$ only ranges over the integers $0$ to $M+L-1$. There are only $M+L$ distinct distances, so only $M+L$ distinct vectors $\mathbf{W}_{k,R}\mathbf{R}_{d}$. I should compute those once and then route them into the right $(i,j)$ slots.

Let me make the routing concrete. Stack the distinct location keys in *reversed* distance order into
$$
\mathbf{Q}=\begin{bmatrix}\mathbf{R}_{M+L-1}^\top\\ \vdots\\ \mathbf{R}_0^\top\end{bmatrix}\mathbf{W}_{k,R}^\top\in\mathbb{R}^{(M+L)\times d},\qquad \mathbf{Q}_k=\mathbf{W}_{k,R}\mathbf{R}_{M+L-1-k}.
$$
The matrix I actually want for term $(b)$ is the $L\times(M+L)$ matrix $\mathbf{B}$ whose $(i,j)$ entry is $q_i^\top\mathbf{W}_{k,R}\mathbf{R}_{i-j}$ — and along row $i$, as $j$ runs over the visible keys, the distance $i-j$ marches through a contiguous run of values, with zeros where $j>i$ (future, masked). Compute instead the dense product $\widetilde{\mathbf{B}}=\mathbf{q}\,\mathbf{Q}^\top$, whose $(i,k)$ entry is $q_i^\top\mathbf{Q}_k$. Writing out $\mathbf{B}$ and $\widetilde{\mathbf{B}}$ side by side, row $i$ of $\mathbf{B}$ is exactly row $i$ of $\widetilde{\mathbf{B}}$ shifted left by a row-dependent amount. So one matmul $\mathbf{q}\mathbf{Q}^\top$ plus a per-row left-shift gives me all of term $(b)$. The shift itself is the cheap trick: pad a zero column on the left of $\widetilde{\mathbf{B}}$, view the $(M+L+1)\times L$ buffer as $L\times(M+L+1)$, drop the first row, view back — the reshape slides each row by one more than the row above, which is precisely the diagonal march of $i-j$. Term $(d)$, $v^\top\mathbf{W}_{k,R}\mathbf{R}_{i-j}$, is the same shape with the constant vector $v$ in place of the per-position query, so I compute $\widetilde{d}=(\mathbf{Q}v)^\top$ once — a matrix-vector product — and apply the same left-shift. Cost is now linear in the sequence length. Good; the recurrence and the relative encoding both pay off in compute, not just in reach.

Let me land this on code. The content terms $(a)+(c)$ and the position terms $(b)+(d)$ separate cleanly because of the $u,v$-into-the-query grouping. The memory is concatenated to form the keys/values but the query is sliced to the current segment; after the forward pass I detach and stash the new states as the next step's memory (the stop-gradient). The relative positions feed in as a sinusoid over distances $\text{klen}-1$ down to $0$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """The relative sinusoid R: parameter-free, so distances unseen in training
    still get a meaningful encoding (this is what lets memory grow at eval)."""
    def __init__(self, demb):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):                       # pos_seq: relative distances
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)[:, None, :]


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0):
        super().__init__()
        self.n_head, self.d_head = n_head, d_head
        # W_q, W_{k,E}, W_v in one matrix; W_{k,R} (location key) is separate.
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)   # W_{k,R}
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop, self.dropatt = nn.Dropout(dropout), nn.Dropout(dropatt)
        self.scale = 1 / (d_head ** 0.5)

    def _rel_shift(self, x):
        # turn the dense q.Q^T into term (b)/(d): each row is left-shifted so the
        # column index becomes the relative distance i-j.
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        return x_padded[1:].view_as(x)

    def forward(self, w, r, u, v, attn_mask=None, mems=None):
        # w: current segment [qlen,bsz,d];  r: relative sinusoid [klen,1,d]
        # u (=r_w_bias), v (=r_r_bias): the two learnable global query vectors.
        qlen, bsz = w.size(0), w.size(1)
        cat = torch.cat([mems, w], 0) if mems is not None else w      # extended context
        w_heads = self.qkv_net(cat)
        r_head_k = self.r_net(r)                                       # W_{k,R} R
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-qlen:]                                    # query: current segment only
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(klen, self.n_head, self.d_head)

        # terms (a)+(c): (q + u) . k_E   — content key shared by both
        AC = torch.einsum('ibnd,jbnd->ijbn', (w_head_q + u, w_head_k))
        # terms (b)+(d): (q + v) . (W_{k,R} R) — then shift to relative distance
        BD = torch.einsum('ibnd,jnd->ijbn', (w_head_q + v, r_head_k))
        BD = self._rel_shift(BD)

        attn_score = (AC + BD).mul_(self.scale)
        if attn_mask is not None:                                     # causal mask
            attn_score = attn_score.float().masked_fill(
                attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
        attn_prob = self.dropatt(F.softmax(attn_score, dim=1))

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(qlen, bsz, self.n_head * self.d_head)
        attn_out = self.drop(self.o_net(attn_vec))
        return self.layer_norm(w + attn_out)                          # residual + LN


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0):
        super().__init__()
        self.attn = RelMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt)
        self.ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, x, r, u, v, attn_mask=None, mems=None):
        return self.ff(self.attn(x, r, u, v, attn_mask=attn_mask, mems=mems))


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tgt_len, mem_len):
        super().__init__()
        self.n_layer, self.d_model, self.mem_len = n_layer, d_model, mem_len
        self.word_emb = nn.Embedding(n_token, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        # u, v: the two global query vectors, shared across positions and layers.
        self.u = nn.Parameter(torch.Tensor(n_head, d_head))
        self.v = nn.Parameter(torch.Tensor(n_head, d_head))
        self.layers = nn.ModuleList([
            DecoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt)
            for _ in range(n_layer)])
        self.out_layer = nn.Linear(d_model, n_token)

    def init_mems(self):
        if self.mem_len <= 0:
            return None
        p = next(self.parameters())
        return [torch.empty(0, dtype=p.dtype, device=p.device) for _ in range(self.n_layer + 1)]

    def _update_mems(self, hids, mems, qlen, mlen):
        # stop-gradient: cache the most recent mem_len states, detached.
        if mems is None:
            return None
        with torch.no_grad():
            end = mlen + qlen
            beg = max(0, end - self.mem_len)
            return [torch.cat([m, h], 0)[beg:end].detach() for m, h in zip(mems, hids)]

    def _forward(self, dec_inp, mems):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        # causal mask over the extended context: a query at i may see keys up to i.
        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]

        # relative distances klen-1 .. 0 (the current token is distance 0).
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        pos_emb = self.drop(self.pos_emb(pos_seq))
        core_out = self.drop(word_emb)                # h^0 = word embeddings, no abs pos added

        hids = [core_out]
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.u, self.v,
                             attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)
        new_mems = self._update_mems(hids, mems, qlen, mlen)
        return self.drop(core_out), new_mems

    def forward(self, data, target, *mems):
        if not mems:
            mems = self.init_mems()
        hidden, new_mems = self._forward(data, mems=mems)
        logit = self.out_layer(hidden)
        loss = F.cross_entropy(logit.view(-1, logit.size(-1)), target.view(-1), reduction='none')
        return [loss] + (new_mems if new_mems is not None else [])
```

So the causal chain, start to finish: I want to model long-range dependency, and attention gives me the short gradient paths an LSTM lacks; but training on fixed-length segments throws that away — the reachable context is capped at one segment, the leading tokens of each segment are starved (fragmentation), and full-context evaluation forces re-encoding a sliding window from scratch. The root is that each segment starts cold, so I cache the previous segment's hidden states (stop-gradient, so the gradient stays local while the forward pass carries history) and let the current queries attend into them as extended keys/values, which lifts the reachable context to $O(N\times L)$ and makes evaluation reuse computation. But carrying state forward exposes that absolute positions repeat every segment, making cached and current positions indistinguishable; so I move to relative distances $i-j$ injected into the score at every layer. To do that surgically I expand the absolute score into four terms — content-content, content-position, position-content, position-position — replace the key's absolute position with the relative sinusoid $\mathbf{R}_{i-j}$, replace the now-referentless query absolute position with two learnable global vectors $u,v$, and give content and location their own key projections $\mathbf{W}_{k,E},\mathbf{W}_{k,R}$; $\mathbf{R}$ stays a parameter-free sinusoid precisely so the model generalizes to longer memory at evaluation. The naive per-pair computation of the position term is quadratic, but since only $M+L$ distinct distances exist, one matmul against the stacked location keys plus a left-shift recovers it in linear time. The result is a self-attention language model whose effective context runs far past a single segment, with coherent positions and cheap evaluation.
