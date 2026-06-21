We want a single neural network that reads a source sentence and emits its translation, trained end to end to maximize the probability of the correct target, replacing the many separately tuned sub-components of a phrase-based statistical system. The established neural way to do this is the encoder-decoder: an encoder RNN walks the source left to right, $h_t = f(x_t, h_{t-1})$, and at the end its final hidden state becomes a single vector $c = h_{T_x}$ that is supposed to be "the meaning of the source"; the decoder is then a conditional language model started from $c$, factorizing the target as $p(y) = \prod_t p(y_t \mid y_{<t}, c)$ and seeing the source only through $c$. On short sentences this works; on long ones it falls apart, and the failure is sharp — BLEU slides downhill as the source grows and drops hardest exactly past the lengths seen in training, while a phrase-based system stays flat. This is not a tuning issue but a structural one: $c$ is the same size whether the sentence has five words or fifty, so everything the decoder will ever need must be squeezed into one fixed-dimensional vector before a single target word comes out, and never revisited. A short sentence fits; a long one overflows. The strongest prior systems all share this defect or sidestep translation entirely. Phrase-based SMT (Moses) is mature and exploits large monolingual data, but it is many separately engineered components with a discrete latent alignment and awkward $\texttt{[NULL]}$ devices for words that align to nothing. The RNN encoder-decoder used for SMT rescoring is a neural feature bolted onto the statistical system, still funneling the source through a fixed $c$. Sequence-to-sequence with deep LSTMs is a standalone neural translator, but the entire source lives in one fixed $v = h_{T_x}$; its reversal trick — feeding the source backwards — helps a lot (perplexity $5.8 \to 4.7$, BLEU $25.9 \to 30.6$), yet it does not enlarge the bottleneck, it only shortens the recurrence distance relevant information must travel to reach the decoder. That clue is the key: the real disease is distance, and the cure is to let the decoder reach source content directly, at the moment it needs it, rather than through a single summary computed once at the start.

I propose RNNsearch, an encoder-decoder translator that keeps every source position's encoder state and lets the decoder softly search the source at each output step, forming a distinct context for every target word instead of one frozen summary. The intuition is how a person actually translates: the gaze shifts back and forth, looking at the source fragment that feeds the word being produced, then moving on. The literal version of this would have the decoder pick one source position $j^\ast$ and read $h_{j^\ast}$ — a cursor sliding along the source. But a hard pick fails twice. It is non-differentiable: argmax has zero gradient almost everywhere and sampling needs a high-variance estimator, so the translation loss cannot tell the cursor where it should have gone, which breaks the whole appeal of one network trained end to end by gradient descent. And it is linguistically wrong: translating "the man" into "l' homme," you cannot decide whether "the" becomes "le", "la", "les" or "l'" without already seeing "man," so reading exactly one position commits before it has the information. Both problems dissolve if, instead of choosing a position, the decoder reads a blend of all positions weighted by how much it wants each one right now. With weights $\alpha_{ij} \ge 0$ summing to one over source positions, the context at step $i$ is
$$c_i = \sum_{j=1}^{T_x} \alpha_{ij}\, h_j,$$
the soft relaxation of the cursor — a hard pick is the corner where one $\alpha_{ij}$ is $1$ and the rest are $0$. It is differentiable, so gradients flow into both the weights and the $h_j$; it is recomputed every step, so the bottleneck is gone and the information stays spread across $h_1,\dots,h_{T_x}$ to be retrieved selectively; and reading $\alpha_{ij}$ as the probability that target word $y_i$ aligns to source word $x_j$ makes $c_i$ the *expected* annotation under a soft alignment — alignment becomes a deterministic, jointly trained function rather than a discrete latent inferred by a separate procedure with $\texttt{[NULL]}$ tokens.

The weights need two things settled: how to turn raw scores into a nonnegative, sum-to-one distribution, and what raw quantity to score. Given scores $e_{i1},\dots,e_{iT_x}$, one per source position, the standard differentiable map onto the simplex is the softmax,
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})},$$
which gives $\alpha_{ij} \ge 0$ and $\sum_j \alpha_{ij} = 1$ automatically, is smooth, and sharpens or flattens as the scores spread or bunch. The normalization is load-bearing, not cosmetic: because the weights sum to one, $c_i$ is a convex combination of the $h_j$, living in the same space and at the same scale whether $T_x$ is $5$ or $50$ — the context does not grow with sentence length. Unnormalized positive weights would pile up a larger context on longer sentences and reintroduce a length-dependent representation. The score $e_{ij}$ must measure how well source position $j$ matches what the decoder is about to do. The decoder knows its recurrent state, carrying everything generated so far and what it intends next; position $j$ is represented by the annotation $h_j$. So $e_{ij}$ is a learned function of those two vectors,
$$e_{ij} = a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j),$$
a one-hidden-layer additive MLP. Two separate matrices, not a dot product $s^\top h_j$, because the query $s_{i-1}$ (dimension $n$) and the key $h_j$ (dimension $2n$, as below) do not even live in the same space, so a raw dot product is undefined; $W_a$ and $U_a$ project both into a shared $n'$-dimensional scoring space, the sum inside the $\tanh$ lets the nonlinearity model "do these match," and $v_a$ reads off a scalar. Only one hidden layer because this scorer runs $T_x \cdot T_y$ times per sentence pair — the dominant new cost — so it must be the cheapest thing that can still learn an arbitrary comparison; and $U_a h_j$ is independent of $i$, so it is precomputed once per sentence and reused across every decoder step, leaving only $W_a s_{i-1}$ to change with $i$. The query is $s_{i-1}$, the state from *before* this step, not $s_i$: since $s_i$ depends on $c_i$, using it would be circular, and $s_{i-1}$ — what has been produced and what is about to be needed — is exactly the right query for "where should I look now." The order resolves to $e_{ij} \to c_i \to s_i = f(s_{i-1}, y_{i-1}, c_i) \to y_i$.

The annotations themselves must be good keys. A plain left-to-right encoder gives an $h_j$ summarizing only $x_1\dots x_j$, but the context that disambiguates position $j$ can lie on either side — to align onto "the" usefully the annotation must know "man" follows. So the encoder is bidirectional: a forward RNN giving $\vec h_j$ (summary of $x_1\dots x_j$) and a backward RNN giving $\overleftarrow h_j$ (summary of $x_j\dots x_{T_x}$), concatenated into $h_j = [\vec h_j; \overleftarrow h_j] \in \mathbb{R}^{2n}$. Because RNNs represent recent inputs most strongly, $\vec h_j$ is most informed about words just before $j$ and $\overleftarrow h_j$ about words just after, so the concatenation is naturally focused near position $j$ while still seeing both sides — a position-tagged key, and the reason the annotation is $2n$-dimensional, which in turn is why the projection matrices in the alignment model were earning their keep. This content-based, non-monotonic scoring is what makes soft search beat the one prior differentiable way to read part of an input sequence, Graves's Gaussian soft window for handwriting: that window is placed by a location coordinate $\kappa$ slid along the input, and $\kappa$ only ever increases, so it is monotonic and forward-only. Translation reorders — adjective/noun order differs across languages, German moves verbs to the end — so the target must jump back to an earlier source word and forward again, which $a(s_{i-1}, h_j)$ scoring any $(i,j)$ pair by content can express and a monotonic window cannot. Softness pays off a second time linguistically: it lets $\alpha$ put weight on both "the" and "man" while emitting "l'," and it handles fertility and source/target length mismatch with no special machinery.

The recurrences are gated to keep gradients alive over long sentences. The decoder is a reset/update gated unit with the context $c_i$ entering every gate alongside the previous target embedding $E y_{i-1}$ and previous state $s_{i-1}$:
$$z_i = \sigma(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i), \qquad r_i = \sigma(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i),$$
$$\tilde s_i = \tanh(W E y_{i-1} + U(r_i \odot s_{i-1}) + C c_i), \qquad s_i = (1 - z_i)\odot s_{i-1} + z_i \odot \tilde s_i.$$
When the update gate $z_i$ is near zero the unit copies $s_{i-1}$ forward unchanged — the constant-derivative path that lets gradients flow back many steps without vanishing — and the reset gate $r_i$ lets the candidate ignore the past and read mostly the current inputs. The context enters all three through its own matrices $C, C_z, C_r$ so the source read drives the gate decisions, not just gets appended. The encoder uses the same gated recurrence in each direction without the $c$ terms, and the decoder is seeded from $s_0 = \tanh(W_s \overleftarrow h_1)$, the backward annotation nearest the start of the source, the natural place a left-to-right generation begins. The output is not a bare affine map to the vocabulary: a deep-output maxout gives the readout nonlinear capacity over the three things known at emission time, forming $\tilde t_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i$, pooling adjacent pairs $t_{i,j} = \max(\tilde t_{i,2j-1}, \tilde t_{i,2j})$, and emitting $p(y_i \mid \cdot) \propto \exp(y_i^\top W_o t_i)$, so the context's source content interacts nonlinearly with what was just generated. Sizes are $n = 1000$ hidden, $m = 620$ embedding, $l = 500$ maxout, $n' = 1000$ alignment. A reassuring sanity check: freezing $c_i$ to a constant such as $\vec h_{T_x}$ for all $i$ recovers exactly the old fixed-vector encoder-decoder, so RNNsearch is a strict generalization — it frees the constant-context constraint and only deviates from the baseline when moving the weights helps. Training is end to end by maximizing the conditional log-likelihood $\frac{1}{N}\sum_n \log p_\theta(y_n \mid x_n)$ (masked cross-entropy over target positions), gradients flowing through the alignment model because everything is differentiable; the global gradient norm is clipped at threshold $1$, optimized by minibatch SGD with Adadelta ($\rho = 0.95$, $\epsilon = 10^{-6}$), minibatches of $80$ pairs grouped by length, with recurrent matrices orthogonal, $W_a, U_a \sim \mathcal{N}(0, 0.001^2)$, $v_a$ and biases zero, and other weights $\mathcal{N}(0, 0.01^2)$, so alignments start near-uniform and sharpen as real matches are found. Decoding is left-to-right beam search, and the alignment matrix $[\alpha_{ij}]$ is directly inspectable as a soft source-target alignment.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRUEncoder(nn.Module):
    """Bidirectional gated encoder: h_j = [forward_j ; backward_j]."""

    def __init__(self, vocab, emb_dim, hid_dim, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.birnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)

    def forward(self, src):                       # src: [B, T_x]
        ann, _ = self.birnn(self.embed(src))      # ann: [B, T_x, 2*hid_dim]
        return ann


class AdditiveAlignment(nn.Module):
    """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j); alpha = softmax_j(e); c_i = sum_j alpha_ij h_j."""

    def __init__(self, dec_dim, ann_dim, align_dim):
        super().__init__()
        self.W_a = nn.Linear(dec_dim, align_dim, bias=False)   # query projection
        self.U_a = nn.Linear(ann_dim, align_dim, bias=False)   # key projection
        self.v_a = nn.Linear(align_dim, 1, bias=False)         # scalar score

    def precompute_keys(self, ann):               # U_a h_j is i-independent: compute once
        return self.U_a(ann)                      # [B, T_x, align_dim]

    def forward(self, s_prev, ann, keys, mask):   # s_prev = s_{i-1}: [B, dec_dim]
        e = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + keys)).squeeze(-1)  # [B, T_x]
        e = e.masked_fill(~mask, float('-inf'))   # ignore padding
        alpha = F.softmax(e, dim=-1)              # distribution over source positions
        c = torch.bmm(alpha.unsqueeze(1), ann).squeeze(1)      # c_i
        return c, alpha


class RNNsearchGRUCell(nn.Module):
    """s_i = (1-z_i) * s_{i-1} + z_i * s_tilde_i, with c_i in every gate."""

    def __init__(self, emb_dim, ann_dim, dec_dim):
        super().__init__()
        self.W_z = nn.Linear(emb_dim, dec_dim, bias=False)
        self.U_z = nn.Linear(dec_dim, dec_dim, bias=False)
        self.C_z = nn.Linear(ann_dim, dec_dim, bias=False)
        self.W_r = nn.Linear(emb_dim, dec_dim, bias=False)
        self.U_r = nn.Linear(dec_dim, dec_dim, bias=False)
        self.C_r = nn.Linear(ann_dim, dec_dim, bias=False)
        self.W = nn.Linear(emb_dim, dec_dim, bias=False)
        self.U = nn.Linear(dec_dim, dec_dim, bias=False)
        self.C = nn.Linear(ann_dim, dec_dim, bias=False)

    def forward(self, y_emb, c, s_prev):
        z = torch.sigmoid(self.W_z(y_emb) + self.U_z(s_prev) + self.C_z(c))
        r = torch.sigmoid(self.W_r(y_emb) + self.U_r(s_prev) + self.C_r(c))
        s_tilde = torch.tanh(self.W(y_emb) + self.U(r * s_prev) + self.C(c))
        return (1.0 - z) * s_prev + z * s_tilde


class AttnDecoder(nn.Module):
    """Gated decoder conditioned on the per-step context, with a maxout deep output."""

    def __init__(self, vocab, emb_dim, dec_dim, ann_dim, align_dim, maxout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.align = AdditiveAlignment(dec_dim, ann_dim, align_dim)
        self.cell = RNNsearchGRUCell(emb_dim, ann_dim, dec_dim)
        self.init_s = nn.Linear(ann_dim // 2, dec_dim)         # s_0 = tanh(W_s backward_1)
        self.U_o = nn.Linear(dec_dim, 2 * maxout, bias=False)
        self.V_o = nn.Linear(emb_dim, 2 * maxout, bias=False)
        self.C_o = nn.Linear(ann_dim, 2 * maxout, bias=False)
        self.W_o = nn.Linear(maxout, vocab)

    def forward(self, ann, mask, tgt_in):         # tgt_in: [B, T_y] teacher-forced prev tokens
        keys = self.align.precompute_keys(ann)
        backward_1 = ann[:, 0, ann.size(-1) // 2:]            # backward annotation at pos 1
        s = torch.tanh(self.init_s(backward_1))              # s_0
        logits = []
        for i in range(tgt_in.size(1)):
            y_emb = self.embed(tgt_in[:, i])                 # E y_{i-1}
            s_prev = s
            c, _ = self.align(s_prev, ann, keys, mask)       # query = s_{i-1} (pre-step state)
            s = self.cell(y_emb, c, s_prev)
            t_tilde = self.U_o(s_prev) + self.V_o(y_emb) + self.C_o(c)
            t = t_tilde.view(t_tilde.size(0), -1, 2).max(dim=-1).values   # maxout
            logits.append(self.W_o(t))
        return torch.stack(logits, dim=1)                    # [B, T_y, vocab]


class RNNSearch(nn.Module):
    def __init__(self, vocab, emb_dim=620, hid=1000, align_dim=1000, maxout=500, pad_id=0):
        super().__init__()
        self.encoder = BiGRUEncoder(vocab, emb_dim, hid, pad_id)
        self.decoder = AttnDecoder(vocab, emb_dim, hid, 2 * hid, align_dim, maxout, pad_id)
        self.pad_id = pad_id

    def forward(self, src, tgt_in):
        ann = self.encoder(src)                              # annotations
        mask = src.ne(self.pad_id)
        return self.decoder(ann, mask, tgt_in)               # next-token logits


def loss_fn(logits, target, pad_id):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target.reshape(-1), ignore_index=pad_id)
```
