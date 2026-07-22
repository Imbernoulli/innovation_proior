Let me start from the thing that actually keeps going wrong. I have an encoder-decoder translator: the encoder RNN walks the source sentence left to right, h_t = f(x_t, h_{t-1}), and at the end I take the last hidden state as a single vector c = h_{T_x} that is supposed to be "the meaning of the source." The decoder is a conditional language model started from c, p(y) = Π_t p(y_t | y_<t, c), and it produces the translation word by word seeing the source only through c. On short sentences this works. On long ones it falls apart, and the failure is sharp: as the source gets longer the BLEU drops, and it drops hardest exactly past the lengths I trained on. Cho and Pouget-Abadie measured this directly — the basic encoder-decoder curve slides downhill with length while a phrase-based system stays flat. So this isn't a tuning issue, it's structural.

Why structural? Because c is the same size whether the sentence has five words or fifty. Everything the decoder will ever need about the source has to be squeezed into that one fixed-dimensional vector before a single target word comes out, and then it can never look at the source again. A short sentence fits; a long one overflows. There is only so much room in a constant-size representation, and I'm asking it to hold an unbounded amount of information. The decoder, generating the thirtieth target word, is still reading from the same vector it had at word one — a vector that had to commit, up front, to a lossy summary of the entire sentence. No wonder the late, far-from-the-start parts of long translations drift.

There's a clue in something Sutskever, Vinyals and Le found that I keep turning over. They reverse the words of the source sentence before feeding it in — map a,b,c → α,β,γ by reading c,b,a — and it helps a lot: test perplexity 5.8 → 4.7, BLEU 25.9 → 30.6. Their explanation is "minimal time lag": normally the first source word and the first target word are T_x steps apart through the recurrence, a long path for gradients and for information to survive; reversing puts the early source words close to the early target words, so some correspondences become short-range and SGD can "establish communication." Stare at that. The trick doesn't make the bottleneck vector bigger — c is still fixed size. It just rearranges *which* information has to travel *how far* to reach the decoder. So the real disease the reverse trick is medicating is distance: relevant source content is too far, in recurrence-steps, from the moment it's needed, and it has to survive the trip squashed inside c. That tells me the cure isn't a cleverer ordering or a bigger vector — it's letting the decoder reach the source content directly, at the moment it needs it, instead of through a single summary computed once at the start.

So let me think about what I actually do when I translate, because I learned this the slow way in school. I don't read the whole sentence, memorize it into one mental blob, and then recite the translation with my eyes closed. My gaze shifts back and forth — I'm producing a target word, I look back to the part of the source it comes from, I produce the next word, I look somewhere else in the source. The "where I'm looking" moves around as I go. What if the decoder could do that — at each step it decides where in the source to look, and reads from there, instead of being handed one frozen summary? Concretely: keep around all the encoder's per-position states h_1, ..., h_{T_x}, not just the last one, and let the decoder, at target step i, search the source for the position it should be reading right now and pull the representation from there.

Let me try the literal version of "decide where to look." At step i the decoder picks one source position, call it j*, and reads h_{j*}. It's like a cursor the decoder slides along the source: emit a word, move the cursor, read, emit, move. That's exactly the back-and-forth gaze. And it removes the bottleneck — the decoder is reading from a per-position state, not a global summary, so a fifty-word sentence is no harder than a five-word one because I only ever read one position's worth at a time.

But how does the decoder "pick" j*? If I make it a hard choice — compute some score over positions and take the argmax, or sample one position — I've put a discrete, non-differentiable operation in the middle of the network. Argmax has zero gradient almost everywhere; sampling needs a high-variance estimator. I can't backpropagate the translation loss through "which position did you choose." The whole appeal of this family is that it's one network trained end to end by gradient descent, and a hard cursor breaks that outright. Fancier hard schemes — say two cursors to bracket a span — don't help either: any discrete selection has the same fatal property, the gradient can't tell the cursor where it should have gone.

Now stand back. The reason I wanted a single position was the school intuition of "look *there*." But why exactly one? When I translate "the man" into French I genuinely have to look at both words at once — I can't decide whether "the" becomes "le", "la", "les" or "l'" without seeing what follows it. A hard one-position cursor can't even express that. So the constraint "pick one position" is both undifferentiable *and* wrong about how translation works. Drop it. Instead of choosing a position, let the decoder read a *blend* of all positions, weighted by how much it wants each one right now. If the weights are α_{i1}, ..., α_{iT_x} with α_{ij} ≥ 0 and Σ_j α_{ij} = 1, then the thing the decoder reads at step i is

  c_i = Σ_{j=1}^{T_x} α_{ij} h_j.

This is the soft version of the cursor. A hard cursor is the special case where one α_{ij} is 1 and the rest are 0; a soft one spreads the weight. And crucially it's differentiable: c_i is a smooth function of the weights and of the h_j, so gradients flow into both. The hard "where to look" became a continuous "how much to look at each place," and continuity is exactly what I needed. Notice what this does to the bottleneck too: c_i is now *different for every target word* — it's recomputed each step from wherever the decoder currently wants to read — instead of one c fixed for the whole sentence. The decoder is no longer forced to encode the whole source into a constant; the information can stay spread across h_1..h_{T_x} and be retrieved selectively.

There's a clean way to read c_i = Σ_j α_{ij} h_j that tells me the weights aren't arbitrary: if I treat α_{ij} as the probability that target word y_i is aligned to — translated from — source word x_j, then c_i is the *expected* source annotation under that alignment distribution. So this is soft alignment: instead of committing to a hard word-to-word alignment as a latent variable the way phrase-based systems do (with their awkward [NULL] tokens for words that align to nothing), I compute a deterministic expected alignment and let the gradient train it. The alignment stops being a hidden discrete thing inferred by a separate procedure and becomes a differentiable, jointly trained part of the network.

Now I need the weights. Two things to settle: what raw quantity scores how much step i wants position j, and how I turn those scores into the nonnegative, sum-to-one α_{ij}. Take the second first, because it forces the first. I have a vector of scores e_{i1}, ..., e_{iT_x}, one per source position, and I need them to become a probability distribution over positions. The standard differentiable map from arbitrary reals to a simplex is the softmax:

  α_{ij} = exp(e_{ij}) / Σ_{k=1}^{T_x} exp(e_{ik}).

It gives me α_{ij} ≥ 0, Σ_j α_{ij} = 1 automatically, it's smooth, and its gradient is well-behaved (the soft-selection sharpens or flattens as the e's spread or bunch). And the normalization isn't cosmetic — because the weights sum to one, c_i is a convex combination of the h_j, so it lives in the same space and at the same scale as a single annotation no matter whether T_x is 5 or 50. The context's magnitude doesn't grow with sentence length. If I'd used unnormalized positive weights, a long sentence would pile up a bigger context vector just by having more terms, and I'd be back to a length-dependent representation. Softmax both makes the cursor soft and keeps the read length-invariant — the differentiable relaxation of the argmax cursor, falling right out of "I need a distribution over positions."

So the score e_{ij}. It has to measure how well source position j matches what the decoder is about to do at step i. What does the decoder know at step i, before it emits y_i? Its recurrent state, which carries everything generated so far and what it intends next. Call it s. And what represents position j? The encoder annotation h_j. So e_{ij} should be a learned function of those two, a(s, h_j) — score the (query, key) pair. The simplest learnable scorer that can compare two vectors and isn't just a fixed similarity is a small neural net. Let me reach for the cheapest one that can actually learn the comparison: a single hidden layer.

  e_{ij} = a(s, h_j) = v_a^T tanh(W_a s + U_a h_j).

Let me make sure each piece is forced and not decorative. Why two separate matrices W_a and U_a instead of, say, a dot product s^T h_j? Because s and h_j don't even live in the same space — s is a decoder state of size n, h_j is an encoder annotation that, as I'll get to, has size 2n — so a raw dot product isn't even dimensionally defined, let alone meaningful. W_a and U_a project the query and the key into a common scoring space of some size n'; adding them inside the tanh, W_a s + U_a h_j, mixes the two so the nonlinearity can model "do these match"; then v_a reads the hidden activation down to a single scalar score. It's literally a one-hidden-layer MLP taking (s, h_j) to a number — additive, because the contributions of query and key are summed before the nonlinearity. Why only one hidden layer? Because of cost. This scorer runs for every (i, j) pair: T_x × T_y evaluations per sentence pair, and that's the dominant new computation I'm adding. A deeper scorer would multiply that. One hidden layer is the cheapest thing that can still learn an arbitrary comparison, so it's the right point on the cost/expressiveness curve. And I get one efficiency gift for free: U_a h_j doesn't depend on i, so I can precompute all the U_a h_j once per sentence and reuse them across every decoder step — only W_a s changes with i. That knocks the per-step alignment cost down a lot.

Which decoder state do I use as the query, s? I want to compute c_i and then use it to produce y_i and to update the decoder. But the decoder's state at step i, s_i, depends on c_i — that's the whole point, the context feeds the recurrence. So I can't use s_i to compute c_i; that's circular. I have to use the state from *before* this step, s_{i-1}, the state just after emitting y_{i-1} and just before emitting y_i. That's also the right semantics: s_{i-1} is "what I've produced and what I'm about to need," exactly the query for "where should I look now." So e_{ij} = v_a^T tanh(W_a s_{i-1} + U_a h_j), and then c_i, and then s_i = f(s_{i-1}, y_{i-1}, c_i), and then y_i. The order resolves the circularity.

Now the annotations h_j themselves. I've been assuming the encoder's per-position states are good keys-and-values for this search, but let me check what a plain left-to-right encoder gives me. h_j there summarizes x_1..x_j — only the words up to and including j. But when I'm deciding whether position j is what target step i wants, the relevant disambiguating context can be on *either* side of x_j. Going back to "the man" → "l' homme": to align the target onto "the" usefully, the annotation at "the" needs to know that "man" follows. A forward-only h_j can't know what comes after j. So I want each annotation to summarize the whole sentence with a focus around position j. Run two RNNs: a forward one giving ⃗h_j (summary of x_1..x_j) and a backward one reading right-to-left giving ⃖h_j (summary of x_j..x_{T_x}), and concatenate, h_j = [⃗h_j ; ⃖h_j]. Now h_j carries both the preceding and the following words. And because RNNs represent recent inputs more strongly, ⃗h_j is most informed about words just before j and ⃖h_j about words just after j, so the concatenation is naturally *focused* near position j while still seeing both sides — which is exactly what makes it a good, position-tagged key for the alignment search. That's why the annotation is 2n-dimensional: it's the concatenation of the two directions, each of size n. (And it's why a raw dot-product score against the n-dimensional s wouldn't have typed — the projection matrices in the alignment model were earning their keep.)

Let me compare this against the one prior way I know to make "look at part of an input" differentiable: Graves's soft window for handwriting synthesis. He convolves the input with a mixture of Gaussians, w_t = Σ_u φ(t,u) c_u with φ(t,u) = Σ_k α_t^k exp(−β_t^k(κ_t^k − u)²), and slides the window forward by a positive learned offset, κ_t = κ_{t-1} + exp(κ̂_t). That's also a soft, differentiable read over an input sequence, so it's the right family. But two of its properties are wrong for translation. First, it's *location-based*: the window is placed by a coordinate κ moving along the input axis, not by matching the content of what's there to what the decoder needs. Second, and fatally, κ only ever *increases* — the window is monotonic, it can only move forward. Translation reorders: French and English put adjectives and nouns in different orders, German moves verbs to the end, so the target frequently has to jump back to an earlier source word and then forward again. A forward-only window can't express that. My e_{ij} = a(s_{i-1}, h_j) scores *any* pair (i, j) by content, with no monotonicity baked in, so α_i can put its mass wherever the content match is, including behind where it just looked. That's the difference that makes content-based soft alignment work for translation where the monotonic location window wouldn't. And unlike the Gaussian window, my α_{ij} is softmax-normalized into an actual distribution over positions, which is what gave me the length-invariant convex-combination read.

Soft alignment also buys something beyond differentiability, past the elided-article case that motivated it: it handles fertility — one source word spawning several target words, or several collapsing to one — and differing source/target lengths, with no special [NULL] machinery the way phrase-based alignment needs. Softness is paying off twice: it's what makes the thing trainable, and it's what matches how alignment actually behaves.

Now let me fix the recurrences concretely, because the gating is what keeps gradients alive over a fifty-word sentence and I shouldn't hand-wave it. I'll use a gated unit of the reset/update kind. For the decoder state, with the context c_i now an input alongside the previous target embedding E y_{i-1} and previous state s_{i-1}:

  z_i = σ(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i)        # update gate
  r_i = σ(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)        # reset gate
  s̃_i = tanh(W E y_{i-1} + U(r_i ⊙ s_{i-1}) + C c_i)    # candidate state
  s_i = (1 − z_i) ⊙ s_{i-1} + z_i ⊙ s̃_i

The update gate z_i interpolates between keeping the old state and taking the candidate — when z_i is near zero the unit copies s_{i-1} forward unchanged, which is the constant-derivative path that lets gradients flow back many steps without vanishing; the reset gate r_i lets the candidate ignore the past and read mostly from the current inputs when a fresh start is wanted. The context c_i enters all three through its own matrices C, C_z, C_r, so the source read can drive the gate decisions and the candidate, not just be appended. The encoder uses the same gated recurrence in each direction (without the c terms, since the encoder has no context to read). I'll seed the decoder from the source: s_0 = tanh(W_s ⃖h_1), the backward annotation at position 1 — the encoder summary nearest the *start* of the source, which is the natural place a left-to-right generation begins. (An LSTM gated unit would serve the same role; the reset/update unit is the lighter choice with the same long-range-gradient property, so I'll go with it.)

For the output, I don't want to go straight from the state to a vocabulary softmax — give the model a bit of nonlinear capacity to combine the three things it knows at emission time: the pre-state s_{i-1}, the previous word E y_{i-1}, and the context c_i. Form a pre-activation t̃_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i, then pool adjacent pairs with a maxout, t_{i,j} = max(t̃_{i,2j-1}, t̃_{i,2j}) for j = 1..l, giving a piecewise-linear nonlinearity, and finally p(y_i | s_i, y_{i-1}, c_i) ∝ exp(y_i^T W_o t_i) — a softmax over the target vocabulary. The maxout is a single deep-output hidden layer; it lets the readout be nonlinear in (state, last word, context) instead of a bare affine map, which matters because the context now carries source content that should interact nonlinearly with what was just generated.

This should be a strict generalization of where I started, not a different architecture — fix c_i to a constant, say ⃗h_{T_x} (the last forward annotation), independent of i. Then every target step reads the same vector, the alignment does nothing, and I'm back to the plain RNN encoder-decoder with a single fixed summary. So the old model is exactly the α_{ij}-frozen corner of the new one: I haven't replaced the architecture, I've freed one constraint inside it (the constant context) and let the data decide.

And the alignment is interpretable for free: the matrix [α_{ij}] is a soft alignment between target and source, so it should read roughly diagonal for a language pair that's mostly monotonic, with off-diagonal mass exactly where reordering happens — jumping over a reordered adjective-noun pair and coming back, the very thing the monotonic window couldn't do, and spreading across two source words when emitting a single fused target word. That's a mechanism-level prediction I can check the trained model against, not a number I'm claiming in advance.

Now the training and decoding glue, which carries over unchanged from the encoder-decoder family. Train end to end by maximizing the conditional log-likelihood (1/N) Σ_n log p_θ(y_n | x_n) — equivalently masked cross-entropy summed over target positions — with gradients flowing through everything, including the alignment model, because every piece is differentiable. RNN gradients can still blow up even when they don't vanish, so clip the global gradient norm to a threshold (rescale g ← g·τ/‖g‖ when ‖g‖ > τ, τ = 1). Optimize with minibatch SGD using a per-parameter adaptive step (Adadelta, ρ=0.95, ε=1e-6) so I don't hand-tune one global rate across the very differently scaled weight groups. Initialize the recurrent matrices as random orthogonal (so the recurrence neither shrinks nor explodes activations at the start), the alignment matrices W_a, U_a from a tiny N(0, 0.001²) and v_a and biases at zero (so alignments start near-uniform and sharpen as training finds real matches), and the other weights from N(0, 0.01²). At decode time, beam search left to right: keep the B best partial hypotheses, extend each with every next word, prune to B, and finish a hypothesis when it emits end-of-sentence.

Let me write it as code, filling the one empty slot in the harness — how the decoder reads the source at each step — with the soft-search alignment, and wiring the gated decoder and maxout output around it. Keeping it to the load-bearing structure (one batch element's worth of logic vectorized over the batch):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRUEncoder(nn.Module):
    """Bidirectional gated encoder: annotation h_j = [forward_j ; backward_j],
    so each h_j summarizes the whole sentence with a focus around position j."""

    def __init__(self, vocab, emb_dim, hid_dim, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.birnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)

    def forward(self, src):                      # src: [B, T_x]
        ann, _ = self.birnn(self.embed(src))     # ann: [B, T_x, 2*hid_dim]  (the h_j)
        return ann


class AdditiveAlignment(nn.Module):
    """The source-conditioning slot: a content-based soft search over source
    positions. Scores each (query s_{i-1}, key h_j) with a one-hidden-layer MLP,
    softmax-normalizes into alignment weights, returns the expected annotation c_i."""

    def __init__(self, dec_dim, ann_dim, align_dim):
        super().__init__()
        self.W_a = nn.Linear(dec_dim, align_dim, bias=False)   # project the query  s_{i-1}
        self.U_a = nn.Linear(ann_dim, align_dim, bias=False)   # project the keys   h_j
        self.v_a = nn.Linear(align_dim, 1, bias=False)         # read off the scalar score

    def precompute_keys(self, ann):              # U_a h_j is independent of i: do it once
        return self.U_a(ann)                     # [B, T_x, align_dim]

    def forward(self, s_prev, ann, keys, mask):  # s_prev: [B, dec_dim], ann/keys: [B, T_x, *]
        # e_{ij} = v_a^T tanh(W_a s_{i-1} + U_a h_j)
        e = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + keys)).squeeze(-1)  # [B, T_x]
        e = e.masked_fill(~mask, float('-inf'))                  # ignore padding positions
        alpha = F.softmax(e, dim=-1)                             # distribution over positions
        c = torch.bmm(alpha.unsqueeze(1), ann).squeeze(1)        # c_i = sum_j alpha_ij h_j
        return c, alpha


class RNNsearchGRUCell(nn.Module):
    """Decoder recurrence with context entering the update gate, reset gate, and candidate."""

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
    """Gated decoder conditioned on a per-step context, with a maxout deep output."""

    def __init__(self, vocab, emb_dim, dec_dim, ann_dim, align_dim, maxout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb_dim, padding_idx=pad_id)
        self.align = AdditiveAlignment(dec_dim, ann_dim, align_dim)
        self.cell = RNNsearchGRUCell(emb_dim, ann_dim, dec_dim)
        self.init_s = nn.Linear(ann_dim // 2, dec_dim)           # s_0 = tanh(W_s backward_1)
        # deep output: t~_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i, then maxout pairs
        self.U_o = nn.Linear(dec_dim, 2 * maxout, bias=False)
        self.V_o = nn.Linear(emb_dim, 2 * maxout, bias=False)
        self.C_o = nn.Linear(ann_dim, 2 * maxout, bias=False)
        self.W_o = nn.Linear(maxout, vocab)

    def forward(self, ann, mask, tgt_in):        # tgt_in: [B, T_y] teacher-forced prev tokens
        keys = self.align.precompute_keys(ann)
        backward_1 = ann[:, 0, ann.size(-1) // 2:]              # backward annotation at pos 1
        s = torch.tanh(self.init_s(backward_1))                # s_0
        logits = []
        for i in range(tgt_in.size(1)):
            y_emb = self.embed(tgt_in[:, i])                   # E y_{i-1}
            s_prev = s
            c, _ = self.align(s_prev, ann, keys, mask)         # query is s_{i-1}: the pre-step state
            s = self.cell(y_emb, c, s_prev)                    # s_i = (1-z)*s_{i-1} + z*s_tilde
            t_tilde = self.U_o(s_prev) + self.V_o(y_emb) + self.C_o(c)
            t = t_tilde.view(t_tilde.size(0), -1, 2).max(dim=-1).values   # maxout over pairs
            logits.append(self.W_o(t))
        return torch.stack(logits, dim=1)                      # [B, T_y, vocab]


class RNNSearch(nn.Module):
    def __init__(self, vocab, emb_dim=620, hid=1000, align_dim=1000, maxout=500, pad_id=0):
        super().__init__()
        self.encoder = BiGRUEncoder(vocab, emb_dim, hid, pad_id)
        self.decoder = AttnDecoder(vocab, emb_dim, hid, 2 * hid, align_dim, maxout, pad_id)
        self.pad_id = pad_id

    def forward(self, src, tgt_in):
        ann = self.encoder(src)                                # annotations h_1..h_{T_x}
        mask = src.ne(self.pad_id)                             # valid source positions
        return self.decoder(ann, mask, tgt_in)                 # next-token logits over target
```

That closes the gap I started from: the fixed-length c was an information bottleneck that the reverse-source trick only ever patched by shortening the distance information had to travel, and letting the decoder softly search all the encoder annotations at every step removes the bottleneck outright rather than rearranging it. Every piece above was forced by that goal, not decorative — the softmax for a length-invariant read, the additive scorer because query and key don't share a space, s_{i-1} as query to avoid circularity, the bidirectional encoder for keys that see both sides of a word — and the old fixed-vector encoder-decoder still sits inside this one as the α-frozen special case.
