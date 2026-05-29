OK, let me think this through from scratch. I want one neural network that reads a sentence in one language and writes it in another, trained end to end. The probabilistic statement is the easy part: translating is finding the target sentence y that makes p(y | x) as large as possible for a source sentence x, and I fit that conditional on a parallel corpus and then search for a good y at test time. So the whole job is to build a good model of p(y | x). The hard part is the shape of that model, because x and y are both variable-length, and a feedforward net wants fixed-size things.

The way everyone does this right now is the encoder–decoder. An encoder RNN walks the source left to right, h_t = f(x_t, h_{t-1}), and at the end you take the whole sentence and crush it into a single vector — Sutskever, Vinyals and Le just take the last LSTM state, c = h_Tx; Cho and colleagues write it more generally as c = q({h_1,…,h_Tx}) but in practice it's still one vector. Then a decoder RNN unrolls the translation out of that vector, factorizing left to right, p(y) = ∏_t p(y_t | y_1,…,y_{t-1}, c), and modeling each factor as g(y_{t-1}, s_t, c) with s_t the decoder's hidden state. It's elegant. One network, one loss, jointly trained.

But stare at that c for a second. It's a fixed number of floats — say a thousand. It has to hold every content word, every modifier, the subject and the object and which is which, the tense, the word order, and any long-range agreement, for a sentence whose length has no upper bound. And then the decoder reuses that same c at step 1 and at step 30 and at step 50. There's something obviously uncomfortable here: the amount of information in a sentence grows with its length, but the channel I'm forcing it through is a constant width. For a short sentence, fine. For a long one, I'm asking a thousand floats to be a lossless summary of fifty words, and then to still be the thing the decoder leans on when it's emitting the fortieth target word and needs one specific source word from the far end.

And this isn't just a hunch — I've seen it measured. When you take the plain encoder–decoder and plot quality against source length, it's fine on short sentences and then it falls off a cliff once the sentence gets past twenty or thirty words, and it's worse still on sentences longer than anything in training. That's the symptom of exactly this bottleneck: the fixed vector can't keep up as the sentence grows. So the fixed-length c isn't a stylistic choice I can shrug at; it's where the model is bleeding.

So how do I get rid of the bottleneck? My first instinct is the cheap one: just make c bigger. Use a 4000-dimensional state instead of 1000. But that's not a fix, it's a delay — whatever width I pick, there's a sentence long enough to overflow it, and I pay the cost on every short sentence too. The bottleneck is structural: it's that the source, no matter how long, gets compressed to a constant size *before the decoder ever looks at it*, and the decoder never gets to go back to the source. Making the bucket bigger doesn't change that I have one bucket.

Let me flip the question. Why am I compressing the source to one vector at all? Because the decoder is a fixed-size machine and I thought it needed a fixed-size input. But the decoder runs step by step. At each step it's producing one target word, and that word corresponds to — depends on — only a small part of the source, not all of it. When I'm emitting the French word for "economic", I mostly care about the English word "economic" and its neighbors; I don't need the other end of the sentence in full detail right then. So the thing the decoder needs at step i isn't a summary of the whole sentence; it's the *relevant piece* of the source for step i. The fixed vector forces the relevant piece and the irrelevant rest to be averaged together once, up front, for all steps at once. That's the waste.

So don't summarize once. Keep the source around as a sequence of vectors — one per source position — and let the decoder, at each of its own steps, reach back and pull out the part it needs right now. The encoder stops being a funnel down to one vector and becomes a memory the decoder can re-read. The decoder's job at step i is to figure out which source positions matter for the word it's about to write, and read those.

That immediately changes the decoder factor. Instead of every step sharing one c, each target word i gets its own context vector c_i: p(y_i | y_1,…,y_{i-1}, x) = g(y_{i-1}, s_i, c_i), with the decoder state s_i = f(s_{i-1}, y_{i-1}, c_i). The structural difference from the old model is exactly that the conditioning vector is now distinct per output word instead of frozen. Good — that's the shape I want. Now I have to actually build two things: the per-position source representations, and the rule that turns "which positions matter for step i" into c_i.

Take the source representations first — call them annotations h_1,…,h_Tx, one per source position. What do I want h_j to be? I want it to represent source position j *in context*. An ordinary forward RNN gives me a state that summarizes x_1 through x_j — everything up to and including j, but nothing after. That's wrong for this purpose: when the decoder is deciding whether source word j is the one it needs, the meaning and even the correct translation of word j can depend on the words that come *after* it. (Think of a determiner whose form depends on the noun that follows.) So the annotation for position j has to see both sides. That's exactly what a bidirectional RNN buys me. Run a forward RNN to get →h_1,…,→h_Tx, where →h_j summarizes x_1…x_j; run a separate backward RNN over the reversed sentence to get ←h_1,…,←h_Tx, where ←h_j summarizes x_j…x_Tx; then glue them together, h_j = [→h_j ; ←h_j]. Now h_j carries both the left context and the right context, and because RNNs naturally emphasize recent inputs, each of the two halves is most strongly about the words near position j — so h_j ends up being a representation centered on x_j but aware of the whole sentence. That's the per-position memory I wanted, and it falls straight out of "the annotation must see both directions."

Now the real question: given the annotations h_1,…,h_Tx and where the decoder is in its own generation, how do I produce c_i? I want c_i to be "the source information relevant for emitting target word i." The honest version of "relevant" is a selection: pick the source position (or few positions) that target word i comes from, and read those annotations. This is exactly the alignment idea from statistical MT — which source word does this target word correspond to. So I could try to make this a hard choice: at step i, select the single best source position j*, set c_i = h_{j*}, and translate from it.

Let me try that and see where it breaks. The selection itself — which j is best for step i — has to be a function of the network's parameters, because I want to *learn* alignment, not hand-code it. So I'd score each position and take the argmax. But the moment I take an argmax, the choice is discrete: c_i jumps from one annotation to another with no in-between, and there's no gradient telling the scorer "you should have leaned a bit more toward position j+1." Backpropagation needs the path from the loss to the alignment scorer to be differentiable, and a hard pick severs it. The classical MT answer to discreteness is to treat the alignment as a latent variable and integrate it out with EM — but that's a separate estimation procedure bolted onto the side, exactly the multi-component pipeline I'm trying to get away from, and it doesn't fit inside one network trained by gradient descent. There's also the awkward NULL-token machinery those models need so that words can align to nothing. A hard, latent alignment is the wrong tool if I want everything trained jointly by backprop.

So relax the selection. Instead of picking one position, take *all* the annotations and form a weighted average, where the weights say how relevant each position is. Let α_ij be a weight on annotation h_j for target step i, with the weights nonnegative and summing to one over j, and set

  c_i = Σ_{j=1}^{Tx} α_ij h_j.

If the weights concentrate on one position, this recovers the hard selection; if they spread, it blends a few positions. And crucially it's smooth: c_i is a differentiable function of every α_ij, so gradient can flow back into whatever produces the weights. There's a clean interpretation too. If I read α_ij as the probability that target word y_i is aligned to — translated from — source word x_j, then c_i = Σ_j α_ij h_j is just the *expected annotation* under that alignment distribution. So I haven't thrown away the alignment idea from statistical MT; I've made it soft. The alignment isn't a discrete latent variable estimated by a separate EM step — it's a deterministic, differentiable quantity computed inside the network, and the gradient of the translation loss flows straight through it and trains the aligner jointly with everything else. The hard-vs-soft distinction also pays off linguistically, almost for free: a hard alignment forces every target word to commit to one source word, but a soft weight can keep mass on two source words at once, which is what you want when a target word's form genuinely depends on more than one source word; for a target word with no single lexical source, the distribution can spread over context instead of requiring a special NULL choice.

Now I have to produce the weights α_ij, and they have to come from a comparison: how well does source position j match what the decoder needs at step i. What represents "what the decoder needs at step i"? The decoder's own state. Here I have to be careful about *which* state. The natural state to use is s_{i-1}, the decoder state just *before* it emits y_i. Why the previous one and not s_i? Because s_i is defined as f(s_{i-1}, y_{i-1}, c_i) — it depends on c_i, which depends on the weights, which is what I'm trying to compute; using s_i to compute c_i would be circular. And s_{i-1} is the right thing conceptually anyway: it summarizes everything generated so far, which is exactly the query "given what I've produced, what do I need to read next?" So I score each source position by comparing s_{i-1} against h_j, producing a raw energy e_ij, and then normalize the energies into weights.

The normalization is the easy half: I want nonnegative weights summing to one over the source positions, and I want the comparison to translate into "share of attention," so a softmax over j is the obvious choice,

  α_ij = exp(e_ij) / Σ_{k=1}^{Tx} exp(e_ik).

The harder half is the scoring function e_ij = a(s_{i-1}, h_j) — how do I compare a decoder state to a source annotation? The first thing I reach for is a dot product, e_ij = s_{i-1} · h_j, since that's the cheapest similarity. But it doesn't even typecheck: s_{i-1} lives in R^n (the decoder has n hidden units), while h_j is a concatenation of a forward and a backward state, so it lives in R^{2n}. I can't dot an n-vector with a 2n-vector. I'd have to insert a projection just to make the shapes line up, and once I'm projecting anyway, there's no reason to restrict the comparison to a bilinear form. So let me just learn the comparison directly with a small neural net: project the query and the key each into a common space and let a nonlinearity and a final vector decide the score. Concretely a one-hidden-layer MLP,

  e_ij = a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j),

with W_a mapping the decoder state (R^n) into the hidden comparison space, U_a mapping the annotation (R^{2n}) into that same space, the tanh giving it the freedom to be a nonlinear match rather than a fixed inner product, and v_a reading the hidden activation out to a single scalar energy. This is a feedforward network jointly trained with the rest — the aligner is just more parameters in the same model, learned by the same backprop, and that's the whole point of having made the alignment soft.

Let me check the cost of this, because it's the one place this design is clearly more expensive than the old encoder–decoder. The score a(s_{i-1}, h_j) has to be evaluated for every source position j and every target position i — that's Tx × Ty evaluations per sentence pair, a full matrix, where the old model paid essentially nothing per step beyond running the decoder. For translation that's tolerable: sentences are mostly fifteen to forty words, so the matrix is small. But I can also shave it. Look at the score: W_a s_{i-1} changes every decoder step, but U_a h_j depends only on the annotation, not on i. So I can compute all the U_a h_j once per sentence, up front, and at each decoder step only recompute W_a s_{i-1} and add. That cuts the per-step alignment work roughly in half and makes the Tx × Ty matrix cheap to fill.

Now wire the context into the decoder. The decoder is a gated recurrent unit — I want gating because a plain tanh RNN can't carry information across the dozens of steps a translation takes; gradients vanish through the recurrence, and the gated cell keeps computation paths whose product of derivatives stays near one so gradient survives. The GRU has an update gate z deciding how much candidate state to write, so 1 − z keeps the old state; a reset gate r decides how much old state feeds the candidate; and the new state is a gated interpolation. The only change from the standard cell is that the per-step context c_i now feeds in alongside the previous output embedding and the previous state, in all three places:

  s̃_i = tanh(W E y_{i-1} + U[r_i ∘ s_{i-1}] + C c_i),
  z_i = σ(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i),
  r_i = σ(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i),
  s_i = (1 − z_i) ∘ s_{i-1} + z_i ∘ s̃_i,

where E is the target embedding matrix so E y_{i-1} is the embedding of the previously generated word, and the C-matrices are the new weights that let the freshly-read source context steer both the candidate state and the two gates. I need to start the recurrence somewhere; s_0 = tanh(W_s ←h_1) is a sensible choice — initialize from the backward encoder state at the first source position, which through the backward RNN already summarizes the whole source sentence from its first word onward.

For the output distribution, rather than a single linear-softmax I use a slightly deeper readout — it helps to give the emission its own nonlinear layer before the vocabulary projection. The detailed readout uses the same pre-emission state that queried the source, the previous word, and the context: t̃_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i. Then a maxout layer takes the max over consecutive pairs, t_i = [ max(t̃_{i,2k-1}, t̃_{i,2k}) ]_k, and a linear-plus-softmax maps to the target vocabulary, p(y_i | y_{<i}, x) ∝ exp(y_i^T W_o t_i). The maxout just gives the readout a cheap nonlinearity by selecting between two linear pieces per unit.

Let me sanity-check that this includes the model I started from. The old encoder–decoder is the special case where the context never changes: fix c_i to a single source summary for every i, with matching context dimensionality; in the usual RNN encoder–decoder that summary is the final forward encoder state →h_Tx. Then c_i no longer depends on i or on any alignment weights, the alignment MLP and the softmax do nothing, and the decoder reduces to g(y_{i-1}, s_i, c) with one frozen c — exactly the original. So the plain encoder–decoder sits inside this model as the degenerate fixed-context configuration, and learning is free to recover it if that were ever best. It won't be, because the frozen-context configuration is precisely the bottleneck I'm trying to escape, but it's reassuring that the new model can't do worse in principle.

There's one more design comparison worth settling, because a soft, learned alignment already exists in a neighboring problem. In handwriting synthesis, Graves makes a generator attend to a character sequence with a differentiable, location-based scheme — a mixture of Gaussian kernels whose centers are predicted — but the centers are constrained to advance monotonically, the attention only ever moves forward. For handwriting that's right, because you write characters in order. For translation it would be a real limitation: getting a grammatical target constantly requires reordering — adjectives and nouns swap between languages, and longer-range movement is common — so the place I need to read in the source can jump backward relative to where I just read. My content-based score e_ij = a(s_{i-1}, h_j) has no monotonicity baked in; it scores every source position on its merits for the current step, so it can attend anywhere and reorder freely. The price is that I score all Tx positions every step instead of nudging a single moving location, but for sentence-length inputs that's the cost I already decided was acceptable.

For training there's nothing exotic: maximize the log-probability of the correct translation, ∑ log p(y | x), by minibatch SGD. RNNs like this are prone to exploding gradients, so I clip the global gradient norm to a threshold of one before each step (Pascanu, Mikolov & Bengio). I use Adadelta to adapt per-parameter learning rates so I don't have to hand-tune a schedule. Initialization matters for the recurrence — random orthogonal matrices for the recurrent weights so the state map preserves gradient norm to start — and the alignment net wants small, near-zero initial weights (sample W_a and U_a from a tight Gaussian, set v_a and the biases to zero) so attention starts roughly uniform and the energies don't saturate the tanh before there's any signal. At decode time I don't take greedy argmaxes; I run a beam search to approximately maximize the conditional probability.

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Bidirectional GRU: turn the source into a sequence of per-position
    annotations the decoder can re-read, plus an initial decoder state."""
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional=True)
        # s_0 = tanh(W_s backward_h_1)
        self.init_state = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):                              # src: [src_len, batch]
        embedded = self.dropout(self.embedding(src))
        # outputs[j] = h_j = [forward_j ; backward_j]  -> the annotations
        outputs, hidden = self.rnn(embedded)            # outputs: [src_len, batch, 2*n]
        # hidden[-1] is the backward state at source position 1
        hidden = torch.tanh(self.init_state(hidden[-1]))
        return outputs, hidden                           # annotations, s_0


class AdditiveAlignment(nn.Module):
    """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j), then softmax over j."""
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.key = nn.Linear(hidden_dim * 2, attn_dim, bias=False)
        self.energy = nn.Linear(attn_dim, 1, bias=False)

    def precompute(self, encoder_outputs):
        return self.key(encoder_outputs)               # U_a h_j: [src_len, batch, attn]

    def forward(self, hidden, projected_annotations):   # hidden: [batch, n]
        query = self.query(hidden).unsqueeze(0)         # [1, batch, attn]
        e = self.energy(torch.tanh(query + projected_annotations)).squeeze(2)
        return torch.softmax(e.transpose(0, 1), dim=1)  # alpha_ij: [batch, src_len]


class DecoderGRUCell(nn.Module):
    """The decoder GRU equations with c_i entering the candidate and both gates."""
    def __init__(self, emb_dim, context_dim, hidden_dim):
        super().__init__()
        self.candidate_x = nn.Linear(emb_dim, hidden_dim)
        self.candidate_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_c = nn.Linear(context_dim, hidden_dim, bias=False)
        self.update_x = nn.Linear(emb_dim, hidden_dim)
        self.update_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_c = nn.Linear(context_dim, hidden_dim, bias=False)
        self.reset_x = nn.Linear(emb_dim, hidden_dim)
        self.reset_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_c = nn.Linear(context_dim, hidden_dim, bias=False)

    def forward(self, embedded, prev_hidden, context):
        z = torch.sigmoid(
            self.update_x(embedded) + self.update_h(prev_hidden) + self.update_c(context)
        )
        r = torch.sigmoid(
            self.reset_x(embedded) + self.reset_h(prev_hidden) + self.reset_c(context)
        )
        candidate = torch.tanh(
            self.candidate_x(embedded) + self.candidate_h(r * prev_hidden) + self.candidate_c(context)
        )
        return (1 - z) * prev_hidden + z * candidate


class Decoder(nn.Module):
    """One GRU step conditioned on the previous word and the per-step context
    c_i = sum_j alpha_ij h_j; maxout readout to the vocabulary."""
    def __init__(self, output_dim, emb_dim, hidden_dim, maxout_dim, dropout, alignment):
        super().__init__()
        self.output_dim = output_dim
        self.maxout_dim = maxout_dim
        self.alignment = alignment
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn_cell = DecoderGRUCell(emb_dim, hidden_dim * 2, hidden_dim)
        self.readout = nn.Linear(hidden_dim + emb_dim + hidden_dim * 2, maxout_dim * 2)
        self.fc_out = nn.Linear(maxout_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, projected_annotations):
        embedded = self.dropout(self.embedding(input))              # [batch, emb]
        alpha = self.alignment(hidden, projected_annotations)       # [batch, src_len]
        annotations = encoder_outputs.permute(1, 0, 2)              # [batch, src_len, 2*n]
        # c_i = sum_j alpha_ij h_j  (expected annotation)
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)  # [batch, 2*n]
        prev_hidden = hidden
        hidden = self.rnn_cell(embedded, prev_hidden, context)           # s_i
        readout = self.readout(torch.cat((prev_hidden, embedded, context), dim=1))
        maxout = readout.view(readout.shape[0], self.maxout_dim, 2).max(dim=2).values
        prediction = self.fc_out(maxout)
        return prediction, hidden, alpha


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, teacher_forcing_ratio):
        trg_len, batch = trg.shape
        outputs = torch.zeros(trg_len, batch, self.decoder.output_dim, device=trg.device)
        encoder_outputs, hidden = self.encoder(src)     # annotations, s_0
        projected_annotations = self.decoder.alignment.precompute(encoder_outputs)
        input = trg[0]                                  # <sos>
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, projected_annotations)
            outputs[t] = output
            teacher = torch.rand((), device=trg.device).item() < teacher_forcing_ratio
            input = trg[t] if teacher else output.argmax(1)
        return outputs
```

The causal chain in one breath: one fixed vector can't hold a whole source sentence, and that shows up as quality collapsing on long inputs; so keep the source as a bidirectional sequence of annotations the decoder can re-read instead of crushing it; let the decoder build a fresh context per step by relevance; "relevance" wants to be a hard pick of one source word but that's non-differentiable and pulls in the separate EM machinery I'm avoiding, so soften it to a softmax-weighted average of annotations — an expected annotation under a differentiable alignment that backprop trains jointly with everything else; score relevance by comparing the pre-emission decoder state s_{i-1} to each annotation with a small MLP because their dimensions don't match for a dot product; feed that context through the gated decoder and the maxout readout — and the plain encoder–decoder drops out as the fixed-context corner of this model.
