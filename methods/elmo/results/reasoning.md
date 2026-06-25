OK, let me think this through from the pain. Every model I build starts the same way: I take a sentence, look up each word in a table of pretrained vectors — word2vec or GloVe — and feed those vectors into whatever encoder the task needs. And there's a thing about that lookup table that has bothered me for a while. The word "play" gets exactly one vector. One. The same vector whether I wrote "Chico Ruiz made a spectacular play on Alusik's grounder" or "Olivia de Havilland signed to do a Broadway play." Those are different words in any sense that matters — different part of speech in some cases, certainly different meaning — and the table cannot tell them apart, because the table is indexed by the string, not by the usage. So whatever vector GloVe assigns to "play" is some averaged smear over the sports sense, the theatrical sense, the verb, the children's-game sense, all collapsed onto one point. A downstream model that receives that point can never recover which one I meant.

Stare at this for a second, because it's the crux and I want to name it precisely. The lookup table assigns a representation to a word *type*. What I actually want is a representation of a word *token* — this particular occurrence, in this particular sentence. And the only thing that distinguishes one occurrence of "play" from another is the rest of the sentence. So what I want is: a representation of a word that is a function of the entire input sentence. Same string, different sentence, different vector. That's the whole goal in one line.

And there's a second thing tangled up in that single GloVe vector that I want to pull apart. A word's "play"-ness has at least two very different facets. One is syntactic — what role does it play in the sentence, is it a noun here or a verb. The other is semantic — which of its meanings is active. A single fixed vector fuses those into one point, and a task can't ask for one without dragging along the other. POS tagging mostly wants the syntax; word-sense disambiguation wants the meaning. If a representation could keep those somewhat separable, a task could lean on whichever it needs. I'll keep that hope on the shelf; I don't yet know if anything I build will actually separate them.

So: where would a sentence-dependent representation of a word come from? I need something that reads the whole sentence and, at each position, produces a vector that has absorbed the surrounding context. And crucially I need it to come from *unlabeled* text, because the labeled datasets for the tasks I care about — entailment, QA, SRL, coreference — are all small, far too small to teach a model what every word means in every context. The knowledge of word usage has to be transferred in from a large corpus, and large corpora are unlabeled.

What reads a whole sentence and is trained on raw unlabeled text? A language model. p(t_1,...,t_N) = prod_k p(t_k | t_1,...,t_{k-1}). You train it by maximum likelihood to predict the next token, no labels at all, on as much text as you can find. And to predict t_{k+1} well, the model has to build, at each position k, an internal state that summarizes everything relevant about t_1,...,t_k. That internal state is a context-dependent representation of the prefix. The modern recipe is: a context-independent token embedding x_k, then L layers of LSTM, and the top LSTM state at position k goes through a softmax over the vocabulary to predict t_{k+1}. The LSTM hidden state h_k is a function of the whole left context by construction. So a language model produces, as a side effect of its training objective, exactly the kind of context-conditioned word state I've been asking for — the hidden states are candidates for the representation I want, if I can get at them.

Has anyone actually used LM internals this way? Yes — there's prior work that pretrains an LM on unlabeled text, freezes it, and concatenates its top-layer hidden state into a sequence tagger, and it helps. And there's the context-vector idea (call it CoVe), which does something adjacent: train a neural machine-translation system, take the top layer of its biLSTM *encoder*, and use that as a contextual vector. CoVe genuinely makes the vectors context-dependent and improves several tasks. So the basic recipe — pretrain an encoder, freeze it, use its hidden states as features — is on the table. But each of these has a piece I want to look hard at.

Take CoVe first. It's trained on parallel translation data. Parallel corpora are *tiny* compared to monolingual text — you need aligned sentence pairs, which someone had to produce. So the encoder sees comparatively little text. If my whole bet is that contextual knowledge comes from scale, training on parallel data is fighting my own thesis. A language model has no such constraint: any raw text in the world is training data. So I lean toward the LM objective, not the MT objective, precisely so I can train on something like a billion words.

Now the deeper question, shared by CoVe *and* the prior biLM-tagging work: they both use only the *top* layer of their encoder. CoVe takes the top encoder layer; the biLM tagger takes the top LM layer. Is the top layer actually the right thing to take? There's a body of observations about what different layers of a deep recurrent network encode. When people supervise a low-level task like POS tagging at the *lower* layers of a deep network, the high-level tasks at the top improve — as if the lower layers naturally want to hold syntax. In an MT encoder, the first-layer representations predict POS tags *better* than the second-layer ones, even though the two-layer system has higher BLEU. And the top layer of a context biLSTM has been shown to capture word sense. Put those together and a picture forms: in a deep contextual encoder, lower layers may lean syntactic, higher layers may lean semantic. That would be the very syntax-vs-semantics split I put on the shelf earlier.

If that picture is right, the top-layer convention is leaving information on the floor: a POS tagger would rather have the lower layer, a WSD model the upper, and no single layer is best for both. But this is exactly the kind of thing I tend to talk myself into by stringing together other people's results — I should say what would actually distinguish "lower=syntax, upper=semantics" from "I'm pattern-matching slogans." The clean test: freeze a two-layer LM encoder; for each layer separately, feed its activations into a tiny probe — a 1-nearest-neighbor classifier over per-sense averages for word sense, a linear classifier for POS — and read off the accuracy. The prediction that would falsify the slogan version: if both layers were equally good at both probes, there'd be nothing to gain from per-layer weighting. The prediction that would support it: the first layer wins on POS, the second wins on WSD, and the gap reverses sign between the two tasks. I can't run that here — it needs a pretrained biLM and the WSD/POS evaluation sets, none of which I have on this page — so I'll hold it as a hypothesis I expect to confirm, not a settled fact, and design so that *if* it holds the design exploits it and *if* it fails I've lost little. The way to be robust to both outcomes is the same: don't hard-code a layer choice; make the layer mixture something the data decides. If the two layers turn out identical, the data will just learn equal weights and I've lost nothing.

So the design pressure points away from picking a layer. But "use all of them" isn't yet a representation — I have to say *how* a downstream model consumes L+1 vectors per token (the token-embedding layer plus the L LSTM layers) instead of one. Let me hold that thought; first I have to settle the encoder itself, because there's an asymmetry problem.

A plain forward LM only ever sees the *left* context: h_k is a function of t_1,...,t_k. But a contextual representation of "play" in the middle of a sentence should obviously depend on the words *after* it too — "a Broadway play for Garson" needs "Broadway" before and arguably the rest after. A unidirectional LM gives me only half the context. The fix is the same trick that makes biLSTM taggers work: also run a *backward* LM, one that factorizes the sequence in reverse, p(t_1,...,t_N) = prod_k p(t_k | t_{k+1},...,t_N), predicting each token from its *future*. Implement it identically to the forward LM but reading the sequence right-to-left; its layer-j hidden state at position k, call it h←_{k,j}, summarizes the right context t_{k+1},...,t_N. Then the full contextual representation of token k at layer j is the concatenation of the two directions, h_{k,j} = [h→_{k,j}; h←_{k,j}] — both sides of the sentence.

Now, do I train two completely separate models? The prior biLM work used fully independent parameters for the two directions. But think about what each direction is: a forward LM and a backward LM are predicting tokens from the *same* vocabulary, using the *same* notion of what a token's input embedding is and the *same* softmax to score vocabulary items. The thing that genuinely must differ between them is the recurrent dynamics — how you accumulate left context is a different function from how you accumulate right context. The token embeddings and the output softmax have no inherent direction. So I'll tie them: share the token-representation parameters Θ_x and the softmax parameters Θ_s across both directions, while keeping separate LSTM parameters Θ_LSTM→ and Θ_LSTM← for each direction. Fewer parameters, and it forces both directions to agree on a common token/output space, which is the inductive bias I want. The biLM then jointly maximizes the sum of the two directions' log-likelihoods:

sum over k of [ log p(t_k | t_1,...,t_{k-1}; Θ_x, Θ_LSTM→, Θ_s) + log p(t_k | t_{k+1},...,t_N; Θ_x, Θ_LSTM←, Θ_s) ].

One objective, one forward pass that produces both directions' states, parameters shared where it makes sense.

Before I go on, the input layer x_k. I could use a normal word-embedding table for the LM input, but that drags the OOV problem right back in — and contextual representations are most valuable for exactly the rare and unseen words a fixed vocabulary can't handle. So make the input purely *character*-based: a small convolutional network over character embeddings. Embed each character, run several 1-D convolutions of different widths across the characters of the word, max-pool each over the character dimension, concatenate the filter outputs, push through a couple of highway layers, and linearly project to the token dimension. This gives a context-independent token vector x_k for *any* string — in or out of vocabulary — and it picks up morphology (prefixes, suffixes, capitalization) for free, which is itself useful. So the biLM provides a representation even for words it never saw in training, which a word-table LM simply cannot.

Now back to the real question: I have, for each token k, a stack of vectors — the input layer and L layers per direction. Let me count them carefully, because I'll need the count to line up with the code. Layer 0 is the context-independent token layer x_k^LM. It is not directional, but once the recurrent layers are represented as [forward; backward] vectors, the implementation can duplicate the token vector as [x_k^LM; x_k^LM] so every layer has the same width and can be mixed by scalars. Then for each of the L LSTM layers I have a forward state and a backward state, concatenated. So the set of representations for token k is

R_k = { x_k^LM, h→_{k,j}^LM, h←_{k,j}^LM : j = 1,...,L } = { h_{k,j}^LM : j = 0,...,L },

where h_{k,0}^LM denotes the token layer (implemented as [x_k^LM; x_k^LM] when the tensors need equal width) and h_{k,j}^LM = [h→_{k,j}^LM; h←_{k,j}^LM] for each biLSTM layer. Count it: 1 token vector + 2 vectors per LSTM layer × L layers = 2L+1 raw direction-vectors, packaged as L+1 layer-vectors. For the L=2 model I have in mind that's 2·2+1 = 5 raw vectors as 3 layer-vectors: token, LSTM-1, LSTM-2. Good — three things to mix.

So how do I turn these L+1 vectors into the single per-token feature a downstream model wants? The most obvious thing is to just pick the top layer, E(R_k) = h_{k,L}^LM. That's what CoVe and the biLM tagger do, and the layer-probing argument above says it discards the lower, more syntactic layers. So set "top only" aside as a thing to beat, not adopt.

Next obvious thing: average them, (1/(L+1)) sum_j h_{k,j}^LM. But a fixed average forces every task to weight syntax and semantics equally, when a POS tagger should lean on the lower layer and a WSD model on the upper. A fixed average can't adapt to which facet the task needs.

So if no fixed choice of layer is right, and the right mixture *depends on the task*, then let the task *learn* the mixture. Give each task its own set of weights over the layers, and let supervised training on that task decide how much of each layer to use. The simplest learnable combination is a weighted sum, sum_j s_j h_{k,j}^LM, with scalar weights s_j. I want these weights to behave like a soft selection over layers — nonnegative and summing to one, so the combination is a genuine convex mixture and the weights read as "how much this task relies on layer j." The clean way to get that from unconstrained parameters w_j is a softmax: s_j = softmax(w)_j. Then the task can learn to put its mass on the lower layer (POS-ish tasks) or the upper layer (sense-ish tasks) or spread it out, all by gradient descent on the task loss, with the biLM itself frozen.

There's one more thing I need, and it's not cosmetic. The hidden states of the biLM live on a completely different scale and distribution than the activations inside the downstream task model — the biLM was trained for language modeling, not for this task, and its layer norms and magnitudes are whatever they happened to be. If I just splice a softmax-weighted sum of biLM layers into the task model, the magnitudes can be badly mismatched and optimization struggles. So I add a single scalar γ that multiplies the whole combined vector, letting the task model rescale the entire biLM feature to a useful magnitude. So the feature is:

ELMo_k^task = E(R_k; Θ^task) = γ^task sum_{j=0}^{L} s_j^task h_{k,j}^LM,

with s^task = softmax(w^task) the normalized layer weights and γ^task the scalar scale, both learned per task, biLM frozen.

Now let me actually check the two claims I've been leaning on, instead of asserting them. First claim: this strictly generalizes the prior top-layer methods — they should be a special case, not a different thing. The special case should be "s puts all mass on j=L and γ=1." Let me verify by plugging numbers into the exact formula. Take L=2, so three layers j=0,1,2, and three little layer vectors h_0,h_1,h_2. To force the softmax onto j=2 I set w = (0, 0, large); softmax((0,0,30)) comes out (1.1e-13, 1.1e-13, ≈1.0) — call it (0,0,1). Then γ·(0·h_0 + 0·h_1 + 1·h_2) with γ=1 is just h_2. I ran exactly this through the LayerCombination code with random h: ‖ELMo − h_top‖ = 0.0. So yes — top-only is literally a setting of my parameters, and ELMo can only do at least as well as it; that's a real generalization, confirmed on the code, not a slogan.

Second claim, which I'll need below: that pushing the weights toward zero turns the mixture into a flat average. softmax of all-equal logits should be uniform. softmax((0,0,0)) = (1/3, 1/3, 1/3) — I computed it, and 1/3 = 0.3333 is exactly the uniform weight over 3 layers. And running the LayerCombination with its default init (all scalar parameters 0, γ=1) on three layer tensors returns exactly their mean — checked, allclose to (h_0+h_1+h_2)/3. So the default state of my module *is* the plain average, and the softmax mechanism degrades gracefully to it. Good to know: my untrained starting point is the sensible baseline, and learning can only move away from it if the task gradient says so.

Let me make sure γ is actually load-bearing and not a free knob I'm adding for symmetry. The worry is the scale mismatch between biLM activations and what the task RNN expects. Concretely: suppose the top biLM layer happens to have activation standard deviation around 5 (it was never normalized for my task), while the task model was tuned for inputs with std around 1. Inject the raw feature and the task RNN sees inputs 5× larger than it's conditioned for — that's exactly the regime where gradients are poorly conditioned. Can a single scalar fix a whole-vector scale gap? It can, because the gap is multiplicative and uniform across the vector: γ = 1/5 = 0.194 maps std 5.15 → std 1.00 (I checked: rescaled std came out 1.0000). And note *where* this bites hardest — in the single-layer special case there's no averaging over layers to soften the mismatch, so the prior-work setting (top layer only, no γ) is precisely the setting most exposed to it. So γ earns its place: it's the one parameter that lets the frozen, foreign-scaled feature meet the task model at the right magnitude. γ stays.

One refinement on the weighting. The L+1 layers don't just differ in overall scale; they have genuinely different activation *distributions* from each other. Softmax-weighting raw layers means a layer with naturally large activations contributes more to the sum regardless of how informative it is — the weight s_j and the layer's intrinsic magnitude both multiply in. So in some cases it should help to normalize each biLM layer tensor *before* the weighted sum — compute its mean and variance over the unmasked batch, time, and feature entries, rescale to zero mean and unit variance, then weight. That puts the layers on equal footing so the learned weights s_j reflect informativeness, not raw magnitude. But it's not obviously always better — normalizing throws away whatever real signal lived in a layer's magnitude — so I'll make it a switch rather than baking it in, and let each task find out.

Now, how does this feature actually enter a task model? The point of the whole approach is that I *don't* want to fine-tune the giant biLM per task — that would couple the task model's size to the biLM and require backprop through a huge network for every task. Instead: freeze the biLM, run it once over the input to get all the layer activations, and treat ELMo_k^task as an extra input feature. Concretely, most supervised NLP models start by forming a context-independent token representation x_k (word embeddings, maybe a char-CNN of their own) and then building a context-sensitive h_k with a biRNN or CNN. I inject ELMo at the bottom by concatenating it onto the token representation: replace x_k with [x_k; ELMo_k^task] and feed that into the task RNN. The rest of the task model is untouched, so this drops into arbitrarily complex architectures — an entailment model with a bi-attention layer on top, a coreference model with a span-clustering head — without any surgery beyond the concatenation.

Should ELMo only go at the input? For architectures that put an attention or span-comparison module *after* the task RNN, I also want the option to inject it at the RNN output — concatenate ELMo onto h_k as well, [h_k; ELMo_k^task], with a *separate* set of learned weights (its own s and γ), so the input and output uses can rely on different layer mixtures. Why would output injection belong in some models and not others? If there's an attention mechanism downstream, giving it direct access to the biLM's internal representations lets it attend straight to that contextual information, which it otherwise only sees filtered through the task RNN. For a task where the task-specific recurrent representation is supposed to do nearly all the contextual work, input injection may be the cleaner choice. So: input as the default, output as an optional second use-site with its own scalar mix.

Two more practical pieces. First, regularization of the layer weights w. I can add λ‖w‖^2 to the loss. Let me trace what that actually does at the limits rather than wave at it, since I already pinned down both ends. As λ → ∞, the penalty pulls w → 0; and I just checked that softmax((0,0,0)) = (1/3,1/3,1/3), so large λ drives the weighting toward a flat *average* of all layers. As λ → 0, the weights are free to move to wherever the task wants them — softmax((2,-1,0.5)) = (0.79, 0.04, 0.18), a sharp pick. So λ is a continuous dial between "trust the task to pick layers" (small λ) and "just average the layers" (large λ), with the two endpoints being exactly the two baselines I considered earlier. Which end is better should depend on data size: a task with a large training set can afford to learn free layer weights (small λ), while a task with a small training set — NER, say — would overfit if it tried to learn the weights, so the inductive bias of averaging (larger λ) is the safer default. Second, dropout on the ELMo vector itself, a moderate amount, since it's a big new feature being injected into a model that's otherwise regularized.

Let me now pin the encoder I'll actually pretrain, balancing language-model quality against the cost of running it inside downstream models. A known strong character-input LM is the CNN-BIG-LSTM architecture; I'll halve its embedding and hidden dimensions to keep it affordable as a frozen feature extractor while staying purely character-based. So: L=2 biLSTM layers, 4096 units each with a 512-dimensional projection, and a residual connection from the first to the second LSTM layer so the second layer can build on rather than replace the first. The input is the char-CNN — on the order of 2048 character n-gram convolution filters, two highway layers, linearly projected down to 512. That yields three layers of representation per token (token, LSTM-1, LSTM-2), available even for tokens outside any training vocabulary because the input is characters. Train it on something like a billion words of raw text for a handful of epochs; since both directions share the token and softmax parameters and differ only in recurrent dynamics, I'd expect forward and backward perplexities to come out in the same ballpark as each other — a useful sanity check at training time, and if they diverge sharply I'd suspect a bug in the reversal. And once it's pretrained, if a downstream domain differs from the pretraining text, I can fine-tune the biLM on that domain's raw text (ignoring the labels, just the LM objective) for an epoch before freezing, reducing the domain mismatch without spending labels.

Let me write the pieces down as code: a character encoder, the bidirectional LM that exposes all of its layers, the learned layer combination, and the injection into a task model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """Highway transform used after the character convolutions."""
    def __init__(self, size, num_layers, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(size, 2 * size) for _ in range(num_layers)])
        self.activation = activation
        for layer in self.layers:
            layer.bias.data[size:].fill_(1.0)

    def forward(self, x):
        for layer in self.layers:
            projected = layer(x)
            nonlinear, gate = projected.chunk(2, dim=-1)
            gate = torch.sigmoid(gate)
            # The gate carries the input path and the complement takes the transform.
            x = gate * x + (1 - gate) * self.activation(nonlinear)
        return x


class CharacterCNNEncoder(nn.Module):
    """Context-independent token vector from characters, so it exists for ANY
    string (in or out of vocabulary) and captures morphology. This is x_k^LM,
    i.e. layer 0 of R_k."""
    def __init__(self, n_chars=262, char_dim=16,
                 filters=((1, 32), (2, 32), (3, 64), (4, 128),
                          (5, 256), (6, 512), (7, 1024)),
                 n_highway=2, proj_dim=512):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(char_dim, n, kernel_size=w) for (w, n) in filters])
        total = sum(n for (_, n) in filters)
        self.highways = Highway(total, n_highway, activation=F.relu)
        self.proj = nn.Linear(total, proj_dim)

    def forward(self, char_ids):                 # (B, T, max_chars)
        B, T, C = char_ids.shape
        mask = (char_ids.gt(0).long().sum(dim=-1) > 0).long()
        x = self.char_emb(char_ids.view(B * T, C))    # (B*T, C, char_dim)
        x = x.transpose(1, 2)                          # (B*T, char_dim, C)
        convs = []
        for conv in self.convs:
            c = conv(x)                                # (B*T, n, C')
            c, _ = c.max(dim=-1)                        # max-pool over characters
            convs.append(F.relu(c))
        tok = torch.cat(convs, dim=-1)                 # (B*T, total)
        tok = self.highways(tok)
        tok = self.proj(tok)                           # (B*T, proj_dim)
        tok = tok.view(B, T, -1) * mask.unsqueeze(-1).float()
        return tok, mask                                # x_k^LM, layer 0


class StackedProjectedLSTM(nn.Module):
    """A stack of projected LSTMs; every layer output is returned for mixing."""
    def __init__(self, input_dim, cell_dim=4096, proj_dim=512, n_layers=2, residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        dim = input_dim
        for _ in range(n_layers):
            self.layers.append(nn.LSTM(dim, cell_dim, batch_first=True, proj_size=proj_dim))
            dim = proj_dim

    def forward(self, x, mask=None):
        outputs = []
        inp = x
        lengths = mask.sum(dim=1).clamp_min(1).cpu() if mask is not None else None
        for layer_index, lstm in enumerate(self.layers):
            if lengths is None:
                out, _ = lstm(inp)
            else:
                packed = nn.utils.rnn.pack_padded_sequence(
                    inp, lengths, batch_first=True, enforce_sorted=False)
                out, _ = lstm(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=x.size(1))
            if self.residual and layer_index != 0:
                out = out + inp
            outputs.append(out)
            inp = out
        return outputs


class BidirectionalLanguageModel(nn.Module):
    """Forward + backward LM sharing token (char-CNN) and softmax params, with
    separate LSTM stacks per direction. Returns ALL L+1 layers per token (the
    token layer plus each biLSTM layer), not just the top -- that stack is R_k."""
    def __init__(self, n_layers=2, proj_dim=512, cell_dim=4096, vocab_size=793471):
        super().__init__()
        self.char_encoder = CharacterCNNEncoder(proj_dim=proj_dim)   # Theta_x (tied)
        self.forward_lm = StackedProjectedLSTM(proj_dim, cell_dim, proj_dim, n_layers, residual=True)
        self.backward_lm = StackedProjectedLSTM(proj_dim, cell_dim, proj_dim, n_layers, residual=True)
        self.softmax = nn.Linear(proj_dim, vocab_size)               # Theta_s (tied)
        self.n_layers = n_layers

    def forward(self, char_ids):                  # (B, T, max_chars)
        x, mask = self.char_encoder(char_ids)      # layer-0 token rep, both directions
        fwd_layers = self.forward_lm(x, mask)       # list of L states (left context)
        bwd_input = self._reverse_sequences(x, mask)
        bwd_layers = self.backward_lm(bwd_input, mask)
        bwd_layers = [self._reverse_sequences(h, mask) for h in bwd_layers]
        # R_k: layer 0 is the token rep (duplicated across directions);
        # each biLSTM layer is the concat of the two directions.
        layers = [torch.cat([x, x], dim=-1) * mask.unsqueeze(-1).float()]
        for f, b in zip(fwd_layers, bwd_layers):
            layers.append(torch.cat([f, b], dim=-1) * mask.unsqueeze(-1).float())
        return {"activations": layers, "mask": mask}   # length L+1, this is R_k

    def lm_loss(self, char_ids, targets_fwd, targets_bwd):
        # jointly maximize forward + backward log-likelihood; softmax (Theta_s)
        # and char encoder (Theta_x) shared across both directions. targets_fwd
        # and targets_bwd are pre-shifted next/previous-token ids.
        x, mask = self.char_encoder(char_ids)
        f_top = self.forward_lm(x, mask)[-1]
        b_top = self.backward_lm(self._reverse_sequences(x, mask), mask)[-1]
        b_top = self._reverse_sequences(b_top, mask)
        loss_f = F.cross_entropy(
            self.softmax(f_top).transpose(1, 2), targets_fwd, ignore_index=0)
        loss_b = F.cross_entropy(
            self.softmax(b_top).transpose(1, 2), targets_bwd, ignore_index=0)
        return 0.5 * (loss_f + loss_b)

    @staticmethod
    def _reverse_sequences(x, mask):
        if mask is None:
            return x.flip(1)
        y = x.clone()
        for batch_index, length in enumerate(mask.sum(dim=1).tolist()):
            y[batch_index, :length] = x[batch_index, :length].flip(0)
        return y


class LayerCombination(nn.Module):
    """Collapse the L+1 biLM layers into one per-token vector:
        gamma * sum_j softmax(w)_j * h_{k,j}
    One scalar parameter per layer, a learned gamma, optional tensor-level
    normalization, and no biLM gradients."""
    def __init__(self, mixture_size, do_layer_norm=False,
                 initial_scalar_parameters=None, trainable=True):
        super().__init__()
        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        if len(initial_scalar_parameters) != mixture_size:
            raise ValueError("initial scalar parameter count must match mixture_size")
        self.scalar_parameters = nn.ParameterList([
            nn.Parameter(torch.tensor([value], dtype=torch.float),
                         requires_grad=trainable)
            for value in initial_scalar_parameters
        ])
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=trainable)
        self.do_layer_norm = do_layer_norm

    def forward(self, layers, mask=None):        # layers: list of (B, T, D)
        if len(layers) != len(self.scalar_parameters):
            raise ValueError("wrong number of layer tensors")
        s = F.softmax(torch.cat([p for p in self.scalar_parameters]), dim=0)
        s = torch.split(s, split_size_or_sections=1)
        if self.do_layer_norm:
            if mask is None:
                raise ValueError("mask is required when do_layer_norm=True")
            layers = [self._layer_norm(h, mask) for h in layers]
        pieces = [s_j * h for s_j, h in zip(s, layers)]
        return self.gamma * sum(pieces)

    @staticmethod
    def _layer_norm(h, mask):
        # zero-mean unit-var over non-masked entries of this layer tensor before weighting.
        m = mask.unsqueeze(-1).float()
        n = m.sum() * h.size(-1)
        mean = (h * m).sum() / n
        var = (((h - mean) * m) ** 2).sum() / n
        return (h - mean) / torch.sqrt(var + 1e-12)


class ContextualFeatureExtractor(nn.Module):
    """Freeze the biLM, run it once, and learn one layer combination per place
    where the task model wants an ELMo vector."""
    def __init__(self, bilm, num_output_representations=1,
                 do_layer_norm=False, dropout=0.5, freeze_bilm=True):
        super().__init__()
        self.bilm = bilm
        self.freeze_bilm = freeze_bilm
        if freeze_bilm:
            for p in self.bilm.parameters():
                p.requires_grad = False
        self.layer_combinations = nn.ModuleList([
            LayerCombination(bilm.n_layers + 1, do_layer_norm=do_layer_norm)
            for _ in range(num_output_representations)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        if self.freeze_bilm:
            with torch.no_grad():
                output = self.bilm(char_ids)
        else:
            output = self.bilm(char_ids)
        layers, mask = output["activations"], output["mask"]
        features = [self.dropout(mix(layers, mask)) for mix in self.layer_combinations]
        return features, mask

    def weight_regularization(self, coefficient):
        # coefficient * ||w||^2; large coefficient -> uniform softmax -> average.
        return coefficient * sum(
            (p ** 2).sum()
            for mix in self.layer_combinations
            for p in mix.scalar_parameters
        )


def add_feature_to_task_model(task_token_repr, contextual_input,
                              task_context_repr=None, contextual_output=None):
    """Concatenate ELMo at the input, and optionally at a later task state."""
    task_input = torch.cat([task_token_repr, contextual_input], dim=-1)
    if task_context_repr is None or contextual_output is None:
        return task_input
    task_output = torch.cat([task_context_repr, contextual_output], dim=-1)
    return task_input, task_output
```

So the causal chain, start to end: a fixed vector per word type can't say which "play" I mean, because it indexes the type, not the token — and the only thing that disambiguates a token is its sentence. A language model trained on unlimited unlabeled text already builds, in its hidden states, a representation of each token conditioned on its context, so its internals are candidates for the contextual representation I want; I make it bidirectional (forward + backward LMs, sharing token and softmax parameters) so both sides of the sentence are seen, and character-based at the input so it works for any word. Prior work used only the top layer, but the layer-probing picture says lower layers carry syntax and upper layers carry word sense, so no single layer is right for every task — and rather than guess, I let each task learn a softmax-normalized weighting over all L+1 layers plus a scalar scale to match magnitudes, optionally layer-normalizing first. I checked the two places that picture had to pay off: the top-only baseline is exactly the one-hot-weights, γ=1 setting (‖ELMo − h_top‖ came out 0 on the code), so this strictly generalizes prior work; and the untrained default with L2 pulling weights to zero is exactly the uniform average (softmax of equal logits = 1/3 each, verified), so the method degrades to the sensible baseline rather than to nonsense. That weighted sum, ELMo_k^task = γ^task sum_j s_j^task h_{k,j}^LM, is concatenated as a frozen feature into the bottom (and sometimes the top) of an otherwise-unchanged task model, with L2 on the weights to trade off free selection against averaging depending on how much labeled data the task has. The one thing I still owe is the layer-probing experiment itself — that lower=syntax, upper=semantics — which I couldn't run here and would want to confirm before trusting the whole story.
