OK, let me think this through from the pain. Every model I build starts the same way: I take a sentence, look up each word in a table of pretrained vectors — word2vec or GloVe — and feed those vectors into whatever encoder the task needs. And there's a thing about that lookup table that has bothered me for a while. The word "play" gets exactly one vector. One. The same vector whether I wrote "Chico Ruiz made a spectacular play on Alusik's grounder" or "Olivia de Havilland signed to do a Broadway play." Those are different words in any sense that matters — different part of speech in some cases, certainly different meaning — and the table cannot tell them apart, because the table is indexed by the string, not by the usage. So whatever vector GloVe assigns to "play" is some averaged smear over the sports sense, the theatrical sense, the verb, the children's-game sense, all collapsed onto one point. A downstream model that receives that point can never recover which one I meant.

Stare at this for a second, because it's the crux and I want to name it precisely. The lookup table assigns a representation to a word *type*. What I actually want is a representation of a word *token* — this particular occurrence, in this particular sentence. And the only thing that distinguishes one occurrence of "play" from another is the rest of the sentence. So what I want is: a representation of a word that is a function of the entire input sentence. Same string, different sentence, different vector. That's the whole goal in one line.

And there's a second thing tangled up in that single GloVe vector that I want to pull apart. A word's "play"-ness has at least two very different facets. One is syntactic — what role does it play in the sentence, is it a noun here or a verb. The other is semantic — which of its meanings is active. A single fixed vector fuses those into one point, and a task can't ask for one without dragging along the other. POS tagging mostly wants the syntax; word-sense disambiguation wants the meaning. If a representation could keep those somewhat separable, a task could lean on whichever it needs.

So: where would a sentence-dependent representation of a word come from? I need something that reads the whole sentence and, at each position, produces a vector that has absorbed the surrounding context. And crucially I need it to come from *unlabeled* text, because the labeled datasets for the tasks I care about — entailment, QA, SRL, coreference — are all small, far too small to teach a model what every word means in every context. The knowledge of word usage has to be transferred in from a large corpus, and large corpora are unlabeled.

What reads a whole sentence and is trained on raw unlabeled text? A language model. p(t_1,...,t_N) = prod_k p(t_k | t_1,...,t_{k-1}). You train it by maximum likelihood to predict the next token, no labels at all, on as much text as you can find. And to predict t_{k+1} well, the model has to build, at each position k, an internal state that summarizes everything relevant about t_1,...,t_k. That internal state is *exactly* a context-dependent representation of the prefix. The modern recipe is: a context-independent token embedding x_k, then L layers of LSTM, and the top LSTM state at position k goes through a softmax over the vocabulary to predict t_{k+1}. The LSTM hidden state h_{k} is a function of the whole left context by construction. So a language model is, almost by accident, a machine that produces contextual word representations — its hidden states *are* the thing I've been asking for.

Has anyone actually used LM internals this way? Yes — there's prior work that pretrains an LM on unlabeled text, freezes it, and concatenates its top-layer hidden state into a sequence tagger, and it helps. And there's the context-vector idea (call it CoVe), which does something adjacent: train a neural machine-translation system, take the top layer of its biLSTM *encoder*, and use that as a contextual vector. CoVe genuinely makes the vectors context-dependent and improves several tasks. So the basic idea — pretrain an encoder, freeze it, use its hidden states as features — is on the table. But each of these has a piece I want to fix.

Take CoVe first. It's trained on parallel translation data. Parallel corpora are *tiny* compared to monolingual text — you need aligned sentence pairs, which someone had to produce. So the encoder sees comparatively little text. If my whole bet is that contextual knowledge comes from scale, training on parallel data is fighting my own thesis. A language model has no such constraint: any raw text in the world is training data. So I want the LM objective, not the MT objective, precisely so I can train on something like a billion words.

Now the deeper problem, shared by CoVe *and* the prior biLM-tagging work: they both use only the *top* layer of their encoder. CoVe takes the top encoder layer; the biLM tagger takes the top LM layer. And I have a nagging reason to think that's wrong. There's a body of observations about what different layers of a deep recurrent network encode. When people supervise a low-level task like POS tagging at the *lower* layers of a deep network, the high-level tasks at the top improve — as if the lower layers naturally want to hold syntax. In an MT encoder, the first-layer representations predict POS tags *better* than the second-layer ones, even though the two-layer system has higher BLEU. And the top layer of a context biLSTM has been shown to capture word sense. Put those together and a picture forms: in a deep contextual encoder, lower layers lean syntactic, higher layers lean semantic. Which is *exactly* the syntax-vs-semantics split I wanted to pull apart at the very start.

So here's where the top-layer convention starts to look like a real mistake. If lower layers carry syntax and upper layers carry word sense, then by taking only the top layer, CoVe and the biLM tagger are throwing away the syntactic information for any task that would have wanted it. POS tagging would rather have the lower layer; WSD would rather have the upper. There is no single layer that is best for everything. Let me make sure I actually believe this and I'm not just repeating a slogan. Suppose I freeze a two-layer LM encoder and, for each layer separately, feed its activations into a tiny probe — a 1-nearest-neighbor classifier for word sense, a linear classifier for POS — and read off the accuracy. If my picture is right, the first layer should win on POS and the second on WSD. And that is what I'd expect to see: the first LM layer noticeably better at POS, the second LM layer noticeably better at WSD, the gap going opposite directions for the two tasks. (And an MT encoder would show the same ordering but lower across the board, because translation is a weaker teacher of monolingual word knowledge than language modeling.) If the two layers genuinely specialize in opposite directions, then committing to either one alone is leaving the other's information on the floor.

So the design pressure is clear: don't pick a layer. Use all of them. But "use all of them" isn't yet a representation — I have to say *how* a downstream model consumes L+1 vectors per token (the token-embedding layer plus the L LSTM layers) instead of one. Let me hold that thought; first I have to settle the encoder itself, because there's an asymmetry problem.

A plain forward LM only ever sees the *left* context: h_k is a function of t_1,...,t_k. But a contextual representation of "play" in the middle of a sentence should obviously depend on the words *after* it too — "a Broadway play for Garson" needs "Broadway" before and arguably the rest after. A unidirectional LM gives me only half the context. The fix is the same trick that makes biLSTM taggers work: also run a *backward* LM, one that factorizes the sequence in reverse, p(t_1,...,t_N) = prod_k p(t_k | t_{k+1},...,t_N), predicting each token from its *future*. Implement it identically to the forward LM but reading the sequence right-to-left; its layer-j hidden state at position k, call it h←_{k,j}, summarizes the right context t_{k+1},...,t_N. Then the full contextual representation of token k at layer j is the concatenation of the two directions, h_{k,j} = [h→_{k,j}; h←_{k,j}] — both sides of the sentence.

Now, do I train two completely separate models? The prior biLM work used fully independent parameters for the two directions. But think about what each direction is: a forward LM and a backward LM are predicting tokens from the *same* vocabulary, using the *same* notion of what a token's input embedding is and the *same* softmax to score vocabulary items. The thing that genuinely must differ between them is the recurrent dynamics — how you accumulate left context is a different function from how you accumulate right context. The token embeddings and the output softmax have no inherent direction. So I'll tie them: share the token-representation parameters Θ_x and the softmax parameters Θ_s across both directions, while keeping separate LSTM parameters Θ_LSTM→ and Θ_LSTM← for each direction. Fewer parameters, and it forces both directions to agree on a common token/output space, which is the right inductive bias. The biLM then jointly maximizes the sum of the two directions' log-likelihoods:

sum over k of [ log p(t_k | t_1,...,t_{k-1}; Θ_x, Θ_LSTM→, Θ_s) + log p(t_k | t_{k+1},...,t_N; Θ_x, Θ_LSTM←, Θ_s) ].

One objective, one forward pass that produces both directions' states, parameters shared where it makes sense.

Before I go on, the input layer x_k. I could use a normal word-embedding table for the LM input, but that drags the OOV problem right back in — and contextual representations are most valuable for exactly the rare and unseen words a fixed vocabulary can't handle. So make the input purely *character*-based: a small convolutional network over character embeddings. Embed each character, run several 1-D convolutions of different widths across the characters of the word, max-pool each over the character dimension, concatenate the filter outputs, push through a couple of highway layers, and linearly project to the token dimension. This gives a context-independent token vector x_k for *any* string — in or out of vocabulary — and it picks up morphology (prefixes, suffixes, capitalization) for free, which is itself useful. So the biLM provides a representation even for words it never saw in training, which a word-table LM simply cannot.

Now back to the real question: I have, for each token k, a stack of vectors — the input layer and L layers per direction. Let me count them. Layer 0 is the context-independent token layer x_k^LM. It is not directional, but once the recurrent layers are represented as [forward; backward] vectors, the implementation can duplicate the token vector as [x_k^LM; x_k^LM] so every layer has the same width and can be mixed by scalars. Then for each of the L LSTM layers I have a forward state and a backward state, concatenated. So the set of representations for token k is

R_k = { x_k^LM, h→_{k,j}^LM, h←_{k,j}^LM : j = 1,...,L } = { h_{k,j}^LM : j = 0,...,L },

where h_{k,0}^LM denotes the token layer (implemented as [x_k^LM; x_k^LM] when the tensors need equal width) and h_{k,j}^LM = [h→_{k,j}^LM; h←_{k,j}^LM] for each biLSTM layer. That's 2L+1 vectors packaged into L+1 layer-vectors. For the L=2 model I have in mind, that's 3 layers: token, LSTM-1, LSTM-2.

So how do I turn these L+1 vectors into the single per-token feature a downstream model wants? The most obvious thing is to just pick the top layer, E(R_k) = h_{k,L}^LM. That's what CoVe and the biLM tagger do, and I just spent several paragraphs arguing it's the wrong call because it discards the lower, more syntactic layers. So scratch "top only."

Next obvious thing: average them, (1/(L+1)) sum_j h_{k,j}^LM. But that's also clearly suboptimal — it forces every task to weight syntax and semantics equally, when a POS tagger should lean on the lower layer and a WSD model on the upper. A fixed average can't adapt.

So if no fixed choice of layer is right, and the right mixture *depends on the task*, then let the task *learn* the mixture. Give each task its own set of weights over the layers, and let supervised training on that task decide how much of each layer to use. The simplest learnable combination is a weighted sum, sum_j s_j h_{k,j}^LM, with scalar weights s_j. Now, I want these weights to behave like a soft selection over layers — nonnegative and summing to one, so the combination is genuinely a convex mixture and the weights are interpretable as "how much this task relies on layer j." The clean way to get that from unconstrained parameters w_j is a softmax: s_j = softmax(w)_j. Then the task can learn to put its mass on the lower layer (POS-ish tasks) or the upper layer (sense-ish tasks) or spread it out, all by gradient descent on the task loss, with the biLM itself frozen.

There's one more thing I need, and it's not cosmetic. The hidden states of the biLM live on a completely different scale and distribution than the activations inside the downstream task model — the biLM was trained for language modeling, not for this task, and its layer norms and magnitudes are whatever they happened to be. If I just splice a softmax-weighted sum of biLM layers into the task model, the magnitudes can be badly mismatched and optimization struggles. So I add a single scalar γ that multiplies the whole combined vector, letting the task model rescale the entire biLM feature to a useful magnitude. So the feature is:

ELMo_k^task = E(R_k; Θ^task) = γ^task sum_{j=0}^{L} s_j^task h_{k,j}^LM,

with s^task = softmax(w^task) the normalized layer weights and γ^task the scalar scale, both learned per task, biLM frozen. (Note the top-only baseline falls right out as the special case where s puts all its mass on j=L and γ=1 — so this strictly generalizes what CoVe and the biLM tagger did.)

Let me double-check that γ is actually load-bearing and not a free knob I'm adding for symmetry. Picture the degenerate case where s selects only the top layer — that's the prior-work setting. Without γ, I'm injecting raw top-layer biLM activations whose scale doesn't match the task model's expectations, and I'd bet that breaks optimization: the task either can't make use of the feature or the gradients are poorly conditioned. With γ free, the model can shrink or grow the whole feature to whatever scale training prefers. So γ matters most exactly in the single-layer case, where there is no averaging over layers to soften the mismatch. That's enough to convince me γ stays.

One refinement on the weighting. The L+1 layers don't just differ in scale; they have genuinely different activation *distributions* from each other. Softmax-weighting raw layers means a layer with naturally large activations dominates the sum regardless of how informative it is. So in some cases it helps to normalize each biLM layer tensor *before* the weighted sum — compute its mean and variance over the unmasked batch, time, and feature entries, rescale to zero mean and unit variance, then weight. That puts the layers on equal footing so the learned weights s_j reflect informativeness, not raw magnitude. It's optional — sometimes the task model is happier with the raw scales — so I'll make it a switch.

Now, how does this feature actually enter a task model? The point of the whole approach is that I *don't* want to fine-tune the giant biLM per task — that would couple the task model's size to the biLM and require backprop through a huge network for every task. Instead: freeze the biLM, run it once over the input to get all the layer activations, and treat ELMo_k^task as an extra input feature. Concretely, most supervised NLP models start by forming a context-independent token representation x_k (word embeddings, maybe a char-CNN of their own) and then building a context-sensitive h_k with a biRNN or CNN. I inject ELMo at the bottom by concatenating it onto the token representation: replace x_k with [x_k; ELMo_k^task] and feed that into the task RNN. The rest of the task model is untouched, so this drops into arbitrarily complex architectures — an entailment model with a bi-attention layer on top, a coreference model with a span-clustering head — without any surgery beyond the concatenation.

Should ELMo only go at the input? For architectures that put an attention or span-comparison module *after* the task RNN, I also want the option to inject it at the RNN output — concatenate ELMo onto h_k as well, [h_k; ELMo_k^task], with a *separate* set of learned weights (its own s and γ), so the input and output uses can rely on different layer mixtures. Why would output injection belong in some models and not others? If there's an attention mechanism downstream, giving it direct access to the biLM's internal representations lets it attend straight to that contextual information, which it otherwise only sees filtered through the task RNN. For a task where the task-specific recurrent representation is supposed to do nearly all the contextual work, input injection may be the cleaner choice. So: input as the default, output as an optional second use-site with its own scalar mix.

Two more practical pieces. First, regularization of the layer weights w. I can add λ||w||^2 to the loss. What does that do? As λ grows large, it pulls the unnormalized weights w toward zero, and softmax of all-equal logits is the uniform distribution — so large λ drives the weighting toward a flat *average* of all layers, while small λ lets the weights move freely to wherever the task wants them. So λ is a dial between "trust the task to pick layers" (small λ) and "just average the layers" (large λ). Which end is better should depend on data size: a task with a large training set can afford to learn free layer weights (small λ), while a task with a small training set — NER, say — overfits if it tries to learn the weights, so the inductive bias of averaging (larger λ) is safer. Second, dropout on the ELMo vector itself, a moderate amount, since it's a big new feature being injected into a model that's otherwise regularized.

Let me now pin the encoder I'll actually pretrain, balancing language-model quality against the cost of running it inside downstream models. A known strong character-input LM is the CNN-BIG-LSTM architecture; I'll halve its embedding and hidden dimensions to keep it affordable as a frozen feature extractor while staying purely character-based. So: L=2 biLSTM layers, 4096 units each with a 512-dimensional projection, and a residual connection from the first to the second LSTM layer so the second layer can build on rather than replace the first. The input is the char-CNN — on the order of 2048 character n-gram convolution filters, two highway layers, linearly projected down to 512. That yields three layers of representation per token (token, LSTM-1, LSTM-2), available even for tokens outside any training vocabulary because the input is characters. Train it on something like a billion words of raw text for a handful of epochs; I'd expect forward and backward perplexities in the same ballpark as each other. And once it's pretrained, if a downstream domain differs from the pretraining text, I can fine-tune the biLM on that domain's raw text (ignoring the labels, just the LM objective) for an epoch before freezing, reducing the domain mismatch without spending labels.

Let me write the pieces down as code, grounded in how this actually gets built — a character encoder, the biLM that stacks LSTMs on top of it, the scalar-mixture that *is* ELMo, and the injection into a task model.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return tok.view(B, T, -1)                       # x_k^LM, layer 0


class BiLM(nn.Module):
    """Forward + backward LM sharing token (char-CNN) and softmax params, with
    separate LSTM stacks per direction. Returns ALL L+1 layers per token (the
    token layer plus each biLSTM layer), not just the top -- that stack is R_k."""
    def __init__(self, n_layers=2, proj_dim=512, hidden=4096, vocab_size=793471):
        super().__init__()
        self.char_encoder = CharacterCNNEncoder(proj_dim=proj_dim)   # Theta_x (tied)
        self.fwd = StackedProjLSTM(proj_dim, hidden, proj_dim, n_layers, residual=True)
        self.bwd = StackedProjLSTM(proj_dim, hidden, proj_dim, n_layers, residual=True)
        self.softmax = nn.Linear(proj_dim, vocab_size)               # Theta_s (tied)
        self.n_layers = n_layers

    def forward(self, char_ids):                  # (B, T, max_chars)
        x = self.char_encoder(char_ids)           # layer-0 token rep, both directions
        fwd_layers = self.fwd(x)                   # list of L states (left context)
        bwd_layers = self.bwd(x.flip(1))           # run on reversed sequence
        bwd_layers = [h.flip(1) for h in bwd_layers]
        # R_k: layer 0 is the token rep (duplicated across directions);
        # each biLSTM layer is the concat of the two directions.
        layers = [torch.cat([x, x], dim=-1)]
        for f, b in zip(fwd_layers, bwd_layers):
            layers.append(torch.cat([f, b], dim=-1))   # h_{k,j} = [h->; h<-]
        return layers                              # length L+1, this is R_k

    def lm_loss(self, char_ids, targets_fwd, targets_bwd):
        # jointly maximize forward + backward log-likelihood; softmax (Theta_s)
        # and char encoder (Theta_x) shared across both directions. targets_fwd
        # and targets_bwd are pre-shifted next/previous-token ids.
        x = self.char_encoder(char_ids)
        f_top = self.fwd(x)[-1]
        b_top = self.bwd(x.flip(1))[-1].flip(1)
        loss_f = F.cross_entropy(self.softmax(f_top).transpose(1, 2), targets_fwd)
        loss_b = F.cross_entropy(self.softmax(b_top).transpose(1, 2), targets_bwd)
        return loss_f + loss_b


class ScalarMix(nn.Module):
    """Collapse the L+1 biLM layers into one per-token vector:
        gamma * sum_j softmax(w)_j * h_{k,j}
    This mirrors the canonical ScalarMix: one scalar parameter per layer, a
    learned gamma, optional tensor-level normalization, and no biLM gradients."""
    def __init__(self, mixture_size, do_layer_norm=False,
                 initial_scalar_parameters=None, trainable=True):
        super().__init__()
        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        self.scalar_parameters = nn.ParameterList([
            nn.Parameter(torch.tensor([value], dtype=torch.float),
                         requires_grad=trainable)
            for value in initial_scalar_parameters
        ])
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=trainable)
        self.do_layer_norm = do_layer_norm

    def forward(self, layers, mask=None):        # layers: list of (B, T, D)
        s = F.softmax(torch.cat([p for p in self.scalar_parameters]), dim=0)
        if self.do_layer_norm:
            if mask is None:
                raise ValueError("mask is required when do_layer_norm=True")
            layers = [self._layer_norm(h, mask) for h in layers]
        pieces = [s_j * h for s_j, h in zip(s, layers)]
        return self.gamma * sum(pieces)

    @staticmethod
    def _layer_norm(h, mask):
        # zero-mean unit-var over non-masked entries of this layer tensor before
        # weighting, matching the AllenNLP ScalarMix behavior.
        m = mask.unsqueeze(-1).float()
        n = m.sum() * h.size(-1)
        mean = (h * m).sum() / n
        var = (((h - mean) * m) ** 2).sum() / n
        return (h - mean) / torch.sqrt(var + 1e-12)


class TaskModelWithELMo(nn.Module):
    """Freeze the biLM, run it, concat ELMo onto the task token rep at the input
    (and optionally the output, with its OWN weights). Rest of the task model is
    unchanged. L2 on ELMo weights (lambda) dials between free weights and average;
    dropout on the ELMo vector."""
    def __init__(self, bilm, task_rnn, n_layers_plus1, use_output=False,
                 elmo_dropout=0.5, lam=0.001):
        super().__init__()
        self.bilm = bilm
        for p in self.bilm.parameters():           # frozen feature extractor
            p.requires_grad = False
        self.elmo_in = ScalarMix(n_layers_plus1)
        self.elmo_out = ScalarMix(n_layers_plus1) if use_output else None
        self.task_rnn = task_rnn
        self.drop = nn.Dropout(elmo_dropout)
        self.lam = lam

    def forward(self, char_ids, x_task, mask=None):
        with torch.no_grad():
            layers = self.bilm(char_ids)            # R_k, biLM frozen
        elmo_in = self.drop(self.elmo_in(layers, mask))
        h = self.task_rnn(torch.cat([x_task, elmo_in], dim=-1))   # [x_k; ELMo_k]
        if self.elmo_out is not None:
            elmo_out = self.drop(self.elmo_out(layers, mask))
            h = torch.cat([h, elmo_out], dim=-1)    # [h_k; ELMo_k] for attention tasks
        return h

    def weight_reg(self):
        # lambda * ||w||^2 ; large lambda -> uniform softmax -> average of layers,
        # small lambda -> task picks layers freely. Larger lambda for small datasets.
        reg = self.lam * sum((p ** 2).sum() for p in self.elmo_in.scalar_parameters)
        if self.elmo_out is not None:
            reg = reg + self.lam * sum((p ** 2).sum() for p in self.elmo_out.scalar_parameters)
        return reg
```

So the causal chain, start to end: a fixed vector per word type can't say which "play" I mean, because it indexes the type, not the token — and the only thing that disambiguates a token is its sentence. A language model trained on unlimited unlabeled text already builds, in its hidden states, a representation of each token conditioned on its context, so its internals *are* the contextual representation I want; I make it bidirectional (forward + backward LMs, sharing token and softmax parameters) so both sides of the sentence are seen, and character-based at the input so it works for any word. Prior work used only the top layer, but lower layers carry syntax and upper layers carry word sense, so no single layer is right for every task — and rather than guess, I let each task learn a softmax-normalized weighting over all L+1 layers plus a scalar scale to match magnitudes, optionally layer-normalizing first. That weighted sum, ELMo_k^task = γ^task sum_j s_j^task h_{k,j}^LM, is concatenated as a frozen feature into the bottom (and sometimes the top) of an otherwise-unchanged task model, with L2 on the weights to trade off free selection against averaging depending on how much labeled data the task has.
