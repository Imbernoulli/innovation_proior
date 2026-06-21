The problem I want to solve is the fundamental limitation of pretrained word embeddings. word2vec and GloVe assign exactly one vector to every word type, so the string "play" is represented by the same point whether it is a verb, a sports highlight, or a theatrical production. A single vector cannot disambiguate polysemy, it forces syntax and semantics into the same coordinates, and it gives nothing for out-of-vocabulary words. What I need instead is a representation of a word token that is a function of the entire sentence it appears in, that captures different facets of meaning at different levels, and that can be pretrained on large unlabeled corpora and reused as a fixed feature in downstream supervised models.

Existing ideas make partial progress but fall short in important ways. Context-independent embeddings are the baseline with the one-vector-per-type problem. CoVe trains a contextual encoder on parallel translation data, but parallel corpora are much smaller than monolingual text and the encoder only exposes its top layer. Prior bidirectional language model features for tagging also use only the top layer. context2vec is another single-top-layer approach. Work that pretrains an encoder and then fine-tunes the whole thing on each task couples the task architecture to the pretrained model and requires expensive end-to-end training. What is missing is a representation that is trained on abundant monolingual text, is genuinely bidirectional so both left and right context matter, and exposes all layers so a downstream task can decide which level of abstraction it needs.

The method I propose is called ELMo, which stands for Embeddings from Language Models. The core idea is to pretrain a deep bidirectional language model on raw text and then use its internal states as contextual word representations. For each token the model produces a stack of representations: a character-CNN token layer plus the hidden states of each bidirectional LSTM layer. ELMo is a task-specific, learned weighted combination of all of those layers, not just the top one. Lower layers of the biLM tend to encode syntactic information such as part-of-speech, while higher layers encode semantic information such as word sense. By letting each supervised task learn its own scalar weights over the layers, ELMo avoids forcing every task to use the same abstraction level. The weights are normalized with a softmax so they form a convex mixture, and a learned scalar gamma rescales the combined vector to match the activation scale of the task model.

The bidirectional language model is built as follows. A forward language model factorizes the probability of a sequence as the product of p(t_k | t_1, ..., t_{k-1}), and a backward language model factorizes it in the reverse direction as the product of p(t_k | t_{k+1}, ..., t_N). They are trained jointly to maximize the sum of the two log-likelihoods. The token representation parameters and the output softmax parameters are shared across the two directions, while each direction has its own LSTM stack. The input to the LMs is a character-level CNN: each word is represented by convolutions over its characters, max-pooled, passed through highway layers, and projected to the token dimension. This means the model can produce a representation for any string, including words that never appeared during pretraining. For an L-layer biLM, each token k has L+1 layer representations: the token layer plus each bidirectional LSTM layer, where each bidirectional layer is the concatenation of the forward and backward states at that depth.

To use ELMo in a downstream task, I freeze the pretrained biLM and run it once over the input. The task learns a separate layer combination for each place it wants to inject ELMo. The most common usage is to concatenate ELMo onto the task's own context-independent token representation at the input of the task RNN. Some architectures also benefit from a second injection at the output of the task RNN, for example when an attention or span-comparison layer follows it. I also apply dropout to the ELMo vector and optionally add L2 regularization on the unnormalized layer weights; a large regularization coefficient pulls the weights toward a uniform average, which is safer on small labeled datasets, while a small coefficient lets the task freely select layers.

The biLM I would pretrain has two LSTM layers with 4096 units and 512-dimensional projections, with a residual connection from the first to the second layer. The character CNN uses a bank of filters over character n-grams of width 1 through 7, two highway layers, and a linear projection to 512. It is trained on the One Billion Word Benchmark. Because the input is purely character-based, it naturally handles rare and unseen words. If the downstream domain differs from the pretraining corpus, the biLM can be fine-tuned on in-domain unlabeled text for an epoch before freezing.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, activation=F.relu):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(size, 2 * size) for _ in range(num_layers)])
        self.activation = activation
        for layer in self.layers:
            layer.bias.data[size:].fill_(1.0)

    def forward(self, x):
        for layer in self.layers:
            nonlinear, gate = layer(x).chunk(2, dim=-1)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * self.activation(nonlinear)
        return x


class CharacterCNNEncoder(nn.Module):
    """Context-independent token vector from characters; works for any string."""
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

    def forward(self, char_ids):
        B, T, C = char_ids.shape
        mask = (char_ids.gt(0).long().sum(dim=-1) > 0).long()
        x = self.char_emb(char_ids.view(B * T, C)).transpose(1, 2)
        convs = []
        for conv in self.convs:
            c, _ = conv(x).max(dim=-1)
            convs.append(F.relu(c))
        tok = self.proj(self.highways(torch.cat(convs, dim=-1)))
        tok = tok.view(B, T, -1) * mask.unsqueeze(-1).float()
        return tok, mask


class StackedProjectedLSTM(nn.Module):
    """Projected LSTM stack that returns every layer output for mixing."""
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
    """Forward + backward LM with shared char encoder and softmax; returns all layers."""
    def __init__(self, n_layers=2, proj_dim=512, cell_dim=4096, vocab_size=793471):
        super().__init__()
        self.char_encoder = CharacterCNNEncoder(proj_dim=proj_dim)
        self.forward_lm = StackedProjectedLSTM(proj_dim, cell_dim, proj_dim, n_layers, residual=True)
        self.backward_lm = StackedProjectedLSTM(proj_dim, cell_dim, proj_dim, n_layers, residual=True)
        self.softmax = nn.Linear(proj_dim, vocab_size)
        self.n_layers = n_layers

    def forward(self, char_ids):
        x, mask = self.char_encoder(char_ids)
        fwd = self.forward_lm(x, mask)
        bwd_input = self._reverse_sequences(x, mask)
        bwd = [self._reverse_sequences(h, mask)
               for h in self.backward_lm(bwd_input, mask)]
        layers = [torch.cat([x, x], dim=-1) * mask.unsqueeze(-1).float()]
        for f, b in zip(fwd, bwd):
            layers.append(torch.cat([f, b], dim=-1) * mask.unsqueeze(-1).float())
        return {"activations": layers, "mask": mask}

    def lm_loss(self, char_ids, tgt_fwd, tgt_bwd):
        x, mask = self.char_encoder(char_ids)
        f = self.softmax(self.forward_lm(x, mask)[-1]).transpose(1, 2)
        b_top = self.backward_lm(self._reverse_sequences(x, mask), mask)[-1]
        b = self.softmax(self._reverse_sequences(b_top, mask)).transpose(1, 2)
        loss_f = F.cross_entropy(f, tgt_fwd, ignore_index=0)
        loss_b = F.cross_entropy(b, tgt_bwd, ignore_index=0)
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
    """ELMo combination: gamma * sum_j softmax(w)_j * layer_j."""
    def __init__(self, mixture_size, do_layer_norm=False,
                 initial_scalar_parameters=None, trainable=True):
        super().__init__()
        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        self.scalar_parameters = nn.ParameterList([
            nn.Parameter(torch.tensor([value], dtype=torch.float), requires_grad=trainable)
            for value in initial_scalar_parameters
        ])
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=trainable)
        self.do_layer_norm = do_layer_norm

    def forward(self, layers, mask=None):
        s = F.softmax(torch.cat([p for p in self.scalar_parameters]), dim=0)
        s = torch.split(s, split_size_or_sections=1)
        if self.do_layer_norm:
            layers = [self._layer_norm(h, mask) for h in layers]
        return self.gamma * sum(s_j * h for s_j, h in zip(s, layers))

    @staticmethod
    def _layer_norm(h, mask):
        m = mask.unsqueeze(-1).float()
        n = m.sum() * h.size(-1)
        mean = (h * m).sum() / n
        var = (((h - mean) * m) ** 2).sum() / n
        return (h - mean) / torch.sqrt(var + 1e-12)


class ContextualFeatureExtractor(nn.Module):
    """Frozen biLM plus one learned layer combination per injection site."""
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
        return coefficient * sum(
            (p ** 2).sum()
            for mix in self.layer_combinations
            for p in mix.scalar_parameters
        )


def add_feature_to_task_model(task_token_repr, contextual_input,
                              task_context_repr=None, contextual_output=None):
    """Concatenate ELMo at the task input and optionally at the task output."""
    task_input = torch.cat([task_token_repr, contextual_input], dim=-1)
    if task_context_repr is None or contextual_output is None:
        return task_input
    task_output = torch.cat([task_context_repr, contextual_output], dim=-1)
    return task_input, task_output
```
