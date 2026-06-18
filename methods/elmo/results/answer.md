# ELMo: Embeddings from Language Models

## Problem

Pretrained word-type embeddings (word2vec, GloVe) give one fixed vector per word string, so they cannot
disambiguate polysemy ("play" the sport vs. the theatre piece), conflate syntax with semantics, and offer
nothing for out-of-vocabulary words. ELMo replaces this with a per-token representation that is a function of
the entire input sentence, learned from large unlabeled corpora and usable as a frozen feature in existing
supervised models.

## Key idea

Pretrain a deep **bidirectional language model (biLM)** on raw text. For each token, the biLM produces a stack
of internal representations (a character-CNN token layer plus each biLSTM layer, in both directions). ELMo is a
**task-specific, learned linear combination of all these layers** — not just the top one. Lower biLM layers
encode syntax (better at POS tagging), higher layers encode word sense (better at WSD); letting each task learn
a soft mixture over all layers avoids committing every task to the same single layer.

## Method

**biLM objective.** Given tokens (t_1,...,t_N), jointly maximize the forward and backward log-likelihoods:

    sum_{k=1}^N [ log p(t_k | t_1,...,t_{k-1}; Θ_x, Θ_LSTM→, Θ_s)
               + log p(t_k | t_{k+1},...,t_N; Θ_x, Θ_LSTM←, Θ_s) ]

The token-representation parameters Θ_x (a character CNN) and the softmax parameters Θ_s are **shared** across
directions; the forward and backward LSTM stacks are separate.

**Layer representations.** An L-layer biLM yields 2L+1 vectors per token, organized as L+1 layer-vectors:

    R_k = { x_k^LM, h→_{k,j}^LM, h←_{k,j}^LM | j=1..L } = { h_{k,j}^LM | j=0..L }

with h_{k,0}^LM denoting the token layer (duplicated as [x_k^LM; x_k^LM] in code so all mixed tensors have the
same width) and h_{k,j}^LM = [h→_{k,j}^LM; h←_{k,j}^LM].

**ELMo combination.** Collapse R_k into one vector with task-specific weights:

    ELMo_k^task = γ^task · sum_{j=0}^L s_j^task · h_{k,j}^LM

where s^task = softmax(w^task) are softmax-normalized (nonnegative, sum-to-one) layer weights and γ^task is a
scalar that rescales the whole vector to match the task model's activation scale. Optionally normalize each
layer tensor before weighting, so naturally large activations don't dominate. Selecting the top layer (s puts
all mass on j=L, γ=1) recovers prior top-layer methods as a special case.

**Usage.** Freeze the biLM. Concatenate ELMo onto the task's context-independent token representation:
[x_k; ELMo_k^task] into the task RNN; optionally also at the output, [h_k; ELMo_k^task], with a separate set of
weights for architectures with attention or comparison layers after the RNN. Add dropout to ELMo and optionally L2-regularize the
weights with λ||w||^2 — large λ drives the mixture toward a flat average (good for small datasets), small λ
lets the task pick layers freely.

**Pretrained biLM.** L=2 biLSTM layers, 4096 units, 512-dim projections, residual connection layer-1→layer-2;
character-CNN input (~2048 char n-gram filters, 2 highway layers, linear projection to 512). Trained on the One
Billion Word Benchmark. Purely character input → a representation for any token, including OOV. Optionally
fine-tune the biLM (LM objective only) on in-domain text before freezing.

## Code

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
    """x_k^LM: context-independent token vector from characters (works for any
    string, captures morphology)."""
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

    def forward(self, char_ids):                       # (B, T, max_chars)
        B, T, C = char_ids.shape
        mask = (char_ids.gt(0).long().sum(dim=-1) > 0).long()
        x = self.char_emb(char_ids.view(B * T, C)).transpose(1, 2)   # (B*T, char_dim, C)
        convs = []
        for conv in self.convs:
            c, _ = conv(x).max(dim=-1)                  # conv + max-pool over chars
            convs.append(F.relu(c))
        tok = self.proj(self.highways(torch.cat(convs, dim=-1)))
        tok = tok.view(B, T, -1) * mask.unsqueeze(-1).float()
        return tok, mask


class StackedProjectedLSTM(nn.Module):
    """Projected LSTM stack returning every layer output."""
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
    """Forward + backward LM; char-CNN (Theta_x) and softmax (Theta_s) shared
    across directions, separate LSTM stacks. Returns all L+1 layers = R_k."""
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
        if len(initial_scalar_parameters) != mixture_size:
            raise ValueError("initial scalar parameter count must match mixture_size")
        self.scalar_parameters = nn.ParameterList([
            nn.Parameter(torch.tensor([value], dtype=torch.float), requires_grad=trainable)
            for value in initial_scalar_parameters
        ])
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=trainable)
        self.do_layer_norm = do_layer_norm

    def forward(self, layers, mask=None):
        if len(layers) != len(self.scalar_parameters):
            raise ValueError("wrong number of layer tensors")
        s = F.softmax(torch.cat([p for p in self.scalar_parameters]), dim=0)
        s = torch.split(s, split_size_or_sections=1)
        if self.do_layer_norm:
            if mask is None:
                raise ValueError("mask is required when do_layer_norm=True")
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
    """Frozen biLM + one learned layer combination per output location."""
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
    task_input = torch.cat([task_token_repr, contextual_input], dim=-1)
    if task_context_repr is None or contextual_output is None:
        return task_input
    task_output = torch.cat([task_context_repr, contextual_output], dim=-1)
    return task_input, task_output
```
