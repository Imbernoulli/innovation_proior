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
        x = self.char_emb(char_ids.view(B * T, C)).transpose(1, 2)   # (B*T, char_dim, C)
        convs = []
        for conv in self.convs:
            c, _ = conv(x).max(dim=-1)                  # conv + max-pool over chars
            convs.append(F.relu(c))
        tok = self.proj(self.highways(torch.cat(convs, dim=-1)))
        return tok.view(B, T, -1)


class BiLM(nn.Module):
    """Forward + backward LM; char-CNN (Theta_x) and softmax (Theta_s) shared
    across directions, separate LSTM stacks. Returns all L+1 layers = R_k."""
    def __init__(self, n_layers=2, proj_dim=512, hidden=4096, vocab=793471):
        super().__init__()
        self.char_encoder = CharacterCNNEncoder(proj_dim=proj_dim)
        self.fwd = StackedProjLSTM(proj_dim, hidden, proj_dim, n_layers, residual=True)
        self.bwd = StackedProjLSTM(proj_dim, hidden, proj_dim, n_layers, residual=True)
        self.softmax = nn.Linear(proj_dim, vocab)
        self.n_layers = n_layers

    def forward(self, char_ids):
        x = self.char_encoder(char_ids)
        fwd = self.fwd(x)
        bwd = [h.flip(1) for h in self.bwd(x.flip(1))]
        layers = [torch.cat([x, x], dim=-1)]            # layer 0 (token), both dirs
        for f, b in zip(fwd, bwd):
            layers.append(torch.cat([f, b], dim=-1))    # h_{k,j} = [h->; h<-]
        return layers                                   # length n_layers + 1

    def lm_loss(self, char_ids, tgt_fwd, tgt_bwd):
        x = self.char_encoder(char_ids)
        f = self.softmax(self.fwd(x)[-1]).transpose(1, 2)
        b = self.softmax(self.bwd(x.flip(1))[-1].flip(1)).transpose(1, 2)
        return F.cross_entropy(f, tgt_fwd) + F.cross_entropy(b, tgt_bwd)


class ScalarMix(nn.Module):
    """ELMo combination: gamma * sum_j softmax(w)_j * layer_j, optional per-layer
    layer-norm first. Learned per task; biLM frozen."""
    def __init__(self, mixture_size, do_layer_norm=False):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(mixture_size))
        self.gamma = nn.Parameter(torch.ones(1))
        self.do_layer_norm = do_layer_norm

    def forward(self, layers, mask=None):
        s = F.softmax(self.w, dim=0)
        if self.do_layer_norm:
            m = mask.unsqueeze(-1).float()
            n = m.sum() * layers[0].size(-1)
            def ln(h):
                mean = (h * m).sum() / n
                var = (((h - mean) * m) ** 2).sum() / n
                return (h - mean) / torch.sqrt(var + 1e-12)
            layers = [ln(h) for h in layers]
        return self.gamma * sum(s_j * h for s_j, h in zip(s, layers))


class ELMo(nn.Module):
    """Frozen biLM + one ScalarMix per output location (e.g. input and output)."""
    def __init__(self, bilm, num_output_representations=1, do_layer_norm=False):
        super().__init__()
        self.bilm = bilm
        for p in self.bilm.parameters():
            p.requires_grad = False
        self.mixes = nn.ModuleList(
            [ScalarMix(bilm.n_layers + 1, do_layer_norm)
             for _ in range(num_output_representations)])

    def forward(self, char_ids, mask=None):
        with torch.no_grad():
            layers = self.bilm(char_ids)
        return [mix(layers, mask) for mix in self.mixes]   # one ELMo vector per location


# Downstream use: concatenate ELMo onto the task token representation.
#   elmo_in, *rest = elmo(char_ids, mask)
#   h = task_rnn(torch.cat([x_task, dropout(elmo_in)], dim=-1))   # [x_k; ELMo_k]
#   # optionally also h = torch.cat([h, dropout(elmo_out)], dim=-1)  # [h_k; ELMo_k]
# Add lambda * ||w||^2 to the loss to trade free layer-selection against averaging.
```
