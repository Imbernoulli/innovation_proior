# HIST, Distilled

HIST forecasts next-day stock returns by decomposing each stock embedding into three additive parts:
shared information from predefined concepts, shared information from hidden concepts, and residual
individual information. Backcasts are subtracted between modules; forecasts are summed before the
final readout.

## Problem

Fixed stock-relation graphs can only reweight edges that already exist. They cannot track that a
stock's relevance to a curated theme changes by date, and they cannot use an unlabeled theme that has
no curated node. HIST turns concepts into date-specific vectors, lets stocks attend to them by
current similarity, discovers additional concepts from the residual signal, and preserves whatever
is left as individual information.

## Mathematical Form

Encoder: `x_i^{t,0} = GRU(s_i^t)`, using the last hidden state of the stock's 60-day, 6-channel
feature sequence.

Predefined concepts:
- initialize each concept from member stocks. With market values, `e_k^{t,0} = Σ_{i∈N_k} α_{ki} x_i^{t,0}`,
  `α_{ki} = c_i^t / Σ_{j∈N_k} c_j^t`; in the qlib membership-only path, binary membership `M` is
  degree-normalized with smoothing, `α_{ik}^{init}=M_{ik}/(M_{ik}Σ_r M_{rk}+1)`, and zero concept rows
  are dropped;
- the full mathematical correction compares all stocks to each initialized concept by cosine,
  `v_{ki}=cos(x_i^{t,0}, e_k^{t,0})`, then normalizes over stocks for each concept,
  `α_{ki}^{t,1}=softmax_i(v_{ki})`, giving
  `e_k^{t,1}=LeakyReLU(W_e Σ_{i∈S^t} α_{ki}^{t,1} x_i^{t,0}+b_e)`;
- the qlib `HISTModel` forward path below does not include that stock-softmax correction stage; it
  uses the smoothed initialized concept rows directly;
- concept-to-stock aggregation normalizes over concepts for each stock,
  `β_{ik}=softmax_k(cos(x_i^{t,0}, e_k))`, then forms shared information from `Σ_k β_{ik} e_k`.

Hidden concepts:
- run on `x_i^{t,1}=x_i^{t,0}-xhat_i^{t,0}`;
- seed one hidden concept per stock, `u_k^{t,0}=x_k^{t,1}`;
- compute `γ_{ki}=cos(x_i^{t,1},u_k^{t,0})`, zero the diagonal to remove the trivial self-score,
  keep the single largest remaining entry in each row, re-add the diagonal only for surviving
  columns, and delete empty columns. The zero diagonal removes the guaranteed self-similarity score;
  a hard exclusion for all-negative off-diagonal rows would require a `-inf` mask, while qlib uses
  the zero mask;
- aggregate each surviving hidden concept from its connected stocks. The full mathematical formula puts a
  LeakyReLU transform on that hidden concept representation; the qlib forward path forms the raw
  weighted hidden-concept rows and applies `fc_is` after concept-to-stock aggregation.

Residual chain and prediction:
`x_i^{t,j}=x_i^{t,j-1}-xhat_i^{t,j-1}` for `j=1,2`, and
`p_i^t = W_p(yhat_i^{t,0}+yhat_i^{t,1}+yhat_i^{t,2})+b_p`.

Loss:
`L = Σ_{t∈T} (1/|S^t|) Σ_{i∈S^t} (p_i^t-d_i^t)^2`, optimized with Adam on daily cross-section
batches.

## Defaults

The selected training setting uses `d_feat=6`, `hidden_size=128`, `num_layers=2`, GRU base,
`dropout=0`, `n_epochs=200`, Adam `lr=2e-4`, `early_stop=30`, gradient value clipping at `3.0`,
and daily batches. The qlib class defaults are more generic (`hidden_size=64`, `lr=1e-3`,
`early_stop=20`) unless the experiment config overrides them. The qlib `HISTModel` path below uses
degree normalization with `+1` smoothing and a single masked argmax (`K=1`) for hidden-concept
assignment.

## Working Code

```python
import torch
import torch.nn as nn


class HISTModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        if base_model == "GRU":
            self.rnn = nn.GRU(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.fc_es = nn.Linear(hidden_size, hidden_size)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        for layer in [
            self.fc_es, self.fc_is, self.fc_es_fore, self.fc_is_fore,
            self.fc_es_back, self.fc_is_back, self.fc_indi,
        ]:
            nn.init.xavier_uniform_(layer.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_t2s = nn.Softmax(dim=1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def cal_cos_similarity(self, x, y):
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        return xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)

    def forward(self, x, concept_matrix):
        device = x.device

        x_hidden = x.reshape(len(x), self.d_feat, -1)
        x_hidden = x_hidden.permute(0, 2, 1)
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        stock_to_concept = concept_matrix
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(
            stock_to_concept.shape[0], 1
        )
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)
        stock_to_concept_sum = stock_to_concept_sum + torch.ones(
            stock_to_concept.shape[0], stock_to_concept.shape[1], device=device
        )
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        hidden = hidden[hidden.sum(1) != 0]

        concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)
        e_shared_info = self.fc_es(concept_to_stock.mm(hidden))
        e_shared_back = self.fc_es_back(e_shared_info)
        output_es = self.leaky_relu(self.fc_es_fore(e_shared_info))

        i_shared_info = x_hidden - e_shared_back
        i_stock_to_concept = self.cal_cos_similarity(i_shared_info, i_shared_info)
        dim = i_stock_to_concept.shape[0]
        diag = i_stock_to_concept.diagonal(0)
        i_stock_to_concept = i_stock_to_concept * (torch.ones(dim, dim, device=device) - torch.eye(dim, device=device))
        row = torch.arange(dim, device=device).long()
        column = i_stock_to_concept.max(1)[1].long()
        value = i_stock_to_concept.max(1)[0]
        i_stock_to_concept[row, column] = 10
        i_stock_to_concept[i_stock_to_concept != 10] = 0
        i_stock_to_concept[row, column] = value
        i_stock_to_concept = i_stock_to_concept + torch.diag_embed(
            (i_stock_to_concept.sum(0) != 0).float() * diag
        )
        hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()
        hidden = hidden[hidden.sum(1) != 0]

        i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
        i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
        i_shared_info = self.fc_is(i_concept_to_stock.mm(hidden))
        i_shared_back = self.fc_is_back(i_shared_info)
        output_is = self.leaky_relu(self.fc_is_fore(i_shared_info))

        individual_info = x_hidden - e_shared_back - i_shared_back
        output_indi = self.leaky_relu(self.fc_indi(individual_info))

        all_info = output_es + output_is + output_indi
        return self.fc_out(all_info).squeeze()
```
