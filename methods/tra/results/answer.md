# Temporal Routing Adaptor (TRA), distilled

TRA is a lightweight extension module that lets a single stock-prediction backbone model *multiple coexisting
trading patterns*. It replaces the backbone's one output head with `K` linear **predictors** (one per latent
pattern) on a shared feature extractor, and adds a **router** that selects, per sample, which predictor to use
using only information available without the current sample's label (the backbone latent plus each predictor's
recent error history), so it works at test time. The routing collapses if trained naively (all samples to one
predictor), so the assignment of samples to predictors is posed as a **balanced optimal-transport problem**
solved by Sinkhorn, and that label-dependent assignment is distilled into the router through a cross-entropy
auxiliary loss.

## Problem it solves

Stock-return prediction is trained as i.i.d. supervised learning with one estimator, but the market is a
mixture of distributions `P = Σ_k ν_k P_k` whose conditional `p(y|x)` differs by regime — momentum (past
winners keep winning) and reversal (past losers rebound) are documented, opposite-signed, and rotate over time,
so a single model can only fit their average. Pattern identifiers are latent and, even if recoverable from
labels in training, are unavailable at test. The goal: represent several relations at once and select the right
one per sample from current-label-free state, cheaply, with `K=1` recovering the plain backbone.

## Key idea

1. **Predictors (cheap experts).** Replace the single output layer with `K` linear heads on a shared backbone
   `ψ`: `ŷ_ik = θ_k ∘ ψ(x_i)`. Negligible extra parameters; `K=1` ⇒ original model (no regression risk).
2. **Router (label-free selector).** Inputs: the latent representation `h_i = ψ(x_i)` (LR) and the temporal
   prediction errors `e_i = [l_{(s,t-T)},…,l_{(s,t-h)}]` of the `K` predictors over a lookback window
   (TPE; `h` exceeds the label horizon to avoid leakage). A small RNN summarizes the error history; its last
   hidden state is concatenated with `h_i` and mapped by a linear `fc` to `K` logits `a_i`.
3. **Discrete, differentiable routing.** Real specialization needs a *discrete* choice (a soft blend re-creates
   the averaging problem), but `argmax` is non-differentiable. Use the **Gumbel-softmax**:
   `choice_ik = exp((a_ik + g_ik)/τ) / Σ_j exp((a_ij + g_ij)/τ)`, `g_ik ~ Gumbel(0,1)`, with a hard
   straight-through sample for training selection. Keep the deterministic router probability
   `q_i = softmax(a_i/τ)` for the auxiliary loss and use `argmax(q_i)` at test.
4. **Balanced assignment as optimal transport (the core).** Naive router training collapses all samples onto
   one predictor (self-reinforcing gate pathology); a soft balance penalty does not say *which* sample belongs
   where. State the real objective: assign each sample to its lowest-error predictor while keeping predictors
   balanced. With loss matrix `L_ik = ℓ(θ_k ∘ ψ(x_i), y_i)` and assignment `P ∈ {0,1}^{N×K}`:

   ```
   min_P  <P, L>
   s.t.   Σ_k P_ik = 1            (one predictor per sample)
          Σ_i P_ik = ν_k · N      (balanced share -> kills collapse)
          P_ik ∈ {0,1}
   ```

   This is an optimal-transport problem. Exact LP is `O(d^3 log d)`; use the **entropic relaxation**
   (Cuturi-style): `P = diag(u)·exp(-L/ε)·diag(v)`, with `u, v` as the row/column scaling vectors. In the qlib
   implementation the fused cost `L_h` is passed as `sinkhorn(-L_h)`, the solver computes `exp((-L_h)/ε)`,
   then alternates column normalization followed by row normalization. The negative sign makes low loss receive
   high mass. At convergence in the equal-share case the rows sum to one and the columns to `N/K`; with the
   configured three iterations, the last row normalization gives exact per-sample mass and a fast balanced
   approximation across predictors.
5. **Distill OT into the router.** `P` depends on labels (via `L`), so it cannot select at test. Train the
   router to imitate it with `reg = Σ_k P_ik log q_ik`, where `q` is the router's plain softmax probability
   (`prob` in code); since `reg = -CE(P_i, q_i)`, the implementation's
   `loss = task_loss - λ*reg` is task loss plus a cross-entropy to the OT teacher:

   ```
   min_{Θ,π,ψ}  E_i [ ℓ(x_i, y_i; Θ,π,ψ)  −  λ Σ_k P_ik log q_ik ]
   ```

6. **Unknowns and schedule.** Shares `ν_k` unknown ⇒ use equal-share transport (`ν_k = 1/K`) to force diversity,
   then relax its effect by **decaying `λ`** as `λ_t = λ_0·ρ^{⌊step/100⌋}` (`ρ` just below 1) so the task loss
   increasingly determines the learned routing.
   `K` unknown ⇒ a hyperparameter (default 3). Fuse current and historical loss into the transport cost,
   `L_h = α·minmax(L) + (1−α)·minmax(hist_loss)`, for a stable target.
7. **Memory + warm-start (implementation).** Computing TPE fresh would need a full forward over all `N` per
   step; cache per-predictor errors in memory `M ∈ R^{N×K}`, refreshed wholesale before each epoch and patched
   per-minibatch on the fly (≈`N/batch` speedup); preserve temporal order at test (router needs earlier
   errors). **Pretrain** the backbone+predictors with the *oracle* transport (route by the balanced `P`
   directly, `λ=0`) to force the predictors apart, then reset the optimizer and train the router with
   router-transport + the OT regularizer.

## Defaults (qlib Alpha158 / CSI300)

Backbone (attentive LSTM): `input_size=20`, `hidden_size=64`, `num_layers=2`, `use_attn=True`, `dropout=0`
(latent size `2·64=128`). TRA: `num_states=3`, `rnn_arch=LSTM`, router `hidden_size=32`, `num_layers=1`,
`tau=1.0`, `src_info=LR_TPE`. Training: Adam `lr=1e-3`, `n_epochs=100`, `early_stop=20`, `lamb=1.0`,
`rho=0.99`, `alpha=0.5`, `transport_method=router`, `memory_mode=sample`, `pretrain=True`. Data: `seq_len=60`,
`batch_size=1024`, `drop_last=True`. Sinkhorn: `n_iters=3`, `epsilon=0.1`.

## Sample-wise qlib implementation

Faithful to the qlib sample-memory configuration: backbone + `K`-predictor router, Sinkhorn OT,
`transport_sample`, two-stage memory, oracle-pretrain then router-train, and label-free argmax inference.

```python
import io, os, copy, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

EPS = 1e-12
device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN(nn.Module):
    """Attentive LSTM/GRU backbone: window [N, seq_len, d_feat] -> latent [N, output_size]."""
    def __init__(self, input_size=16, hidden_size=64, num_layers=2,
                 rnn_arch="LSTM", use_attn=True, dropout=0.0, **kwargs):
        super().__init__()
        self.rnn_arch, self.use_attn = rnn_arch, use_attn
        self.input_proj = nn.Linear(input_size, hidden_size) if hidden_size < input_size else None
        self.rnn = getattr(nn, rnn_arch)(
            input_size=min(input_size, hidden_size), hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout)
        if use_attn:
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        rnn_out, last_out = self.rnn(x)
        if self.rnn_arch == "LSTM":
            last_out = last_out[0]
        last_out = last_out.mean(dim=0)
        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)
            att_out = (rnn_out * scores).sum(dim=1)
            last_out = torch.cat([last_out, att_out], dim=1)
        return last_out


class TRA(nn.Module):
    """K linear predictors + a router (LR and/or TPE) emitting a discrete gumbel choice."""
    def __init__(self, input_size, num_states=1, hidden_size=32, rnn_arch="LSTM",
                 num_layers=1, dropout=0.0, tau=1.0, src_info="LR_TPE"):
        super().__init__()
        assert src_info in ["LR", "TPE", "LR_TPE"], "invalid `src_info`"
        self.num_states, self.tau = num_states, tau
        self.rnn_arch, self.src_info = rnn_arch, src_info
        self.predictors = nn.Linear(input_size, num_states)
        if num_states > 1:
            if "TPE" in src_info:
                self.router = getattr(nn, rnn_arch)(
                    input_size=num_states, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size + input_size if "LR" in src_info else hidden_size,
                                    num_states)
            else:
                self.fc = nn.Linear(input_size, num_states)

    def forward(self, hidden, hist_loss):
        preds = self.predictors(hidden)
        if self.num_states == 1:
            return preds, None, None
        if "TPE" in self.src_info:
            out = self.router(hist_loss)[1]
            if self.rnn_arch == "LSTM":
                out = out[0]
            out = out.mean(dim=0)
            if "LR" in self.src_info:
                out = torch.cat([hidden, out], dim=-1)
        else:
            out = hidden
        out = self.fc(out)
        choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)
        prob = torch.softmax(out / self.tau, dim=-1)
        return preds, choice, prob


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    if len(pred.shape) == 2:
        label = label[:, None]
    return (pred[mask] - label[mask]).pow(2).mean(dim=0)


def minmax_norm(x):
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    mask = (xmin == xmax).squeeze()
    x = (x - xmin) / (xmax - xmin + EPS)
    x[mask] = 1
    return x


def shoot_infs(inp_tensor):
    """Replace inf by max finite value (numerical guard for exp)."""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            inp_tensor[tuple(ind)] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            inp_tensor[tuple(ind)] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.1):
    """Entropic OT: qlib passes -cost, then exp(Q/eps), column norm, row norm."""
    with torch.no_grad():
        Q = torch.exp(Q / epsilon)
        Q = shoot_infs(Q)
        for _ in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q


def transport_sample(all_preds, label, choice, prob, hist_loss, count,
                     transport_method, alpha, training=False):
    """Sample-wise transport: build L, fuse with history, Sinkhorn -> P, routed pred + loss."""
    assert all_preds.shape == choice.shape
    assert transport_method in ["oracle", "router"]
    all_loss = torch.zeros_like(all_preds)
    mask = ~torch.isnan(label)
    all_loss[mask] = (all_preds[mask] - label[mask, None]).pow(2)        # L_ik [N, K]

    L = minmax_norm(all_loss.detach())
    Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)
    Lh = minmax_norm(Lh)
    P = sinkhorn(-Lh)                                                    # low loss -> high mass

    if transport_method == "router":
        if training:
            pred = (all_preds * choice).sum(dim=1)                       # gumbel selection
        else:
            pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)] # label-free argmax
        loss = loss_fn(pred, label)
    else:                                                               # oracle: route by P
        pred = (all_preds * P).sum(dim=1)
        loss = (all_loss * P).sum(dim=1).mean()
    return loss, pred, L, P


def evaluate(pred):
    pred = pred.rank(pct=True)
    diff = pred.score - pred.label
    return {"MSE": (diff ** 2).mean(), "MAE": diff.abs().mean(),
            "IC": pred.score.corr(pred.label, method="spearman")}


class TRAModel:
    """qlib-style model interface. The workflow supplies MTSDatasetH sequence batches with memory."""

    def __init__(self):
        self.model_config = dict(input_size=20, hidden_size=64, num_layers=2,
                                 rnn_arch="LSTM", use_attn=True, dropout=0.0)
        self.tra_config = dict(num_states=3, rnn_arch="LSTM", hidden_size=32,
                               num_layers=1, dropout=0.0, tau=1.0, src_info="LR_TPE")
        self.lr, self.n_epochs, self.early_stop = 1e-3, 100, 20
        self.lamb, self.rho, self.alpha = 1.0, 0.99, 0.5
        self.transport_method, self.pretrain = "router", True
        self.transport_fn = transport_sample
        self.model = RNN(**self.model_config).to(device)
        self.tra = TRA(self.model.output_size, **self.tra_config).to(device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()),
                                    lr=self.lr)
        self.fitted, self.global_step = False, -1

    def train_epoch(self, data_set, is_pretrain=False):
        self.model.train(); self.tra.train(); data_set.train()
        self.optimizer.zero_grad()
        for batch in tqdm(data_set):
            if not is_pretrain:
                self.global_step += 1
            data, state, label = batch["data"], batch["state"], batch["label"]
            index = batch["index"]
            hidden = self.model(data)
            all_preds, choice, prob = self.tra(hidden, state)
            loss, pred, L, P = self.transport_fn(
                all_preds, label, choice, prob, state.mean(dim=1), None,
                self.transport_method if not is_pretrain else "oracle",  # oracle for pretrain
                self.alpha, training=True)
            data_set.assign_data(index, L)                              # patch memory
            decay = self.rho ** (self.global_step // 100)               # relax equal-share prior
            lamb = 0 if is_pretrain else self.lamb * decay
            reg = prob.log().mul(P).sum(dim=1).mean()                   # log-likelihood under OT target
            loss = loss - lamb * reg
            loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()

    def test_epoch(self, data_set, return_pred=False):
        self.model.eval(); self.tra.eval(); data_set.eval()
        preds, metrics = [], []
        for batch in tqdm(data_set):
            data, state, label = batch["data"], batch["state"], batch["label"]
            index = batch["index"]
            with torch.no_grad():
                hidden = self.model(data)
                all_preds, choice, prob = self.tra(hidden, state)
                loss, pred, L, P = self.transport_fn(
                    all_preds, label, choice, prob, state.mean(dim=1), None,
                    self.transport_method, self.alpha, training=False)
            data_set.assign_data(index, L)
            X = np.c_[pred.cpu().numpy(), label.cpu().numpy()]
            df = pd.DataFrame(X, index=batch["index"], columns=["score", "label"])
            metrics.append(evaluate(df))
            if return_pred:
                preds.append(df)
        m = pd.DataFrame(metrics)
        out = {"MSE": m.MSE.mean(), "MAE": m.MAE.mean(),
               "IC": m.IC.mean(), "ICIR": m.IC.mean() / m.IC.std()}
        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
            return out, preds["score"]
        return out

    def _fit(self, train_set, valid_set, test_set, is_pretrain):
        best_score, stop = -1, 0
        best = {"model": copy.deepcopy(self.model.state_dict()),
                "tra": copy.deepcopy(self.tra.state_dict())}
        if not is_pretrain:
            self.test_epoch(train_set)                                  # init memory
        for epoch in range(self.n_epochs):
            self.train_epoch(train_set, is_pretrain=is_pretrain)
            valid_metrics = self.test_epoch(valid_set)
            if valid_metrics["IC"] > best_score:
                best_score, stop = valid_metrics["IC"], 0
                best = {"model": copy.deepcopy(self.model.state_dict()),
                        "tra": copy.deepcopy(self.tra.state_dict())}
            else:
                stop += 1
                if stop >= self.early_stop:
                    break
        self.model.load_state_dict(best["model"])
        self.tra.load_state_dict(best["tra"])
        return best_score

    def fit(self, dataset):
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
        self.fitted, self.global_step = True, -1
        if self.pretrain:                                              # 1) oracle: specialize preds
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.tra.predictors.parameters()), lr=self.lr)
            self._fit(train_set, valid_set, test_set, is_pretrain=True)
            self.optimizer = optim.Adam(                               # 2) reset, learn router
                list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)
        self._fit(train_set, valid_set, test_set, is_pretrain=False)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        test_set = dataset.prepare(segment)
        _, preds = self.test_epoch(test_set, return_pred=True)         # router argmax, label-free
        return preds
```

The workflow provides the time-series dataset (sequence batches over a 60-day window, an external memory
holding per-sample per-predictor errors that the router reads as TPE, and index bookkeeping to map predictions
back to `(datetime, instrument)`). `daily` transport (assigning whole trading days rather than individual
samples) is a variant of the same Sinkhorn machinery, summing each day's per-predictor loss before transport.
