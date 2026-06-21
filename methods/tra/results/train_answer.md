The thing that actually fails is the assumption hiding inside the usual recipe. I train a sequence model on stock features to predict the next-period cross-sectional return rank in the standard way: collect $\{(x_i, y_i)\}$, encode each lookback window with an attentive LSTM into a latent vector, push it through a linear head, and regress to the label by mean-squared error — empirical risk minimization under the silent premise that every $(x_i, y_i)$ is an i.i.d. draw from one fixed joint distribution $P$, so there is a single relation $p(y\mid x)$ to learn. The market is not that. It is the aggregate of many participants running different strategies, and two of its most documented cross-sectional effects point in *opposite* directions at once: momentum (stocks that rose keep rising) and reversal (stocks that fell rebound). I can make the contradiction concrete without any new experiment. Fit a plain linear model of next-month return rank on three features — size rank, value rank, twelve-month momentum rank — separately on different years: the momentum coefficient comes out negative on 2009 (high past return predicted low future return, pure reversal) and positive on 2013 (momentum). One coefficient vector cannot be both signs at once. So the ceiling I keep hitting is not model capacity; it is that I am asking one function with one parameter vector to hold two relations that contradict, and the best it can do is average them, fitting neither. The data is really a mixture $P = \sum_k \nu_k P_k$, each component a distinct pattern $p_k(y\mid x)$ with its own share $\nu_k$, and which one is active rotates over time.

If I knew the pattern label of every sample this would be ordinary multi-domain learning, but the pattern identifiers are latent, and worse than latent: even if I cheated and used the labels to find each training sample's best-fitting pattern, that trick dies at test time, where there are no labels — and the test period is the whole point. The prior tools each fall short on exactly this. A single backbone (attentive LSTM or Transformer) is one estimator with one head: it learns the average relation and has no mechanism to recognize which regime a sample is in. Frequency-decomposed recurrence splits the cell state along *frequency*, fixed by architecture, not arbitrary discovered patterns, and remains one fused predictor. Mixture-of-experts gating is structurally right — several sub-functions, a gate that routes — but its gate has a notorious self-reinforcing collapse, and the standard cure, a soft importance penalty equal to the squared coefficient of variation of per-expert mass, only nudges toward balance, does not equalize the number of examples per expert, and has no opinion about *which* sample belongs to which expert. Balanced self-labelling via optimal transport fixes collapse with a hard equipartition constraint, but it assigns clusters in a representation-learning setting where the assignment is used directly — it does not supply a supervised predictor whose assignment must be reproduced at test without the current label. That last constraint — select from current-label-free state — is the one that kills the naive solutions, and it must stay front of mind.

I propose the Temporal Routing Adaptor (TRA): a lightweight module that lets a single shared backbone carry several coexisting patterns and route each sample to the right one using only label-free state. The first design choice is what to replicate. The obvious move is $K$ full parallel backbones, but that multiplies parameters by $K$ and overfits a noisy, smallish financial dataset hard; and the patterns differ in the *relation* $p(y\mid x)$ far more than in how to encode the window, so the feature extractor can be shared. I keep one shared backbone $\psi$ and replicate only the cheap final linear layer into $K$ linear predictors, $\hat y_{ik} = \theta_k \circ \psi(x_i)$ — negligible extra parameters, and with $K=1$ this collapses exactly to the original single-head backbone, so by construction I can never do worse than the baseline. On top sits the router, which must run at test time and therefore can see only label-free signals: the latent representation $h_i = \psi(x_i)$ itself (call it LR — routing on $h$, which is already tuned toward the prediction task, is more informative about the active $p(y\mid x)$ than the raw features), and the temporal prediction errors (TPE), the history of each predictor's recent losses $e_i = [\,l_{(s,t-T)}, \dots, l_{(s,t-h)}\,]$ where each $l_{(s,t')}$ is the $K$-vector of all predictors' errors at an earlier time. The intuition for TPE is how a real investor rotates strategies — they watch which strategies have been working lately rather than deducing the macro regime from today's prices. One subtlety keeps the router causal: the most recent usable entry is $t-h$, not $t$, with $h$ exceeding the label horizon, because the loss at $t$ needs a future return I do not know yet. A small RNN summarizes the variable-length error history into its last hidden state, which is concatenated with $h_i$ and mapped by a linear $fc$ to $K$ router logits $a_i$.

How the router *uses* those logits is the next load-bearing choice. A soft attention blend $\hat p_i = \mathrm{softmax}(a_i)^\top \hat y_i$ is differentiable and easy, but it re-creates the original disease: every predictor receives gradients from every sample, so they all drift toward the same average relation — a smeared committee instead of one model. To force genuine specialization the routing must be *discrete*: each sample goes to one predictor, which is then trained only on its kind of sample and is free to swing its coefficients to, say, negative-momentum without the momentum samples dragging it back. Discrete selection is $\arg\max$, which has no gradient. The escape is the Gumbel-softmax: draw $g_{ik} \sim \mathrm{Gumbel}(0,1)$ and form
$$\mathrm{choice}_{ik} = \frac{\exp((a_{ik} + g_{ik})/\tau)}{\sum_j \exp((a_{ij} + g_{ij})/\tau)},$$
which is an exact categorical draw from $\mathrm{softmax}(a_i)$ in the $\tau\to 0$ limit and a smooth, differentiable point on the simplex for finite $\tau$. I use a hard straight-through sample for the training selection (genuine one-hot forward, soft gradient backward) and keep the deterministic probability $q_i = \mathrm{softmax}(a_i/\tau)$ for the auxiliary loss and for $\arg\max$ selection at test, with $\tau = 1$.

Training this end to end naively collapses: every sample routes to one predictor and the rest starve, the self-reinforcing gate pathology — whichever predictor is marginally better early gets more samples, trains faster, and is favored even more. A soft balance penalty does not fix it because it cannot say which sample belongs where; arbitrary balanced assignment does not discover patterns. So I state the real objective precisely. Pack every predictor's error on every sample into a loss matrix $L_{ik} = \ell(\theta_k \circ \psi(x_i), y_i)$ and seek an assignment $P \in \{0,1\}^{N\times K}$ that minimizes total assigned loss while giving each sample exactly one predictor and keeping the predictors balanced:
$$\min_P \ \langle P, L\rangle \quad \text{s.t.} \quad \sum_k P_{ik} = 1, \quad \sum_i P_{ik} = \nu_k \cdot N, \quad P_{ik} \in \{0,1\}.$$
These are marginal constraints on a nonnegative matrix with a linear cost — an optimal-transport problem, transporting unit mass from the $N$ samples to the $K$ predictors at cost $L$. The anti-collapse requirement *is literally* the column-marginal constraint: a prescribed amount of mass is forced into every predictor, so none can be starved to zero, while $\langle P, L\rangle$ simultaneously pulls each sample toward the predictor that fits it best. This is the same equipartition-constrained transport that cures collapse in self-labelling. The exact LP is combinatorial and about $O(d^3 \log d)$ — too slow per minibatch — so I use the entropic relaxation, whose positive solution has the scaling form $P = \mathrm{diag}(u)\,\exp(-L/\varepsilon)\,\mathrm{diag}(v)$. In implementation I build a fused cost $L_h$, pass $-L_h$ into Sinkhorn, compute $\exp(-L_h/\varepsilon)$, and then alternate column normalization (equalize predictor mass) followed by row normalization (restore one unit per sample), in exactly that order, for three iterations; the final row normalization makes the per-sample mass exact while the columns are a fast balanced approximation. The negative sign is non-negotiable — low loss must become high transport mass. Because $\varepsilon$ sits in the exponential, $L$ is min-max normalized per row to a consistent scale, with a degenerate all-equal row mapping to a flat (uniform) assignment rather than blowing up.

But $P$ depends on $L$, which depends on the labels, so $P$ is a label-dependent oracle — perfect as a teacher, useless as the test-time selector. So I distill it. The router, fed only LR and TPE, is trained to imitate the OT assignment as a supervised classification problem with target $P_{i\cdot}$ and prediction $q_i$, via cross-entropy. Writing $\mathrm{reg}_i = \sum_k P_{ik}\log q_{ik} = -\mathrm{CE}(P_i, q_i)$, the objective is
$$\min_{\Theta,\pi,\psi} \ \mathbb{E}_i\!\left[\,\ell(x_i, y_i; \Theta,\pi,\psi) \;-\; \lambda \sum_k P_{ik}\log q_{ik}\,\right],$$
so $\text{loss} = \text{task\_loss} - \lambda\cdot\mathrm{reg}$ is exactly task loss plus a cross-entropy to the OT teacher. The training selection still uses the hard Gumbel sample, but the regularizer uses the plain probability $q$, because the target is a distribution to imitate. At test I discard $P$, the loss machinery, and the regularizer, and simply run $\arg\max(q_i)$ — the label-free selector the router has learned to be.

Two unknowns in the objective I pin down by reasoning rather than guessing. The shares $\nu_k$ are latent; early in training the predictors are undifferentiated, so the maximally noncommittal and maximally anti-collapse choice is equal shares $\nu_k = 1/K$, which forces every predictor to take a $1/K$ slice in the teacher. Equal shares is surely wrong in the limit, so I do not let the teacher dominate forever: the transport target stays balanced but its grip weakens, decaying $\lambda_t = \lambda_0\,\rho^{\lfloor \text{step}/100\rfloor}$ with $\rho = 0.99$, so the model balances hard early and lets the task loss settle into the data's true share structure later. The count $K$ is also unknown and treated as a hyperparameter — too few and one head still averages momentum and reversal, too many and the data shreds into overfitting slices — with $K=3$ a modest default and the $K=1$ safety net. Three implementation walls remain. The TPE input would, if recomputed fresh, need a full forward over all $N$ samples every step; instead I cache per-predictor errors in an external memory $M \in \mathbb{R}^{N\times K}$, refreshing it wholesale before each epoch and patching it for the minibatch on the fly — an $N/\text{batch}$ speedup, at the cost of slightly stale errors, which is fine since the error history is slow-moving. Temporal order must be respected strictly so the router only ever sees earlier errors. The single-batch loss matrix is noisy, so I fuse it with the smoother memory, $L_h = \alpha\cdot\mathrm{minmax}(L) + (1-\alpha)\cdot\mathrm{minmax}(\text{hist\_loss})$ with $\alpha = 0.5$, then renormalize. And a cold router cannot bootstrap off random predictors, so I warm-start: pretrain the backbone and predictors against the *oracle* transport (route by $P$ directly, $\lambda=0$) to force the predictors apart, then reset the optimizer and train the router with router-transport plus the OT cross-entropy. The defaults: an attentive-LSTM backbone with `input_size=20`, `hidden_size=64`, two layers, attention pooling (latent size $2\cdot 64 = 128$); a router with $K=3$ states, a one-layer LSTM of hidden size 32 over the error history, $\tau=1$, `src_info=LR_TPE`; Adam at $10^{-3}$, $\lambda=1$, $\rho=0.99$, $\alpha=0.5$, sequence length 60, batch 1024, Sinkhorn with three iterations and $\varepsilon=0.1$.

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
