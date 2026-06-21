The problem is forecasting a single long time series whose statistical properties drift over time. In air quality, power demand, or financial returns, the input distribution P(x) changes with regime, season, or macro shocks, while the conditional P(y|x) — the law that turns today's conditions into tomorrow's outcome — tends to stay much more stable. A standard recurrent forecaster trained to minimize average error over the whole stream silently assumes the data are i.i.d., so when the test window comes from a different regime than the training bulk, the model degrades and has no way to know why. Domain-adaptation methods could in principle help, but they are built for a labelled source/target pair and usually match only the final network representation, throwing away the hidden trajectory that is the whole reason for using an RNN.

The right move is to split the stream into distinct temporal periods, treat the problem as temporal covariate shift, and force the recurrent net to learn what is invariant across those periods. I would call this method AdaRNN, short for Adaptive RNN. It has two stages. First, Temporal Distribution Characterization discovers the periods. Because I have no prior on the unseen future, the least-committal stance is to train under the worst case: split the stream so the periods are as distributionally different as possible, then learn what survives that diversity. Concretely, I pre-split the stream into a small number of candidate units and greedily add boundaries that maximize the average pairwise distribution distance among the resulting periods, sweeping a few choices of K and selecting by validation performance. Second, Temporal Distribution Matching adds a distribution-alignment regularizer to the prediction loss, but unlike endpoint-only adaptation it matches the distribution at every hidden state of the recurrent trajectory. Matching only the final summary wastes the trajectory: the shift between two periods is not a single fact about the endpoints but plays out across the whole sequence, with early states still dominated by the input embedding and late states carrying the integrated history. Each layer therefore contributes its own sequence of hidden states, and the loss sums a chosen distribution distance over all of them. The distance itself can be cosine on the period means, linear or RBF maximum mean discrepancy, CORAL covariance alignment, or even a domain-adversarial discrepancy; the framework does not depend on which one is used. The states are weighted by an importance vector alpha, one per layer and per period pair, because early and late states differ in how much they drift and how useful alignment is.

The trickiest part is learning alpha. A separate neural network for alpha fails early because the hidden states are meaningless before the GRU has trained, and it is expensive when the number of period pairs grows. Instead I pre-train the GRU on prediction loss alone for a warm-up phase so the states become informative. Then I switch to a boosting update: initialize alpha uniformly, and after each epoch increase the weight on any state whose cross-period distance grew relative to the previous epoch, using a multiplier 1 plus a sigmoid of the distance increase. This keeps the multiplier in (1, 2), so weights only ratchet up on the worst-aligned states and a noisy epoch cannot blow them up; I then L1-normalize so lambda alone controls the matching strength. At inference time all of this machinery drops away — I just run the GRU and the prediction head, one forward pass.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance(source, target):
    source = source.mean(dim=0)
    target = target.mean(dim=0)
    return 1.0 - F.cosine_similarity(source, target, dim=0).mean()


class MMDLoss(nn.Module):
    def __init__(self, kernel_type="linear"):
        super().__init__()
        self.kernel_type = kernel_type

    def forward(self, source, target):
        if self.kernel_type == "linear":
            delta = source.mean(0) - target.mean(0)
            return delta.dot(delta)
        # rbf fallback: squared MMD with a single Gaussian kernel
        total = torch.cat([source, target], dim=0)
        n = total.size(0)
        dist = torch.cdist(total, total, p=2).pow(2)
        bandwidth = dist.sum().item() / (n * n - n) + 1e-5
        kernel = torch.exp(-dist / bandwidth)
        b = source.size(0)
        return kernel[:b, :b].mean() + kernel[b:, b:].mean() - 2 * kernel[:b, b:].mean()


class TransferLoss:
    def __init__(self, loss_type="cosine"):
        self.loss_type = loss_type
        self.mmd = MMDLoss()

    def compute(self, X, Y):
        if self.loss_type == "cosine":
            return cosine_distance(X, Y)
        if self.loss_type in ("mmd", "mmd_lin"):
            return self.mmd(X, Y)
        raise ValueError(f"unknown loss_type {self.loss_type}")


class AdaRNN(nn.Module):
    def __init__(self, n_input, n_hiddens=(64, 64), n_output=1, len_seq=9,
                 trans_loss="cosine", dropout=0.0):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.num_layers = len(n_hiddens)
        self.len_seq = len_seq
        self.trans_loss = trans_loss

        in_size = n_input
        cells = []
        for h in n_hiddens:
            cells.append(nn.GRU(in_size, h, num_layers=1,
                                batch_first=True, dropout=dropout))
            in_size = h
        self.cells = nn.ModuleList(cells)
        self.head = nn.Linear(n_hiddens[-1], n_output)

        # per-layer gate for the warmup phase
        self.gates = nn.ModuleList([
            nn.Linear(len_seq * h * 2, len_seq) for h in n_hiddens])
        self.bns = nn.ModuleList([nn.BatchNorm1d(len_seq) for _ in n_hiddens])
        self.softmax = nn.Softmax(dim=0)

    def gru_features(self, x, predict=False):
        out = x
        states = []
        weights = [] if not predict else None
        for i, cell in enumerate(self.cells):
            out, _ = cell(out.float())
            states.append(out)
            if not predict:
                s = out[: out.size(0) // 2]
                t = out[out.size(0) // 2:]
                cat = torch.cat([s, t], dim=2).view(s.size(0), -1)
                w = torch.sigmoid(self.bns[i](self.gates[i](cat)))
                weights.append(self.softmax(w.mean(dim=0)))
        return out, states, weights

    def forward_pre_train(self, x, len_win=0):
        out, states, weights = self.gru_features(x)
        pred = self.head(out[:, -1, :]).squeeze()
        src, tar = [], []
        for s in states:
            src.append(s[: s.size(0) // 2])
            tar.append(s[s.size(0) // 2:])
        loss_transfer = 0.0
        for i, (s, t) in enumerate(zip(src, tar)):
            crit = TransferLoss(self.trans_loss)
            for j in range(self.len_seq):
                start = max(j - len_win, 0)
                end = min(j + len_win, self.len_seq - 1)
                for k in range(start, end + 1):
                    loss_transfer = loss_transfer + weights[i][j] * crit.compute(
                        s[:, j, :], t[:, k, :])
        return pred, loss_transfer, weights

    def forward_boosting(self, x, weight_mat=None):
        out, states, _ = self.gru_features(x, predict=False)
        pred = self.head(out[:, -1, :]).squeeze()
        src, tar = [], []
        for s in states:
            src.append(s[: s.size(0) // 2])
            tar.append(s[s.size(0) // 2:])
        if weight_mat is None:
            weight_mat = torch.ones(self.num_layers, self.len_seq) / self.len_seq
        dist_mat = torch.zeros(self.num_layers, self.len_seq)
        loss_transfer = 0.0
        for i, (s, t) in enumerate(zip(src, tar)):
            crit = TransferLoss(self.trans_loss)
            for j in range(self.len_seq):
                d = crit.compute(s[:, j, :], t[:, j, :])
                loss_transfer = loss_transfer + weight_mat[i, j] * d
                dist_mat[i, j] = d
        return pred, loss_transfer, dist_mat, weight_mat

    def update_weights(self, weight_mat, dist_old, dist_new):
        eps = 1e-5
        with torch.no_grad():
            grew = dist_new > dist_old + eps
            weight_mat[grew] *= 1.0 + torch.sigmoid(dist_new[grew] - dist_old[grew])
            weight_mat = weight_mat / weight_mat.norm(p=1, dim=1, keepdim=True)
        return weight_mat

    def predict(self, x):
        out, _, _ = self.gru_features(x, predict=True)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_adarnn(model, period_loaders, n_epochs=200, pre_epoch=40,
                 dw=0.5, lr=1e-3, len_win=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    weight_mat, dist_old = None, None

    for epoch in range(n_epochs):
        model.train()
        dist_mat = torch.zeros(model.num_layers, model.len_seq)
        for batches in zip(*period_loaders):
            optimizer.zero_grad()
            feats = [b[0] for b in batches]
            labels = [b[1] for b in batches]
            # simple two-period pairing; extend to all pairs for K > 2
            x = torch.cat([feats[0], feats[1]], dim=0)
            y_s, y_t = labels[0], labels[1]

            if epoch < pre_epoch:
                pred, loss_trans, _ = model.forward_pre_train(x, len_win=len_win)
            else:
                pred, loss_trans, dist, weight_mat = model.forward_boosting(x, weight_mat)
                dist_mat = dist_mat + dist

            pred_s = pred[: feats[0].size(0)]
            pred_t = pred[feats[0].size(0):]
            loss = criterion(pred_s, y_s) + criterion(pred_t, y_t) + dw * loss_trans
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()

        if epoch >= pre_epoch and epoch > pre_epoch and dist_old is not None:
            weight_mat = model.update_weights(weight_mat, dist_old, dist_mat)
        dist_old = dist_mat.detach().clone()
```
