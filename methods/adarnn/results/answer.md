# AdaRNN, distilled

AdaRNN (Adaptive RNN) is a framework for forecasting non-stationary time series under what it
formalizes as *Temporal Covariate Shift* (TCS): the input marginal `P(x)` drifts over the series
while the conditional `P(y|x)` is shared across time. It builds an RNN forecaster robust to that
drift in two stages — **Temporal Distribution Characterization (TDC)**, which discovers the
distinct distributional periods inside the training stream, and **Temporal Distribution Matching
(TDM)**, which trains the RNN to align those periods' representations while keeping the temporal
dependency, weighting each hidden state by a boosting-learned importance vector.

## Problem it solves

A single long, non-stationary stream where the test stretch's input distribution differs from the
training bulk. Train under the I.I.D. assumption and the forecaster degrades on the future. The
goal is to extract the cross-period invariant — the shared `P(y|x)` — so the model generalizes to
an unseen future regime, given that the periods (their number `K` and boundaries) are unknown.

## Key idea

1. **TCS framing.** Decompose the stream into `K` periods `D_1..D_K`; within a period the
   distribution is fixed, across periods `P_{D_i}(x) != P_{D_j}(x)` but `P_{D_i}(y|x) =
   P_{D_j}(y|x)`. The test period shares the conditional, differs in the marginal.

2. **TDC — discover the periods by maximizing diversity.** With no prior on the unseen test
   distribution, the maximum-entropy (least-committal) choice is to train under the worst case:
   the most distributionally diverse split. Solve
   `max_{0<K<=K_0} max_{n_1..n_K} (1/K) sum_{i!=j} d(D_i, D_j)` s.t. `Delta_1 < |D_i| < Delta_2`,
   `sum_i |D_i| = n`. Intractable exactly, so a greedy search: evenly pre-split into `N=10` units
   (9 candidate cuts), sweep `K in {2,3,5,7,10}`, and greedily add the cut that most increases the
   summed pairwise period distance.

3. **TDM — match all hidden states, weighted.** Add a distribution-matching regularizer to the
   prediction loss. Matching only the final RNN state wastes the recurrent trajectory, so match at
   every hidden state `t = 1..V`, weighted by an importance vector `alpha` (one per period pair,
   per layer):
   `L_tdm(D_i, D_j) = sum_t alpha_{ij}^t d(h_i^t, h_j^t)`. Full objective for one layer:
   `L(theta, alpha) = L_pred(theta) + lambda (2/(K(K-1))) sum_{i<j} L_tdm(D_i, D_j; theta, alpha)`,
   with `L_pred` the per-period MSE and `2/(K(K-1))` averaging over the `K(K-1)/2`
   unordered period pairs.

4. **Learn `alpha` by pre-training + boosting.** A learned `alpha`-network fails: `alpha` and
   `theta` are coupled (early hidden states are meaningless) and one network per period-pair is
   costly. Instead: (a) pre-train `theta` with `lambda=0` to get meaningful states; (b) update
   `alpha` by boosting — initialize uniform `1/V`, and for each state whose cross-period distance
   *grew* from the previous epoch to the current epoch, multiply its weight by
   `G = 1 + sigma(d_new^t - d_old^t)`
   (so `G in (1,2)`, importance only ratchets up on the worst-aligned states), then L1-normalize.
   Inference uses only `L_pred` — one forward pass, no matching cost.

5. **Distance-agnostic.** `d` can be cosine, linear/RBF MMD, CORAL, or domain-adversarial
   discrepancy; the same family serves TDC and TDM. Main results use MMD; the qlib wrapper defaults
   to cosine for cheapness on large data.

## Final algorithm

```
1. TDC: split the training stream into K periods by maximizing avg pairwise distribution distance
        (greedy over the 9 candidate cuts of the N=10 even pre-split, sweeping K in {2,3,5,7,10}).
2. Pre-train theta on the prediction loss alone (lambda = 0) across all periods -> theta_0.
3. For each epoch:
     for each period pair (D_i, D_j):
         run both through the stacked GRU; compute prediction MSE on each period;
         compute the importance-weighted matching loss summed over all hidden states;
     update theta by SGD/Adam on L(theta, alpha) (grad clipped to value 3.0);
     update alpha by the boosting rule from this epoch's per-state distances vs last epoch's.
   Keep the best (theta, alpha) by validation metric.
4. Inference: forward pass through GRU + head (L_pred only).
```

## Working code (qlib `ADARNN`, faithful to `pytorch_adarnn.py`)

The network (`AdaRNN`) — stacked single-layer GRUs, optional bottleneck, per-layer gate weights for
warmup, boosting-updated weights, and the per-state matching — is exactly the form derived in the
reasoning trace (see `reasoning.md` for the full network and the distance functions
`cosine`/`CORAL`/`MMD_loss`/`adv`/`TransferLoss`). The qlib wrapper below drives that network. The
general greedy period search lives in the transferlearning `data_process.TDC` code; the qlib finance
wrapper uses an even split of training days into `n_splits` periods before running the same
pre-epoch and boosting phases.

```python
class ADARNN(Model):
    """qlib Model wrapper around the AdaRNN network."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0,
                 n_epochs=200, pre_epoch=40, dw=0.5, loss_type="cosine",
                 len_seq=60, len_win=0, lr=0.001, metric="mse",
                 batch_size=2000, early_stop=20, loss="mse", optimizer="adam",
                 n_splits=2, GPU=0, seed=None, **_):
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.pre_epoch = pre_epoch
        self.dw = dw                         # lambda: matching trade-off
        self.loss_type = loss_type           # distribution distance d
        self.len_seq = len_seq
        self.len_win = len_win
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_splits = n_splits
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        n_hiddens = [hidden_size for _ in range(num_layers)]
        self.model = AdaRNN(use_bottleneck=False, bottleneck_width=64, n_input=d_feat,
                            n_hiddens=n_hiddens, n_output=1, dropout=dropout,
                            model_type="AdaRNN", len_seq=len_seq, trans_loss=loss_type)
        if self.optimizer == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    def train_AdaRNN(self, train_loader_list, epoch, dist_old=None, weight_mat=None):
        self.model.train()
        criterion = nn.MSELoss()
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        out_weight_list = None
        for data_all in zip(*train_loader_list):                 # one minibatch per period
            self.train_optimizer.zero_grad()
            list_feat, list_label = [], []
            for data in data_all:
                feature, label_reg = data[0].to(self.device).float(), data[1].to(self.device).float()
                list_feat.append(feature); list_label.append(label_reg)
            index = get_index(len(data_all) - 1)                 # unordered period pairs
            if any(list_feat[s1].shape[0] != list_feat[s2].shape[0] for s1, s2 in index):
                continue
            total_loss = torch.zeros(1).to(self.device)
            for s1, s2 in index:
                feature_s, feature_t = list_feat[s1], list_feat[s2]
                label_s, label_t = list_label[s1], list_label[s2]
                feature_all = torch.cat((feature_s, feature_t), 0)
                if epoch < self.pre_epoch:                      # warmup path: gate-weighted matching
                    pred_all, loss_transfer, out_weight_list = self.model.forward_pre_train(
                        feature_all, len_win=self.len_win)
                else:                                           # boosting-weighted matching
                    pred_all, loss_transfer, dist, weight_mat = self.model.forward_Boosting(
                        feature_all, weight_mat)
                    dist_mat = dist_mat + dist
                pred_s = pred_all[0: feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0):]
                loss_s, loss_t = criterion(pred_s, label_s), criterion(pred_t, label_t)
                total_loss = total_loss + loss_s + loss_t + self.dw * loss_transfer
            self.train_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
        if epoch >= self.pre_epoch:
            if epoch > self.pre_epoch:                          # boosting update of alpha
                weight_mat = self.model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
            return weight_mat, dist_mat
        weight_mat = self.transform_type(out_weight_list)       # seed weights from gate
        return weight_mat, None

    def fit(self, dataset: DatasetH, evals_result=dict(), save_path=None):
        df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"],
                                             data_key=DataHandlerLP.DK_L)
        days = df_train.index.get_level_values(level=0).unique()
        train_splits = np.array_split(days, self.n_splits)        # TDC: even split into K periods
        train_splits = [df_train[s[0]: s[-1]] for s in train_splits]
        train_loader_list = [get_stock_loader(df, self.batch_size) for df in train_splits]

        save_path = get_or_create_path(save_path)
        self.fitted = True
        stop_steps = 0
        best_score = -np.inf
        weight_mat, dist_mat = None, None
        for step in range(self.n_epochs):
            weight_mat, dist_mat = self.train_AdaRNN(train_loader_list, step, dist_mat, weight_mat)
            train_metrics = self.test_epoch(df_train)
            valid_metrics = self.test_epoch(df_valid)
            valid_score = valid_metrics[self.metric]
            if valid_score > best_score:
                best_score, stop_steps, best_epoch = valid_score, 0, step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        return best_score

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return self.infer(x_test)

    def infer(self, x_test):
        index = x_test.index
        self.model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        x_values = x_values.reshape(sample_num, self.d_feat, -1).transpose(0, 2, 1)
        preds = []
        for begin in range(sample_num)[:: self.batch_size]:
            end = min(begin + self.batch_size, sample_num)
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                preds.append(self.model.predict(x_batch).detach().cpu().numpy())
        return pd.Series(np.concatenate(preds), index=index)

    def transform_type(self, init_weight):                        # gate weights -> [num_layers, len_seq] matrix
        weight = torch.ones(self.num_layers, self.len_seq).to(self.device)
        for i in range(self.num_layers):
            for j in range(self.len_seq):
                weight[i, j] = init_weight[i][j].item()
        return weight
```

`get_index` enumerates unordered period pairs `(i, j)`; `get_stock_loader` wraps each period's
`(feature, label)` in a `DataLoader`. The loader reshapes qlib features to
`[batch, len_seq, d_feat]`, and inference uses the same reshape before calling `predict`.
`test_epoch` computes IC/ICIR/RankIC and the selected validation metric for early stopping. The
boosting rule `update_weight_Boosting` and the network (`AdaRNN`, `TransferLoss`, and the distance
functions) are listed in full in `reasoning.md`.

## Why the choices

- **Maximize (not minimize) cross-period distance in TDC** — maximum entropy / worst-case: no
  prior on the unseen test regime, so train on the most diverse split for best generalization;
  the opposite (similar-period) split underperforms.
- **Greedy split** — the exact diversity-maximizing split is combinatorially intractable at scale.
- **Match all hidden states, not just the last** — an RNN's value is its hidden trajectory; each
  state carries partial, drifting distribution information.
- **Importance vector `alpha`** — states differ in drift and in alignment value; equal weights
  underperform.
- **Pre-train then boosting (not a learned `alpha`-net)** — `alpha`/`theta` coupling makes a
  learned net train on meaningless early states, and a per-pair net is expensive; boosting needs
  no extra parameters and concentrates on the worst-aligned states. `G = 1 + sigma(.)` guarantees
  importance only increases on stalling states, bounded by 2.
- **L1-normalize `alpha`** — keep it a convex weighting so `lambda` alone controls matching
  strength.
- **Two stages (TDC then TDM)** — controls computation; end-to-end joint optimization is left open.
