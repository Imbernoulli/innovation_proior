The problem is forecasting a single long time series whose statistical properties drift over time. In air quality, power demand, or financial returns, the input distribution P(x) changes with regime, season, or macro shocks, while the conditional P(y|x) — the law that turns today's conditions into tomorrow's outcome — tends to stay much more stable. A standard recurrent forecaster trained to minimize average error over the whole stream silently assumes the data are i.i.d., so when the test window comes from a different regime than the training bulk, the model degrades and has no way to know why. Domain-adaptation methods could in principle help, but they are built for a labelled source/target pair and usually match only the final network representation, throwing away the hidden trajectory that is the whole reason for using an RNN.

The right move is to split the stream into distinct temporal periods, treat the problem as temporal covariate shift, and force the recurrent net to learn what is invariant across those periods. I would call this method AdaRNN, short for Adaptive RNN. It has two stages. First, Temporal Distribution Characterization discovers the periods. Because I have no prior on the unseen future, the least-committal stance is to train under the worst case: split the stream so the periods are as distributionally different as possible, then learn what survives that diversity. Concretely, I pre-split the stream into a small number of candidate units and greedily add boundaries that maximize the average pairwise distribution distance among the resulting periods, sweeping a few choices of K and selecting by validation performance. Second, Temporal Distribution Matching adds a distribution-alignment regularizer to the prediction loss, but unlike endpoint-only adaptation it matches the distribution at every hidden state of the recurrent trajectory. Matching only the final summary wastes the trajectory: the shift between two periods is not a single fact about the endpoints but plays out across the whole sequence, with early states still dominated by the input embedding and late states carrying the integrated history. Each layer therefore contributes its own sequence of hidden states, and the loss sums a chosen distribution distance over all of them. The distance itself can be cosine on the period means, linear or RBF maximum mean discrepancy, CORAL covariance alignment, or even a domain-adversarial discrepancy; the framework does not depend on which one is used. The states are weighted by an importance vector alpha, one per layer and per period pair, because early and late states differ in how much they drift and how useful alignment is.

The trickiest part is learning alpha. A separate neural network for alpha fails early because the hidden states are meaningless before the GRU has trained, and it is expensive when the number of period pairs grows. Instead I pre-train the GRU on prediction loss alone for a warm-up phase so the states become informative. Then I switch to a boosting update: initialize alpha uniformly, and after each epoch increase the weight on any state whose cross-period distance grew relative to the previous epoch, using a multiplier 1 plus a sigmoid of the distance increase. This keeps the multiplier in (1, 2), so weights only ratchet up on the worst-aligned states and a noisy epoch cannot blow them up; I then L1-normalize so lambda alone controls the matching strength. At inference time all of this machinery drops away — I just run the GRU and the prediction head, one forward pass.

The artifact I actually ship is a qlib-compatible model wrapper, `ADARNN`, that owns this two-phase loop and turns it into the `fit`/`predict` interface qlib expects: while `epoch < pre_epoch` it calls `forward_pre_train` to get the gate-weighted matching loss and seed the initial alpha, and once past the warm-up it calls `forward_Boosting` to get per-state distances that `update_weight_Boosting` folds into the next epoch's alpha, all under the same gradient-clipped Adam/SGD step. For the finance deployment the expensive diversity-maximizing TDC search is replaced by its cheap default — an even split of trading days into `n_splits` periods — since the periods here are already well-separated calendar chunks and the boosting/warm-up machinery is where the real gain comes from; the training loop, the loss, and the alpha update run exactly as derived above:

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
