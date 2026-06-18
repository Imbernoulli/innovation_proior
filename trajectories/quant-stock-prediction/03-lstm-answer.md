**Problem.** LightGBM fixed the optimization fragility (csi300 IR turned positive, 0.280; IC 0.0399) but
discarded the temporal structure of the 60×6 window. On csi100 it had a fine IC (0.0363) yet a *negative*
portfolio IR (−0.334): ranking quality the per-row tabular view extracts but cannot push past the
backtest. The gap is the time axis — recoverable by a sequence model, if trained robustly rather than on
the transformer's high-wire schedule.

**Key idea.** A 2-layer LSTM over the window. Interpret the same flat 360-dim Alpha360 row as a 60×6
sequence (`x.reshape(N,6,60).permute(0,2,1)`), run the gated memory cell over the 60 steps, take the
final hidden state, and read it to one score. The cell's constant-error carousel
(`c_t = f_t⊙c_{t-1} + i_t⊙g_t`) carries gradient across the window at unit gain when the forget gate is
open, so a day at the start can both influence and receive gradient from the end — the temporal credit the
tree could not assign and a plain RNN would lose to the vanishing-gradient product.

**Why.** Keep the optimization robustness that made the tree beat the transformer, but restore the
temporal inductive bias the tree threw away. The LSTM is far less initialization-sensitive than the
attention stack, so it trains at a healthy `lr=1e-3` (10× the transformer) with no warmup and generous
early-stop patience.

**Task-specific note.** This is the qlib *non-TS* LSTM on ordinary `DatasetH`: each sample is still the
flat 360-vector (same rows the tree saw), with the sequence reconstructed *inside* the model, not by a
`TSDatasetH` of genuine overlapping windows. The processor block is *kept at the default* neural pipeline
(`RobustZScoreNorm` + `Fillna`) — the inverse of the lgbm rung, which stripped it; a gradient-descent
model needs the standardized inputs the tree did not.

**Hyperparameters (qlib Alpha360 benchmark).** `d_feat=6`, `hidden_size=64`, `num_layers=2`,
`dropout=0.0`; Adam `lr=1e-3`; `batch_size=800`, MSE loss, `n_epochs=200`, early-stop patience `20`, grad
value-clip `3.0`.

**What to watch.** The decisive number is csi300 IR against the tree's 0.280 — clearing it decisively is
the evidence that bringing back the time axis (done robustly) was the right top of the ladder, with IC
expected toward the mid-0.04s. csi100 IR should improve over −0.334, but may not turn positive if its
weakness is a portfolio-construction artifact of a small universe rather than a signal-quality problem.

```python
# =====================================================================
# EDITABLE: CustomModel — implement your stock prediction model here
# =====================================================================
import copy
import torch.optim as optim


class LSTMModel(nn.Module):
    """LSTM network — verbatim from qlib/contrib/model/pytorch_lstm.py."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T] — Alpha360 gives 360 flat features
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class CustomModel(Model):
    """LSTM model — faithful to qlib's official LSTM (pytorch_lstm.py).

    Uses DatasetH with Alpha360 features. The LSTMModel reshapes the flat
    360-dim feature vector internally: [N, 360] -> [N, 6, 60] -> [N, 60, 6].

    Hyperparameters from official benchmark:
    examples/benchmarks/LSTM/workflow_config_lstm_Alpha360.yaml
    """

    def __init__(self):
        super().__init__()
        # Official Alpha360 benchmark hyperparameters
        self.d_feat = 6
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.0
        self.n_epochs = 200
        self.lr = 0.001
        self.metric = "loss"
        self.batch_size = 800
        self.early_stop = 20
        self.loss = "mse"
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.lstm_model = LSTMModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.train_optimizer = optim.Adam(
            self.lstm_model.parameters(), lr=self.lr
        )
        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.lstm_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_train_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.lstm_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = (
                torch.from_numpy(x_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )
            label = (
                torch.from_numpy(y_values[indices[i : i + self.batch_size]])
                .float()
                .to(self.device)
            )

            pred = self.lstm_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError(
                "Empty data from dataset, please check your dataset config."
            )

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        self.fitted = True

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print(
                "Epoch%d: train %.6f, valid %.6f"
                % (step, train_score, val_score)
            )

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.lstm_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index
        self.lstm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size
            x_batch = (
                torch.from_numpy(x_values[begin:end]).float().to(self.device)
            )
            with torch.no_grad():
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)
```
