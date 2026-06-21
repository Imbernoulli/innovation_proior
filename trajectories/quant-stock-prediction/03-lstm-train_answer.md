The LightGBM rung did what I expected on the axis that mattered most and left a precise gap on another. On csi300 the IR turned **positive** — 0.280, up from the transformer's −0.44 — and the IC climbed to 0.0399 from 0.0117, with Rank IC 0.0492; csi300_recent told the same story (IR 0.266). That confirms the robustness diagnosis: a learner with no optimization fragility formed real signal where the brittle transformer formed noise. But csi100 came back exactly as I worried it might — a perfectly respectable IC of 0.0363, yet a *negative* portfolio return (−0.017) and IR (−0.334). The run is deterministic, so there is no variance to blame; the tree simply ranks csi100 about as well as csi300 by IC, but that ranking quality does not survive the TopkDropout backtest on the smaller universe. The tree threw away the temporal structure of the 60×6 window, and on csi100 especially that lost structure is leaving ranking quality on the table that a model *aware* of the time axis could recover. The directed move is to go back to a sequence model — but emphatically *not* the brittle transformer. I want the temporal inductive bias *with* the optimization robustness the tree showed me matters most here: a recurrent net trained at a healthy learning rate with generous early-stop patience, reading the window as a sequence but training the forgiving way the tree was robust.

The question is which recurrence, and the obvious one fails for a reason I can write down. Take a plain recurrent net: its state is $h_t = f(h_{t-1}, x_t)$, and I want a feature from day 1 of the window to influence the read-out at day 60. Follow a single error signal backward through the sixty steps. The error that lands on a unit at the end and reaches a unit $q$ steps earlier is, telescoped, a *product* of $q$ factors, each of the form $f'(\text{net})\cdot w$. With the logistic squashing the derivative peaks at 0.25, so each factor is below 1 whenever $|w| < 4$ — and any reasonable weight is — so the product over $q$ steps decays geometrically. The gradient from sixty days ago is exponentially attenuated by the time it reaches the read-out, and bigger weights only push the unit toward saturation where $f'$ collapses faster than $w$ grows, making it worse. A bigger learning rate scales long-range and short-range credit identically, so the *ratio* is unchanged and recent days still dominate every update. The vanishing is structural — the lag sits in the exponent, and no ordinary knob touches it. A plain RNN over a 60-step window would learn mostly from the last few days, nearly as blind to day 1 as the tree was. That is not the fix.

I propose a **2-layer LSTM** — a gated recurrent regressor. The cure has to make the product over the lag *not* shrink, ideally exactly 1 for any $q$. In the simplest setting, a single unit with a self-connection of weight $w_{jj}$, the per-step backward factor is $f'(\text{net})\cdot w_{jj}$; demanding it equal 1 forces $f'(\text{net}) = 1/w_{jj}$, a constant — the squashing must be *linear*. Take the cleanest case, the identity with $w_{jj} = 1$: the activation simply *persists* unchanged step to step, and the backpropagated error riding through it is multiplied by exactly 1 each step. A linear self-loop of weight one — a constant error carousel — is a channel down which gradient survives an arbitrarily long lag. That is the seed.

A bare carousel cannot be wired to the rest of the net without two conflicts appearing. A single incoming weight has to do two opposed jobs: at the moment the relevant day's information arrives it must let it *in* (write to memory), but on all other days it must *not* let irrelevant inputs overwrite what is stored (protect) — one weight receiving contradictory update signals, the input weight conflict. The mirror image is on the output side: one outgoing weight must release the stored content when needed and shield downstream units from it when not — the output weight conflict. A static weight cannot resolve either, because it is one number and cannot be context-sensitive. But another *unit* can be context-sensitive, and a *multiplicative* gate can do what an additive bias cannot: a control in $[0,1]$ multiplying the input can zero an irrelevant signal *completely* (perfect protection) or pass it through entirely. So I wrap the carousel in multiplicative gates that are themselves learned sigmoid units reading the rest of the net — an input gate deciding when to write, an output gate deciding when to read. A third issue — on a stream that is never reset the linear state accumulates without bound and the output squashing saturates — is fixed by a forget gate multiplying the carried state, recovering the exact carousel when it is open (weight 1) and letting the cell reset when it is closed.

That gives the gated memory cell, in compact form
$$i_t = \sigma(W_{ix} x_t + W_{ih} h_{t-1} + b_i),\quad f_t = \sigma(\cdots),\quad g_t = \tanh(\cdots),$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t,\quad o_t = \sigma(\cdots),\quad h_t = o_t \odot \tanh(c_t),$$
with $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ the carousel, now gated. The backward pass carries the state error as $\varepsilon_s^t = o_t \odot \tanh'(c_t) \odot \varepsilon_h^t + f_{t+1} \odot \varepsilon_s^{t+1}$ — when the forget gate is near 1, the state error flows back across the lag at unit gain, the constant carousel restated as a fact about gradients. So over the 60-day window a feature from day 1 can influence the day-60 read-out *and* its gradient can flow back to day 1 undamped — precisely the temporal credit the tree could not assign and the plain RNN would have lost to the vanishing product. This is the right recurrence.

I have to respect one same-named subtlety in how this is wired to the edit surface. This rung uses qlib's **non-TS** LSTM on the ordinary `DatasetH` (not `TSDatasetH`). Each training sample is *still* the flat 360-dim Alpha360 row — the same rows the tree saw — and the time axis is recovered *inside* the model by `x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]`, not by the dataset handing over genuine overlapping windows. So the LSTM reconstructs the 60-step sequence from the same flat vector the tree treated as tabular; the difference is purely that the LSTM *interprets* those 360 numbers as a 60×6 sequence and runs the gated recurrence over it, while the tree saw 360 unordered columns. I stack two LSTM layers (`num_layers=2`) of `hidden_size=64`, take the final step's hidden state — the cell's summary of the whole window conditioned on the most recent day — and read it through `Linear(64, 1)` to one score. Width 64 and depth 2 are again modest by design: on a low-signal regression the danger is fitting noise, and a compact recurrent net is a regularizer, the same logic that drove the narrow transformer and the heavily-penalized trees.

The training loop is where this rung deliberately differs from the transformer rung, and it is the whole point. Adam at `lr = 1e-3` — *ten times* the transformer's $10^{-4}$ — because the LSTM is far less initialization-sensitive than the attention stack and can take a healthy step from the start; no warmup is needed. Early-stop patience is `20`, four times the transformer's 5, so the run is not cut off on a noisy validation dip the way I suspect the transformer was. Batch size 800, MSE loss masked over the finite labels, gradient value-clipped at 3.0 — the clip is the one nod to the *exploding* side of the error-flow analysis, capping an update without touching the cell equations — up to 200 epochs, restoring the best-validation parameters. Every choice is the robustness lesson from the tree applied to a sequence model: train it the forgiving way, not the high-wire way. And the processor note is the inverse of the lgbm rung's edit, which I must not get backwards: the LSTM, like the transformer, is a gradient-descent model that needs standardized inputs to train stably, so this rung *keeps* the default workflow processor block — `RobustZScoreNorm` + `Fillna` on the features — exactly as the transformer rung did, and does *not* strip it the way the tree rung did. The neural preprocessing I removed for the trees is the preprocessing the LSTM depends on.

So the delta from the LightGBM rung is: keep the optimization robustness that made the tree beat the transformer, but restore the temporal inductive bias the tree discarded — interpret the same 360 numbers as a 60×6 sequence and run a gated memory cell over it so a day at the start of the window can both influence and receive gradient from the end, trained at a healthy learning rate with generous patience and the neural feature normalization the tree did not need. The cleanest falsifiable claim is on csi300: the LSTM should beat both prior rungs on IC and IR together — IC above the tree's 0.0399 (toward the mid-0.04s) and IR well above the tree's 0.280 — because the temporal credit assignment should sharpen the ranking and the forgiving training should let it actually converge, unlike the transformer. If csi300 IR does not clear 0.280, the temporal bias did not pay and the tree's robust tabular view was already extracting most of the signal. The sharper, more uncertain test is csi100, where the tree had a fine IC but a *negative* IR: if the LSTM's temporal awareness genuinely orders the ranking better, csi100's IR should improve over −0.334 — but I am honestly unsure it will turn positive, because csi100's problem may be structural (holding 50 of 100 names and churning 5 a day is punishing regardless of signal quality); if csi100 stays negative even as its IC improves, that confirms the weakness is a portfolio-construction artifact, not a model failure. The single number I will judge this rung on is the csi300 information ratio against the tree's 0.280 — clearing it decisively is the evidence that bringing back the time axis, done robustly, was the right top of the ladder.

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
