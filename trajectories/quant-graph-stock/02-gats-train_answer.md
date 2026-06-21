The boosting floor landed where it should: on csi300 LightGBM gave IC $0.0399$, ICIR $0.308$, Rank IC $0.0492$ — real, positive, modest factor signal. But the portfolio side exposes the row-independence assumption. The information ratio is only $0.280$ on csi300 and the annualized return $0.0203$, and on the harder, lower-breadth csi100 the portfolio actually loses money: IR $-0.334$, return $-0.0172$, even though the *signal* IC there is still positive at $0.0363$. That gap is the tell. The model can rank stocks well enough on average to post a positive IC, but the TopkDropout backtest lives or dies on getting the *top of the daily ranking* right day after day, and a model that scores each stock from its own 360 features — blind to what the rest of the cross-section is doing — produces a daily top-50 too noisy to compound, worst exactly where breadth is thinnest. The assumption to break is row-independence, and the minimal way to break it is to let the stocks present on a given day *look at each other* before each prediction is read out.

I propose **GATs**: encode each stock temporally, run *one all-to-all attention over the whole day's cross-section*, add the attended information back to each stock with a residual, and read out a per-stock score. The form matters, and it deliberately differs from the textbook graph-attention layer. The canonical layer attends a node over a *masked* neighborhood — you supply an adjacency and only edges in the graph get a score. I have a candidate adjacency here, the stock-concept membership matrix the harness loads, but I do not use it, for two reasons. First cost: masking exists to avoid all-pairs $O(N^2)$ on graphs with millions of nodes, but a single day's cross-section is at most a few hundred stocks (a hundred for csi100), so the full $N\times N$ attention matrix is tiny and trivially affordable — the cost reason for masking simply does not bite. Second diagnosis: the curated concept graph is static and incomplete (its edges are frozen the same way every day; an unlabeled emerging theme or a freshly listed name it can never propagate), and I want to test whether *any* learned cross-stock interaction beats the row-independent floor before committing to that structure. All-to-all learned attention isolates "does cross-section help" from "does the curated graph help," leaving the latter to the next rung.

The encoder comes first, because the attention needs a vector per stock and the raw input is a sequence. Each stock-day is 360 Alpha360 features reshaped to a $60\times 6$ window, which I run through a recurrent backbone, taking the last hidden state as the representation $h_i \in \mathbb{R}^{64}$. The backbone is an LSTM (`hidden_size=64`, `num_layers=2`, `dropout=0.7`), and it is *warm-started*: I load a pretrained LSTM (`model_lstm_csi300.pkl`) into the matching backbone weights before training, so the recurrent encoder begins as a sensible single-stock predictor and the attention layer learns the cross-stock correction on top of it. That structure matters — the attention is not learning temporal encoding and cross-sectional mixing simultaneously from scratch; the temporal part is warm-started and gradient descent spends its capacity on the new thing, the neighbor weighting. The heavy dropout of $0.7$ is the published Alpha360 setting, there for the same reason LightGBM needed enormous $\lambda$: the signal is faint and the model will overfit the training window without aggressive regularization.

The attention itself lands an exact form that differs from the canonical multi-head masked layer in several deliberate ways. After the backbone produces $h_i$ for every stock in the day, the layer passes each through a single shared linear transform $t(h)=W_t h$ — one transform, no per-head projections — and scores every ordered pair with the additive form
$$e_{ij} = \mathrm{LeakyReLU}\!\left(a^\top\big[t(h_i)\,\|\,t(h_j)\big]\right),\qquad a\in\mathbb{R}^{128}.$$
Additive, not dot-product, because the concatenation lets $a$ hold separate weights for the query half and the key half: the score is *asymmetric* (stock $i$'s relevance to $j$ need not equal $j$'s to $i$ — a large-cap leading a sector is more relevant to its followers than the reverse), and the comparison has its own trainable capacity decoupled from $W_t$. LeakyReLU rather than ReLU so that a low (negative) score — the layer saying "this stock is *un*important to that one" — still carries gradient and can be learned down rather than dead-zeroed. A *single* head, not the eight-head concatenation of the citation-network setup, because at this data scale with heavy dropout on a pretrained encoder one head is the stable choice. The implementation materializes the full pairwise score matrix directly: expand the $[N,\dim]$ transformed hidden into $[N,N,\dim]$ on the query side, transpose for the key side, concatenate along the feature axis to $[N,N,2\dim]$, and apply $a$ to get the $[N,N]$ score matrix in one shot.

Then the crucial no-mask step: softmax across the *entire* row, over all $N$ stocks, not a masked neighborhood,
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k}\exp(e_{ik})},$$
the sum running over every stock in the day, so each stock's attention is a full convex combination over the whole cross-section with the curated graph nowhere in sight. The per-stock normalization makes a 300-name day and a 100-name day produce comparable weights summing to one — the same degree-invariance the masked version gets, now over the full set. The aggregation adds the attended information back with a residual:
$$\text{hidden}_i = \Big(\sum_j \alpha_{ij}\, t(h_j)\Big) + h_i.$$
The residual is doing real work. Without it, a stock's representation after one attention layer is purely a mixture of the cross-section, and if early attention is noisy the stock loses its own signal entirely; with it, the attended cross-stock information is a *correction* added to the stock's own warm-started hidden state — the floor signal is preserved and the cross-section only adjusts it, exactly the inductive bias I want with a pretrained encoder. After the residual, a learned `fc` linear layer plus another LeakyReLU mixes the combined representation and a final `fc_out` reads out the scalar score.

Training is per-day cross-sections — a batch is one day's full set, since mixing dates would mix unrelated cross-sections and make the attention meaningless. The loss is MSE against the `CSRankNorm`-normalized next-day-return label, masked to finite labels, optimized with Adam at `lr=1e-4`, gradient-value clipping at $3.0$, early stopping on validation loss with patience 20, up to 200 epochs. The risk I leave explicitly on the table for the next rung: all-to-all attention with no structure can over-smooth — pull every stock toward the day's mean and wash out the idiosyncratic spread that, on csi100 especially, is much of the edge. If csi300 IR rises strongly while csi100 stays negative, that is the over-smoothing signature, and the fix is to stop attending to everything and aggregate through concept structure that keeps a stock's own and its theme's signal separate.

```python
# =====================================================================
# EDITABLE: CustomModel -- implement your stock prediction model here
# =====================================================================
import copy
import torch.optim as optim


class LSTMModel(nn.Module):
    """LSTM backbone -- verbatim from qlib/contrib/model/pytorch_lstm.py."""

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
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GRUModel(nn.Module):
    """GRU backbone -- verbatim from qlib/contrib/model/pytorch_gru.py."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GATModel(nn.Module):
    """GAT network -- verbatim from qlib/contrib/model/pytorch_gats.py."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x: [N, F*T] -- DatasetH provides flattened features
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


class CustomModel(Model):
    """GATs model -- faithful to qlib's official GATs (pytorch_gats.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/GATs/workflow_config_gats_Alpha360.yaml

    Uses DatasetH with Alpha360 (d_feat=6, 360 features reshaped to 60x6).
    Daily batching via get_daily_inter() for graph attention across stocks.
    """

    def __init__(self):
        super().__init__()
        # Official benchmark hyperparameters (Alpha360)
        self.d_feat = 6
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.7
        self.n_epochs = 200
        self.lr = 1e-4
        self.metric = "loss"
        self.early_stop = 20
        self.loss = "mse"
        self.base_model = "LSTM"
        self.model_path = "examples/benchmarks/LSTM/model_lstm_csi300.pkl"
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.train_optimizer = optim.Adam(
            self.GAT_model.parameters(), lr=self.lr
        )
        self.fitted = False
        self.GAT_model.to(self.device)

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

    def get_daily_inter(self, df, shuffle=False):
        # organize the train data into daily batches
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            # shuffle data
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        self.GAT_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)

            pred = self.GAT_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GAT_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.GAT_model.eval()

        scores = []
        losses = []

        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        best_param = None

        # optionally load pretrained base_model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers)
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path:
            pretrained_model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)

        # train
        self.fitted = True

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GAT_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GAT_model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.GAT_model.eval()
        x_values = x_test.values
        preds = []

        # organize the data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)

            with torch.no_grad():
                pred = self.GAT_model(x_batch).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)
```
