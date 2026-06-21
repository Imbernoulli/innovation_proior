All-to-all attention did exactly what I hoped on the broad universes and exactly what I feared on the narrow one, and the split is the whole story. On csi300 GATs lifted information ratio from LightGBM's $0.280$ to $1.362$ and annualized return from $0.0203$ to $0.106$, with signal IC also up to $0.0495$; csi300_recent held the same shape (IR $1.324$). But csi100 is the tell I was watching: IR went from $-0.334$ to only $-0.252$ and the return stayed negative at $-0.0141$, even though signal IC there is fine at $0.0465$. So the model *can* rank csi100 stocks; it just cannot build a profitable top-50 from them. That is the over-smoothing signature — with only a hundred names and one all-to-all attention with no structure, the softmax over the full cross-section pulls every stock toward the day's mean, and on a low-breadth universe where most of the edge lives in the *idiosyncratic* spread between names, washing out that spread keeps IC positive while killing IR. The attention shares information indiscriminately, every stock borrowing from every other including the irrelevant ones, and the curated concept graph — deliberately ignored last rung — is sitting unused, holding exactly the structure for *which* stocks to borrow from.

I propose **HIST**, concept-aware shared-information mining. The move is to bring the concept structure back, but not as a frozen adjacency mask — the curated graph is static and incomplete, and I still believe that. Instead I make a concept a first-class, date-specific object whose representation and whose edges to stocks are *computed from the current embeddings* rather than read from a fixed table, and I do this for both the curated concepts *and* a set of concepts discovered from the data. That yields three sources of signal per stock — what it shares with predefined themes, what it shares with discovered hidden themes, and its own idiosyncratic residual — and crucially keeps them separate, so the idiosyncratic part that all-to-all attention washed out is preserved in its own channel. The backbone is unchanged: each stock's $60\times 6$ window goes through an LSTM (`hidden_size=64`, `num_layers=2`, `dropout=0.0`), warm-started by loading the pretrained `model_lstm_csi300.pkl` so the relational modules learn on top of a sensible single-stock encoder. The new machinery is what happens after the last hidden state $x_i$.

The **predefined-concept module** fixes the first wall with the curated graph: membership is binary and frozen while a stock's relevance to a theme drifts day to day. So membership only *initializes* a concept's representation. Take the binary stock-concept matrix $M$ (rows = the day's stocks, columns = concepts) and form each concept's vector as a degree-normalized aggregate of its member stocks' hidden states. The normalization is a smoothed membership degree: divide each member's contribution by $(\text{membership-masked column-sum} + 1)$, so a singleton concept does not explode into a full-strength copy of one stock, and an empty concept produces a zero row I drop. That gives a concept representation $e_k$ per surviving column, $\text{hidden} = M^{\top} x_{\text{hidden}}$ after the normalization. Then — the drift fix — concept information flows *back* to the stocks not through the frozen membership but through cosine similarity on the *current* embeddings: each stock scores its cosine similarity to every concept vector, softmax over concepts (each stock choosing a mixture over columns), and the concept vectors are aggregated by those weights into a per-stock shared vector, passed through a learned `fc_es`. A stock can now borrow from a known theme even when its original membership row did not attach it strongly, and the weights are recomputed every date — the dynamic edge editing the frozen mask could never express. Instead of attending to all 300 stocks, a stock attends to the *themes* it currently resembles: a far lower-dimensional, structured neighborhood.

The **backcast/forecast residual chain** is what keeps the three sources separate and what preserves csi100's idiosyncratic spread. The predefined module emits not only a forecast head `output_es` (a LeakyReLU on a learned transform of the shared vector) but a *backcast* `e_shared_back` (a learned linear head, no nonlinearity) representing the part of $x_i$ it has accounted for. The next module runs on the residual $x^1_i = x_i - e_{\text{shared\_back}}$. Without this subtraction, three parallel modules would all see the full embedding and redundantly re-encode the same sector move three times — double-counting the shared trend and starving the idiosyncratic part, which is over-smoothing in another guise. The backcast forces a decomposition: each module only sees what the previous ones could not explain.

The **hidden-concept module** attacks the second wall — edges that *aren't there*, emerging themes the curators never labeled — by discovering concepts from the residual $x^1_i$ with no labels. The construction is parameter-free: posit one hidden concept seeded by each stock (initialize seed $k$ as $x^1_k$), measure every stock's cosine similarity to every seed, and connect each stock to its single most similar seed; seeds many stocks point at *are* emergent groups, seeds nobody points at are spurious and get deleted. There is an immediate bug to handle: every stock's similarity to its own seed is exactly $1$ (cosine of a vector with itself), so a row-argmax would always pick itself and no grouping happens. The fix is to zero the diagonal of the similarity matrix before the row-max, take each row's argmax over the remaining columns as that stock's chosen concept, keep only that one connection per row (mark it $10$, zero everything else, then restore the kept value), and re-add a surviving seed's own originating stock to its membership via $\text{diag\_embed}\big((\text{col-sum}\ne 0)\cdot\text{diag}\big)$. Aggregate each surviving concept's members' residual embeddings (similarity-weighted) into the hidden concept's representation, drop empty columns, and send the hidden shared information back to stocks via the same cosine → softmax → weighted-sum → `fc_is` path. This stage has essentially no concept-specific parameters — a similarity-driven clustering rerun fresh every date — which is what I want, because the themes it finds should be free to change as the market changes. This is the channel that should catch the co-movement on csi100 that the curated concepts miss and that indiscriminate attention drowned.

The **individual module** runs the residual discipline once more: the hidden module emits its own backcast `i_shared_back`, and the individual information is $x^2_i = x_i - e_{\text{shared\_back}} - i_{\text{shared\_back}}$ — the part explained by neither predefined nor hidden themes. A nonlinear head `fc_indi` on this residual is the idiosyncratic forecast. *This* is the channel all-to-all attention destroyed on csi100: the stock's own peculiar signal, kept separate and never averaged against the cross-section. Because the backcasts subtract on the way down, the three slices add back to the original embedding exactly, so the forecasts *sum* on the way out,
$$\text{all\_info} = \text{output\_es} + \text{output\_is} + \text{output\_indi},$$
and a final `fc_out` reads the scalar score: predefined-shared trend plus hidden-shared trend plus individual trend, the additive structure the residual stack guarantees.

The harness specifics I honor: early stopping is on validation **IC**, not loss — `metric_fn` computes the Pearson correlation between predictions and labels on each day's batch, the right objective for a ranking task and a change from the GATs rung's loss-based stopping. Training is per-day cross-sections (a batch that mixed dates would mix unrelated graphs), MSE loss masked to finite labels, Adam at `lr=1e-4`, grad-value clip $3.0$, up to 200 epochs, patience 20. The per-day concept matrix is fetched from `qlib_csi300_stock2concept.npy` via the `stock_index` map, with index $733$ as the padding fallback for unknown instruments. The sharpest test against GATs is csi100 IR, which the bare attention left at $-0.252$: I expect HIST to close most of that gap toward zero, driven by the preserved individual channel and the hidden-concept channel catching unlabeled co-movement, with IC rising above GATs on every universe. I will not be surprised if HIST trades a little raw csi300 IR — the structured model spends some sharing on the narrow universe and the idiosyncratic channel rather than maximally smoothing the broad one — and that is a good trade if csi100 crosses toward zero and IC rises everywhere.

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


class HISTModel(nn.Module):
    """HIST network -- verbatim from qlib/contrib/model/pytorch_hist.py."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

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

        self.fc_es = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es.weight)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is.weight)

        self.fc_es_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_middle.weight)
        self.fc_is_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_middle.weight)

        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_fore.weight)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_fore.weight)
        self.fc_indi_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi_fore.weight)

        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_back.weight)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim=0)
        self.softmax_t2s = torch.nn.Softmax(dim=1)

        self.fc_out_es = nn.Linear(hidden_size, 1)
        self.fc_out_is = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def cal_cos_similarity(self, x, y):  # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
        cos_similarity = xy / (x_norm.mm(torch.t(y_norm)) + 1e-6)
        return cos_similarity

    def forward(self, x, concept_matrix):
        device = torch.device(torch.get_device(x))

        x_hidden = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1)  # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        # Predefined Concept Module

        stock_to_concept = concept_matrix

        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)

        stock_to_concept_sum = stock_to_concept_sum + (
            torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device)
        )
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)

        hidden = hidden[hidden.sum(1) != 0]

        concept_to_stock = self.cal_cos_similarity(x_hidden, hidden)
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        e_shared_info = concept_to_stock.mm(hidden)
        e_shared_info = self.fc_es(e_shared_info)

        e_shared_back = self.fc_es_back(e_shared_info)
        output_es = self.fc_es_fore(e_shared_info)
        output_es = self.leaky_relu(output_es)

        # Hidden Concept Module
        i_shared_info = x_hidden - e_shared_back
        hidden = i_shared_info
        i_stock_to_concept = self.cal_cos_similarity(i_shared_info, hidden)
        dim = i_stock_to_concept.shape[0]
        diag = i_stock_to_concept.diagonal(0)
        i_stock_to_concept = i_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        row = torch.linspace(0, dim - 1, dim).to(device).long()
        column = i_stock_to_concept.max(1)[1].long()
        value = i_stock_to_concept.max(1)[0]
        i_stock_to_concept[row, column] = 10
        i_stock_to_concept[i_stock_to_concept != 10] = 0
        i_stock_to_concept[row, column] = value
        i_stock_to_concept = i_stock_to_concept + torch.diag_embed((i_stock_to_concept.sum(0) != 0).float() * diag)
        hidden = torch.t(i_shared_info).mm(i_stock_to_concept).t()
        hidden = hidden[hidden.sum(1) != 0]

        i_concept_to_stock = self.cal_cos_similarity(i_shared_info, hidden)
        i_concept_to_stock = self.softmax_t2s(i_concept_to_stock)
        i_shared_info = i_concept_to_stock.mm(hidden)
        i_shared_info = self.fc_is(i_shared_info)

        i_shared_back = self.fc_is_back(i_shared_info)
        output_is = self.fc_is_fore(i_shared_info)
        output_is = self.leaky_relu(output_is)

        # Individual Information Module
        individual_info = x_hidden - e_shared_back - i_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)

        # Stock Trend Prediction
        all_info = output_es + output_is + output_indi
        pred_all = self.fc_out(all_info).squeeze()

        return pred_all


class CustomModel(Model):
    """HIST model -- faithful to qlib's official HIST (pytorch_hist.py).

    Hyperparameters from official benchmark:
    examples/benchmarks/HIST/workflow_config_hist_Alpha360.yaml
    """

    def __init__(self):
        super().__init__()
        # Official benchmark hyperparameters
        self.d_feat = 6
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.0
        self.n_epochs = 200
        self.lr = 1e-4
        self.metric = "ic"
        self.early_stop = 20
        self.loss = "mse"
        self.base_model = "LSTM"
        self.model_path = "examples/benchmarks/LSTM/model_lstm_csi300.pkl"
        self.stock2concept = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock2concept.npy")
        self.stock_index = os.path.expanduser("~/.qlib/qlib_data/qlib_csi300_stock_index.npy")
        self.optimizer_name = "adam"
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.HIST_model = HISTModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.train_optimizer = optim.Adam(
            self.HIST_model.parameters(), lr=self.lr
        )
        self.fitted = False
        self.HIST_model.to(self.device)

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
        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
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

    def train_epoch(self, x_train, y_train, stock_index):
        stock2concept_matrix = np.load(self.stock2concept)
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        stock_index = stock_index.values
        stock_index[np.isnan(stock_index)] = 733
        self.HIST_model.train()

        # organize the train data into daily batches
        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index[batch]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            pred = self.HIST_model(feature, concept_matrix)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.HIST_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y, stock_index):
        # prepare training data
        stock2concept_matrix = np.load(self.stock2concept)
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        stock_index = stock_index.values
        stock_index[np.isnan(stock_index)] = 733
        self.HIST_model.eval()

        scores = []
        losses = []

        # organize the test data into daily batches
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index[batch]]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.HIST_model(feature, concept_matrix)
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
            raise ValueError("Empty data from dataset, please check your dataset config.")

        stock_index_map = np.load(self.stock_index, allow_pickle=True).item()
        df_train["stock_index"] = 733
        df_train["stock_index"] = df_train.index.get_level_values("instrument").map(stock_index_map)
        df_valid["stock_index"] = 733
        df_valid["stock_index"] = df_valid.index.get_level_values("instrument").map(stock_index_map)

        x_train, y_train, stock_index_train = df_train["feature"], df_train["label"], df_train["stock_index"]
        x_valid, y_valid, stock_index_valid = df_valid["feature"], df_valid["label"], df_valid["stock_index"]

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
            pretrained_model.load_state_dict(torch.load(self.model_path))

        model_dict = self.HIST_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.HIST_model.load_state_dict(model_dict)

        # train
        self.fitted = True

        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train, stock_index_train)
            train_loss, train_score = self.test_epoch(x_train, y_train, stock_index_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid, stock_index_valid)
            print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.HIST_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    print("early stop")
                    break

        print("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.HIST_model.load_state_dict(best_param)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset: DatasetH, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        stock2concept_matrix = np.load(self.stock2concept)
        stock_index_map = np.load(self.stock_index, allow_pickle=True).item()
        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        df_test["stock_index"] = 733
        df_test["stock_index"] = df_test.index.get_level_values("instrument").map(stock_index_map)
        stock_index_test = df_test["stock_index"].values
        stock_index_test[np.isnan(stock_index_test)] = 733
        stock_index_test = stock_index_test.astype("int")
        df_test = df_test.drop(["stock_index"], axis=1)
        index = df_test.index

        self.HIST_model.eval()
        x_values = df_test.values
        preds = []

        # organize the data into daily batches
        daily_index, daily_count = self.get_daily_inter(df_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index_test[batch]]).float().to(self.device)

            with torch.no_grad():
                pred = self.HIST_model(x_batch, concept_matrix).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)
```
