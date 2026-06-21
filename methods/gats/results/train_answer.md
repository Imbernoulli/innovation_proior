The first thing to fix is the row-independence assumption. A gradient-boosted floor like LightGBM scores every (stock, day) pair on its own engineered alphas, so two stocks that move together on the same day are treated as unrelated points. That is fine for a baseline, but it wastes the single strongest source of structure in financial cross-sections: on any given day the returns of hundreds of stocks are correlated through shared risk, sectors, and sentiment. When the model cannot see the day's other stocks, its top-fifty ranking is noisy and its information ratio stays weak, even turning negative on a narrow universe such as csi100. The question is therefore not whether to add cross-sectional information, but how to let each stock look at the rest of the day's stocks without imposing a static, hand-curated graph that may be incomplete or wrong.

The right tool is attention, because attention is designed for exactly this situation: a variable-sized, unordered set of items that must be combined with learned, data-dependent weights. Earlier graph methods either require an eigendecomposition of a fixed graph, use degree-fixed coefficients such as GCN's 1/sqrt(d_i d_j), or presuppose a known concept graph. We want none of that. We want each stock to attend directly to every other stock in the same day's cross-section, with the relevance learned from their current hidden representations. A daily stock universe is at most a few hundred names, so a dense all-pairs attention matrix is cheap, and dropping the mask lets the model discover relationships that a static concept graph would miss. The method is Graph Attention Networks, implemented in this setting as the qlib GATs benchmark.

GATs begins by encoding each stock's raw Alpha360 features through an LSTM. The flattened input is reshaped from (N, F*T) to (N, T, F), run through a two-layer LSTM with hidden size 64, and the last time-step hidden state h_i is taken as the stock's temporal summary. This encoder is warm-started from a pretrained LSTM checkpoint so that gradient does not have to rediscover temporal structure from scratch and can instead spend its capacity on the new cross-stock attention layer. The LSTM output is then projected through a linear transformation and an additive attention scorer is applied across all pairs of stocks in the batch. For each pair (i, j) the raw attention score is e_ij = LeakyReLU(a^T [t(h_i) || t(h_j)]), where t is the learned projection, a is a trainable vector, and the concatenation makes the score asymmetric so that stock j can matter more to stock i than the reverse. The scores are normalized with a softmax across the whole day's cross-section, producing attention weights alpha_ij that sum to one per stock. The output for stock i is the attention-weighted sum of all stocks' projected features, added back as a residual to stock i's own hidden state, passed through another linear layer and LeakyReLU, then mapped to a single return forecast.

The design choices all serve the same purpose. All-to-all attention with no graph mask removes any structural assumption; the model learns which stocks to borrow from rather than being told by a concept matrix. The additive scorer, instead of a dot product, gives the comparison its own capacity and breaks symmetry. LeakyReLU keeps gradient flowing even for low scores, so the model can learn to down-weight irrelevant peers rather than zeroing them and losing the gradient. Softmax over the full day normalizes per stock, making the coefficients comparable whether the universe has 100 or 300 names. The residual add is crucial: without it the attention output would replace each stock's own signal and early, noisy attention could erase the temporal forecast; with it the cross-section acts as a correction. A single attention head is used here rather than the multi-head variant common in citation-network GAT, because at this scale a single head with heavy dropout is more stable and the pretrained encoder already provides a strong starting point.

Training follows the qlib Alpha360 protocol. The model is trained per-day across the training segment, with each day's stocks forming one attention graph. MSE loss is used against the rank-normalized return label, optimized with Adam at learning rate 1e-4, gradient-value clipping at 3.0, and dropout 0.7 applied to the LSTM and the attention inputs. Early stopping monitors validation loss with patience 20, and the best checkpoint is restored. Because the LSTM is pretrained on csi300, the model starts with a sensible temporal representation and refines it for the cross-stock objective.

```python
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class LSTMModel(nn.Module):
    """LSTM backbone for warm-starting the temporal encoder."""

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
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GATModel(nn.Module):
    """Graph Attention Network for daily stock cross-sections."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="LSTM"):
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
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


class CustomModel(Model):
    """GATs model faithful to qlib's official pytorch_gats.py benchmark."""

    def __init__(self):
        super().__init__()
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.GAT_model = GATModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            base_model=self.base_model,
        )
        self.train_optimizer = optim.Adam(self.GAT_model.parameters(), lr=self.lr)
        self.fitted = False
        self.GAT_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def loss_fn(self, pred, label):
        if self.loss == "mse":
            return self.mse(pred, label)
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred, label)
        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        self.GAT_model.train()
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
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        self.GAT_model.eval()
        scores, losses = [], []
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)
        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.GAT_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                scores.append(self.metric_fn(pred, label).item())
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

        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(
                d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers
            )
        elif self.base_model == "GRU":
            pretrained_model = LSTMModel(
                d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers
            )
        else:
            raise ValueError("unknown base model name `%s`" % self.base_model)

        if self.model_path:
            pretrained_model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

        model_dict = self.GAT_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.GAT_model.load_state_dict(model_dict)

        self.fitted = True
        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            _, train_score = self.test_epoch(x_train, y_train)
            _, val_score = self.test_epoch(x_valid, y_valid)
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
        daily_index, daily_count = self.get_daily_inter(x_test, shuffle=False)
        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.GAT_model(x_batch).detach().cpu().numpy()
            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=index)
```
