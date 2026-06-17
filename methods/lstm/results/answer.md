# Long Short-Term Memory (LSTM), distilled

LSTM is a recurrent architecture built around a linear memory state with a fixed unit self-loop —
the **constant error carousel (CEC)** — wrapped in learned multiplicative **gates**. The CEC lets
the backpropagated error ride across an arbitrarily long time lag at unit gain, instead of
decaying or exploding exponentially the way it does in an ordinary recurrent net. The gates make
the read/write/reset decisions context-sensitive, resolving the conflicts a single static weight
cannot.

## Problem it solves

Training a recurrent net by exact gradient (BPTT/RTRL) fails on long-range temporal dependencies:
error propagated `q` steps back through the recurrence is a product of `q` factors `f'(net)·w`,
which decays exponentially when those factors are below 1 (the ordinary sigmoid case, since
`max f' = 0.25`, so `|f'·w| < 1` whenever `|w| < 4`) and explodes when they are above 1. The lag
sits in the exponent; larger weights (saturation kills `f'` faster) and larger learning rates
(same long/short ratio) do not help. LSTM is designed to bridge minimal time lags well past 1000
steps, robustly to noisy inputs, at `O(1)` cost per weight per time step (local in space and time).

## Key idea

1. **Constant error carousel.** For constant one-step error flow through a self-connected unit `j`,
   need `f_j'(net_j) w_jj = 1`. Solving the ODE forces `f_j` linear and (with `w_jj = 1`) the
   activation to persist: `y_j(t+1) = y_j(t)`. Error riding this self-loop is multiplied by exactly
   1 per step — no decay, no blow-up, for any lag.

2. **Multiplicative gates resolve the read/write conflicts.** Once the carousel connects to the
   net, a single incoming weight must both *store* relevant inputs and *protect* the memory from
   irrelevant ones (input weight conflict); a single outgoing weight must both *expose* the content
   when needed and *protect* downstream units otherwise (output weight conflict). A static weight
   cannot do both. The fix is a context-sensitive *multiplicative* control — a sigmoid gate unit in
   `[0,1]`, because only multiplication by ~0 can block a signal completely (an additive bias only
   shifts it). Input gate = when to write; output gate = when to read.

3. **Truncated backprop preserves the carousel.** Error arriving at the cell input or either gate's
   net input is not propagated further back in time (set those cross-time derivatives to 0). This
   forbids backprop from routing error around the carousel and reintroducing the vanishing product;
   the only multi-step error path is the state's CEC, with `∂s(t)/∂s(t-1) = 1` verified directly.
   It also makes the update `O(W)` per step and local in space and time.

4. **Forget gate (the modern cell).** The original two-gate state `s(t) = s(t-1) + i·g` can only grow; on
   continual/unsegmented streams it grows without bound until `tanh` saturates and the cell dies.
   Replace the fixed self-weight 1 with a learned forget gate `f_t ∈ [0,1]` multiplying the carried
   state. `f_t = 1` recovers the exact CEC; `f_t = 0` resets the cell; the cell learns its own reset
   points.

## Final cell (modern LSTM, the form `nn.LSTM` computes — no peephole connections)

Per step, with input `x_t`, previous hidden `h_{t-1}`, previous state `c_{t-1}`, sigmoid `σ`,
`tanh` for cell input/output squashing, `⊙` elementwise:

```
i_t = σ(W_ix x_t + W_ih h_{t-1} + b_i)        # input gate   (when to write)
f_t = σ(W_fx x_t + W_fh h_{t-1} + b_f)        # forget gate  (when to reset)
g_t = tanh(W_gx x_t + W_gh h_{t-1} + b_g)      # candidate cell input
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t               # cell state = gated CEC
o_t = σ(W_ox x_t + W_oh h_{t-1} + b_o)        # output gate  (when to read)
h_t = o_t ⊙ tanh(c_t)                          # exposed cell output / hidden state
```

Backward pass (state error carries the carousel):

```
δ_o^t  = σ'(a_o^t) ⊙ tanh(c_t) ⊙ ε_h^t
ε_s^t  = o_t ⊙ tanh'(c_t) ⊙ ε_h^t + f_{t+1} ⊙ ε_s^{t+1}     # CEC: unit gain back when f_{t+1}=1
δ_g^t  = i_t ⊙ tanh'(a_g^t) ⊙ ε_s^t
δ_f^t  = σ'(a_f^t) ⊙ c_{t-1} ⊙ ε_s^t
δ_i^t  = σ'(a_i^t) ⊙ g_t ⊙ ε_s^t
```

where `ε_h^t = ∂L/∂h_t` is collected from the read-out and the next step's gate inputs. The term
`f_{t+1} ⊙ ε_s^{t+1}` is the constant error carousel in gradient form: when the forget gate is
open the state error flows back across the lag undamped.

Stack `num_layers` of these (each layer's hidden sequence is the next layer's input sequence) for
a deeper representation; apply dropout between stacked layers.

## Relation to prior methods

- **vs. BPTT/RTRL:** the original truncated update has `O(W)` per-step cost (RTRL is `O(W^2)`),
  and the CEC makes the long-range state error flow constant instead of exponentially decaying.
  The qlib/PyTorch implementation below trains the same gated cell with autograd through a fixed
  input window.
- **vs. time constants / time delays (Mozer; Lang et al.; Plate):** the gates *learn* when to hold
  and release rather than using a tuned, fixed memory horizon, and the multiplicative input gate
  stops new input from perturbing held memory.
- **vs. hierarchical chunkers (Schmidhuber):** the carousel works even on noisy, incompressible
  streams with no local predictability to exploit.

## Working code (sequence-to-one regression over a fixed feature window)

Grounded in qlib's LSTM model: `nn.LSTM` over a `[N, time, features]` view of an Alpha360-style
window (6 base ratios × 60 days), read out the final hidden state to one score, train with masked
MSE and Adam, value-clip gradients (the exploding side of the error-flow analysis), early-stop on
a validation score. The cell uses sigmoid gates, tanh candidate/output squashing, and no peephole
connections.

```python
import copy
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model


class LSTMModel(nn.Module):
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
        # x: [N, F*T] flat -> recover the time axis the recurrence needs
        x = x.reshape(len(x), self.d_feat, -1)   # [N, 6, 60]
        x = x.permute(0, 2, 1)                    # [N, 60, 6]
        out, _ = self.rnn(x)                      # [N, 60, hidden]
        return self.fc_out(out[:, -1, :]).squeeze()   # final hidden -> one score


class LSTM(Model):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=800,
        early_stop=20,
        loss="mse",
        GPU=0,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.lstm_model = LSTMModel(
            d_feat=self.d_feat, hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout,
        ).to(self.device)
        self.train_optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.lr)
        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        return torch.mean((pred - label) ** 2)

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
        x_values = x_train.values
        y_values = np.squeeze(y_train.values)
        self.lstm_model.train()
        indices = np.arange(len(x_values))
        np.random.shuffle(indices)
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
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
        scores, losses = [], []
        indices = np.arange(len(x_values))
        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)
            pred = self.lstm_model(feature)
            losses.append(self.loss_fn(pred, label).item())
            scores.append(self.metric_fn(pred, label).item())
        return np.mean(losses), np.mean(scores)

    def fit(self, dataset: DatasetH):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"], col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]
        stop_steps, best_score, best_epoch, best_param = 0, -np.inf, 0, None
        self.fitted = True
        for step in range(self.n_epochs):
            self.train_epoch(x_train, y_train)
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
            print("Epoch%d: train %.6f, valid %.6f" % (step, train_score, val_score))
            if val_score > best_score:
                best_score, stop_steps, best_epoch = val_score, 0, step
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

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.lstm_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []
        for begin in range(sample_num)[:: self.batch_size]:
            end = sample_num if sample_num - begin < self.batch_size else begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.lstm_model(x_batch).detach().cpu().numpy()
            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=index)
```
