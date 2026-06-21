The reason to use a recurrent network at all is that its feedback connections can hold information across time: the activation pattern now can encode an input seen many steps ago, and the network can in principle learn to use it. In practice this only half works. When the relevant input and the moment its information is needed are a few steps apart, the exact-gradient training rules — Back-Propagation Through Time and Real-Time Recurrent Learning — train fine. But as the lag stretches to tens, hundreds, or a thousand steps, training stalls or crawls so slowly it never arrives. The cause is not mysterious once you follow a single error signal backward through time. An output unit $k$ at time $t$ has error $\delta_k(t)=f_k'(\mathrm{net}_k(t))(d_k(t)-y_k(t))$, and a non-output unit collects what flows back from the next step, $\delta_j(t)=f_j'(\mathrm{net}_j(t))\sum_i w_{ij}\,\delta_i(t+1)$. Tracing the error that lands on unit $u$ at time $t$ back $q$ steps to a unit $v$ and expanding the recursion telescopes into a sum over paths,

$$\frac{\partial \delta_v(t-q)}{\partial \delta_u(t)}=\sum_{l_1=1}^{n}\cdots\sum_{l_{q-1}=1}^{n}\ \prod_{m=1}^{q} f'_{l_m}(\mathrm{net}_{l_m}(t-m))\,w_{l_m l_{m-1}},$$

with $l_0=u$, $l_q=v$. Each of the $n^{q-1}$ paths contributes a product of $q$ factors, each of the form $f'(\mathrm{net})\cdot w$, and the lag $q$ sits in the exponent. If every factor on a path exceeds $1$ in magnitude the product explodes and arriving errors oscillate; if every factor is below $1$ — the ordinary case — the product vanishes geometrically. For the logistic sigmoid $\max f'=0.25$, so a single factor $|f'(\mathrm{net})\,w|\le 0.25\,|w|<1$ whenever $|w|<4$; with any reasonable weight the error to long-past inputs decays exponentially. Tightening the bound with the matrix norm $\lVert W\rVert_A:=\max_r\sum_s|w_{rs}|$ gives $|\partial\delta_v(t-q)/\partial\delta_u(t)|\le n\,(f'_{\max}\lVert W\rVert_A)^q$, and if $|w_{ij}|\le w_{\max}<4/n$ then with $\tau:=n\,w_{\max}/4<1$ the whole thing is bounded by $n\,\tau^q$ — exponential decay in the lag. The usual knobs do not touch the exponent: bigger weights drive the unit into saturation where $f'$ collapses faster than $w$ grows, and a bigger learning rate scales long-range and short-range error by the same factor, leaving the starving ratio unchanged. The prior partial fixes do not close the gap either — time-delay nets and time constants cover only a fixed short window and need hand-tuning, the additive "old activation plus current input" form lets new input keep perturbing what is stored, and hierarchical sequence chunkers bridge long lags only when the sequence is locally predictable, degrading exactly as inputs get noisy and incompressible. What is wanted is an architecture and matching learning rule that bridges minimal time lags well past $1000$ steps, survives noisy incompressible inputs, keeps short-lag ability, and costs $O(1)$ per weight per step.

I propose LSTM. The design starts from inverting the failure: if a product of $q$ factors is the disease, the cure is a channel whose per-step factor is exactly $1$, for any $q$. In the simplest setting — a single unit $j$ with one self-connection $w_{jj}$ — the one-step backward multiplier is $f_j'(\mathrm{net}_j(t))\,w_{jj}$, and demanding it equal $1$ is a differential equation $f_j'(\mathrm{net})=1/w_{jj}$, a constant, whose solution forces $f_j$ to be linear. Taking the cleanest choice, the identity with $w_{jj}=1$, the forward dynamics become $y_j(t+1)=y_j(t)$: the activation persists unchanged and the backpropagated error riding it is multiplied by exactly $1$ each step. This is the constant error carousel (CEC) — error circulates the self-loop at unit gain, surviving an arbitrarily long lag. But a unit cannot be an island; the moment it is wired to the net two conflicts appear that a single static weight cannot resolve. A lone incoming weight $w_{ji}$ must let the relevant input *in* when it arrives yet *block* irrelevant inputs at every other moment to protect what is stored — the input weight conflict. A lone outgoing weight $w_{kj}$ must let the stored content *out* when it is needed yet *protect* downstream units from it otherwise — the output weight conflict. One number cannot do two opposed jobs, and both conflicts worsen exactly in the long-lag regime where memory must be protected longest. The resolution is to replace static control with a context-sensitive *multiplicative* gate: a sigmoid unit in $[0,1]$ reading the rest of the network, multiplying the signal it controls. Multiplication is essential rather than incidental — a control of $0$ zeroes a signal completely (perfect protection), a control of $1$ passes it, and anything between is a soft valve; an additive bias could only shift a signal, never block it. The two conflicts are genuinely distinct, so they need two gates: an input gate $y^{\mathrm{in}_j}=f_{\mathrm{in}_j}(\mathrm{net}_{\mathrm{in}_j})$ deciding when to write, and an output gate $y^{\mathrm{out}_j}=f_{\mathrm{out}_j}(\mathrm{net}_{\mathrm{out}_j})$ deciding when to read.

The cell wraps these around the carousel. The new input is squashed through $g$, multiplied by the input gate, and accumulated onto the linear state, $s_{c_j}(t)=s_{c_j}(t-1)+y^{\mathrm{in}_j}(t)\,g(\mathrm{net}_{c_j}(t))$, where the $s(t-1)$ term carries an implicit weight of exactly $1$ — that is the CEC, untouched. The cell exposes its state through a squashing $h$ scaled by the output gate, $y^{c_j}(t)=y^{\mathrm{out}_j}(t)\,h(s_{c_j}(t))$. The state is squashed through $h$ rather than emitted raw because $s$ is a linear accumulator that can grow large, and the cell's output should live on the same bounded scale ($[-1,1]$) as the ordinary units it feeds; $g$ likewise (range $[-2,2]$) keeps a single huge input from slamming the state, the wider $g$/narrower $h$ asymmetry letting the cell be driven over a useful range while its exposed output stays tight. The remaining danger is subtle: every new connection is a route by which error can leak. If standard backprop ran free, error leaving through the output gate could loop back into the cell through the input gate at an earlier step, and that loop is again a product of $f'\cdot w$ — the vanishing/exploding factor returns. So the gradient is *truncated*: error arriving at the cell input or either gate's net input is not propagated further back in time, $\partial \mathrm{net}_{\mathrm{in}_j}(t)/\partial y^u(t-1)\approx0$ and likewise for the output gate and cell input. The only multi-step error path left alive is the one riding the state $s$ along the CEC. Checking it directly: with the back-in-time terms killed, the only surviving piece of $\partial s(t)/\partial s(t-1)$ is the explicit carried term with derivative $1$, so $\partial\delta_{s_c}(t)/\partial\delta_{s_c}(t+1)\approx 1$ — constant flow, realized inside a real gated cell. Error is scaled once on entry (by the output gate and $h'$) and once on exit (by the input gate and $g'$), and by $1$ across the lag between. Truncation also makes the algorithm cheap: only the cell and input gate's forward sensitivities $\partial s(t)/\partial w$ persist across steps, giving $O(W)$ per step — far below RTRL's $O(W^2)$ — and locality in space and time.

One failure remains. The two-gate state can only ever add to itself; with the self-loop pinned at $1$ it never decays, so on a continual stream that is never reset, $s$ grows without bound until $h$ saturates, $h'$ collapses, and the cell dies as a pegged accumulator. The cell needs to learn *when to forget*. The fixed self-weight is what blocks this, so make it adaptive in the same context-sensitive multiplicative way: introduce a third gate, the forget gate $y^{\varphi_j}\in[0,1]$, multiplying the carried state,

$$s_{c_j}(t)=y^{\varphi_j}(t)\cdot s_{c_j}(t-1)+y^{\mathrm{in}_j}(t)\cdot g(\mathrm{net}_{c_j}(t)).$$

At $y^{\varphi_j}=1$ this is the original CEC exactly, so the long-lag highway is not lost; at $y^{\varphi_j}=0$ the cell wipes its state; in between it decays at a learned rate, and because the gate is learned from context the cell discovers its own reset points. That gives the modern memory cell — input, forget, output gates around a protected linear state — which in the compact vectorized form I would actually compute, with $\sigma$ the logistic gate nonlinearity and $\tanh$ for the candidate input $g$ and output squashing $h$, is

$$
\begin{aligned}
i_t &= \sigma(W_{ix}x_t + W_{ih}h_{t-1} + b_i), &\quad f_t &= \sigma(W_{fx}x_t + W_{fh}h_{t-1} + b_f),\\
g_t &= \tanh(W_{gx}x_t + W_{gh}h_{t-1} + b_g), &\quad o_t &= \sigma(W_{ox}x_t + W_{oh}h_{t-1} + b_o),\\
c_t &= f_t\odot c_{t-1} + i_t\odot g_t, &\quad h_t &= o_t\odot\tanh(c_t),
\end{aligned}
$$

each gate a sigmoid unit reading both the new input and the recurrent feedback, $c_t$ the protected linear memory, $h_t$ what the rest of the net sees. The backward pass makes the carousel claim explicit in this form: with $\varepsilon_h^t=\partial L/\partial h_t$ arriving at the exposed output and $\varepsilon_s^t=\partial L/\partial c_t$ on the state, the output gate gets $\delta_o^t=\sigma'(a_o^t)\odot\tanh(c_t)\odot\varepsilon_h^t$, and the state error collects the part returning through $h_t$ plus the part carried from the next step's state — which, because $c_{t+1}=f_{t+1}\odot c_t+\dots$, arrives multiplied by the forget gate,

$$\varepsilon_s^t = o_t\odot\tanh'(c_t)\odot\varepsilon_h^t + f_{t+1}\odot\varepsilon_s^{t+1}.$$

That $f_{t+1}\odot\varepsilon_s^{t+1}$ term is the constant error carousel in gradient form: when the forget gate is open the state error flows back across the lag undamped, and when it is closed the cell drops the gradient too because it has chosen to forget. The remaining updates fall out of differentiating each multiplication: $\delta_g^t=i_t\odot\tanh'(a_g^t)\odot\varepsilon_s^t$, $\delta_f^t=\sigma'(a_f^t)\odot c_{t-1}\odot\varepsilon_s^t$, $\delta_i^t=\sigma'(a_i^t)\odot g_t\odot\varepsilon_s^t$, with the recurrent contribution to $h_{t-1}$ being $W_{ih}^{\top}\delta_i^t+W_{fh}^{\top}\delta_f^t+W_{gh}^{\top}\delta_g^t+W_{oh}^{\top}\delta_o^t$.

I put this to work on a concrete sequence-to-one regression: predict a forward return for each instrument from a window of six base price/volume ratios over sixty trading days. The flat feature vector arrives as $[N,360]$; I reshape it to $[N,6,60]$ and transpose to $[N,60,6]$ to recover the time axis, run a stack of LSTM layers (each layer's hidden sequence feeding the next as input, with dropout between layers), and read out the *last* step's hidden state — the cell's summary of the whole sixty-day window, reachable because the protected memory kept a feature from sixty days ago from having its gradient vanish — through a linear layer to one score. Training is masked mean-squared error over the finite targets with Adam at $1\mathrm{e}{-3}$, gradients value-clipped at $3.0$ because the *forward* analysis warned of the exploding side too — a bad batch can still spike a gradient, and value-clipping caps updates without changing the cell equations — and early stopping on the validation score with the best parameters restored.

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
