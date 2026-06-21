We are given a multivariate time series with $N$ correlated variables — loop detectors on a road network, or clients on a meter grid — and we must produce the standard direct multi-step forecast $\hat Y \in \mathbb{R}^{F \times N}$ for the next $F$ steps from the last $P$ observed steps, using only past values and the calendar covariates available at the forecast origin. The dominant answer to this problem is the Spatial-Temporal Graph Neural Network: a graph module (a fixed road graph, a learned adjacency, or a data-adaptive dependency matrix) bolted to a recurrent or convolutional temporal module, with message passing at every layer. DCRNN runs bidirectional diffusion walks inside a recurrent encoder-decoder; STGCN sandwiches temporal convolutions around graph convolutions; Graph WaveNet learns adjacency from node embeddings and pairs it with dilated convolutions; AGCRN learns node-specific recurrent parameters and an adaptive graph. These models are strong, but they repeatedly pay for graph construction, message passing, and recurrent or convolutional sequence processing, and the comparisons have grown far more elaborate than the accuracy gaps they chase. The right question is not whether spatial and temporal information matter — they obviously do — but whether this much machinery is the only way to put those distinctions in front of a regressor.

To find the load-bearing ingredient, take the simplest channel-shared direct regressor and look at what it is forced to do. It maps a recent value trace to a future trace through one shared set of weights. If sensor $i$ and sensor $j$ happen to have nearly the same last $P$ values, the regressor sees nearly the same input and is compelled to emit nearly the same prediction — yet their futures can genuinely differ because they are different variables sitting in different parts of the system. This is not a capacity shortfall that universal approximation could cure; it is a representation collision. The identical failure recurs along the time axis: a twelve-step trace during morning buildup and a similar trace during evening recovery demand different continuations, but a model that sees only the trace has no clean way to know which periodic phase it occupies. Seen this way, what a graph convolution really supplies is not topology for its own sake but a context that depends on a variable's place in the system, and what a sequence module supplies is temporal context. The load-bearing property is distinguishability: the regressor must know which variable a sample is from and which periodic time state it is anchored at. If I supply that directly, I can test whether the heavy machinery was one route to this information rather than the necessary mechanism.

I propose STID, Spatial-Temporal IDentity: a graph-free direct forecaster that injects distinguishability through learned identity embeddings instead of through message passing. The most direct spatial marker is a trainable table with one vector per variable, $E \in \mathbb{R}^{N \times D}$; when I forecast variable $i$ I concatenate its row $E_i$ onto the representation, so two identical value traces from two different variables stop being identical inputs downstream. This is deliberately less structured than a graph — it never states who is upstream or downstream — but gradient descent can arrange the coordinates so that variables with similar forecasting behavior land near each other, and that is the only thing the regressor needed. The temporal marker is just as direct because the preprocessing already provides periodic covariates: a five-minute dataset has $N_d = 288$ slots per day and $N_w = 7$ days per week. I keep two more trainable tables, $T^{\mathrm{TiD}} \in \mathbb{R}^{N_d \times D}$ and $T^{\mathrm{DiW}} \in \mathbb{R}^{7 \times D}$, look up the current time-of-day and day-of-week rows at the forecast origin, and concatenate them to every node's representation. A table beats feeding the raw normalized scalar because a continuous coordinate would force the model to discover the periodic regimes through one smooth dimension, whereas a table lets slot 84 and slot 228 carry completely unrelated signatures whenever the data demand it.

The value trace still needs an embedding, supplied by a single shared fully connected layer. For node $i$ with past values $X^i_{t-P:t} \in \mathbb{R}^P$,
$$H^i_t = \mathrm{FC}_{\text{embedding}}\!\left(X^i_{t-P:t}\right), \qquad H^i_t \in \mathbb{R}^D.$$
The sharing matters: I am not handing every node its own forecaster, I am giving one forecaster a way to tell otherwise-colliding samples apart. Concatenating the four pieces gives the augmented representation
$$Z^i_t = H^i_t \;\Vert\; E_i \;\Vert\; T^{\mathrm{TiD}}_t \;\Vert\; T^{\mathrm{DiW}}_t.$$
With all three identity tables enabled and equal width $D$, each term contributes $D$ and $Z^i_t \in \mathbb{R}^{4D}$; in the implementation the width is left general, $\text{embed\_dim} + \text{node\_dim}\cdot\mathbb{1}_{\text{node}} + \text{temp\_dim\_tid}\cdot\mathbb{1}_{\text{tid}} + \text{temp\_dim\_diw}\cdot\mathbb{1}_{\text{diw}}$, so any branch can be ablated and the widths need not match — but the clean derivation is exactly the four-way concatenation. The encoder is then a plain residual MLP that preserves the width across $L$ layers,
$$\left(Z^i_t\right)^{l+1} = \mathrm{FC}^l_2\!\left(\sigma\!\left(\mathrm{FC}^l_1\!\left(\left(Z^i_t\right)^l\right)\right)\right) + \left(Z^i_t\right)^l,$$
with $\sigma = \mathrm{ReLU}$ and no graph convolution, attention, or recurrence concealed inside it. The residual connection is purely an optimization choice — each block learns a correction at fixed dimension. A final linear head regresses the whole horizon at once,
$$\hat Y^i_{t:t+F} = \mathrm{FC}_{\text{regression}}\!\left(\left(Z^i_t\right)^L\right) \in \mathbb{R}^F,$$
and the objective is the mean absolute error averaged over nodes and steps, $\frac{1}{NF}\sum_i\sum_j \lvert \hat Y^i_j - Y^i_j \rvert$, which a benchmark harness may evaluate masked over missing values but with the same direction. There is no diffusion normalization, no Laplacian sign convention, and no recurrence state to reconcile; every constant is a table size, a width, or a horizon length.

Two pieces of bookkeeping are load-bearing because an off-by-one would silently change the method. The temporal indices are anchored at the last observed step, so I read time-of-day from channel $1$ and day-of-week from channel $2$ of the final history step. Because the generated features are $k/\text{steps\_per\_day}$ and $d/7$ for integers $k, d$ and never reach $1.0$, multiplying by the table size and casting to integer gives $\mathrm{floor}(\text{tod}\cdot N_d)\in\{0,\dots,N_d-1\}$ and $\mathrm{floor}(\text{dow}\cdot 7)\in\{0,\dots,6\}$ — valid indices with no clamping. I keep these as per-node $[B, N]$ index tensors even though the preprocessing tiles the same time value across nodes, so the lookup returns $[B, N, D]$. For the history embedding I select the first $\text{input\_dim}$ forward channels, flatten each node's $P$-step history to length $P\cdot\text{input\_dim}$, and apply a $1\times1$ convolution to $\text{embed\_dim}$ — a shared linear map per node, written as a convolution for tensor convenience. The claim is falsifiable in exactly the right way: remove the node table and similar histories from different variables collide again; remove the temporal tables and similar histories at different periodic phases collide again. If the full model competes with the graph stack, the lesson is that a simple regressor suffices once its samples are made distinguishable — topology is useful context, but not the only way to supply it. For the short-paper traffic experiments $P = F = 12$, $D = 32$, $\text{num\_layer} = 3$, $N_d = 288$, $N_w = 7$; for Electricity $P = 168$, $F = 12$, and the time table shrinks to $24$ for hourly data.

```python
import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        return hidden + input_data


class STID(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)

        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )

        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )

        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        # history_data: [B, input_len, N, C]
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            ]
        else:
            day_in_week_emb = None

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            node_emb.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)
        return prediction
```
