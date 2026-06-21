The problem I am solving is multivariate time series forecasting: from a lookback window $X \in \mathbb{R}^{C \times L}$ of $C$ channels over $L$ past steps, predict the next $H$ steps $Y \in \mathbb{R}^{C \times H}$. The datasets that matter — traffic networks, power grids — have $C$ in the hundreds, pushing toward nine hundred on the PEMS sets, and the data drifts: each channel's level and scale wander across the day and the week, and individual channels go anomalous when a sensor sticks or a meter drops out. Two facts about this setting pull in opposite directions. Channels are correlated, and the correlation is genuine signal — adjacent road sensors rise and fall together, so one channel's recent history is informative about another, and a model that refuses to look across channels throws that away. But the field learned the hard way that models which mix channels are fragile: the per-timestamp-token Transformers, which jam all $C$ values at one time step into a token and attend along time, were beaten by a one-layer linear model. The reading is non-stationarity — when you fit cross-channel structure you fit the joint distribution of the channels, and that joint distribution is exactly what shifts between train and test, so you overfit to correlations that do not hold at test time, and higher capacity over the channels buys *less* robustness, not more.

The field's escape was channel independence — process each channel alone with shared weights and never let channels talk. That is robust, cheap, and strong, but it is strictly capped, because by construction it cannot use the correlation that is plainly present on dense sensor grids. The direct precursors that tried to recover correlation, the inverted-token attention models, make each channel's whole series into one token and run self-attention across the $C$ channel tokens. This keeps each channel as a robust unit and recovers cross-channel correlation, but it costs $O(C^2)$ in time and memory, which is the binding constraint when $C$ is in the hundreds, and — the deeper problem — its entire content is the *per-pair* attention weight: channel $i$ decides how much to take from channel $j$ through an extracted $i$–$j$ correlation, the very quantity the robustness literature shows is untrustworthy under non-stationarity and corrupted when $j$ goes anomalous. So attention is $O(C^2)$ *and* leans on the one thing I have the most reason to distrust. The honest statement of the problem is to put *some* cross-channel information back, without re-importing the fragility channel independence was invented to escape, at a cost that survives $C$ in the hundreds — correlation *and* robustness *and* linear scaling, together.

I propose SOFTS, the Series-cOre Fused Time Series forecaster, built around a centralized cross-channel module I call STAR, for STar Aggregate-Redistribute. The first move is representational: make a token a *whole channel* rather than a timestamp. Embed one channel's $L$ past values through a single linear map $L \to d$ into one $d$-dim token, giving $C$ tokens, one per channel. With the per-timestamp token, channel mixing happens structurally inside every token before anything is learned; with the channel-as-token view, each channel is a self-contained unit and any cross-channel operation becomes an explicit, separable step over the set of tokens. Before embedding I neutralize the drift with reversible instance normalization — subtract each channel's mean over the lookback, divide by its standard deviation — and undo that transform on the prediction at the end, so the tokens carry shape rather than the slowly drifting level and scale that cause most of the train/test mismatch. The skeleton is then forced everywhere except the middle: RevIN, embed each channel to a token, $N$ layers that let the $C$ tokens exchange information, a linear head $d \to H$ per token, denormalize. The interaction in the middle is the whole game.

The key idea is to replace all-pairs attention with a *star-shaped* topology instead of a mesh. Rather than $C^2$ pairwise conversations each weighted by a fragile correlation, route everything through a hub: every channel contributes to one shared summary — the *core* — and every channel reads back from that one summary. This is $O(C)$ links instead of $O(C^2)$, and it is not merely a cost trick. A single summary aggregated over *all* channels is a *statistic*, and a statistic over many channels is far more stable than any one channel or any one pair; averaging over channels is the robust operation, the opposite of trusting a brittle per-pair weight. If a sensor goes anomalous it is one contributor out of hundreds to the core, drowned out, rather than a correlation partner some other channel ingests at full weight. So the hub fixes both founding problems at once: linear cost, and an interaction that depends on a robust aggregate. The shape is therefore to aggregate the $C$ channel tokens into one global core, redistribute that core back to every channel, and let each channel fuse the global context with its own local token, with a residual so the layer refines each channel rather than overwriting it.

Defining "aggregate the $C$ tokens into one vector" correctly is where the robustness argument earns its keep. The core summarizes an unordered *set* of channel tokens, so its value must not depend on the arbitrary index assigned to each channel — relabeling sensor 5 as sensor 200 must not change the summary of the whole network. The set-function theory offers exactly two candidate shapes. The general continuous form $\rho(\sum_m \lambda_m \, \phi(x_m))$ applies an inner map $\phi$ to each element but sums with a *per-coordinate* weight $\lambda_m$, making the output depend on which slot each element sits in — permutation-*variant*. The DeepSets form $\rho(\sum_x \phi(x))$ applies the same $\phi$ to every element, sums with *equal* weight, and approximates any continuous permutation-*invariant* set function. I want the invariant form. A per-channel weight $\lambda_m$ is a parameter glued to channel identity — it spends capacity learning "channel 5 specifically tends to look like this" — and the channel index is meaningless under distribution shift, exactly the relationship that has drifted by test time. Forcing the model to summarize channels by *what they contain* rather than *which slot they occupy* is the robust thing channel independence taught us to value. So the core takes the form $\rho(\sum \phi(x))$, with $\phi = \text{MLP}_1$ a shared two-layer map with a GELU, $\mathbb{R}^d \to \mathbb{R}^{d'}$, applied to every channel token alike.

Two prunings follow. The outer $\rho$ after pooling is redundant: right after pooling I concatenate the core onto each token and push the result through a fusion MLP, which can absorb any outer nonlinearity, so I drop $\rho$ and let the fusion MLP do the redistribution-side nonlinearity. And the $\sum$ in the set form is, up to scale, mean pooling — but mean is not obviously right. Mean pooling is maximally stable yet washes a strong, informative channel into the crowd; max pooling preserves the salient signal but bets each summary coordinate on a single channel and inherits its garbage if that channel is the anomalous one. On different datasets each wins; neither dominates. So I want a pooling that interpolates between "average everyone" and "take the strongest" and that also regularizes, because the core is a single shared object every channel reads from — if it overfits, everyone overfits together. Stochastic pooling over the channel axis does exactly this. For activations $A \in \mathbb{R}^{C \times d'}$ (channel by core-feature), form a distribution over channels per feature $j$ with a softmax *across channels*,

$$p_{ij} = \frac{e^{A_{ij}}}{\sum_{k=1}^{C} e^{A_{kj}}}.$$

During training, for each feature $j$ sample a channel $c \sim \text{Multinomial}(p_{\cdot j})$ and set the core coordinate to that channel's value, $o_j = A_{cj}$; at test time, take the expectation under the same distribution,

$$o_j = \sum_{i=1}^{C} p_{ij}\, A_{ij}.$$

The sampling probability is monotone in the activation, so high-activation channels are picked often (the max tendency) while low-activation channels are still occasionally picked (the mean tendency) — it lives between max and mean by construction. The training-time sampling injects noise into the shared core every step, regularizing it instead of letting it lock onto whichever channels were loud on the training set, and the test-time expectation is a clean deterministic magnitude-weighted average — the mean of the very distribution sampled from in training, so train and test are consistent and it is precisely the soft aggregate over all channels the robustness argument wanted, with weight concentrated on the salient channels but spread so no single channel dominates and no dead channel hijacks the core. Softmax over the channel axis is the right axis: I normalize across the things I am pooling, the channels, for each feature independently. In batched code the indexing must be careful — $A$ is $[B, C, d']$, softmaxed on $C$, permuted to $[B, d', C]$ and flattened to $[B \cdot d', C]$ so $\texttt{torch.multinomial}$ draws one channel per $(\text{batch}, \text{feature})$ pair, reshaped to $[B, 1, d']$ to line up with $A$'s channel dimension, then used as the index for $\texttt{gather}$, giving one core $[B, 1, d']$ that is repeated over $C$ channels for redistribution.

One STAR layer then takes the channel tokens $S \in \mathbb{R}^{C \times d}$, computes the core $o = \text{StochPool}(\text{MLP}_1(S))$, repeats $o$ to all $C$ channels and concatenates it onto each token to get $F \in \mathbb{R}^{C \times (d+d')}$, and fuses with a residual, $S' = \text{MLP}_2(F) + S$, where $\text{MLP}_2 : \mathbb{R}^{d+d'} \to \mathbb{R}^d$ is again a two-layer GELU map. Around this interaction sublayer I wrap ordinary transformer-block hygiene — a position-wise feed-forward network applied per channel token with its own residual, and post-LayerNorm after each sublayer to keep the stack stable as layers are added — and stack $N$ of them. The cost is linear throughout: $\text{MLP}_1$ is $O(Cd^2)$, the channel-axis softmax is $O(Cd)$, the fusion $\text{MLP}_2$ is $O(Cd^2)$, the head $O(CdH)$, and no $C \times C$ matrix is ever formed — the only place all channels meet is the pooled core, $O(C)$ to build and $O(C)$ to broadcast. With $d$ a fixed model width the interaction is $O(C)$ in channels against attention's $O(C^2)$, so the full encoding is $O(CLd + Cd^2 + CdH)$, linear in $C$, $L$, and $H$. Wrapping it in the fixed skeleton — RevIN-normalize, embed each channel's lookback $L \to d$, run the $N$ STAR layers, final LayerNorm, linear head $d \to H$ per channel, transpose back, RevIN-denormalize, train with MSE under Adam — gives correlation, robustness, and linear scaling in one architecture. In the implementation the core and fusion MLPs are spelled out as their constituent linear layers ($\texttt{gen1..gen4}$ with GELU between), with $\text{MLP}_1 = \texttt{gen2} \circ \text{GELU} \circ \texttt{gen1}$ and $\text{MLP}_2 = \texttt{gen4} \circ \text{GELU} \circ \texttt{gen3}$, exactly the two-layer-with-GELU shape argued for.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAR(nn.Module):
    """STar Aggregate-Redistribute: O(C) centralized cross-channel interaction."""

    def __init__(self, d_series, d_core):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)            # MLP_1 : R^d -> R^{d'}
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)   # MLP_2 : R^{d+d'} -> R^d
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, x, *args, **kwargs):                    # x: [B, C, d]
        B, C, d = x.shape
        a = self.gen2(F.gelu(self.gen1(x)))                  # A: [B, C, d']

        if self.training:                                    # sample one channel per feature
            ratio = F.softmax(a, dim=1).permute(0, 2, 1).reshape(-1, C)   # [B*d', C]
            idx = torch.multinomial(ratio, 1).view(B, -1, 1).permute(0, 2, 1)  # [B, 1, d']
            core = torch.gather(a, 1, idx).repeat(1, C, 1)   # o_j = A_{cj}
        else:                                                # expectation o_j = sum_i p_ij A_ij
            p = F.softmax(a, dim=1)
            core = (a * p).sum(dim=1, keepdim=True).repeat(1, C, 1)

        f = torch.cat([x, core], dim=-1)                     # redistribute + fuse
        return self.gen4(F.gelu(self.gen3(f))), None


class EncoderLayer(nn.Module):
    def __init__(self, interaction, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.interaction = interaction
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, _ = self.interaction(x, x, x)                 # STAR ignores extra args
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class SOFTS(nn.Module):
    def __init__(self, input_len, output_len, num_features,
                 hidden_size=512, core_size=128, num_layers=2, dropout=0.05):
        super().__init__()
        self.eps = 1e-5
        self.embed = nn.Linear(input_len, hidden_size)       # L -> d, per channel
        self.layers = nn.ModuleList([
            EncoderLayer(STAR(hidden_size, core_size), hidden_size, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_len)       # d -> H, per channel

    def forward(self, inputs):                               # inputs: [B, L, C]
        mean = inputs.mean(dim=1, keepdim=True).detach()     # RevIN normalize
        x = inputs - mean
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x = x / stdev

        N = x.size(-1)
        h = self.embed(x.transpose(1, 2))                    # [B, C, d]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        pred = self.head(h).transpose(1, 2)[:, :, :N]        # [B, H, C]
        return pred * stdev + mean                           # RevIN denormalize
```
