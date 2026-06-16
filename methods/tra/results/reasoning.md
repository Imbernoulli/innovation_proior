Let me start from the thing that actually fails. I train a sequence model on stock features to predict next-period return ranks, the usual way: collect `{(x_i, y_i)}`, encode each lookback window into a latent vector with an attentive LSTM, push it through a linear head, regress to the label by mean-squared error, and call it empirical risk minimization. Buried in that pipeline is an assumption I never questioned — that every `(x_i, y_i)` is an i.i.d. draw from one fixed joint distribution `P`, so there is one relation `p(y|x)` to learn. And I keep hitting a ceiling. Let me figure out whether the ceiling is the model capacity or the assumption.

Here is what makes me suspect the assumption. The market is the sum of a lot of people running different strategies, and two of the most documented cross-sectional effects point in opposite directions at once. Momentum: stocks that went up keep going up. Reversal / mean-reversion: stocks that went down rebound. Both are real, both are in the same market, so the sign of "past return → future return" is not fixed across samples. I can make this brutally concrete without any new experiment — it's a known property of the data. Take three features, size rank, value rank, twelve-month momentum rank, and fit a plain linear model of next-month return rank, but fit it on different years. The momentum coefficient comes out negative when fit on 2009 — that year, high past return predicted low future return, pure reversal — and positive when fit on 2013 — momentum. A single coefficient vector cannot be both negative and positive. So it is not that my model is too small; it is that I am asking *one* function with *one* parameter vector to hold two relations that contradict each other, and the best it can do is average them, which fits neither. And these regimes rotate over time — size leads in one stretch, momentum in another — so it is not a one-off either. The i.i.d. single-`P` assumption is the wall. The data is really `P = Σ_k ν_k P_k`, a mixture of patterns, each with its own `p_k(y|x)` and its own share `ν_k`, and which one is active drifts.

So what would actually solve it. If I knew, for every sample, which pattern `k` it came from, this is easy: it is just multi-domain learning — split the data by domain identifier, fit a model per domain, done. But I have no pattern identifiers. They are latent. Worse than latent: even if I cheated and used the *labels* to figure out which pattern each training sample best fits, that trick dies at test time, because at test there are no labels — and the whole point of a predictor is the test period. So the real constraint is sharp: I need a thing that (a) holds several distinct relations at once, (b) decides per-sample which relation applies, and (c) can make that decision from signals available without the current label: the sample's latent representation and whatever past prediction-error state has already been cached. Property (c) is the one that kills the naive solutions, and I should keep it front of mind.

Let me look at the tools for "one model, several behaviors." There is the conditional-computation / mixture-of-experts idea: keep `K` expert sub-networks and a trainable gating network `G(x)` that, per input, picks a sparse combination of experts, and train it all jointly by backprop. That is structurally exactly what I want — several functions, a gate that routes. Let me try to adopt it directly. First decision: what are the experts? The obvious move is `K` full parallel backbones, one per pattern. But that multiplies my parameter count by `K`, and I want this to be cheap, and more importantly when I have `K` times the parameters on a noisy, smallish financial dataset I overfit hard — I can already feel that happening in my head, `K` LSTMs each memorizing its slice of a few thousand stocks. So back up. The patterns differ in the *relation* `p(y|x)`, the mapping from a representation to a return, far more than in how to *encode* the window in the first place; the feature extraction can be shared. So I keep one shared backbone `ψ` and only replicate the cheap part — the final linear output layer — into `K` linear predictors. The `k`-th prediction is `ŷ_ik = θ_k ∘ ψ(x_i)`, where `ψ` is shared and `θ_k` is just a linear head. Negligible extra parameters, and a free bonus: if I set `K = 1` this collapses to exactly the original single-head backbone, so I can never do worse than the baseline by construction. Good. So: `predictors = Linear(hidden, K)`, producing all `K` scores at once.

Now the gate — the router that decides which predictor a sample belongs to. What does it get to look at? It must run at test time, so it can only use things available without the label. The first input is the latent representation `h_i = ψ(x_i)` itself, the backbone's encoding of the window; certain patterns should be distinguishable straight from `h`. I deliberately route on `h` rather than the raw features because `h` is already tuned toward the prediction task, so it is more informative about which `p(y|x)` is in play than the raw inputs are. But I worry that the latent alone underdetermines the pattern — these patterns are genuinely hard to read off a single window; the difference between "we are in a momentum regime" and "we are in a reversal regime" is not obviously written in one stock's 60-day feature vector. I need a second signal.

Here is where I should think about how a real investor switches strategies, because that is the missing signal. An investor does not stare at today's prices and deduce the macro regime from first principles; they watch how their strategies have been *performing lately* and rotate toward whatever has been working — the "investment clock" intuition, momentum strategies leading in one phase of the cycle, value in another. Translate that: I have `K` predictors, each one is a strategy; the analogue of "how a strategy has been performing lately" is the *recent prediction error of that predictor*. So feed the router the history of each predictor's errors. For sample `i = (s, t)` — stock `s` at time `t` — collect the vector of past per-predictor losses over a lookback window, `e_i = [l_{(s,t-T)}, l_{(s,t-T+1)}, ..., l_{(s,t-h)}]`, where `l_{(s,t')} = [ℓ(ŷ_{1}, y), ..., ℓ(ŷ_{K}, y)]` is the `K`-vector of all predictors' errors at that earlier time. One subtlety I have to get right or I leak the future: the most recent entry I can use is `t-h`, not `t`, where `h` exceeds the label horizon — because the loss `l_{(s,t)}` needs `y` at `t`, which is a future return I do not know yet at decision time. So the gap `h` is not cosmetic; it is what keeps the router causal. I'll call these two information sources LR (latent representation) and TPE (temporal prediction errors), and let the router consume `h_i`, `e_i`, or both. To turn the variable-length error history into a fixed vector I run it through a small recurrent net and take its last hidden state, then concatenate that with `h_i`. So `out = concat(h_i, RNN(e_i)[-1])`, then a linear `fc` maps `out` to `K` logits `a_i = π(h_i, e_i)`.

Now, how does the router *use* the logits to produce a prediction? The lazy answer is a soft attention: `q_i = softmax(a_i)`, final prediction `p̂_i = q_i^T ŷ_i`, a convex blend of the `K` predictors. Let me sit with that, because it is differentiable and easy, but I think it is wrong for my goal. If the router always blends, then each predictor sees gradients from *every* sample weighted by `q`, and they all get pulled toward the same average relation again — I have re-created the single-model problem with extra steps, just a smeared committee instead of one model. To actually force the predictors to *specialize* into distinct patterns, the routing has to be *discrete*: each sample goes to *one* predictor, so that predictor is trained only on its kind of sample and can swing its coefficients to, say, negative-momentum without being dragged back by the momentum samples. Discrete selection is `argmax(q_i)`. But `argmax` has zero gradient almost everywhere — I cannot train the router through it. Wall.

The escape is the Gumbel-softmax. The Gumbel-Max trick says I can draw a categorical sample as `one_hot(argmax_k (a_ik + g_ik))` with `g_ik` i.i.d. `Gumbel(0,1)` (and I can sample `Gumbel(0,1)` as `-log(-log(Uniform(0,1)))`), which is an exact draw from `softmax(a_i)`. The non-differentiable part is still the `argmax`, so replace *that* with a temperature softmax: `q_ik = exp((a_ik + g_ik)/τ) / Σ_j exp((a_ij + g_ij)/τ)`. As `τ → 0` this peaks into a one-hot vector — discrete selection — and for finite `τ` it is a smooth point on the simplex with a real gradient. That gives me differentiable-but-discrete routing. In practice I want the forward pass to be genuinely one-hot (a hard choice) while the backward pass uses the soft gradient — the straight-through estimator. So the router emits a hard gumbel-softmax `choice` (a one-hot used as the training selection weight) and also a plain `prob = softmax(a/τ)` that I keep around for test-time, where I do not want noise and just take `argmax(prob)`. Temperature `τ = 1` to start. So the prediction during training is `pred = (all_preds * choice).sum`, picking out the chosen predictor's score (with straight-through gradient), and at test it is `all_preds[argmax(prob)]`.

Let me try to just train this end to end now: backbone, `K` predictors, router with gumbel-softmax, minimize the task loss on the chosen predictor. And it collapses. Every sample gets routed to the *same* predictor; the others sit unused. I should have seen this coming, because it is exactly the self-reinforcing failure mode of gated mixtures — whichever predictor is marginally better early gets all the samples, so it trains faster, so it gets even more samples, and the rest starve. The router found the trivial solution: one predictor, the others dead. So the architecture is necessary but not sufficient; the *learning* of the router is the hard part, and left to itself it destroys the multi-pattern capacity I built. Wall.

What is the standard patch? The mixture-of-experts people add a soft balancing penalty — an "importance" loss equal to the squared coefficient of variation of the per-expert gate mass, pushing all experts toward equal total weight. Let me consider bolting that on: add `w · CV(per-predictor selection mass)^2` to the loss. I can see two problems with it before I even run it. First, it is *soft* — it nudges toward balance but does not enforce it, and the gate can trade a little penalty for a lot of collapse if collapse lowers the task loss. Second, and worse for me: even if it equalizes the *importance* (the summed gate weight) per predictor, that does not equalize the *number of samples* each predictor actually owns — one predictor can hoard many small-weight samples — and, crucially, the CV penalty only cares about balance, it has *no opinion about which sample should go to which predictor*. It will happily balance by assigning samples arbitrarily. But arbitrary balanced assignment does not discover patterns; I need the *right* samples grouped together, namely the ones a given predictor predicts *well*. So balance alone is not the objective. The objective is: assign each sample to the predictor that predicts it best, subject to keeping the predictors from collapsing into one. Two requirements at once — low cost and balance.

Now state that precisely and see what it is. Pack every predictor's error on every sample into a loss matrix `L ∈ R^{N×K}`, `L_ik = ℓ(θ_k ∘ ψ(x_i), y_i)`. I want an assignment matrix `P ∈ {0,1}^{N×K}` that (i) minimizes the total assigned loss `<P, L>` (Frobenius inner product — sum of the chosen predictors' errors), (ii) gives each sample exactly one predictor, `Σ_k P_ik = 1`, and (iii) keeps the predictors balanced, with the number of samples sent to predictor `k` proportional to that pattern's share, `Σ_i P_ik = ν_k · N`. Write it out:

  minimize over `P`   `<P, L>`
  subject to   `Σ_k P_ik = 1`  for all `i`,
               `Σ_i P_ik = ν_k · N`  for all `k`,
               `P_ik ∈ {0,1}`.

Stare at the constraints. Row sums all equal one, column sums fixed to prescribed totals, nonnegative entries minimizing a linear cost — these are *marginal* constraints on a nonnegative matrix with a linear objective. That is an optimal-transport problem: transport unit mass from the `N` samples (row marginal all-ones) to the `K` predictors (column marginal `ν_k N`) at cost `L`, `min_{P ∈ U(r,c)} <P, L>` over the transportation polytope `U(r,c) = {P ≥ 0 : P\mathbf{1} = r, P^T\mathbf{1} = c}`. The "keep predictors from collapsing" requirement is *literally the column-marginal constraint* — it forces a prescribed amount of mass into every predictor, so none can be starved to zero. This is the same structure that makes balanced self-labelling work: minimizing the model's own loss with the labels as free targets collapses everyone to one cluster, and the cure is to add an equipartition (balanced-marginal) constraint, which turns the assignment into exactly this transport problem. So the collapse I hit and the collapse in unsupervised clustering are the same disease, and the cure is the same — encode "do not collapse" as a hard marginal constraint rather than a soft penalty, and let the cost term `<P, L>` simultaneously pull each sample toward the predictor that actually fits it. Balance and correctness in one optimization.

Can I solve it? The constraints `P_ik ∈ {0,1}` make it combinatorial, and an exact transport LP costs about `O(d^3 log d)` — far too slow to solve inside every minibatch of training. The relaxation I need is the entropic one. Replace the hard LP by the regularized transport problem with the same marginals and a smoothing term; its positive solution has the scaling form `P = diag(u) · exp(-L/ε) · diag(v)`, so the cost enters with a minus sign inside the exponential and the vectors `u, v` are just the row and column rescalings needed to hit the marginals. The implementation-friendly form is the same calculation with the sign pushed into the argument: build the fused cost `L_h`, pass `-L_h` into Sinkhorn, and inside the solver compute `exp(Q/ε) = exp(-L_h/ε)`. Then repeat column normalization followed by row normalization, exactly in that order. The column step equalizes predictor mass; the row step restores one unit of mass per sample. In the equal-share case, the positive-matrix fixed point has row sums one and column sums `N/K`; after a finite three iterations, the last row normalization makes the per-sample mass exact and the columns are the fast balanced approximation. The negative sign is non-negotiable: low loss must become high transport mass, not the other way around. `ε` controls the smoothing; it has to be scaled to the magnitude of the cost entries, which is why I will normalize `L` to a fixed range before exponentiating.

Now a real obstacle: `P` depends on `L`, and `L` is built from the prediction *errors*, which need the labels. So `P` is a label-dependent oracle assignment — fantastic for training, useless at test, where I have no labels and cannot compute `L`. I cannot ship `P` as the router. But I can use `P` as a *teacher* for the router. The router, fed only LR and TPE (both label-free), should learn to *predict* the assignment that the OT oracle would have produced. That is a supervised classification problem: target distribution `P_i·` (the oracle's one-hot-ish assignment for sample `i`), prediction `q_i` (the router's softmax over predictors), trained by cross-entropy. So I add an auxiliary regularization term to the objective:

  minimize over `Θ, π, ψ`   `E_i [ ℓ(x_i, y_i; Θ, π, ψ) − λ Σ_k P_ik log q_ik ]`.

The first term is the task loss on the routed prediction. The second term uses `reg_i = Σ_k P_ik log q_ik`, the router's log-likelihood under the OT target; since cross-entropy is `CE(P_i, q_i) = -Σ_k P_ik log q_ik`, minimizing `task_loss − λ·reg` is the same as minimizing `task_loss + λ·CE(P, q)`. For training selection I still use the hard straight-through Gumbel sample, but for this regularizer I use the router's plain probability vector, because the target is a distribution to imitate. This trains the router to imitate the OT assignment using only the current-sample latent and the available error-history state, not the current label. At test time I throw away `P` and the whole loss machinery and just run the router: `argmax(q_i)` picks the predictor without the current label. That is exactly property (c) from the start — the router distills the label-dependent optimal assignment into a selector that can run during inference. In code the regularizer is `reg = prob.log().mul(P).sum(dim=1).mean()` and the total loss is `loss = task_loss − λ · reg`.

Two parameters in that objective still need pinning down, and I should derive them, not guess. First, `ν_k`, the pattern shares in the column marginal — I do not know them; they are latent. What is a safe prior? Early in training the predictors are undifferentiated, so I have no business claiming one pattern is rarer than another; the maximally noncommittal choice is equal shares, `ν_k = 1/K`, which also maximally fights collapse (it forces every predictor to take a `1/K` slice in the OT teacher). But equal shares is surely wrong in the limit — real pattern frequencies are not uniform — so I do not want the teacher to dominate forever. The transport target itself stays balanced, but its grip on the learned router weakens over time: `λ` is the strength with which that OT teacher is imposed, and I decay it as `λ_t = λ_0 · ρ^{⌊step/100⌋}` with `ρ` just below one (say `ρ = 0.99`). Early on, strong `λ` and equal-share transport force diverse, balanced predictors; later, weak `λ` lets the task loss and the router settle into the share structure the data supports. That is the role of `ρ`.

Second, `K`, the number of predictors — also unknown, also a hyperparameter. I cannot derive the true number of patterns, so I treat it as a knob, with the safety net that `K = 1` recovers the plain backbone. Reasoning about the trade-off: too few and I cannot represent the distinct patterns (one head still averages momentum and reversal); too many and I shred the data into tiny per-predictor slices that overfit. A modest `K` — three by default for this setup — gives the predictors enough room to separate the main regimes without starving each on data.

There is one more practical wall, and it is about cost. The router needs TPE — every predictor's *recent* errors for each sample — and computing those errors fresh would require a full forward pass over *all* `N` training samples on *every* training step, since each predictor's parameters change each step. That is hopeless. So I cache. Keep an external memory `M ∈ R^{N×K}` holding the latest per-sample per-predictor errors, and update it in two stages: refresh the *whole* memory once before each epoch (one full pass to get a consistent snapshot of every predictor's errors), and then patch the memory *on the fly* for just the minibatch samples touched in each step. The router reads its TPE input from this memory rather than recomputing. That turns an `O(N)`-per-step cost into `O(batch)`-per-step, an `N/batch` speedup, at the price of the TPE being slightly stale within an epoch — acceptable, because the error history is a slowly-varying signal anyway. And I must respect temporal order strictly, especially at test: the router needs errors from *earlier* timestamps to choose the predictor for the current one, so batches have to flow forward in time, never shuffled across the time axis.

One refinement on the transport cost itself. The single-batch loss matrix `L` is noisy — it is one snapshot of how the predictors did on this batch. The memory already holds a smoother history of errors. So instead of transporting against the raw batch loss, I fuse the two: `L_h = α · normalize(L) + (1−α) · normalize(hist_loss)`, then renormalize, and run Sinkhorn on `−L_h`. With `α` mixing current-batch error and historical error, the transport target is stabilized; `α = 0.5` weights them equally. The normalization here is a min-max squash of each row into `[0,1]` so that `ε` in the Sinkhorn exponential is on a consistent scale and a degenerate all-equal row (no information) maps to a flat assignment rather than blowing up.

And one warm-start, because a cold router cannot bootstrap. If I begin with random predictors and a random router simultaneously, the router is trying to imitate an OT assignment over predictors that are all equally bad, so the targets are meaningless and it has nothing to learn from — and the predictors, never receiving cleanly separated samples, never differentiate. Break the chicken-and-egg by pretraining: first train the backbone and the `K` predictors with the OT assignment used directly (the oracle transport — form the balanced `P` from losses and route by that `P`, with `λ = 0` so there is no router regularizer), which forces the predictors apart into distinct specializations before the router has to carry inference. Then, with diverse predictors in hand, reset the optimizer and train the router (router transport plus the OT cross-entropy regularizer) to learn the current-label-free selection. So `pretrain` runs the loop once in "oracle" mode (predictors specialize), then again in "router" mode (router learns to mimic the now-meaningful assignment).

Let me also fix the default architecture sizes, since they have to be concrete. The backbone is an attentive LSTM: project the 20 input features, run a two-layer LSTM with hidden size 64, attention-pool the sequence (`score = softmax_t(u^T tanh(W · rnn_out))`, `att = Σ_t score_t · rnn_out_t`) and concatenate the pooled vector with the mean last hidden state — so the latent fed to the predictors and router has size `2 · 64 = 128`. The router's TPE encoder is a small one-layer LSTM with hidden size 32 over the `K`-dim error history; its last hidden state is concatenated with the 128-dim latent and mapped by `fc` to `K` logits. `τ = 1`. The whole thing trains with Adam at `1e-3`, early-stopping on validation IC, `λ = 1`, `ρ = 0.99`, `α = 0.5`, `K = 3`, sequence length 60, batch 1024, Sinkhorn with three iterations and `ε = 0.1`.

Now I have the causal chain and can write the code that fills the empty head/training slots, grounded in the structure above. The pieces: the backbone encoder; the router that holds `K` linear predictors plus a TPE-RNN-and-`fc` that emits the gumbel-softmax `choice` and the softmax `prob`; the Sinkhorn solver; the transport routine that builds the fused loss matrix, runs Sinkhorn to get `P`, forms the routed prediction, and returns the task loss and `P`; and the training loop with two-stage memory, the `−λ·reg` regularizer with decaying `λ`, the oracle-pretrain then router-train, and the label-free `argmax` inference.

```python
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPS = 1e-12
device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN(nn.Module):
    # shared backbone psi: attentive LSTM. window -> latent h_i = psi(x_i).
    def __init__(self, input_size=16, hidden_size=64, num_layers=2,
                 rnn_arch="LSTM", use_attn=True, dropout=0.0, **kwargs):
        super().__init__()
        self.rnn_arch = rnn_arch
        self.use_attn = use_attn
        # only project when compressing (hidden < input); otherwise feed raw features
        self.input_proj = nn.Linear(input_size, hidden_size) if hidden_size < input_size else None
        self.rnn = getattr(nn, rnn_arch)(
            input_size=min(input_size, hidden_size), hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout)
        if use_attn:                                   # temporal attention pooling
            self.W = nn.Linear(hidden_size, hidden_size)
            self.u = nn.Linear(hidden_size, 1, bias=False)
            self.output_size = hidden_size * 2         # [mean last hidden ; attended]
        else:
            self.output_size = hidden_size

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        rnn_out, last_out = self.rnn(x)
        if self.rnn_arch == "LSTM":
            last_out = last_out[0]                      # (h, c) -> h
        last_out = last_out.mean(dim=0)                 # average over layers
        if self.use_attn:
            laten = self.W(rnn_out).tanh()
            scores = self.u(laten).softmax(dim=1)      # softmax over time
            att_out = (rnn_out * scores).sum(dim=1)
            last_out = torch.cat([last_out, att_out], dim=1)
        return last_out


class TRA(nn.Module):
    # K linear predictors on the shared latent + a router that emits a discrete choice.
    def __init__(self, input_size, num_states=1, hidden_size=32, rnn_arch="LSTM",
                 num_layers=1, dropout=0.0, tau=1.0, src_info="LR_TPE"):
        super().__init__()
        assert src_info in ["LR", "TPE", "LR_TPE"]
        self.num_states = num_states
        self.tau = tau
        self.rnn_arch = rnn_arch
        self.src_info = src_info
        self.predictors = nn.Linear(input_size, num_states)   # the K patterns (cheap heads)
        if num_states > 1:
            if "TPE" in src_info:                             # encode error-history into router
                self.router = getattr(nn, rnn_arch)(
                    input_size=num_states, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden_size + input_size if "LR" in src_info else hidden_size,
                                    num_states)
            else:
                self.fc = nn.Linear(input_size, num_states)   # LR only

    def forward(self, hidden, hist_loss):
        preds = self.predictors(hidden)                       # all K predictions [N, K]
        if self.num_states == 1:                              # falls back to plain backbone
            return preds, None, None
        if "TPE" in self.src_info:
            out = self.router(hist_loss)[1]                   # TPE: summarize error history
            if self.rnn_arch == "LSTM":
                out = out[0]
            out = out.mean(dim=0)
            if "LR" in self.src_info:
                out = torch.cat([hidden, out], dim=-1)        # LR + TPE
        else:
            out = hidden                                       # LR only
        out = self.fc(out)                                     # router logits a_i [N, K]
        choice = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=True)  # discrete (straight-through)
        prob = torch.softmax(out / self.tau, dim=-1)          # soft probs (argmax at test)
        return preds, choice, prob


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    if len(pred.shape) == 2:
        label = label[:, None]
    return (pred[mask] - label[mask]).pow(2).mean(dim=0)       # per-predictor MSE


def minmax_norm(x):
    # squash each row to [0,1] so Sinkhorn's epsilon sits on a consistent scale;
    # a row with no spread (no information) maps to a flat 1 (uniform assignment).
    xmin = x.min(dim=-1, keepdim=True).values
    xmax = x.max(dim=-1, keepdim=True).values
    mask = (xmin == xmax).squeeze()
    x = (x - xmin) / (xmax - xmin + EPS)
    x[mask] = 1
    return x


def shoot_infs(inp_tensor):
    # replace any inf produced by exp(.) with the max finite value (numerical guard)
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            inp_tensor[tuple(ind)] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            inp_tensor[tuple(ind)] = m
    return inp_tensor


def sinkhorn(Q, n_iters=3, epsilon=0.1):
    # entropic-OT solution: exp(-cost/eps), then qlib's column-then-row normalization.
    with torch.no_grad():
        Q = torch.exp(Q / epsilon)        # Q is -L_h on input, so low loss -> high mass
        Q = shoot_infs(Q)
        for _ in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)   # column marginal (per predictor) -> balance
            Q /= Q.sum(dim=1, keepdim=True)   # row marginal (per sample) -> one unit
    return Q


def transport_sample(all_preds, label, choice, prob, hist_loss, count,
                     transport_method, alpha, training=False):
    # build loss matrix L, fuse with history, Sinkhorn -> assignment P, form routed pred.
    all_loss = torch.zeros_like(all_preds)
    mask = ~torch.isnan(label)
    all_loss[mask] = (all_preds[mask] - label[mask, None]).pow(2)   # L_ik [N, K]

    L = minmax_norm(all_loss.detach())
    Lh = L * alpha + minmax_norm(hist_loss) * (1 - alpha)           # fuse current + memory
    Lh = minmax_norm(Lh)
    P = sinkhorn(-Lh)                                               # OT assignment (oracle target)

    if transport_method == "router":
        if training:
            pred = (all_preds * choice).sum(dim=1)                  # pick chosen predictor (gumbel)
        else:
            pred = all_preds[range(len(all_preds)), prob.argmax(dim=-1)]   # label-free argmax
        loss = loss_fn(pred, label)
    else:                                                          # oracle: route by P directly
        pred = (all_preds * P).sum(dim=1)
        loss = (all_loss * P).sum(dim=1).mean()
    return loss, pred, L, P


class TRAModel:
    """qlib-style model interface filled with the predictors+router method."""

    def __init__(self):
        self.model_config = dict(input_size=20, hidden_size=64, num_layers=2,
                                 rnn_arch="LSTM", use_attn=True, dropout=0.0)
        self.tra_config = dict(num_states=3, rnn_arch="LSTM", hidden_size=32,
                               num_layers=1, dropout=0.0, tau=1.0, src_info="LR_TPE")
        self.lr, self.n_epochs, self.early_stop = 1e-3, 100, 20
        self.lamb, self.rho, self.alpha = 1.0, 0.99, 0.5           # reg strength, its decay, fusion
        self.transport_method, self.pretrain = "router", True
        self.transport_fn = transport_sample
        self.model = RNN(**self.model_config).to(device)
        self.tra = TRA(self.model.output_size, **self.tra_config).to(device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()),
                                    lr=self.lr)
        self.fitted, self.global_step = False, -1

    def train_epoch(self, data_set, is_pretrain=False):
        self.model.train(); self.tra.train(); data_set.train()
        self.optimizer.zero_grad()
        for batch in data_set:
            if not is_pretrain:
                self.global_step += 1
            data, state, label = batch["data"], batch["state"], batch["label"]
            index = batch["index"]
            hidden = self.model(data)
            all_preds, choice, prob = self.tra(hidden, state)
            # pretrain uses the oracle assignment to force predictors apart; then the router.
            loss, pred, L, P = self.transport_fn(
                all_preds, label, choice, prob, state.mean(dim=1), None,
                "oracle" if is_pretrain else self.transport_method, self.alpha, training=True)
            data_set.assign_data(index, L)                         # patch memory on the fly
            decay = self.rho ** (self.global_step // 100)          # relax the equal-share prior
            lamb = 0 if is_pretrain else self.lamb * decay
            reg = (prob.log() * P).sum(dim=1).mean() if prob is not None else 0.0
            loss = loss - lamb * reg                               # distill OT target into router
            loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()

    def test_epoch(self, data_set, return_pred=False):
        self.model.eval(); self.tra.eval(); data_set.eval()
        preds = []
        for batch in data_set:
            data, state, label = batch["data"], batch["state"], batch["label"]
            index = batch["index"]
            with torch.no_grad():
                hidden = self.model(data)
                all_preds, choice, prob = self.tra(hidden, state)
                loss, pred, L, P = self.transport_fn(
                    all_preds, label, choice, prob, state.mean(dim=1), None,
                    self.transport_method, self.alpha, training=False)
            data_set.assign_data(index, L)                         # keep memory fresh
            if return_pred:
                X = np.c_[pred.cpu().numpy(), label.cpu().numpy()]
                preds.append(pd.DataFrame(X, index=batch["index"], columns=["score", "label"]))
        if return_pred:
            preds = pd.concat(preds, axis=0)
            preds.index = data_set.restore_index(preds.index)
            preds.index = preds.index.swaplevel()
            preds.sort_index(inplace=True)
            return preds["score"]

    def _fit(self, train_set, valid_set, is_pretrain):
        best = -1
        if not is_pretrain:
            self.test_epoch(train_set)                             # init memory before router training
        for _ in range(self.n_epochs):
            self.train_epoch(train_set, is_pretrain=is_pretrain)
            # (validation IC tracked here for early stopping; omitted for brevity)
        return best

    def fit(self, dataset):
        train_set, valid_set, test_set = dataset.prepare(["train", "valid", "test"])
        self.fitted, self.global_step = True, -1
        if self.pretrain:                                          # 1) specialize predictors (oracle)
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.tra.predictors.parameters()), lr=self.lr)
            self._fit(train_set, valid_set, is_pretrain=True)
            self.optimizer = optim.Adam(                           # 2) reset, then learn the router
                list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)
        self._fit(train_set, valid_set, is_pretrain=False)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        test_set = dataset.prepare(segment)
        return self.test_epoch(test_set, return_pred=True)         # label-free: router argmax
```

The whole arc: one model with one head can only fit the average relation, but the data carries opposite-signed patterns whose signs flip across regimes, so averaging fails. Replicate only the cheap output head into `K` predictors on a shared backbone so several relations coexist at near-zero extra cost, with `K = 1` recovering the baseline. A gate must pick *one* predictor per sample and must run without the current label at test, so route on the latent representation plus each predictor's recent error history, made discrete-but-differentiable by the gumbel-softmax. Training the router naively collapses everyone onto one predictor — the self-reinforcing gate pathology — and a soft balance penalty does not fix it because it cannot say *which* sample belongs where. State the real objective — assign each sample to its low-error predictor while keeping the predictors balanced — and it is exactly a balanced optimal-transport problem, the same equipartition-constrained transport that cures collapse in self-labelling, solved cheaply by a few Sinkhorn iterations of `exp(-L_h/ε)` through column-then-row normalization after passing `-L_h` into the solver. The transport assignment needs labels, so it cannot be the test-time selector; instead distill it into the router with `reg = Σ_k P_ik log q_ik` and `loss = task_loss − λ·reg`, which is task loss plus a cross-entropy to the OT teacher. Decay that regularizer's strength so the model first balances hard then lets the task loss dominate; cache per-predictor errors in an external memory with a two-stage refresh so TPE is affordable; fuse current and historical loss for a stable transport target; and warm-start by pretraining the predictors against the oracle assignment before the router carries inference. At test, drop the current labels, the transport target, and the regularizer, and just run the router's argmax over the `K` predictors.
