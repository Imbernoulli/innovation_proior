Let me start from the thing that actually fails. I train a sequence model on stock features to predict next-period return ranks, the usual way: collect `{(x_i, y_i)}`, encode each lookback window into a latent vector with an attentive LSTM, push it through a linear head, regress to the label by mean-squared error, and call it empirical risk minimization. Buried in that pipeline is an assumption I never questioned — that every `(x_i, y_i)` is an i.i.d. draw from one fixed joint distribution `P`, so there is one relation `p(y|x)` to learn. And I keep hitting a ceiling. Let me figure out whether the ceiling is the model capacity or the assumption.

Here is what makes me suspect the assumption. The market is the sum of a lot of people running different strategies, and two of the most documented cross-sectional effects point in opposite directions at once. Momentum: stocks that went up keep going up. Reversal / mean-reversion: stocks that went down rebound. Both are real, both are in the same market, so the sign of "past return → future return" need not be fixed across samples. I can probe this without any new architecture, just a regression I can reason about: take three features, size rank, value rank, twelve-month momentum rank, and fit a plain linear model of next-month return rank, but fit it separately on different years and read off the momentum coefficient. If the single-`P` assumption held, that coefficient would be roughly stable from year to year. It is not: fit on 2009 it comes out *negative* — that year high past return predicted low future return, reversal — and fit on 2013 it comes out *positive*, momentum. A single coefficient vector cannot be both negative and positive at once. So the problem is not that my model is too small; it is that I am asking *one* function with *one* parameter vector to hold two relations that contradict each other, and the best it can do is land somewhere between them — a coefficient near zero — which fits neither year. And these regimes rotate over time — size leads in one stretch, momentum in another — so it is not a one-off either. The i.i.d. single-`P` assumption is the wall. The data looks more like `P = Σ_k ν_k P_k`, a mixture of patterns, each with its own `p_k(y|x)` and its own share `ν_k`, and which one is active drifts.

So what would actually solve it. If I knew, for every sample, which pattern `k` it came from, this is easy: it is just multi-domain learning — split the data by domain identifier, fit a model per domain, done. But I have no pattern identifiers. They are latent. Worse than latent: even if I cheated and used the *labels* to figure out which pattern each training sample best fits, that trick dies at test time, because at test there are no labels — and the whole point of a predictor is the test period. So the real constraint is sharp: I need a thing that (a) holds several distinct relations at once, (b) decides per-sample which relation applies, and (c) can make that decision from signals available without the current label: the sample's latent representation and whatever past prediction-error state has already been cached. Property (c) is the one that kills the naive solutions, and I should keep it front of mind.

Let me look at the tools for "one model, several behaviors." There is the conditional-computation / mixture-of-experts idea: keep `K` expert sub-networks and a trainable gating network `G(x)` that, per input, picks a sparse combination of experts, and train it all jointly by backprop. That is structurally exactly what I want — several functions, a gate that routes. Let me try to adopt it directly. First decision: what are the experts? The obvious move is `K` full parallel backbones, one per pattern. But that multiplies my parameter count by `K`, and I want this to be cheap, and more importantly when I have `K` times the parameters on a noisy, smallish financial dataset I overfit hard — I can already feel that happening in my head, `K` LSTMs each memorizing its slice of a few thousand stocks. So back up. The patterns differ in the *relation* `p(y|x)`, the mapping from a representation to a return, far more than in how to *encode* the window in the first place; the feature extraction can be shared. So I keep one shared backbone `ψ` and only replicate the cheap part — the final linear output layer — into `K` linear predictors. The `k`-th prediction is `ŷ_ik = θ_k ∘ ψ(x_i)`, where `ψ` is shared and `θ_k` is just a linear head. The extra parameter count is `K` copies of a `hidden→1` linear map; with `hidden = 128` and `K = 3` that is `3·129 = 387` scalars on top of a backbone that already has on the order of `10^5` — negligible. And there is a containment property I want to lean on, so let me check it actually holds rather than just hope it does: set `K = 1`. Then `predictors = Linear(hidden, 1)` is a single linear head, the router has nothing to choose between (one column), and the whole selection apparatus below switches off; what remains is exactly "encode the window, apply one linear head, regress" — the original baseline. So this construction *contains* the baseline as its `K=1` instance, which means a tuned version should not underperform it for reasons of architecture. Good. So: `predictors = Linear(hidden, K)`, producing all `K` scores at once.

Now the gate — the router that decides which predictor a sample belongs to. What does it get to look at? It must run at test time, so it can only use things available without the label. The first input is the latent representation `h_i = ψ(x_i)` itself, the backbone's encoding of the window; certain patterns should be distinguishable straight from `h`. I deliberately route on `h` rather than the raw features because `h` is already tuned toward the prediction task, so it is more informative about which `p(y|x)` is in play than the raw inputs are. But I worry that the latent alone underdetermines the pattern — these patterns are genuinely hard to read off a single window; the difference between "we are in a momentum regime" and "we are in a reversal regime" is not obviously written in one stock's 60-day feature vector. I need a second signal.

Here is where I should think about how a real investor switches strategies, because that is the missing signal. An investor does not stare at today's prices and deduce the macro regime from first principles; they watch how their strategies have been *performing lately* and rotate toward whatever has been working — the "investment clock" intuition, momentum strategies leading in one phase of the cycle, value in another. Translate that: I have `K` predictors, each one is a strategy; the analogue of "how a strategy has been performing lately" is the *recent prediction error of that predictor*. So feed the router the history of each predictor's errors. For sample `i = (s, t)` — stock `s` at time `t` — collect the vector of past per-predictor losses over a lookback window, `e_i = [l_{(s,t-T)}, l_{(s,t-T+1)}, ..., l_{(s,t-h)}]`, where `l_{(s,t')} = [ℓ(ŷ_{1}, y), ..., ℓ(ŷ_{K}, y)]` is the `K`-vector of all predictors' errors at that earlier time. One subtlety I have to get right or I leak the future: the most recent entry I can use is `t-h`, not `t`, where `h` exceeds the label horizon — because the loss `l_{(s,t)}` needs `y` at `t`, which is a future return I do not know yet at decision time. So the gap `h` is not cosmetic; it is what keeps the router causal. I'll call these two information sources LR (latent representation) and TPE (temporal prediction errors), and let the router consume `h_i`, `e_i`, or both. To turn the variable-length error history into a fixed vector I run it through a small recurrent net and take its last hidden state, then concatenate that with `h_i`. So `out = concat(h_i, RNN(e_i)[-1])`, then a linear `fc` maps `out` to `K` logits `a_i = π(h_i, e_i)`.

Now, how does the router *use* the logits to produce a prediction? The lazy answer is a soft attention: `q_i = softmax(a_i)`, final prediction `p̂_i = q_i^T ŷ_i`, a convex blend of the `K` predictors. Let me sit with that, because it is differentiable and easy. The thing I need to check is what gradient each predictor `θ_k` actually receives under the blend. With `p̂_i = Σ_k q_ik ŷ_ik` and an MSE task loss `(p̂_i − y_i)^2`, the gradient flowing into predictor `k` from sample `i` is `∂loss/∂θ_k = 2(p̂_i − y_i)·q_ik·∂ŷ_ik/∂θ_k`. The factor `q_ik` is nonzero for *every* predictor on *every* sample — a soft blend never zeros it. So each `θ_k` is updated by a `q_ik`-weighted mix of all samples, including the reversal samples and the momentum samples together, and its fitted coefficients relax toward whatever minimizes that weighted average error. That is the single-head averaging problem coming back, just split across a committee whose members all drift toward the same compromise. To actually force the predictors to *specialize* into distinct patterns, predictor `k` must receive gradient from *only* its kind of sample — the weight on the other samples has to be exactly zero, not small. That means the routing has to be *discrete*: each sample goes to *one* predictor, `q_ik ∈ {0,1}`, so a momentum-pattern sample contributes nothing to the reversal predictor and that predictor is free to swing its coefficients to negative-momentum without being dragged back. Discrete selection is `argmax(a_i)`. But `argmax` has zero gradient almost everywhere — I cannot train the router through it. Wall.

The escape is the Gumbel-softmax. The Gumbel-Max trick says I can draw a categorical sample as `one_hot(argmax_k (a_ik + g_ik))` with `g_ik` i.i.d. `Gumbel(0,1)` (and I can sample `Gumbel(0,1)` as `-log(-log(Uniform(0,1)))`), which is an exact draw from `softmax(a_i)`. The non-differentiable part is still the `argmax`, so replace *that* with a temperature softmax: `q_ik = exp((a_ik + g_ik)/τ) / Σ_j exp((a_ij + g_ij)/τ)`. As `τ → 0` this peaks into a one-hot vector — discrete selection — and for finite `τ` it is a smooth point on the simplex with a real gradient. That gives me differentiable-but-discrete routing. In practice I want the forward pass to be genuinely one-hot (a hard choice) while the backward pass uses the soft gradient — the straight-through estimator. So the router emits a hard gumbel-softmax `choice` (a one-hot used as the training selection weight) and also a plain `prob = softmax(a/τ)` that I keep around for test-time, where I do not want noise and just take `argmax(prob)`. Temperature `τ = 1` to start. So the prediction during training is `pred = (all_preds * choice).sum`, picking out the chosen predictor's score (with straight-through gradient), and at test it is `all_preds[argmax(prob)]`.

Let me try to just train this end to end now: backbone, `K` predictors, router with gumbel-softmax, minimize the task loss on the chosen predictor. Nothing in this objective pins the predictors to *different* patterns — the only pressure is "make the chosen predictor's error small." Walk the dynamics. At initialization all `K` predictors are about equally bad, so the router's choice is roughly random; but some predictor is marginally ahead by chance and gets a slightly larger share of samples. That predictor therefore receives more gradient, improves faster, and on the next pass the router — which is *rewarded* for picking low-error predictors — sends it even more samples. The share is self-reinforcing: ahead → more samples → more improvement → further ahead. There is no force in the loss pushing back on that feedback loop, so its fixed point is the corner where one predictor takes everything and the rest, never selected, never train and stay dead. This is precisely the documented gated-mixture pathology, and the objective as written has no term that opposes it. So the architecture is necessary but not sufficient; the *learning* of the router is the hard part, and left to itself it destroys the multi-pattern capacity I built. Wall.

What is the standard patch? The mixture-of-experts people add a soft balancing penalty — an "importance" loss equal to the squared coefficient of variation of the per-expert gate mass, pushing all experts toward equal total weight. Let me consider bolting that on: add `w · CV(per-predictor selection mass)^2` to the loss, and think through what it buys before committing. First, it is *soft* — it nudges toward balance but does not enforce it, so the gate can trade a small constant penalty for the large task-loss reduction of collapse whenever collapse pays, and the feedback loop I just traced can win at a slightly higher loss. Second, and the one that actually decides it against this patch: the CV term is a function of the *column totals* of the assignment only — `Σ_i q_ik` for each `k`. It is completely blind to *which* rows make up each total. So two assignments with the same balanced column sums are scored identically by CV, whether each predictor got the samples it fits well or a uniformly random scramble. Penalizing imbalance therefore does not push the assignment toward the *correct* grouping; it only flattens the totals. But arbitrary balanced assignment does not discover patterns — I need the *right* samples grouped together, namely the ones a given predictor predicts *well*. So balance alone is not the objective, and a term that only sees column totals cannot supply the missing half. The objective has to couple two things at once: assign each sample to the predictor that predicts it best (a per-entry cost, the part CV throws away) *and* keep the column totals balanced (the part CV keeps). Low cost and balance, jointly.

Now state that precisely and see what it is. Pack every predictor's error on every sample into a loss matrix `L ∈ R^{N×K}`, `L_ik = ℓ(θ_k ∘ ψ(x_i), y_i)`. I want an assignment matrix `P ∈ {0,1}^{N×K}` that (i) minimizes the total assigned loss `<P, L>` (Frobenius inner product — sum of the chosen predictors' errors), (ii) gives each sample exactly one predictor, `Σ_k P_ik = 1`, and (iii) keeps the predictors balanced, with the number of samples sent to predictor `k` proportional to that pattern's share, `Σ_i P_ik = ν_k · N`. Write it out:

  minimize over `P`   `<P, L>`
  subject to   `Σ_k P_ik = 1`  for all `i`,
               `Σ_i P_ik = ν_k · N`  for all `k`,
               `P_ik ∈ {0,1}`.

Stare at the constraints. Row sums all equal one, column sums fixed to prescribed totals, nonnegative entries minimizing a linear cost — these are *marginal* constraints on a nonnegative matrix with a linear objective. That is the shape of an optimal-transport problem: transport unit mass from the `N` samples (row marginal all-ones) to the `K` predictors (column marginal `ν_k N`) at cost `L`, `min_{P ∈ U(r,c)} <P, L>` over the transportation polytope `U(r,c) = {P ≥ 0 : P\mathbf{1} = r, P^T\mathbf{1} = c}`. The "keep predictors from collapsing" requirement is *literally the column-marginal constraint* — it forces a prescribed amount of mass into every predictor, so none can be starved to zero. This is the same structure that appears in balanced self-labelling, where minimizing the model's own loss with the labels as free targets collapses everyone to one cluster and the cure is to add an equipartition (balanced-marginal) constraint. Encoding "do not collapse" as a hard marginal constraint rather than a soft penalty lets the cost term `<P, L>` simultaneously pull each sample toward the predictor that actually fits it: balance and correctness in one optimization. I am asserting two things about this transport problem that I should not take on faith — that the column constraint really does prevent collapse even when one predictor is uniformly ahead, and that a cheap Sinkhorn approximation actually produces the marginals I want. I will check both on a small case once I have the solver written down, rather than assume them.

Can I solve it? The constraints `P_ik ∈ {0,1}` make it combinatorial, and an exact transport LP costs about `O(d^3 log d)` — far too slow to solve inside every minibatch of training. The relaxation I need is the entropic one. Replace the hard LP by the regularized transport problem with the same marginals and a smoothing term; its positive solution has the scaling form `P = diag(u) · exp(-L/ε) · diag(v)`, so the cost enters with a minus sign inside the exponential and the vectors `u, v` are just the row and column rescalings needed to hit the marginals. The implementation-friendly form is the same calculation with the sign pushed into the argument: build the fused cost `L_h`, pass `-L_h` into the solver, and inside it compute `exp(Q/ε) = exp(-L_h/ε)`. Then repeat column normalization followed by row normalization, in that order — the column step pushes mass toward equal predictor totals, the row step restores one unit of mass per sample.

Let me actually run this on a tiny case to see whether it does what I claim, because the marginal behavior is the whole point and I have only been asserting it. Take `N = 4` samples, `K = 2` predictors, the cleanly-separated situation: predictor 0 fits samples 0,1 well and 2,3 badly, predictor 1 the reverse. So `L = [[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]]`, and `ε = 0.1`. First `exp(-L/ε)`: with `ε = 0.1` the exponent is `-10·L`, giving roughly `[[0.368, 0.0001], [0.135, 0.0003], [0.0001, 0.368], [0.0003, 0.135]]` — already nearly diagonal-blocked because the spread in `L` is large relative to `ε`. Initial column sums are `[0.504, 0.504]`, row sums `[0.368, 0.135, 0.368, 0.135]` — neither marginal is right yet. Now one iteration. Column-normalize (divide each column by its sum): the two columns become balanced. Row-normalize (divide each row by its sum): the matrix is now `[[0.9997, 0.0003], [0.9975, 0.0025], [0.0003, 0.9997], [0.0025, 0.9975]]`. Reading the marginals off this: row sums are `[1, 1, 1, 1]` exactly (the row step just enforced that), and column sums are `[2, 2]` — exactly `N/K = 4/2 = 2`. Running two more iterations changes nothing to four decimals; it is already at the fixed point. And the assignment is `argmax` per row = `[0, 0, 1, 1]`: each sample landed on its low-loss predictor. So the marginal claim is real — row sums one, column sums `N/K` — and three iterations is plenty here.

That was the easy case where the answer was obvious anyway. The case I actually care about is collapse: what if one predictor is uniformly better, the exact situation that wrecked the naive router? Set `L = [[0.2, 0.5], [0.3, 0.6], [0.4, 0.5], [0.1, 0.7]]` — predictor 0 has lower loss on *every* one of the four samples, so naive "route to the argmin predictor" sends all four to predictor 0 and starves predictor 1. Run the same three Sinkhorn iterations on `-L`. The result is `P ≈ [[0.46, 0.54], [0.46, 0.54], [0.10, 0.90], [0.94, 0.06]]`, with column sums `[1.96, 2.04]` — close to the balanced `[2, 2]` despite predictor 0 dominating the raw losses. So the column constraint did force roughly half the mass into the starved predictor, exactly as intended. And it did not do so blindly: the one row it left on predictor 0 is sample 3 (`argmax → 0`), the sample where predictor 0's *relative* advantage is largest (loss gap 0.1 vs 0.7), while the three samples where the two predictors are closer got handed to predictor 1. That is the behavior I wanted and could not get from a soft penalty — balance enforced as a hard constraint, and within that constraint the assignment still tracks *relative* fit. Good; the OT formulation is doing real work, not just relabeling the problem.

The negative sign is non-negotiable: I confirmed above that passing `-L` makes low loss become high transport mass; passing `+L` would invert it and route every sample to its *worst* predictor. `ε` controls the smoothing; it has to be scaled to the magnitude of the cost entries — in the trace above the spread of `L` was order-1 and `ε = 0.1` gave a crisp near-one-hot assignment, but if `L` entries were order-100 the same `ε` would make `exp(-L/ε)` underflow to all-zeros — which is why I will min-max normalize `L` to `[0,1]` before exponentiating, so `ε = 0.1` always sits at a sane scale.

Now a real obstacle: `P` depends on `L`, and `L` is built from the prediction *errors*, which need the labels. So `P` is a label-dependent oracle assignment — fantastic for training, useless at test, where I have no labels and cannot compute `L`. I cannot ship `P` as the router. But I can use `P` as a *teacher* for the router. The router, fed only LR and TPE (both label-free), should learn to *predict* the assignment that the OT oracle would have produced. That is a supervised classification problem: target distribution `P_i·` (the oracle's one-hot-ish assignment for sample `i`), prediction `q_i` (the router's softmax over predictors), trained by cross-entropy. So I add an auxiliary regularization term to the objective:

  minimize over `Θ, π, ψ`   `E_i [ ℓ(x_i, y_i; Θ, π, ψ) − λ Σ_k P_ik log q_ik ]`.

The first term is the task loss on the routed prediction. The second term uses `reg_i = Σ_k P_ik log q_ik`. I want to make sure the sign in front of it is right, because a flipped sign here would train the router to *flee* the OT target. Cross-entropy of the router prediction `q_i` against target `P_i·` is `CE(P_i, q_i) = −Σ_k P_ik log q_ik`, so `reg_i = −CE(P_i, q_i)`. Substituting, the per-sample objective is `ℓ − λ·reg_i = ℓ − λ·(−CE) = ℓ + λ·CE(P_i, q_i)`. So the minus sign in front of `reg` is exactly what turns "maximize the router's log-likelihood under the OT target" into "add a cross-entropy penalty pulling `q` toward `P`" — minimizing `task_loss − λ·reg` *is* minimizing `task_loss + λ·CE(P, q)`, the thing I want. A quick numeric sanity check on the signs: take a near-one-hot target `P = [0, 1, 0]` and a router prediction `q = [0.2, 0.7, 0.1]`. Then `reg = 1·log(0.7) = −0.357`, `CE = −reg = +0.357`, and `task_loss − λ·reg = 0.5 − (−0.357) = 0.857 = 0.5 + 0.357 = task_loss + λ·CE` (with `λ = 1`, `task_loss = 0.5`). The two forms agree and the penalty is positive when `q` disagrees with `P`, as a penalty should be. For training selection I still use the hard straight-through Gumbel sample, but for this regularizer I use the router's plain probability vector, because the target is a distribution to imitate. This trains the router to imitate the OT assignment using only the current-sample latent and the available error-history state, not the current label. At test time I throw away `P` and the whole loss machinery and just run the router: `argmax(q_i)` picks the predictor without the current label. That is exactly property (c) from the start — the router distills the label-dependent optimal assignment into a selector that can run during inference. In code the regularizer is `reg = prob.log().mul(P).sum(dim=1).mean()` and the total loss is `loss = task_loss − λ · reg`.

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
