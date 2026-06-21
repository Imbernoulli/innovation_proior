Let me start from the claim I actually want to break. The accepted wisdom on data like M4 is that
pure machine learning loses to statistics — the competition's own winner is a hybrid that grafts an
exponential-smoothing model onto a recurrent net, and the pure-ML entrants came in *behind* the
classical benchmarks. If I take that at face value, the conclusion is "you need the statistical
component." But I do not believe the conclusion follows from the evidence. The pure-ML entrants were
mostly off-the-shelf MLPs and RNNs applied with no inductive bias suited to short series; they
overfit. The hybrid wins not necessarily because the statistics are essential, but because *some*
structure — de-seasonalization, level normalization, residual handling — is essential, and the
hybrid happens to supply it through a hand-built statistical core. So the real question is: can I
build those structural biases *into a pure deep network*, with no hand-set seasonality and no
statistical model bolted on, and have the same architecture work across every M4 frequency? That is
the target — a single, frequency-agnostic, pure-deep, direct-multi-step forecaster.

First, the output strategy, because it constrains everything. The horizon is short but not one step,
and a recurrent roll-out — predict one step, feed it back — accumulates error: each step is fed its
own imperfect prediction, and a small per-step bias compounds over the horizon. I would rather emit
the whole horizon at once: a direct-multi-step map from the length-`L` look-back to the length-`H`
forecast, no recursion, no compounding. A feed-forward network whose output layer has `H` units does
exactly this. So the skeleton is a feed-forward network from the look-back vector to the horizon
vector. The whole design question is what to put between input and output so that this network has
the right inductive biases for short, noisy, single series instead of just being a big MLP that
overfits.

The first bias I want is *sequential refinement*. A single network forced to explain the entire
look-back at once has to model trend, seasonality, and noise simultaneously, and it will spend
capacity wherever the loss is largest — exactly the trend/season starvation problem that motivates
classical decomposition. Boosting and deep residual learning both solve a version of this: fit the
target as a *sum of successive corrections*, each new learner handling what the running sum left
unexplained. Let me carry that idea to forecasting. Build the network as a stack of blocks. Each
block looks at a *residual* input — what the previous blocks have not yet explained — and produces
two things: a partial *forecast* (its contribution to the horizon) and a *backcast* (its best
reconstruction of the part of the look-back it just used). The block's backcast is subtracted from
the residual before passing it on, so the next block sees only the leftover signal; the block's
forecast is added into a running forecast sum. After the last block, the residual look-back has been
peeled apart piece by piece and the forecast is the accumulation of every block's contribution.

Let me write the double-residual loop precisely, because the two residual streams are the heart of
it and easy to get backwards. Let `r_0` be the look-back (I will come back to its orientation).
Block `b` receives `r_{b-1}`, computes a backcast `x̂_b` and a forecast `ŷ_b`, and the loop is
`r_b = r_{b-1} − x̂_b` (subtract what this block explained from the look-back residual) and
`y = y + ŷ_b` (accumulate the forecast). The backcast residual flows *backward* — each block strips
its explained component out of the input so downstream blocks face a cleaner, simpler signal — and
the forecast residual flows *forward* as a sum. This double-residual stacking is what makes a very
deep stack trainable (each block only has to model a small correction, like a ResNet) *and* what
makes the decomposition emergent: the look-back is sequentially decomposed, not by a hand-set
moving average, but by whatever each block learns to peel off.

Now, what is inside one block? I want each block to map its residual input to *coefficients*, not
directly to forecast values, and then expand those coefficients through a basis. Here is why that
matters. If a block outputs the `H` forecast values directly, it is unconstrained — it can produce
any horizon, including an overfit wiggle that fits training noise. If instead the block outputs a
small vector of coefficients `θ` and the forecast is `ŷ = V·θ` for a fixed basis matrix `V`, then
the forecast is *constrained to the span of that basis*. Choose the basis to be a meaningful family
and the block can only produce forecasts of that shape. So a block is: a small fully-connected stack
(a few `Linear`+`ReLU` layers of width `W`) that reads the residual look-back and produces a hidden
representation, then two *separate* linear projections from that hidden representation to a backcast
coefficient vector and a forecast coefficient vector, then a basis expansion of each into the
backcast and forecast. The fully-connected stack is the learned part; the basis is the inductive
bias on the output.

What basis? Two designs, and I want both because they answer different needs. The *generic* design
puts no structure on the basis at all: let `θ` itself be the forecast, i.e. the basis is the
identity and the block's forecast projection maps the hidden vector straight to `H` values (and the
backcast projection to `L` values). This is maximally flexible — it is the pure-deep, no-prior
version, and it is what I will lean on when raw accuracy across heterogeneous series matters more
than interpretability. The *interpretable* design constrains each block's basis to a classical
family. For a **trend** block, use a low-degree polynomial basis: the forecast is
`ŷ = Σ_{p=0}^{P} θ_p · t^p` where `t` is normalized time over the horizon, `t = [0, 1, ..., H−1]/H`,
and `P` is small (2 or 3). A low-degree polynomial is monotonic-ish and smooth on the scale of the
horizon — exactly a trend — so a block with this basis *can only* produce a trend, and its `θ` are
the polynomial coefficients, directly interpretable. For a **seasonality** block, use a Fourier
basis: the forecast is a sum of sines and cosines at the harmonics of the horizon,
`ŷ = Σ_l θ_l cos(2π l t) + θ'_l sin(2π l t)`, which can only produce a periodic pattern, and its
`θ` are the harmonic amplitudes. Stack trend blocks then seasonality blocks: the trend stack peels
the trend off the look-back (backcast) and contributes the trend forecast, the seasonality stack
then works on the de-trended residual and contributes the seasonal forecast. That is a *learned*
seasonal-trend decomposition — the same split STL imposes by hand, here emerging from the residual
loop and the constrained bases, with no per-frequency seasonality supplied by me. Crucially, the two
designs share the *exact same* architecture — blocks, double residual, basis expansion — differing
only in the basis matrices; so "one architecture across all frequencies" holds for both.

A detail in the basis expansion that I must get right: the forecast and backcast share the block's
hidden representation but use *different* basis matrices, because the backcast must span the
look-back length `L` and the forecast must span the horizon length `H`. For the generic block that
is two independent linear maps (`hidden → L` and `hidden → H`). For the interpretable blocks the
polynomial/Fourier templates are computed over `L` normalized time points for the backcast and `H`
for the forecast, and the `θ` split into a backcast part and a forecast part. The backcast's only
job is to clean the residual for downstream blocks — its accuracy is a means, not the end — but it
must use the *same* basis family as the forecast so that the component the block removes from the
input is the same *kind* of component it adds to the forecast (a trend block removes a trend and
predicts a trend). That coupling is what makes the emergent decomposition coherent.

Now the input orientation, which I flagged. The most recent look-back values matter most, and I want
the network to read them in a stable position regardless of `L`. Reverse the look-back so the most
recent point is first; feed that reversed vector to the blocks. (If the harness supplies a mask for
missing/padded history, the backcast subtraction is masked too, so padded positions never pollute
the residual.) This is a small thing but it stabilizes training across series of different lengths.

There is one more structural choice that turns out to matter as much as the blocks: organize blocks
into *stacks*, and let each stack carry one basis type, and additionally let blocks *within a stack
share weights*. Weight-sharing inside a stack is a strong regularizer — it forces every block in,
say, the trend stack to use the same fully-connected map, so the stack as a whole learns one trend
operator applied residually rather than many independent ones — and on short M4 series, where
overfitting is the enemy, that regularization is exactly what the pure-ML entrants lacked. The
generic configuration uses several stacks of generic blocks; the interpretable configuration uses
one trend stack followed by one seasonality stack.

Let me reason about the objective. M4 is scored by sMAPE and MASE; I should train on the metric I am
judged by rather than on raw squared error, because the series live at wildly different magnitudes
and an MSE would let the few large series dominate. sMAPE, `(200/H) Σ |y − ŷ| / (|y| + |ŷ|)`, is
scale-free per series, so training on it (or on a close relative like MAPE/MASE) puts every series
on equal footing. Direct-multi-step, so the loss is computed over the whole horizon at once, which
is what the DMS output was for.

Finally, robustness across the 100,000 series. A single trained model is sensitive to
initialization and to the chosen look-back length on short data. The clean fix that does not touch
the architecture is to *ensemble*: train the model at several look-back lengths (multiples of the
horizon, `L = 2H, 3H, ..., 7H`) and with several random initializations and several loss functions
(sMAPE, MASE, MAPE), and average the forecasts (median or mean). Each member is the same pure-deep
architecture; the ensemble just averages away the variance that a single short-series fit carries.
This is a property of the *training procedure*, not a new model component, and it keeps the "pure
deep, one architecture" claim intact while delivering the stability the statistical ensembles got
from combining many distinct methods.

Step back and check the thesis. The accepted wisdom said pure ML loses to statistics on M4. My
architecture is pure deep — fully-connected blocks, no statistical core, no hand-set seasonality —
yet it carries every structural bias that made the hybrid work: direct-multi-step output (no error
accumulation), sequential residual refinement (each block a small correction, like boosting/ResNet),
basis-constrained outputs (which, in the interpretable configuration, *emerge* as a learned
trend/seasonality decomposition), weight-sharing within stacks (regularization for short series),
training on the percentage metric, and ensembling over look-back/init/loss for stability. None of
those is statistics; all of them are the structure that short-series forecasting rewards. If the bet
is right, the same network — unchanged across Yearly through Hourly — should match or beat the hybrid
winner *and* expose an interpretable trend and seasonality as a free by-product, refuting "you need
the statistical component." The interpretable configuration trades a little accuracy for that
transparency; the generic configuration, with several generic stacks, is the one to reach for when
raw accuracy across heterogeneous series is what matters, since the residual loop alone — even
without imposed bases — already supplies the decomposition-by-refinement that the data wants.

Let me write it as the architecture I would ship: a stack-of-blocks module where each block is an
FC stack feeding two basis projections, wired by the double-residual loop, in both the generic and
interpretable basis variants.
