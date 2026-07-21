The linear floor told me exactly what I suspected it would, and it told me in numbers. On **ETTh1** the
decomposition-linear map landed MSE 0.0644 / MAE 0.1878 — respectable, the smooth oil-temperature
target is largely recoverable from its own history, just as I predicted. But on **Weather** it sat at
MSE 0.005652 / MAE 0.0627 and on **ECL** at MSE 0.3873 / MAE 0.4512, and those two are loose in
precisely the way I set up the control to expose. The Weather MSE is an order of magnitude above the
small numbers I will get once the model can actually read the other channels; the ECL number is the
worst of the three relative to what these panels are known to support. So the channel-independence I
built in on purpose is now visibly costing me on exactly the datasets where the target leans on its
covariates — that part of the diagnosis is clean. But before I jump straight to cross-channel fusion, I
want to read the result more carefully, because there are *two* things a channel-independent linear map
gives up, and conflating them would send me up the wrong rung.

Before I even split the two effects apart, let me squeeze the three numbers for what they say on their
own, because the *texture* of the error is a clue to its cause. Convert each MSE to an RMSE and compare
it to the MAE: on ETTh1, $\sqrt{0.0644}=0.254$ against MAE $0.1878$, a MAE/RMSE ratio of $0.74$; on ECL,
$\sqrt{0.3873}=0.622$ against $0.4512$, ratio $0.73$; on Weather, $\sqrt{0.005652}=0.0752$ against
$0.0627$, ratio $0.83$. A Gaussian error would sit near $0.80$; a heavy-tailed error dominated by a few
big misses pulls the ratio *down* (RMSE inflates faster than MAE). So ETTh1 and ECL carry relatively
more large, spiky errors, while Weather's error is comparatively *uniform* — the model is wrong by a
similar amount almost everywhere. That last part is the tell I care about: a broadly uniform error is
the signature of a *systematic* miss, a bias the model cannot represent, not a handful of hard windows.
And a systematic miss on Weather is exactly what I would expect if the target is being driven by
covariates the channel-independent map cannot see — it is off by a roughly constant push at each step
because it is missing an input, not because a few windows are intrinsically hard. That reading survives
only as a hypothesis, but it points the same direction as the design argument, and it is the kind of
thing I want on the record before I choose the next rung.

Now to the two effects themselves. The first thing the channel-independent map gives up is cross-channel
information — the exogenous signal. The second thing it gives up is any *nonlinear* read of the target's
own temporal structure: the linear map
is a single
affine combination of the look-back, and that is a genuinely limited hypothesis class for a target
whose own dynamics are nonlinear and multi-scale. The ETTh1 number is the tell here. ETTh1's target is
smooth and strongly autocorrelated, and its panel is small and homogeneous, so cross-channel fusion has
the *least* to add there — yet 0.0644 is not the floor ETTh1 can reach. That residual gap on the one
dataset where fusion barely matters is telling me there is room left in the *per-channel temporal
model itself*, before I touch channels at all. If I leapt to a cross-variate machine now and it beat
DLinear, I would not know whether the win came from fusion or just from finally giving the target a
nonlinear temporal model. So the disciplined next rung is the one that upgrades the temporal model of
each channel to its strongest channel-independent form, and *holds the channel-independence fixed* — a
clean control that isolates "better per-channel temporal modeling" from "cross-channel fusion." Only
after I know what the best channel-independent forecaster gets can I attribute the next gap to fusion.

So the question becomes: what is the strongest way to model one channel's temporal structure, still
without looking at other channels? There is an obvious cheaper option I should dispatch first, because
if it works I do not need a Transformer at all: just make the linear map nonlinear — an MLP along time,
$96\to h\to 96$ with a GELU in the middle. It is strictly more expressive than DLinear's affine map and
it stays perfectly channel-independent. But walk it a step and it stalls. An MLP still swallows the
length-96 window as one flat vector and produces the horizon as one flat vector; it can bend the map
nonlinearly, but it has no operation that *compares* a sub-shape near the start of the window with a
similar sub-shape near the end — the thing I actually want when a ramp recurs each cycle. Its first
layer is also $96\cdot h$ weights, so a longer look-back costs linearly in width with no structural
payoff, and with $h$ large enough to matter it just overfits these small panels. The MLP buys
nonlinearity but not *relational, multi-scale* temporal modeling, which is the specific capability the
linear floor is missing. So I keep looking, and the reach is a Transformer — but I have to be careful,
because the embarrassing fact behind this whole ladder is that the linear map I just ran *beats* the
deployed temporal-attention Transformers — Informer, Autoformer, FEDformer — on these benchmarks. If
attention over the time axis were earning its keep, a plain linear map would have no business winning.
Either attention is the wrong tool for this data, or — and this is the possibility I want to chase — we
have been feeding the series into it in a way that destroys the structure attention is good at finding.
I refuse to touch the attention kernel. I want to ask instead: what is the *token*, and is the token
the bug?

Look at what a token is in every one of those models. It is a single time step — one scalar at time
$t$ for one channel. Sit with that, because I think it is broken. In language a token is a word, a
coherent semantic unit, and attention between two words compares two meanings. What is the meaning of
a sensor's value at 14:03? Nothing, in isolation — less than a character, which at least belongs to a
small alphabet. The information in a series lives in *shapes over short stretches*: a rising edge, a
dip, a local oscillation, the slope of a ramp. Point-wise attention asks "how does the scalar at 14:03
relate to the scalar at 09:17?" and the answer is almost always noise, because neither scalar means
anything by itself. The attention map is being computed over the wrong objects. That would explain the
linear model winning: the linear map reads the whole window at once and so it gets to see the *shape*,
while point-wise attention never does. The failure is a *tokenization* failure, not an attention
failure.

There is a second symptom pointing the same way, and it matters for the look-back the substrate fixed
at 96. With one token per step, the token count $N$ equals the sequence length $L$, the attention map
is $L\times L$, and lengthening the look-back is quadratically punished — so everyone defaults to short
windows and throws away older history that a forecaster could use. And people have found that you can
*down-sample* the window — take every fourth step — and still forecast about as well, sometimes better
at the same token count. Read that carefully: it says the time axis is *redundant*, that neighboring
steps carry overlapping, compressible structure rather than independent information. Put a number on it: down-sampling
96 steps by four leaves 24 tokens, a $16\times$ smaller attention map, and the forecast barely moves —
which is only possible if roughly three of every four steps were carrying information their neighbors
already had. That is the same compression a stride-8 patching exploits when it turns 96 steps into 12
tokens, except patching keeps the discarded detail *inside* each token rather than throwing it away. So
the move is staring at me. If neighboring steps are redundant and a single step is meaningless, don't
tokenize per step — group a local stretch of steps into one token.

This is exactly what vision did at the same wall. An image has $H\times W$ pixels; one pixel is
meaningless and per-pixel attention is hopeless, so the Vision Transformer cut the image into $16\times
16$ patches, called each patch a token, and the token count collapsed while each token gained a local
visual concept. The analogy to a series is immediate: cut the length-$L$ series into contiguous
sub-series **patches**, and let each patch be a token. A patch of, say, sixteen consecutive steps is a
little shape — a ramp, a bump, a level — which is the kind of thing attention should be comparing. Make
it concrete: take one channel's history $x^{(i)}\in\mathbb{R}^{L}$, pick a patch length $P=16$ and a
stride $S=8$, slide a width-$P$ window in steps of $S$, and each placement is one patch in
$\mathbb{R}^{P}$. The patch count is $N=\lfloor(L-P)/S\rfloor+1$ plus one for the end padding the
embedding adds — far fewer than $L$, so attention is cheap, *and* each token is now a meaningful local
shape.

Concretely, with $L=96$, $P=16$, $S=8$, `PatchEmbedding` replication-pads $S=8$ on the end and unfolds
width-16 windows at stride 8, giving $\lfloor(104-16)/8\rfloor+1 = 12$ patch tokens — which is exactly
the $12$ the flatten head's `head_nf` $=d_{model}\cdot 12$ assumes, so the head lines up. Twelve tokens
instead of ninety-six collapses the attention map from $96\times96$ to $12\times12$, and the stride-8
half-overlap keeps a shape that straddles a patch boundary whole inside a neighboring window, so I am
not severing local structure at the cuts. Each patch is embedded by a shared $\mathbb{R}^{16}\to
\mathbb{R}^{128}$ linear map plus a positional embedding, and the twelve tokens run through the encoder.

And the query-key dot product now computes the right thing: each patch token embeds a 16-step local
shape, so the score between patch $a$ and patch $b$ is a learned similarity between two local waveforms
— a matched-filter over shapes that can learn the ramp starting a daily cycle predicts the peak eight
patches later. With only 12 tokens the head can learn a sharp routing rather than spreading its softmax
over ninety-six mostly-meaningless positions. Same attention kernel, same softmax; the only change is
that the compared objects went from meaningless scalars to meaningful shapes — the entire thesis of the
rung.

One honest caveat about the efficiency half of the argument: the substrate fixes `seq_len=96`, so the
"cheap long window" benefit — collapsing $L$ into $L/S$ tokens to afford more history — is real but
*latent* here; I cannot spend it. The win this rung claims must be the *semantic* half alone: at the
same 96-step look-back, patch tokens carry meaningful local shapes that point-wise tokens did not. If
PatchTST beats DLinear, that gain is attributable to better token semantics, not a longer window, which
keeps the comparison clean. Run the patch tokens through a stack of vanilla encoder layers (the
substrate's `Encoder`/`EncoderLayer`/`FullAttention`) and flatten to the 96-step horizon.

The crucial design decision — the one that keeps this a *clean control* — is that the backbone is
**channel-independent**: the same patch-embedding, encoder, and head are applied to every channel
separately, implemented by folding the channel axis into the batch so each channel becomes its own
sequence of patch tokens, run through the shared backbone, then reshaped back. On ECL that is $B\cdot
321$ independent length-12 sequences in one forward pass, none of which share an attention score — the
covariates are as invisible to the target here as in DLinear. This is deliberate: it does for the
temporal model what DLinear could not — patches plus attention give a strong nonlinear, multi-scale read
of one channel's own dynamics — while holding channel-independence fixed exactly where DLinear had it.
So whatever this rung gains over DLinear is attributable to *better per-channel temporal modeling*, not
fusion. That is why it comes before the cross-variate rung.

The cost is modest: with $e\_layers=3$, $d\_model=128$, $d\_ff=256$, on the order of $0.55$M parameters
— roughly $30\times$ DLinear's $18.6$k but still small, and crucially shared across all channels and
*independent of $C$*, so the same size on ETTh1's 7 channels and ECL's 321. The extra capacity goes into
the temporal model, once, not the channel count.

Two pieces of plumbing I keep because they matter and cost little. First, **instance normalization** in
the Non-stationary-Transformer style: subtract the per-instance mean and divide by the per-instance std
of the look-back before patching, add them back after the head, to remove the train/test distribution
shift the model would otherwise chase. It reduces over the *time* axis only, giving one mean and std per
channel — so it stays per-channel and smuggles in no cross-channel coupling, and it happens before the
channel axis folds into the batch. Second, **BatchNorm** (transpose-BN-transpose form) instead of
LayerNorm inside the encoder. Patch embeddings vary enormously in magnitude across a batch — a flat
window embeds to a small-norm token, a steep ramp to a large one — and LayerNorm, normalizing *within*
each token across its 128 features, leaves that between-token spread intact; transpose-BatchNorm
normalizes each feature *across* the token-and-batch axis, centering that spread away, which converges
more smoothly on patch tokens. With `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`, `patch_len=16`,
`stride=8`, dropout 0.1, the model is small and fast. The full scaffold module is in the answer.

Now the expectations against DLinear's measured numbers, as directions I can check. On **ETTh1**, where
fusion has little to add, this rung should *still beat* DLinear's 0.0644 / 0.1878 — and if it does, that
gain is evidence the per-channel temporal model was the ETTh1 bottleneck, the residual I flagged in the
linear result. On **Weather**, a real improvement over DLinear's 0.005652 because attention-over-patches
is a far better temporal model, but a residual should *remain* because it still cannot read the other
channels — that residual is the quantity the next, cross-variate rung must close. On **ECL** I expect
the biggest absolute drop from DLinear's 0.3873, since 321 clients give the shared linear map the most
trouble, again with a residual only cross-client fusion can take. If instead this rung *fails* to beat
DLinear on ETTh1, my diagnosis was wrong and the linear floor was already the best per-channel model —
that comparison is the experiment that decides it. And the diagnostic that licenses the *next* rung is
not "did it beat DLinear" but "where did a gap survive": if it closes on ETTh1 and persists on Weather
and ECL, that surviving gap is the fusion gap, now cleanly separated from the temporal-model gap this
rung paid off.
