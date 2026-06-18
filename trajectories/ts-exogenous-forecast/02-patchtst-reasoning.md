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

The first thing it gives up is cross-channel information — the exogenous signal. The second thing it
gives up is any *nonlinear* read of the target's own temporal structure: the linear map is a single
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
without looking at other channels? The obvious reach is a Transformer, but I have to be careful,
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
steps carry overlapping, compressible structure rather than independent information. So the move is
staring at me. If neighboring steps are redundant and a single step is meaningless, don't tokenize per
step — group a local stretch of steps into one token.

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
shape. Embed each patch with a shared linear map $\mathbb{R}^{P}\to\mathbb{R}^{d}$ plus a positional
embedding, run a stack of vanilla Transformer encoder layers over the patch tokens (the substrate's
`Encoder`/`EncoderLayer`/`FullAttention`), and flatten the resulting $N$ patch representations through
a linear head to the 96-step horizon.

The crucial design decision — the one that keeps this rung a *clean control* — is that the backbone is
**channel-independent**: the same patch-embedding, the same encoder, the same head are applied to every
channel separately, and channels never attend to each other. I implement that by folding the channel
axis into the batch: reshape $(B,L,C)$ so each channel becomes its own sequence of patch tokens, run
the shared backbone, then reshape back. This is deliberate. It does for the temporal model what
DLinear could not — patches plus attention give a strong *nonlinear, multi-scale* read of one channel's
own dynamics — while holding the channel-independence axis fixed at exactly where DLinear had it. So
whatever this rung gains over DLinear is attributable to *better per-channel temporal modeling*, not to
fusion. That is the whole reason it comes before the cross-variate rung.

Two pieces of plumbing I keep because they are known to matter and cost little. First, **instance
normalization** in the Non-stationary-Transformer style: subtract the per-instance mean and divide by
the per-instance std of the look-back before patching, then add them back after the head. These
benchmarks have heavy distribution shift between train and test windows, and normalizing each instance
removes the shift the model would otherwise waste capacity chasing — and importantly it is applied
per-channel, so it does not smuggle in any cross-channel coupling. Second, **BatchNorm** (in the
transpose-BN-transpose form) instead of LayerNorm inside the encoder, which is the reference choice for
patch tokens and trains more stably here. With `e_layers=3`, `n_heads=4`, `d_model=128`, `d_ff=256`,
`patch_len=16`, `stride=8`, dropout 0.1 — the standard configuration — the model is small and fast. The
full scaffold module is in the answer.

Now the falsifiable expectations against DLinear's measured numbers, dataset by dataset. On **ETTh1**,
where fusion has little to add, this rung should *still beat* DLinear's 0.0644 / 0.1878 — and if it
does, that improvement is pure evidence that the per-channel temporal model was the bottleneck on ETTh1,
exactly the residual gap I flagged in the linear result; I expect ETTh1 MSE to drop into the high
0.05s. On **Weather**, where the target leans hard on its covariates, I expect a real improvement over
DLinear's 0.005652 simply because attention-over-patches is a far better temporal model — but I also
expect this rung to *leave a gap*, because it still cannot read the other channels; that residual gap
on Weather is precisely the quantity the next, cross-variate rung must close, and I want it on the
record. On **ECL** I expect the biggest absolute drop from DLinear's 0.3873, since 321 clients give the
shared linear map the most trouble and a patch-attention temporal model the most room — but again with
a residual that only cross-client fusion can take. If instead this rung *fails* to beat DLinear on
ETTh1, my diagnosis was wrong and the linear floor was already the best per-channel model, which would
send the next rung straight to fusion with no temporal upgrade. I do not expect that, but the ETTh1
comparison is the experiment that decides it.
