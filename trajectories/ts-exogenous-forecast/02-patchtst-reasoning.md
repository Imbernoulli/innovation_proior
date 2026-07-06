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

Let me put real numbers on "far fewer," because the whole efficiency argument lives there. With $L=96$,
$P=16$, $S=8$: `PatchEmbedding` first replication-pads $S=8$ steps on the end, taking the length to
$104$, then unfolds width-16 windows at stride 8, giving $\lfloor(104-16)/8\rfloor+1 = 11+1 = 12$ patch
tokens — which is exactly the $\text{int}((96-16)/8 + 2)=12$ that the flatten head's `head_nf`
$=d_{model}\cdot 12$ assumes, so the head lines up. Twelve tokens instead of ninety-six. Point-wise
attention builds a $96\times 96$ score matrix, $9216$ entries per channel per layer; patch attention
builds $12\times 12$, $144$ entries — a $64\times$ reduction, and quadratic, so if I ever wanted a
longer look-back the gap only widens. The stride-8 half-overlap between adjacent 16-step patches is a
deliberate small redundancy: a shape that straddles a patch boundary still appears whole inside a
neighboring window, so I am not arbitrarily severing local structure at the cuts. Each patch is embedded
by a shared $\mathbb{R}^{16}\to\mathbb{R}^{128}$ linear map plus a positional embedding, and the twelve
tokens run through the encoder.

It is worth being explicit about what a query-key dot product now *computes*, because that is where the
tokenization change pays off. Each patch token is a $128$-dim embedding of a 16-step local shape, so the
score between patch $a$ and patch $b$ is a learned similarity between two local waveforms — "does the
shape around step $a$ resemble the shape around step $b$?" That is a matched-filter operation over
shapes, exactly the thing a forecaster wants: it can learn that the ramp at the start of a daily cycle
predicts the peak eight patches later, and route information from the one to the other. With only 12
tokens, the attention head has a small enough set that it can learn a sharp, near-deterministic routing
rather than a diffuse average — a further reason the collapsed token count is a feature, not just a cost
saving, and one that a 96-token point-wise map, spreading its softmax mass over eight times as many
mostly-meaningless positions, could never enjoy. Contrast the
point-wise score, which asked "does the scalar at step $a$ equal the scalar at step $b$?" — a comparison
of two structureless numbers whose answer is dominated by noise. Same attention kernel, same softmax;
the only thing I changed is that the objects being compared went from meaningless to meaningful, and
that is the entire thesis of the rung.

I owe myself one honest caveat about the efficiency half of the patching argument, because the substrate
fixes `seq_len=96` and I cannot lengthen it. The "cheap long window" benefit — that collapsing $L$ into
$L/S$ tokens lets you afford a much longer look-back — is real but *latent* here; I am not allowed to
spend it. So the win this rung actually claims on these three datasets cannot be "I fed the model more
history." It must be the *semantic* half alone: at the same fixed 96-step look-back, patch tokens carry
meaningful local shapes that point-wise tokens did not, so attention finally has the right objects to
compare. If PatchTST beats DLinear here, that improvement is attributable purely to better token
semantics, not to a longer window — which keeps this a clean comparison against the linear floor. Embed each patch with a shared linear map $\mathbb{R}^{P}\to\mathbb{R}^{d}$ plus a positional
embedding, run a stack of vanilla Transformer encoder layers over the patch tokens (the substrate's
`Encoder`/`EncoderLayer`/`FullAttention`), and flatten the resulting $N$ patch representations through
a linear head to the 96-step horizon.

The crucial design decision — the one that keeps this rung a *clean control* — is that the backbone is
**channel-independent**: the same patch-embedding, the same encoder, the same head are applied to every
channel separately, and channels never attend to each other. I implement that by folding the channel
axis into the batch: reshape $(B,L,C)$ so each channel becomes its own sequence of patch tokens, run
the shared backbone, then reshape back. Let me trace the shapes so I am sure no channel ever attends to
another: $x_{enc}$ is $(B,96,C)$; permute to $(B,C,96)$; `PatchEmbedding` folds the channel axis into
the batch and patches, giving $(B\cdot C, 12, 128)$; the encoder runs self-attention over the $12$-token
axis only, so its attention map is per-$(B\cdot C)$-row — on ECL that is $B\cdot 321$ independent
length-12 sequences in one forward pass, none of which share an attention score. Reshape back to
$(B,C,128,12)$, and the flatten-head maps $128\cdot 12 = 1536$ down to the $96$-step horizon per channel.
The channel axis passed through as a pure batch dimension from start to finish; the exogenous covariates
are as invisible to the target here as they were in DLinear. This is deliberate. It does for the
temporal model what
DLinear could not — patches plus attention give a strong *nonlinear, multi-scale* read of one channel's
own dynamics — while holding the channel-independence axis fixed at exactly where DLinear had it. So
whatever this rung gains over DLinear is attributable to *better per-channel temporal modeling*, not to
fusion. That is the whole reason it comes before the cross-variate rung.

I should size the jump in cost I am accepting, because "add a Transformer" can quietly mean a hundredfold
more parameters and I want it to be a small model. With $e\_layers=3$, $d\_model=128$, $d\_ff=256$: each
encoder layer is roughly $4d^2$ for the attention projections ($4\cdot128^2 = 65{,}536$) plus $2\cdot d
\cdot d\_ff$ for the feed-forward ($2\cdot128\cdot256 = 65{,}536$), about $131$k per layer, so three
layers is $\approx 393$k. The flatten head is $1536\cdot96 \approx 147$k, and the patch embedding is a
tiny $16\cdot128 = 2048$. Total is on the order of $0.55$ million parameters — roughly $30\times$
DLinear's $18.6$k, but still a small model, and crucially it is *shared across all channels and
independent of $C$*, so it is the same size on ETTh1's 7 channels and ECL's 321. That is the right kind
of scaling: I am spending the extra capacity on the temporal model, once, not on the channel count.

Two pieces of plumbing I keep because they are known to matter and cost little. First, **instance
normalization** in the Non-stationary-Transformer style: subtract the per-instance mean and divide by
the per-instance std of the look-back before patching, then add them back after the head. These
benchmarks have heavy distribution shift between train and test windows, and normalizing each instance
removes the shift the model would otherwise waste capacity chasing — and importantly it is applied
per-channel, so it does not smuggle in any cross-channel coupling. Let me check that last claim in the
shapes, because if the normalization leaked across channels it would quietly break the control. The mean
is `x_enc.mean(1, keepdim=True)`, reducing over the *time* axis only, so it has shape $(B,1,C)$ — one
mean and one std per channel, computed from that channel's look-back alone. The subtract, the divide, and
the symmetric de-normalization after the head (`stdev[:,0,:]` and `means[:,0,:]` broadcast over the 96
horizon steps) all act channel-wise. So a channel is centered and scaled by its own statistics and never
sees another's — the normalization respects channel-independence exactly, and it happens *before* the
channel axis is folded into the batch, so each of the $B\cdot C$ folded sequences arrives already
standardized. Second, **BatchNorm** (in the
transpose-BN-transpose form) instead of LayerNorm inside the encoder, which is the reference choice for
patch tokens and trains more stably here. The reason is a property of the tokens: patch embeddings vary
enormously in magnitude across a batch — a flat, quiet window embeds to a small-norm token, a steep ramp
to a large one — and LayerNorm, which normalizes *within* each token across its 128 features, leaves that
between-token magnitude spread intact. Transpose-BatchNorm instead normalizes each of the 128 features
*across* the token-and-batch axis, so it explicitly centers and scales away that spread, which is why it
converges more smoothly on patch tokens than LayerNorm does. With `e_layers=3`, `n_heads=4`,
`d_model=128`, `d_ff=256`,
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

There is a second reading of these numbers I want prepared in advance, because it is the one that
licenses the *next* rung rather than this one. Even in the best case where PatchTST beats DLinear on all
three, the shape I am predicting is that the ETTh1 improvement roughly *exhausts* that dataset's
headroom — the residual there was almost all weak-temporal-model, and a strong per-channel model should
take most of it — whereas on Weather and ECL a visible residual should *remain* after the temporal
upgrade, because no amount of per-channel modeling can manufacture the covariate signal. So the
diagnostic on this feedback is not just "did it beat DLinear" but "where did a gap survive": if the gap
closes on ETTh1 and persists on Weather and ECL, that surviving gap is the fusion gap, now cleanly
separated from the temporal-model gap that this rung just paid off. That separation is the whole reason
this rung exists between the linear floor and the first cross-variate model — it converts an ambiguous
"DLinear is loose everywhere" into a specific, addressable "the remaining looseness on Weather and ECL
is exactly the missing exogenous signal," which is the mandate the next rung inherits.
