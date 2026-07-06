Patched temporal attention came back and it resolved the fork I left open. ETTh1 dropped to MSE 0.3794
/ MAE 0.3986 — below the linear control's 0.3962, on the very dataset the linear model was strongest on,
which is the clean confirmation I asked for: attention over time, when the token is a local patch shape
instead of a meaningless single step, *does* extract more than a linear temporal map. So temporal
modeling was not exhausted; it was mis-tokenized. Weather fell hard too, 0.1962 → 0.1738, exactly where
patch shapes plus instance norm should help the noisy data. The mean MSE went 0.2676 → 0.2451, a 0.0225
drop. Reading the per-dataset moves confirms the mechanism rather than just the headline: ETTh1 fell
0.0168 (about 4% of its value), Weather fell 0.0224 (about 11%), and both are the datasets where the
temporal-only fix should register, so patched temporal attention paid where the theory said it would. But the
diagnosis that actually decides the next rung is ECL: 0.2104 → 0.1819. It improved — in absolute terms
the largest drop of the three — and yet it is still the dataset carrying the most unmodeled structure,
and the tell is the same one the linear control showed: its MAE stays disproportionately high relative to
its MSE. On ECL the MAE of 0.2743 sits about 51% above the MSE of 0.1819, versus roughly 24% on Weather
(0.2157 over 0.1738) and 5% on ETTh1 (0.3986 over 0.3794). A pervasive MAE-over-MSE gap is the fingerprint
of a systematic, spread-out bias rather than a few large misses — exactly what a channel-independent model
leaves on the 321-channel dataset by nudging every client the same wrong way — and it is the one symptom
patching did not close, just as it did not for the linear map. That is the signal I predicted would tell
me where to go: ECL is the 321-channel dataset, and PatchTST, like the linear control, is still
channel-independent — it never lets one channel inform another. The temporal lever is now well used and
the channel lever is untouched. So the next rung is not a better temporal operator; it is finally
modeling cross-variate structure.

The question is *how* to add cross-variate attention without throwing away what just worked. Three
routes are on the table and I want to reason each to its cost. The first keeps the per-step token and
lets its N-dimensional variate slice do the channel mixing implicitly — but that is exactly the layout
already in use across the prior lineage, and its ECL residue is the evidence it does not surface channel
correlation cleanly, so it is not a fix, it is the status quo I am trying to escape. The second is the
two-stage route: run temporal attention along time and then a separate channel attention across variates,
alternating the two, which is what a Crossformer-style design does. It is explicit about channels, but it
pays both an O(T²) temporal cost and an O(N²) channel cost at every layer and, worse, it keeps the
per-step temporal token whose incoherence I am about to indict — so it inherits the disease while doubling
the attention budget, and on ECL's 321 channels the channel stage alone is 321² per layer stacked on top
of the temporal one. The third route is more radical and cheaper: drop the temporal attention entirely and
run attention *only* across variates, by changing what a token is rather than adding a second attention.
Before I can trust that third route I have to look again at what a temporal token even is, because I think the layout
itself is the bug, and PatchTST only sidestepped it by going channel-independent. In the standard
layout I have a panel X ∈ R^{T×N}: T timesteps, N variates. The reflex from language is "a token per
position in the sequence," and the sequence is time, so the token is the slice X_{t,:} ∈ R^N — all N
variates at one instant t — embedded into a d_model vector, giving T tokens, self-attention over time.
It feels natural by analogy. But look at what is *inside* one of those tokens. A word token is a
coherent semantic unit; the embedding has something real to represent. X_{t,:} is whatever every sensor
happened to read at the same wall-clock instant: in ECL, electricity consumption from 321 different
clients; in Weather, temperature next to rainfall next to pressure — different physical quantities,
different units, different distributions, jammed into one vector. Worse, they are not even synchronized
in the way that matters: an event hits one channel and only later the next, so the "same timestamp"
lumps together different phases of the same process across channels. The temporal token is a fruit salad
of time-misaligned, incommensurable numbers with a receptive field of a single instant. There is barely
any temporal information inside it; the temporal content lives *across* tokens, and the model is asked to
reconstruct it from a sequence of scrambled snapshots.

This also explains a harm I had not named. LayerNorm in this layout normalizes across the feature
dimension of a token — which here is the variate mixture at fixed t. So at every timestamp the model
centers and scales temperature against rainfall against pressure together, blending channels that have
nothing to do with each other; that is not removing nuisance variation, it is injecting interaction
noise between unrelated, possibly lagged processes, and across timesteps it oversmooths. And the
attention map over T temporal tokens reports which *instants* resemble which other instants in this
scrambled feature space — not which *variables* drive which, which is the multivariate structure ECL is
screaming for. Make the normalization harm concrete on Weather: at a fixed timestamp the token holds
temperature, pressure, humidity, wind, rainfall, all in one vector, and LayerNorm subtracts their shared
mean and divides by their shared spread — so a hot dry hour and a cool wet hour get their *cross-quantity*
contrast rescaled together as if temperature and rainfall were commensurable readings of one thing. On
ECL it is milder in kind but worse in degree: 321 clients whose only commonality is being electricity are
centered against each other every hour, so a single client's genuine level is contaminated by whatever
the other 320 happened to be doing at that instant. Neither is nuisance removal; both are structured
interference injected by normalizing along the wrong axis.

There is a cleaner way to see the temporal axis is the wrong place for attention. Attention is
permutation-invariant in its tokens by construction — softmax(QKᵀ/√d)V does not know the order its
tokens came in, which is exactly why language bolts on positional encodings. But time *has* order;
shuffling timesteps destroys the series. So a permutation-invariant operator on the temporal axis is a
structural mismatch — I am fighting the operator's own symmetry and patching it with positional
encodings just to undo the damage. PatchTST mitigated this by making each token a local patch (so the
within-patch order is preserved inside the token and only N≈12 ordered patches need positions), but it
still ran attention along time and still refused channels. The ECL result says that refusal is now the
binding constraint.

So I ask the inverse question. The thing with order I should not permute is the time axis. The thing
*without* an inherent order, where permutation invariance is *correct*, is the set of variates — channel
5 and channel 12 have no canonical ordering. What if I make the token the whole series of one variate
and run attention *across* variates? Take the column X_{:,n} ∈ R^T — the entire look-back of channel n —
and embed *that* into a d_model vector h_n with one linear map R^T → R^D. Now I have N tokens, one per
variate, each a description of a single physically coherent series over its whole history. The receptive
field of a token is no longer one instant; it is the entire look-back window. This is the extreme of
patching: PatchTST enlarged the receptive field by grouping a handful of consecutive steps into a token;
push that all the way and the patch is the complete series. Each token finally has something real and
homogeneous to represent, and — crucially — attention between two of these tokens is now a clean
cross-variate correlation: h_i · h_j measures how channel i's whole history relates to channel j's,
which is exactly the structure ECL needs and which neither the linear control nor PatchTST could touch.

I should be honest about what this does to the attention budget, because the inversion is not free — it
relocates the cost. Temporal-token attention is O(T²) = 96² = 9216 pairwise scores per layer, fixed by
the look-back. Variate-token attention is O(N²), and N is the channel count: 7² = 49 on ETTh1, 21² = 441
on Weather, but 321² = 103041 on ECL — over eleven times the temporal cost there. So on the wide dataset
the inversion is actually the *more* expensive operator per layer, and I accept that trade knowingly:
the whole point is that ECL's error is cross-channel, and paying 321² to model 321 coupled channels is
spending the budget on the axis that carries the signal, whereas the 96² temporal attention was spending
it on the axis PatchTST already handled. The compensating structural win is that this cost is independent
of T — lengthening the look-back only widens the R^T → R^D embedding's input, never the attention map —
so unlike the temporal-token regime, more history is nearly free, which is the opposite of the wall that
made the field default to short windows.

Three things fall out of the inversion and each is a fix to a harm I just named. First, attention now
runs over the unordered variate set, so permutation invariance is *correct* and I do not need a temporal
positional encoding at all — the structural mismatch is gone, not patched. Second, LayerNorm now
normalizes across the feature dimension of a *variate* token — the learned representation of one
channel's series — which is a sensible per-series normalization, not a cross-channel blend; the
interaction noise I diagnosed is removed by construction. Third, the cost is O(N²) in the number of
*variates*, independent of T; lengthening the look-back no longer blows up attention (it only widens the
embedding's input), so the model can use more history cheaply — the opposite of the temporal-token
regime where longer windows hurt.

I want to verify the permutation claim concretely, because it is the load-bearing symmetry argument.
Suppose I relabel the channels — swap channel 5 and channel 12 before embedding. In the variate-token
layout that just swaps two rows of the token matrix; self-attention, being permutation-equivariant, then
produces the same two output tokens swapped, and the per-token projection maps each to the same horizon
it would have had under the original labels, so the final forecast for every channel is identical up to
the relabeling. That is the correct behavior: channel identity carries no order, so the answer must not
depend on the arbitrary column order the loader happened to use. Now run the same test on the temporal
axis in the old layout — swap timestep 5 and timestep 12 — and the series is corrupted: the evening
shoulder now sits where the morning rise belongs, and a faithful model *must* change its forecast. Yet
the old layout's attention, being permutation-invariant over its tokens, would be blind to that swap
unless the positional encoding rescued it. So the two tests come out opposite, and they confirm the axis
choice: permutation invariance is a bug on the time axis and a feature on the variate axis, and moving
attention to the variate axis makes the operator's built-in symmetry match the data's actual symmetry
instead of fighting it.

The feed-forward and the head follow from the layout. After attention mixes variate tokens, the
position-wise feed-forward acts *within* each variate token — it is the per-series temporal feature
extractor, learning a nonlinear representation of that channel's history from the d_model embedding,
shared across channels. So the division of labor is clean and inverts the standard one: attention does
cross-variate correlation, the FFN does per-series temporal representation.

I can sanity-check the division by pushing it to a degenerate limit. Take a single channel: N = 1, one
variate token. The self-attention over one token is trivial — a token attends only to itself, softmax of
a single score is 1, so attention is the identity and contributes nothing. What remains is the embedding
R^96 → R^512, the position-wise FFN, and the R^512 → R^96 projection: a plain per-series MLP mapping the
look-back to the horizon. That is the *right* thing to reduce to — with no other channels there is no
cross-variate structure to find, and the model gracefully becomes a nonlinear temporal regressor rather
than doing something ill-defined. It also locates the ETTh1 worry precisely: at 7 channels the attention
map is a 7×7 that has almost nothing to correlate, so nearly all the model's expressive work falls on the
FFN and embedding, and those are exactly the parts the fixed d_model = 512 inflates. The limit confirms
the labor split and, read the other way, confirms why a near-degenerate channel count wastes the
architecture's whole point. For the head, I keep the
direct multi-step generation that has worked at every rung: one linear map d_model → pred_len applied to
each variate token, then transpose so the output is [B, pred_len, N]. No decoder, no autoregression. I
wrap the whole thing in the same reversible per-window instance normalization PatchTST used — subtract
the look-back mean, divide by its std, restore after the projection — because variate tokens still need
the level/scale drift removed before embedding.

One more thing the inverted embedding buys almost for free: the calendar covariates. In the temporal
layout the time-feature marks (hour, day-of-week, and the rest) had to be added into every per-step token
as a positional side channel. Here I embed them as their own tokens alongside the variate tokens — the
`DataEmbedding_inverted` block maps the look-back of each mark series into the same d_model space and
appends it to the variate set — so a calendar signal becomes just another token that the cross-attention
can correlate against the real channels. That is why the projection slices the first N tokens back out at
the end: the extra mark tokens participate in attention as informative pseudo-variates but are not
themselves forecast. It is a clean way to let "which hour is it" inform every channel's prediction without
reintroducing a positional-encoding patch over the time axis I deliberately removed.

I trace the shapes once, because the inversion permutes the panel in a way that is easy to get wrong. The
window arrives [B, L, N] with L = 96. Instance-normalize along the time axis, then the inverted embedding
transposes to treat each variate's length-L history as one token and maps R^96 → R^512, giving
[B, N(+marks), 512] — B batches, one 512-vector per variate (plus the mark tokens). Self-attention runs
over that middle axis of length N(+marks), so its map is (N+marks)² and every entry is a channel-to-channel
(or channel-to-calendar) correlation. The projection is d_model → pred_len applied per token: [B, N(+marks),
512] → [B, N(+marks), 96], then transpose to [B, 96, N(+marks)] and slice the first N along the last axis
to drop the mark tokens, leaving [B, 96, N] — exactly the horizon shape the loop wants. Denormalize by
the stored per-channel mean and std and return. The dimensions close, and the one place a bug would hide —
forecasting the mark tokens or slicing the wrong axis — is exactly the slice I just pinned down.

I have to be explicit about the edit-surface gap again, because it shapes what I should expect on the
small dataset. The iTransformer training script tunes `d_model` per dataset: 128 on ETTh1 (only 7
channels, so 7 tokens — a small model is plenty), 512 on Weather and ECL. The Custom edit gets the fixed
scaffold config `d_model=512`, `e_layers=2` for all three. On ETTh1 that is badly over-parameterized,
and the arithmetic says how badly: the R^96 → R^512 embedding alone is about 49k weights, each encoder
layer carries roughly four d_model² attention projections plus a two-layer d_model×d_ff feed-forward —
on the order of 1.5M parameters — so two layers plus the embedding and the d_model→96 head is several
million weights, all tasked with representing a 7×7 = 49-entry cross-channel attention over seven weakly
coupled series. Seven variate tokens of dimension 512 is far more capacity than seven channels of
structure can support, so I should expect iTransformer to *under*-perform on ETTh1 relative to its tuned recipe and
possibly relative to PatchTST here — the inversion's whole advantage is cross-variate correlation, and
ETTh1 has only 7 channels with weak coupling, so there is almost nothing for the cross-variate attention
to find and a lot of capacity to overfit. On ECL, by contrast, with 321 richly coupled channels, the
inversion should finally pay: this is the dataset built to reward cross-variate modeling.

Falsifiable expectations against the PatchTST numbers. The headline claim is that ECL — PatchTST's
channel-limited residue at 0.1819 — is where the inverted attention wins: I expect iTransformer to drop ECL MSE
clearly below 0.1819, into the mid-0.14s, and to cut its MAE sharply below PatchTST's 0.2743, because
cross-variate correlation is exactly the missing lever there. On ETTh1 I expect the opposite sign: with
only 7 channels and a fixed `d_model=512`, iTransformer should be roughly level with or slightly *worse*
than PatchTST's 0.3794 — if it loses on ETTh1 while winning big on ECL, that is the cleanest possible
confirmation that the inversion buys cross-variate structure and nothing temporal. Weather, at 21
channels, sits between: a modest improvement over 0.1738 if its channels couple, a wash if they do not.
The mean MSE should fall below PatchTST's 0.2451, carried mostly by ECL. The arithmetic of that bet is
telling: if ECL alone drops from 0.1819 to the mid-0.14s — call it ~0.033 off ECL — while ETTh1 gives
back a little and Weather barely moves, the three-dataset mean still falls by roughly a third of that ECL
gain, enough to clear 0.2451. So a single-dataset win, concentrated exactly where the mechanism predicts,
is sufficient to move the headline — which is precisely the shape of result that would confirm the
inversion buys cross-variate structure and nothing temporal. The uncomfortable version of the same
prediction is the honest risk I am taking: if ETTh1 regresses far enough to cancel the ECL gain, the mean
stalls and the lesson becomes that cross-variate attention cannot be bought without a temporal cost at
this fixed capacity — which would itself be the instruction that the two levers need to be held together,
not traded.
