Patched temporal attention came back and it resolved the fork I left open. ETTh1 dropped to MSE 0.3794
/ MAE 0.3986 — below the linear control's 0.3962, on the very dataset the linear model was strongest on,
which is the clean confirmation I asked for: attention over time, when the token is a local patch shape
instead of a meaningless single step, *does* extract more than a linear temporal map. So temporal
modeling was not exhausted; it was mis-tokenized. Weather fell hard too, 0.1962 → 0.1738, exactly where
patch shapes plus instance norm should help the noisy data. The mean MSE went 0.2676 → 0.2451. But the
diagnosis that actually decides the next rung is ECL: 0.2104 → 0.1819. It improved, but it is now the
*relative laggard* of the three — ETTh1 and Weather both moved more, and ECL's MAE (0.2743) is again the
worst by a wide margin, just as it was for the linear model. That is the signal I predicted would tell
me where to go: ECL is the 321-channel dataset, and PatchTST, like the linear control, is still
channel-independent — it never lets one channel inform another. The temporal lever is now well used and
the channel lever is untouched. So the next rung is not a better temporal operator; it is finally
modeling cross-variate structure.

The question is *how* to add cross-variate attention without throwing away what just worked. The reflex,
the one the whole prior lineage took, is to keep the temporal-token layout and somehow bolt channel
mixing on top — but I want to look again at what a temporal token even is, because I think the layout
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
screaming for.

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

Three things fall out of the inversion and each is a fix to a harm I just named. First, attention now
runs over the unordered variate set, so permutation invariance is *correct* and I do not need a temporal
positional encoding at all — the structural mismatch is gone, not patched. Second, LayerNorm now
normalizes across the feature dimension of a *variate* token — the learned representation of one
channel's series — which is a sensible per-series normalization, not a cross-channel blend; the
interaction noise I diagnosed is removed by construction. Third, the cost is O(N²) in the number of
*variates*, independent of T; lengthening the look-back no longer blows up attention (it only widens the
embedding's input), so the model can use more history cheaply — the opposite of the temporal-token
regime where longer windows hurt.

The feed-forward and the head follow from the layout. After attention mixes variate tokens, the
position-wise feed-forward acts *within* each variate token — it is the per-series temporal feature
extractor, learning a nonlinear representation of that channel's history from the d_model embedding,
shared across channels. So the division of labor is clean and inverts the standard one: attention does
cross-variate correlation, the FFN does per-series temporal representation. For the head, I keep the
direct multi-step generation that has worked at every rung: one linear map d_model → pred_len applied to
each variate token, then transpose so the output is [B, pred_len, N]. No decoder, no autoregression. I
wrap the whole thing in the same reversible per-window instance normalization PatchTST used — subtract
the look-back mean, divide by its std, restore after the projection — because variate tokens still need
the level/scale drift removed before embedding.

I have to be explicit about the edit-surface gap again, because it shapes what I should expect on the
small dataset. The iTransformer training script tunes `d_model` per dataset: 128 on ETTh1 (only 7
channels, so 7 tokens — a small model is plenty), 512 on Weather and ECL. The Custom edit gets the fixed
scaffold config `d_model=512`, `e_layers=2` for all three. On ETTh1 that is badly over-parameterized: 7
variate tokens of dimension 512 with 2 encoder layers is far more capacity than 7 channels of structure
can support, so I should expect iTransformer to *under*-perform on ETTh1 relative to its tuned recipe and
possibly relative to PatchTST here — the inversion's whole advantage is cross-variate correlation, and
ETTh1 has only 7 channels with weak coupling, so there is almost nothing for the cross-variate attention
to find and a lot of capacity to overfit. On ECL, by contrast, with 321 richly coupled channels, the
inversion should finally pay: this is the dataset built to reward cross-variate modeling.

Falsifiable expectations against the PatchTST numbers. The headline claim is that ECL — PatchTST's
relative laggard at 0.1819 — is where the inverted attention wins: I expect iTransformer to drop ECL MSE
clearly below 0.1819, into the mid-0.14s, and to cut its MAE sharply below PatchTST's 0.2743, because
cross-variate correlation is exactly the missing lever there. On ETTh1 I expect the opposite sign: with
only 7 channels and a fixed `d_model=512`, iTransformer should be roughly level with or slightly *worse*
than PatchTST's 0.3794 — if it loses on ETTh1 while winning big on ECL, that is the cleanest possible
confirmation that the inversion buys cross-variate structure and nothing temporal. Weather, at 21
channels, sits between: a modest improvement over 0.1738 if its channels couple, a wash if they do not.
The mean MSE should fall below PatchTST's 0.2451, carried mostly by ECL.
