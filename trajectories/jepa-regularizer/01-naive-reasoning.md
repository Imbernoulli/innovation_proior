The regularizer is the whole point, but it sits on top of an invariance objective, and before I can
judge any anti-collapse term I have to see what the invariance objective does *without* one. So the
floor I start from is the barest possible fill of the contract: pull the two views together and add
nothing to stop them from imploding. That is the lower bound by construction, and running it is the
diagnostic that tells me, in measured numbers, how bad collapse is on this harness — the failure every
later step has to beat.

`forward(z1, z2)` receives two `[B, D]` projected embeddings — two augmented views of the same batch,
each through the same backbone and projector — and must return a dict with a scalar `"loss"`. That is
the entire surface: no negative-sample slot beyond the batch, no second network, no predictor, no
stop-gradient hook, no teacher branch. So the simplest honest thing to optimize is "make the two views
agree," and the cleanest scalar for that is the mean squared distance between paired embeddings,
`F.mse_loss(z1, z2)` — no coefficient, no margin, no second term. Whatever survives crop, color jitter,
grayscale and flip is what this loss declares to be content, and it asks the encoder to map both views
of an image to the same embedding.

There is a *different* candidate for barest fill, and ruling it out is what tells me the MSE version is
the one to run. The literal barest fill is the placeholder, `loss = 0`. But the training step
backpropagates the regularizer plus the online probe's cross-entropy, and the probe reads the backbone
through a `detach()`. A zero-gradient regularizer plus a severed probe gradient means the encoder
receives *no gradient at all* — it never leaves random init. The metric would then measure a
random-feature ResNet, and random features on CIFAR-10 sit well above chance: a random-init conv stack
carries plenty of low-level, input-correlated structure, so a linear probe on it plausibly lands in the
thirties or forties. That is not the floor I want — this step is meant to measure how far the *invariance
objective itself* drags the representation down, and a zero-loss encoder never engages that objective.
Worse, active collapse can score *below* random init, because it spends gradient steps destroying the
very input-correlated variance random init handed over for free. So MSE is the better floor: it exercises
the objective under test and reports its genuine worst case. I keep it.

The invariance term is, in the metric-learning lineage, the *attract-only* half of a contrastive pair
loss: keep the similar-pair branch `½‖G(x₁) − G(x₂)‖²`, treat two views as the similar pair, delete the
margin repulsion. Picture point masses joined by zero-rest-length springs — every spring pulls toward
coincidence, nothing pushes apart. There is a configuration with exactly zero energy: all points at the
same location. If the encoder emits one constant vector `c`, every paired distance is zero and the MSE
is zero — a global minimum. Not a bad local optimum I might dodge with luck; the constant map is *the*
minimizer, and it carries no information about the image. The missing counter-pressure — repulsion,
negatives, decorrelation, whatever a real method supplies — is exactly what this baseline omits.

And gradient descent does not merely tolerate the constant, it flows toward it. The gradient
`∂L/∂z1_{b,i} = (2/BD)(z1_{b,i} − z2_{b,i})` contracts each paired pair toward its midpoint; since the
two views share the encoder, coincidence requires insensitivity to whatever augmentation separated them.
There are two routes. The *hard* way learns features that survive crop, jitter, grayscale and flip while
still varying across *different* images — real invariant content. The *easy* way ignores the input
entirely: a constant output depends on nothing, so every paired distance is zero at once with no
representational work. Both reach `L = 0`, nothing rewards the hard way, and the easy way is a low-effort
direction in parameter space that weight decay also favors. So the constant is the path of least
resistance to the same zero the informative solution reaches only with effort. This harness's aggressive
augmentation — a scale-0.2 crop can keep as little as a fifth of the image, so the two views can be
near-disjoint patches — only widens that effort gap, since a constant is invariant to everything for
free, so the collapse should be thorough. And the 10 warmup epochs won't save it: `val_acc` is read at
the endpoint, by which point the encoder has had the full schedule to reach the constant.

The metric sharpens all this. It is not the regularizer's own loss but a linear probe trained online on
the *frozen, detached* backbone features — the probe never sees `z`, it reads the 512- or 2048-dim
backbone representation and fits a 10-way classifier each step. So the metric asks: after this loss
drives the encoder, do the backbone features linearly separate the ten classes? Under collapse they do
not, and the probe lands near the 10% chance floor. And the probe cannot push back against collapse:
with `L_total = L_reg(z1, z2) + CE(W · sg(h), y)`, the cross-entropy's gradient to any encoder parameter
runs through the detached `h` and is zero, so the encoder is shaped purely by my MSE term while the
probe's gradient flows only into `W`. The probe is a passive reader of an encoder free to march to the
constant — which confirms the floor is a clean test of the objective, uncontaminated by the metric's own
training signal.

I deliberately leave `CONFIG_OVERRIDES` empty: reshaping the projector cannot rescue an objective with
no anti-collapse term, because collapse is a property of the objective, not the width — narrowing or
widening only changes how many coordinates share the same zero. And I add no epsilon, temperature,
embedding normalization, or stabilizer, since each is a partial anti-collapse mechanism smuggled in. The
tempting one is swapping MSE for negative cosine similarity, `−cos(z1, z2)` — the invariance form the
predictor methods use. But cosine first l2-normalizes onto the unit sphere, and that normalization is
itself a weak variance constraint: it forbids the all-zero vector outright (you cannot normalize zero),
ruling out one flavor of collapse for free. It does not rule out the important one — every image can
still map to the *same* unit vector, cosine 1, fully collapsed — but it puts a thumb on the scale. I
want the floor to carry no such thumb, so I keep the raw unnormalized MSE, whose minimizer is the honest
constant `c` with no sphere to hide on. `F.mse_loss` reduces by the mean over all `B·D` elements, an
`O(1)` objective LARS handles without numerical trouble, so a broken number here is a statement about
the objective, not the optimizer. I return the scalar plus an `"invariance_loss"` echo so the log shows
the two are identical — there is nothing else in the objective.

Across the three backbones the loss is identical; only backbone capacity and probe feature width differ
(512 for ResNet-18/34, 2048 for ResNet-50). Collapse is not a capacity problem — a bigger network
collapses just as completely, arguably more cleanly for having more freedom to find the constant — so I
expect no rescue and no clean capacity ordering: a tight low band with only noise separating the three,
and if anything ResNet-50's wider features giving its probe a slightly larger cushion to overfit
spurious separation out of the low-variance residue. That residue — a rank-1 mean component plus a thin
skirt from BatchNorm's per-feature rescaling, not retained class signal — is why the number should sit a
few points above 10% rather than exactly at it. The one clean way to falsify the collapse reading is any
backbone clearing well into the forties: that would mean the invariance term was not collapsing there,
and the whole floor premise would be wrong.

One distinction matters for what the next step must fix: collapse has two shapes. The loud one is total
collapse — one constant vector, feature covariance rank zero, every dimension dead. But the same
zero-repulsion objective also permits dimensional collapse, where variance survives but crowds into a
low-rank subspace, a handful of active directions with the rest flat. The invariance term is as content
with a low-rank representation as with a point, as long as paired distances are small, and nothing
rewards *using* all the dimensions. So the failure I am measuring is really "the feature covariance has
collapsed toward low rank," with the constant as its extreme corner — and a fix has to raise the loss
both when variance vanishes *and* when it survives but is confined and redundant. That makes the next
step's instruction specific: not "add something that stops collapse" but "add a term that keeps the
embedding's variance both nonzero *and* spread across dimensions."

So the fix is forced in *kind* before I pick the exact one: add an anti-collapse term making the
constant map — and the softer low-rank collapse — high-loss states rather than minimizers. The
metric-learning ancestor would restore margin repulsion against negatives; the information-maximization
line would instead demand the embedding dimensions stay spread and decorrelated, the branch I expect to
climb here, since it needs no negatives and no second network — exactly the machinery this contract
withholds. Step 1's job is done the moment it produces a broken number: it converts "collapse is a
theoretical worry" into "collapse costs N points of `val_acc` on this exact setup," and that gap is the
budget every later regularizer must recover. I expect the naive baseline to be unambiguously the weakest
thing on the board — a two-digit accuracy near the ten-class chance floor on all three backbones. (The
distilled module and the one-line scaffold edit are in the answer.)
