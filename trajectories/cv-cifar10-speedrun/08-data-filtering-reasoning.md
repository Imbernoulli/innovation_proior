airbench96 reaches 96% but it is expensive — 34.7 seconds, about 9× the 3.83-second airbench94 — and
before I attack that number I want to read where it comes from, because a 9× jump is exactly what I
predicted the 96% recipe would cost and I should confirm the cost has the shape I think. Spread over its
37 epochs, 34.7 seconds is 34.7 / 37 = 0.94 s per epoch, versus airbench94's 3.83 / 9.9 = 0.39 s per
epoch — about 2.4× more expensive per epoch from the wider, deeper network. And 2.4× per epoch times 3.7×
more epochs is 8.9×, which is essentially the observed 34.7 / 3.83 = 9.06×. So my rung-7 accounting was
right: the 96% cost is the per-step growth times the epoch stretch, and it is dominated by one thing —
training over the *full* 50,000-image dataset for 37 epochs. That is where the seconds live, so that is
where a speedup has to come from. The question for reaching 96% *faster* is not "can the net be smaller"
(it cannot, without dropping below 96%) but "am I spending those 37 full-dataset epochs on the right
examples?"

Up to now every training step has treated all examples as equally worth a gradient — a uniform pass over
all 50,000 images, every epoch. But CIFAR-10 is not uniform in *difficulty*, and that non-uniformity is
the opening. A large fraction of the images are easy: once the network has learned them in the first few
epochs, it classifies them confidently and correctly, their loss sits near zero, and — this is the key —
their *gradient* sits near zero too, because the gradient of a cross-entropy that is already near zero is
tiny. So every subsequent forward+backward pass on an already-solved easy image contributes almost nothing
to the weights while costing the same compute as a hard one. The hard examples — the ambiguous, atypical,
near-boundary images — are where the remaining error lives and where the gradient signal still is. In a
37-epoch run, most of the late-training compute is being spent re-solving images that were solved by epoch
5, which is precisely the "don't spend effort on what is already done" waste I have been hunting in every
rung, now wearing the costume of per-example difficulty rather than per-layer conditioning or per-epoch
view redundancy.

It is worth asking why this technique belongs at the 96% bar and would have done little at the 94% bar,
because that is a check on the diagnosis. The 94% run is only ~10 epochs — short enough that the network is
still actively learning even the "easy" examples for most of it, so there is no long tail of solved-and-
idle examples to prune; filtering would be discarding gradient the short run still needs. The 96% run is 37
epochs, and it is precisely the *back* two-thirds of that long schedule where a large population of easy
images sits fully solved, contributing near-zero gradient while still costing full forward+backward. So
data filtering is intrinsically a *long-run* technique: its saving scales with how much of training happens
after the easy examples are done, which is negligible at 10 epochs and large at 37. That it surfaces now,
at the 96% bar, is not a coincidence — it is the first rung whose schedule is long enough for the easy-tail
waste to be worth harvesting.

So the idea is *data filtering*: within each batch, train only on the examples that still carry signal —
the high-loss ones — and skip the gradient on the easy, low-loss rest. Concretely, for a batch of B
examples, compute the per-example loss, keep the top-k highest-loss examples, and backprop only those.
With batch_size 1024, if I keep the hardest 512 (batch_size_masked), I roughly halve the per-step compute
while losing almost nothing, because the dropped 512 are the near-zero-loss examples whose gradients were
negligible anyway — I am not throwing away signal, I am throwing away the examples that had no signal left
to give. This is hard-example mining, and it is the natural way to make a *long* training cheaper without
lowering the accuracy bar: the bar is set by the hard examples, and I keep all my compute on them.

But there is a chicken-and-egg problem I have to solve before this is real, and it is where the naive
version fails. To know which examples are hard, I need a *trained* network to score them by loss — but the
whole point is to speed up training the network in the first place. If I compute per-example loss with the
*current* model on every batch and then mask, I have already paid for a full forward pass on all 1024
examples just to decide which 512 to backprop: I save the backward compute on the easy half but not the
forward compute, so a batch that filters to 512 still costs a 1024-forward plus a 512-backward instead of
a clean 512-forward-plus-512-backward. Worse, the masking decision is being made by a half-trained model
whose notion of "hard" is itself unreliable early on, when its losses are noisy for every example. Online
self-masking saves less than it looks and trusts a bad judge. I need a cleaner, cheaper source of
difficulty scores.

The fix is a two-stage scheme, and the design choice is *what scores the difficulty*. I could try to score
once with an external pretrained model, but I do not have one cheaply on hand and it would be its own
expense. Instead I run a small *proxy network* — a cheap, narrow version of the model — quickly over the
data, and record, for each batch position, which examples it found hardest: the mask of top-loss examples
at each step. That proxy run is fast because the network is small (a narrow net's conv FLOPs scale like
the square of its width, so a proxy at, say, a quarter width is roughly a sixteenth the compute per
forward). Then I run the *full-size* model and, at each step, reuse the *pre-computed* proxy masks to
select which examples to train on — the big model never has to score anything itself; it just consumes the
mask, filters its batch to the hardest 512 *before* the forward pass, and does a clean forward+backward on
512. So the full model's per-step cost is genuinely halved, forward and backward both, and the only new
cost is the cheap proxy.

Whether this ledger closes positive is the crux, so let me do the arithmetic honestly. The main run's
saving is large: half of the big-model per-step compute over all 37 epochs, since every full-model step now
processes 512 instead of 1024. The proxy's cost is small on two counts. It is narrow — call it ~1/16 the
per-forward FLOPs of the full model — and it further skips most of its own backward passes: the loop does
a real backward only when `current_steps % 4 == 0` (one step in four) and merely records the mask under
`no_grad` on the other three. So the proxy pays a full-model-shaped schedule of steps but at a small
fraction of the per-step cost — roughly a sixteenth for the forwards, plus a quarter of that again for the
occasional backward. Set against a main-run saving of ~half the *full* model's training, the proxy
overhead is a small tax on a large rebate. The ledger closes positive precisely *because* the proxy is
small and mostly-forward; if I had made the proxy full-size or let it do every backward, it would eat the
saving. Keeping it narrow and quarter-backward is what makes data filtering a net win rather than a wash.

```python
# proxy run: collect a difficulty mask per step (top-loss examples), cheaply
if current_steps % 4 == 0:
    outputs = model(inputs)
    loss1 = loss_fn(outputs, labels)
    mask = torch.zeros(len(inputs)).cuda().bool()
    mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True   # keep the hardest
    masks.append(mask)
    loss = (loss1 * mask.float()).sum()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
else:
    with torch.no_grad():
        outputs = model(inputs)
        loss1 = loss_fn(outputs, labels)
        mask = torch.zeros(len(inputs)).cuda().bool()
        mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
        masks.append(mask)
```

```python
masks = iter(train_proxy(hyp, model_proxy, data_seed))   # collected from the small proxy
...
mask = next(masks)
inputs = inputs[mask]
labels = labels[mask]
outputs = model(inputs)
loss = loss_fn(outputs, labels).sum()
```

A couple of details in that loop are worth reading carefully. The selection is `loss1.argsort()[-512:]`:
argsort returns indices that would sort the losses in *ascending* order, so the last 512 indices are the
512 *largest* losses — the hardest examples — and `mask[...] = True` marks exactly them. And notice that
in the proxy's backward step the loss is `(loss1 * mask.float()).sum()`, so even the proxy trains only on
its own masked-in hard 512, not on all 1024. That is deliberate and matters: it keeps the proxy learning on
the *same filtered distribution* the full model will train on, so the proxy is not a differently-trained
model whose difficulty judgments come from a different data regime — it is scoring hardness from inside the
same hard-example loop it is building masks for. Proxy and main see the same seed-aligned data and both
learn on the filtered half, which keeps their notions of "hard" consistent.

The choice to keep exactly half — 512 of 1024 — is a real balance, not a round number I reached for. Keep
too few (say the top 128) and the "hard" set collapses onto the extreme tail, which is disproportionately
outliers and mislabeled or corrupted images whose gradients are closer to noise than to signal; I would be
throwing away the large, informative *middle* of moderately-hard examples that are still teaching the
network real structure, and chasing the noisy tail. Keep too many (say 900) and I have saved almost no
compute, because I am still processing most of the batch. Half sits at the balance: it drops the clearly
already-solved easy portion, whose gradients really are negligible, while retaining the whole
still-learning population — the genuinely ambiguous middle *and* the hard tail — so the compute I keep is
spent where learning is still happening. It also halves the per-step cost cleanly, which is the arithmetic
the ledger needs.

That said, hard-example mining has a known failure direction I should keep in view: the very highest-loss
examples are exactly where label noise and corruption concentrate, so a scheme that always trains on the
top-loss risk over-weighting mislabeled images and fitting noise. Keeping the top *half* rather than the
top handful is itself the mitigation — the mislabeled tail is diluted among a large majority of clean,
genuinely-hard examples, so its influence is bounded — and using the proxy to *score* difficulty, rather
than letting the full model recursively over-focus on its own worst losses, keeps the selection one step
removed from the model's own overfitting. So the half-keep does double duty: it balances compute-saving
against signal-retention, and it caps how much the noisy tail can dominate.

One more property makes this fit a 37-epoch run rather than a one-shot subset selection: difficulty is not
static across training. An image that is hard at epoch 5 may be comfortably solved by epoch 20, and a
different image may become the new frontier. The proxy does not compute a single fixed "hard subset" and
apply it throughout — it records a mask *per step*, one boolean vector for each of the total training
steps, so the sequence of masks is *time-varying*: it captures the hard set shrinking and shifting as
training proceeds. The main run consumes them step-for-step (`next(masks)`), so at step t the full model
filters to whatever the proxy found hard at step t, which tracks the moving frontier of difficulty rather
than freezing it. Storing the masks costs almost nothing — a boolean per example per step, total steps ×
1024 bits, a few megabits — so keeping the full time-resolved schedule of masks is free.

There is a subtlety in the alignment that, if I get it wrong, silently poisons the whole thing, so I have
to reason it through. The masks are recorded *by batch position* — `mask[j]` says "the example at position
j in this step's batch is hard" — and they are consumed the same way in the main run (`inputs[mask]`). For
that to select the *same images* in both runs, the proxy and the main run must present the same images in
the same order, with the same augmentation, at every step. If the two runs shuffled differently, the
proxy's "position 37 of step 12 is hard" would point at a completely different image in the main run, and
I would be training the full model on a mask that has nothing to do with its own batch — noise, not hard
examples. This is why the loader seeds its ordering and augmentation with a shared `data_seed`: with the
same seed, both runs walk the 50,000 images in the identical permuted order and apply the identical
per-image flips and crops, so position j at step t is the *same picture* in both, and the mask lines up.
Let me state the correctness condition plainly: mask alignment holds iff (proxy ordering, proxy
augmentation) == (main ordering, main augmentation), which the shared seed guarantees. It is the kind of
bug that would not crash — it would just quietly drop accuracy — so it is worth being explicit that the
seed is load-bearing, not incidental.

The whole scheme rests on one empirical assumption I should name because it is where the risk lives:
that example *hardness transfers across model scale* — that the images a small narrow proxy finds hard are
substantially the same images the big model finds hard. That is plausible because hardness is largely a
property of the *data* (a blurry, atypical, or mislabeled-looking image is hard for any reasonable model)
rather than of a particular network's size, so a small model is a decent difficulty oracle for a big one.
But it is an assumption, and if it fails — if the proxy's hard set diverges from the full model's — then I
would be filtering each full-model batch down to the *wrong* 512, training on examples that are not
actually the ones carrying the big model's remaining error, and accuracy would fall below 96%. The bet is
that hardness transfers well enough across this scale gap that the wrong-half contamination is small.

There is a second metric this should move, and predicting it sharpens the claim. The airbench points so far
traced a log-log FLOPs↔error line — more accuracy bought with more FLOPs. Data filtering is a different kind
of move: it holds the error fixed (still 96%) and cuts the FLOPs, by removing wasted compute rather than
buying accuracy. So I expect the filtered 96% point to sit *below and to the left* of airbench96's 4.9
PFLOPs at the same accuracy — a point that is *off* the previous line, because the previous line was drawn
by un-filtered training and this rung changes the efficiency of training itself. A win that moves left at
constant error, rather than up along the error curve, is exactly the signature of removing waste, and it is
what I should see in the PFLOPs column if the easy-example gradients really were the dead weight I claim.

The prediction, and it is a 96%-target *speedup* rung, same bar as airbench96, less time. If the easy
examples really were contributing negligible gradient in the back half of training, then dropping them via
the proxy's cheap difficulty scores should cut the full model's per-step compute by roughly half while
holding 96% mean accuracy — pushing the 96% record below airbench96's 34.7 seconds. The cost is the proxy
run, which the ledger above says pays for itself only because it is narrow and quarter-backward, so I
expect the net saving to be substantial but *less* than a clean 2× (the proxy tax and the fact that not
every epoch's easy-half is equally droppable both eat into it). And there is a structural ceiling on the
wall-clock win even before those: only the *training* compute is filterable. The fixed overheads — data
loading, the whitening initialization, evaluation with its multi-crop TTA — do not shrink when I train on
half the batch, and the proxy run adds its own (small) block of time. So even if I halved the full model's
training compute perfectly, the *wall-clock* would fall by less than 2× because training is only part of
the 34.7 seconds; a realistic expectation is a meaningful fraction off 34.7, not half of it. Being honest
about that ceiling up front is what keeps the prediction falsifiable rather than hopeful. The falsifiable outcomes are two: if
accuracy holds at 96% and the seconds drop below 34.7, the difficulty-transfer bet and the wasted-easy-
compute diagnosis were both right; if accuracy slips below 96%, the proxy mis-scored hardness and I was
training the wrong half. The 96% seconds-and-accuracy table, together with the PFLOPs
column, will separate those cases — the seconds and FLOPs falling while accuracy holds at 96% is the
outcome that confirms both the wasted-easy-compute diagnosis and the difficulty-transfer bet at once. The
change is the proxy-mask data-filtering loop; the code is in the answer.
