airbench96 reaches 96% but costs 34.7 seconds, about 9× the 3.83-second airbench94, and the cost has the
shape I predicted. Over its 37 epochs, 34.7 s is 0.94 s per epoch versus airbench94's 3.83/9.9 = 0.39 s —
about 2.4× more per epoch from the wider, deeper network — and 2.4× per epoch times 3.7× more epochs is
8.9×, essentially the observed 9.06×. So the 96% cost is per-step growth times epoch stretch, dominated by
one thing: training over the *full* 50,000-image dataset for 37 epochs. That is where the seconds live, so
that is where a speedup has to come from. The question for reaching 96% *faster* is not "can the net be
smaller" (it cannot, without dropping below 96%) but "am I spending those 37 full-dataset epochs on the
right examples?"

Up to now every step has treated all examples as equally worth a gradient — a uniform pass over all
50,000 images every epoch. But CIFAR-10 is not uniform in *difficulty*, and that is the opening. A large
fraction of images are easy: once the network has learned them in the first few epochs it classifies them
confidently, their loss sits near zero, and — the key — their *gradient* sits near zero too, because the
gradient of a cross-entropy already near zero is tiny. So every subsequent forward+backward on an
already-solved easy image contributes almost nothing while costing the same compute as a hard one. The
error and the remaining signal live in the hard, ambiguous, near-boundary examples. In a 37-epoch run,
most of the late-training compute is spent re-solving images that were solved by epoch 5 — the same "don't
spend effort on what is already done" waste, now wearing the costume of per-example difficulty.

This belongs at the 96% bar and would have done little at 94%, which is a check on the diagnosis. The 94%
run is only ~10 epochs, short enough that the network is still actively learning even the "easy" examples
for most of it, so there is no long tail of solved-and-idle examples to prune. The 96% run is 37 epochs,
and it is precisely the *back* two-thirds where a large population sits fully solved, contributing
near-zero gradient at full forward+backward cost. Data filtering is intrinsically a *long-run* technique:
its saving scales with how much of training happens after the easy examples are done, negligible at 10
epochs and large at 37.

So: within each batch, train only on the examples that still carry signal — the high-loss ones — and skip
the gradient on the easy rest. For a batch of 1024, keep the hardest 512 and backprop only those, roughly
halving per-step compute while losing almost nothing, because the dropped 512 are the near-zero-loss
examples whose gradients were negligible anyway. This is hard-example mining, the natural way to make a
*long* training cheaper without lowering the bar: the bar is set by the hard examples, so keep all the
compute on them.

But there is a chicken-and-egg problem where the naive version fails. To know which examples are hard I
need a *trained* network to score them by loss — but the point is to speed up training that network.
Computing per-example loss with the *current* model on every batch and then masking pays for a full
forward on all 1024 just to decide which 512 to backprop: I save the backward on the easy half but not the
forward, so the step still costs a 1024-forward plus a 512-backward. Worse, the masking decision is made
by a half-trained model whose notion of "hard" is unreliable early on. Online self-masking saves less than
it looks and trusts a bad judge.

The fix is a two-stage scheme, and the design choice is *what scores the difficulty*. An external
pretrained model would be its own expense and I do not have one cheaply on hand. Instead I run a small,
narrow *proxy network* quickly over the data and record, per batch position, which examples it found
hardest — the top-loss mask at each step. The proxy is fast because a narrow net's conv FLOPs scale like
the square of its width, so a quarter-width proxy is roughly a sixteenth the compute per forward. Then I
run the *full-size* model and, at each step, reuse the *pre-computed* proxy masks to select which examples
to train on — the big model never scores anything itself, it just filters its batch to the hardest 512
*before* the forward pass and does a clean forward+backward on 512, halving its per-step cost forward and
backward both. Whether the ledger closes positive is the crux: the main saving is ~half the big model's
compute over all 37 epochs, while the proxy pays a full-model-shaped schedule of steps but at ~1/16 the
per-step cost, and further skips its own backward except one step in four (recording the mask under
`no_grad` on the other three). A small tax on a large rebate. It closes positive *because* the proxy is
narrow and mostly-forward; a full-size or every-backward proxy would eat the saving (code in the answer).

Two details in that loop: the selection is `loss1.argsort()[-512:]`, and since argsort returns ascending
indices the last 512 are the *largest* losses — the hardest — which the mask marks. And the proxy's own
backward trains on `(loss1 * mask).sum()`, i.e. only its masked-in hard 512, not all 1024 — deliberately,
so the proxy learns on the *same filtered distribution* the full model will train on and scores hardness
from inside the same loop it is building masks for, rather than being a differently-trained model whose
judgments come from a different regime.

Keeping exactly half — 512 of 1024 — is a real balance. Keep too few (top 128) and the "hard" set
collapses onto the extreme tail, which is disproportionately outliers and mislabeled or corrupted images
whose gradients are closer to noise than signal, throwing away the large, informative *middle* of
moderately-hard examples still teaching real structure. Keep too many (900) and I save almost no compute.
Half drops the clearly already-solved easy portion while retaining the whole still-learning population,
and it halves per-step cost cleanly. The half-keep also mitigates the known failure direction of
hard-example mining — label noise and corruption concentrate in the very highest losses, so always
training on the top handful would over-weight mislabeled images; keeping the top *half* dilutes the
mislabeled tail among a majority of clean, genuinely-hard examples, and scoring with the proxy keeps the
selection one step removed from the model's own overfitting.

Difficulty is not static across a 37-epoch run — an image hard at epoch 5 may be solved by epoch 20, and a
different one becomes the new frontier — so the proxy records a mask *per step*, one boolean vector per
training step, and the main run consumes them step-for-step, so at step t the full model filters to
whatever the proxy found hard at step t. Storing them is nearly free (a boolean per example per step). The
alignment is subtle and, if wrong, silently poisons everything: the masks are recorded *by batch position*
and consumed the same way, so selecting the *same images* in both runs requires the proxy and the main run
to present the same images in the same order with the same augmentation at every step. The loader seeds
its ordering and augmentation with a shared `data_seed` precisely for this — with the same seed, position
j at step t is the same picture in both runs and the mask lines up. It is the kind of bug that would not
crash, just quietly drop accuracy, so the seed is load-bearing, not incidental.

The whole scheme rests on one empirical assumption where the risk lives: that example *hardness transfers
across model scale* — that the images a small narrow proxy finds hard are substantially the ones the big
model finds hard. That is plausible because hardness is largely a property of the *data* (a blurry,
atypical, or mislabeled-looking image is hard for any reasonable model), so a small model is a decent
difficulty oracle. But if it fails, I would be filtering each full-model batch down to the wrong 512 and
accuracy would fall below 96%.

This should also move a second metric in a telling way. The airbench points so far traced a log-log
FLOPs↔error line — accuracy bought with FLOPs. Data filtering is a different move: it holds error fixed
(still 96%) and cuts FLOPs by removing wasted compute, so I expect the filtered point to sit *below and to
the left* of airbench96's 4.9 PFLOPs at the same accuracy — off the previous line, because that line was
drawn by un-filtered training and this changes the efficiency of training itself.

So if the easy examples really were contributing negligible gradient in the back half, dropping them via
the proxy's cheap difficulty scores should cut the full model's per-step compute by ~half while holding
96%, pushing the record below 34.7 seconds. The net saving will be *less* than a clean 2×: the proxy tax
and the fact that not every epoch's easy-half is equally droppable both eat into it, and there is a
structural ceiling because only the *training* compute is filterable — the fixed overheads (data loading,
the whitening initialization, multi-crop TTA evaluation) do not shrink, and the proxy adds its own small
block of time. So a realistic expectation is a meaningful fraction off 34.7, not half of it. If accuracy
holds at 96% while seconds and PFLOPs fall, both the wasted-easy-compute diagnosis and the difficulty-
transfer bet were right; if accuracy slips, the proxy mis-scored hardness. The code is in the answer.
