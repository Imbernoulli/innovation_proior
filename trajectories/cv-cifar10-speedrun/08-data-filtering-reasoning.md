airbench96 reaches 96% but it's expensive — 34.7 seconds, ~10× the 94% run — and the cost is dominated by
the long training over the *full* 50,000-image dataset for 37+ epochs. So the question for getting to 96%
*faster* is: am I spending those epochs on the right examples? Up to now every training step has treated
all examples as equally worth a gradient. But CIFAR-10 is not uniform in difficulty. A large fraction of
the images are easy — once the network has learned them in the first few epochs, it classifies them
confidently and correctly, and every subsequent gradient step on those easy images contributes almost
nothing: the loss is already near zero, the gradient is tiny, and the forward+backward pass on them is
mostly wasted compute. The hard examples — the ambiguous, atypical, near-boundary images — are where the
remaining error lives and where the gradient signal is. If most of my late-training compute is being spent
on easy images that are already solved, I'm wasting epochs.

So the idea is *data filtering*: within each batch, train only on the examples that still carry signal —
the high-loss ones — and skip the gradient on the easy, low-loss rest. Concretely, for a batch of B
examples, compute the per-example loss, keep the top-k highest-loss examples, and backprop only those. If
I keep, say, half the batch (the hardest half), I roughly halve the backward-pass compute per step while
losing almost nothing, because the dropped half were the near-zero-loss examples whose gradients were
negligible anyway. This is hard-example mining, and it's the natural way to make a *long* training cheaper
without lowering the accuracy bar.

But there's a chicken-and-egg problem I have to solve. To know which examples are hard, I need a trained
network to score them — but the whole point is to speed up training the network. If I compute per-example
loss with the *current* model on every batch and then mask, I've paid for a full forward pass on all B
examples just to decide which to backprop; I save backward compute but not forward compute, and the
masking decision is made by a half-trained model that may misjudge which examples are truly hard. I need a
cleaner source of difficulty scores.

The fix is a two-stage scheme. First run a *small proxy network* — a cheap, narrow version of the model —
quickly over the data, and record, for each batch position, which examples it found hardest (the mask of
top-loss examples). That proxy run is fast because the network is small. Then run the *full-size* model,
and at each step reuse the *pre-computed proxy masks* to select which examples to train on — no need for
the big model to score everything itself. The proxy is a good enough difficulty oracle: example hardness
is largely a property of the data and transfers across model sizes, so the small model's judgment of "this
image is hard" mostly holds for the big one. The proxy also skips every other backward pass while still
recording masks on the no-backward steps, so collecting the masks is cheap:

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

Then in the main run I just consume those masks — `batch_size` is 1024, `batch_size_masked` is 512, so
each full-model step trains on the hardest 512 of every 1024-image batch:

```python
masks = iter(train_proxy(hyp, model_proxy, data_seed))   # collected from the small proxy
...
mask = next(masks)
inputs = inputs[mask]
labels = labels[mask]
outputs = model(inputs)
loss = loss_fn(outputs, labels).sum()
```

A subtlety I have to respect: the mask selection has to be aligned with the data ordering and
augmentation between the proxy and the main run, or the proxy's "example 37 is hard" won't point at the
same image in the main run. The loader seeds its augmentation and ordering with a shared `data_seed`, and
the masks are indexed by batch position, so the two runs see the same images in the same order and the
masks line up. With that handled, the full model trains on roughly half the examples per step — the half
that actually matters — at the same 96% accuracy.

The prediction. This is a 96%-target *speedup* rung: same bar as airbench96, less time. The bet is that
the easy examples really were contributing negligible gradient late in training, so dropping them (via a
cheap proxy's difficulty scores) cuts the per-step backward compute substantially while holding 96% mean
accuracy — pushing the 96% record below airbench96's 34.7 seconds. The cost is the proxy run itself, which
must stay cheap enough that the masks it buys save more than it spends; using a small narrow proxy and
skipping half its backward passes is what keeps that ledger positive. The risk is that the proxy
mis-scores difficulty — that the small model's hard set diverges from the big model's — in which case I'd
be training on the wrong half and accuracy would drop below 96%; the bet is that hardness transfers across
scale well enough. The change is the proxy-mask data-filtering loop; code in the answer.
