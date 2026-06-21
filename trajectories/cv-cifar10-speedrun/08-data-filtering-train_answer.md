airbench96 reaches 96%, but it is expensive — 34.7 A100-seconds, roughly ten times the 94% run — and that cost is dominated by the long training over the *full* 50,000-image dataset for 37-plus epochs. So the question for getting to 96% *faster* is whether I am spending those epochs on the right examples. Up to now every training step has treated all examples as equally worth a gradient, but CIFAR-10 is not uniform in difficulty. A large fraction of the images are easy: once the network has learned them in the first few epochs it classifies them confidently and correctly, and every subsequent gradient step on them contributes almost nothing — the loss is already near zero, the gradient is tiny, and the forward-plus-backward pass is mostly wasted compute. The hard examples — the ambiguous, atypical, near-boundary images — are where the remaining error lives and where the gradient signal is. If most of my late-training compute is being spent on easy images that are already solved, I am wasting epochs.

I propose **data filtering** — hard-example mining made cheap by a two-stage proxy scheme. Within each batch I want to train only on the examples that still carry signal, the high-loss ones, and skip the gradient on the easy, low-loss rest: for a batch of $B$ examples, compute the per-example loss, keep the top-$k$ highest-loss examples, and backprop only those. Keeping the hardest half roughly halves the backward-pass compute per step while losing almost nothing, because the dropped half were the near-zero-loss examples whose gradients were negligible anyway. There is a chicken-and-egg problem, though: to know which examples are hard I need a trained network to score them, but the whole point is to speed up training the network. If I score every batch with the *current* model and then mask, I have paid for a full forward pass over all $B$ examples just to make a masking decision, and that decision is made by a half-trained model that may misjudge which examples are truly hard. The fix is a cleaner source of difficulty scores: first run a small, narrow **proxy network** quickly over the data and record, for each batch position, which examples it found hardest (the mask of top-loss examples); then run the *full-size* model and reuse those *pre-computed* proxy masks to select which examples to train on, so the big model never has to score everything itself. The proxy is a good enough difficulty oracle because example hardness is largely a property of the data and transfers across model sizes — the small model's judgment that "this image is hard" mostly holds for the big one.

Two details make the ledger positive and the masks valid. The proxy is cheap not only because the network is small but because it skips every other backward pass — concretely, on steps where `current_steps % 4 == 0` it does a real backward step (and records the mask), and otherwise it computes the loss under `torch.no_grad` and records the mask alone — so collecting the masks costs far less than a full training run. The mask is the top-loss `batch_size_masked` indices of each batch (`loss1.argsort()[-batch_size_masked:]`), with `batch_size` 1024 and `batch_size_masked` 512, so each full-model step trains on the hardest 512 of every 1024-image batch. The subtlety I have to respect is alignment: the mask selection has to match the data ordering and augmentation between the proxy and the main run, or the proxy's "example 37 is hard" will not point at the same image in the main run. The loader seeds its augmentation and ordering with a shared `data_seed`, and the masks are indexed by batch position, so the two runs see the same images in the same order and the masks line up. With that handled, the full model trains on roughly half the examples per step — the half that actually matters — at the same 96% accuracy, pushing the 96% record below airbench96's 34.7 seconds. The one risk is that the proxy mis-scores difficulty and its hard set diverges from the big model's, in which case I would be training on the wrong half and accuracy would drop; the bet is that hardness transfers across scale well enough, and it does.

```python
def train_proxy(...):
    masks = []
    for indices, inputs, labels in train_loader:
        if current_steps % 4 == 0:                     # backward step: train + record mask
            outputs = model(inputs)
            loss1 = loss_fn(outputs, labels)
            mask = torch.zeros(len(inputs)).cuda().bool()
            mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True   # keep the hardest
            masks.append(mask)
            loss = (loss1 * mask.float()).sum()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        else:                                          # no-backward step: still record the mask
            with torch.no_grad():
                outputs = model(inputs)
                loss1 = loss_fn(outputs, labels)
                mask = torch.zeros(len(inputs)).cuda().bool()
                mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
                masks.append(mask)
        scheduler.step(); current_steps += 1
        if current_steps == total_train_steps: break
    return masks

# full-size run reuses the proxy's masks:
masks = iter(train_proxy(hyp, model_proxy, data_seed))
...
mask = next(masks)
inputs = inputs[mask]; labels = labels[mask]
outputs = model(inputs)
loss = loss_fn(outputs, labels).sum()
```
