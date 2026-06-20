**Problem (from step 7).** airbench96 reaches 96% but costs 34.7 s — a long training over the *full* 50k set
for 37+ epochs. CIFAR-10 is non-uniform in difficulty: many images are easy, solved in the first few epochs,
after which every gradient step on them is near-zero loss and wasted compute. The error and the signal live
in the hard, near-boundary examples. So much late-training compute is spent re-solving already-solved images.

**Key idea.** *Data filtering* (hard-example mining), made cheap by a two-stage scheme. First run a small,
narrow **proxy network** quickly over the data and record, per batch position, which examples it found
hardest (the top-loss mask). Then train the full-size model and reuse those *pre-computed* masks to keep
only the hardest half of each batch (`batch_size` 1024 → `batch_size_masked` 512), backpropagating just
those. The proxy is a good difficulty oracle because example hardness is mostly a property of the data and
transfers across model scale; collecting masks is cheap (small net, skips every other backward pass). The
loader shares a `data_seed` so proxy and main runs see the same images in the same order and the masks align.

**Why it works.** The dropped half are the low-loss easy examples whose gradients were negligible, so
halving the per-step backward compute loses almost nothing while cutting wall-clock — pushing the 96% record
below airbench96's 34.7 s. The proxy run pays for itself only because it is small and half-backward, keeping
the ledger positive. Risk: if the proxy mis-scores hardness relative to the big model, training the wrong
half would drop below 96% — the bet is that hardness transfers across scale.

**Change / code.** The proxy collects masks; the main loop consumes them to filter each batch.

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
