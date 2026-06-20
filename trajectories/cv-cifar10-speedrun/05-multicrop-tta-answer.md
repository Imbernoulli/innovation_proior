**Problem (from step 4).** At 12.0 epochs / 4.6 s, every win so far worked on *training*. But inference
quality can be bought cheaply, and the speedrun allows arbitrary TTA: squeezing more accuracy from the same
trained net at eval time means the net needn't be trained as far to clear 94% — a favorable trade of
expensive training epochs for cheap extra forward passes over the 10k test set. The baseline only averages
two views (flip-TTA).

**Key idea.** Multi-crop TTA with **six views**: the untranslated image plus a one-pixel up-left and a
one-pixel down-right translation, each in both flip states. Combine with a weighted average — the two
untranslated views weighted 0.25 each, the four translated views 0.125 each — so the clean, untranslated
image carries half the total weight and the reflection-padded translated views (slightly degraded) carry
less. The translations mirror the training-time augmentation the net is only *approximately* invariant to.

**Why it works.** Averaging the net over label-preserving views it is only approximately invariant to
cancels nuisance variance and lowers test error — the same variance-reduction logic as the Lookahead weight
averaging, applied to input views. That error reduction is cashed in as fewer training epochs to hit 94%, so
seconds drop even after paying for six-view inference. Six is the trade's sweet spot: more crops keep
helping (classic ImageNet multi-crop scales to dozens), but the extra inference cost would eat the saved
training time.

**Change / code.** A hierarchical TTA: average each view with its flip (`infer_mirror`), then blend the
untranslated mirror-result 50/50 with the mean of the two translated mirror-results.

```python
def infer_mirror(inputs, net):
    return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

def infer_mirror_translate(inputs, net):
    logits = infer_mirror(inputs, net)
    pad = 1
    padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
    inputs_translate_list = [
        padded_inputs[:, :, 0:32, 0:32],   # up-and-to-the-left
        padded_inputs[:, :, 2:34, 2:34],   # down-and-to-the-right
    ]
    logits_translate_list = [infer_mirror(t, net) for t in inputs_translate_list]
    logits_translate = torch.stack(logits_translate_list).mean(0)
    return 0.5 * logits + 0.5 * logits_translate

def infer(model, loader, tta_level=0):
    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]   # level 2 = multi-crop
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])
```
