Lookahead brought me to 12.0 epochs and 4.6 A100-seconds. Every rung so far has worked on *training* — initialization, optimizer, learning rates — but there is a second place to spend effort that I have been treating as fixed: *inference*. The speedrun's timer runs from first data access to emitting test predictions, and arbitrary test-time augmentation is allowed, so inference cost counts against me but inference *quality* can be bought, and that is leverage. If I can squeeze more accuracy out of the *same* trained network at evaluation time, the network does not have to be trained as far to clear 94%: I can shorten the training and let a smarter inference make up the last fraction of a percent — trading expensive training epochs for a few cheap extra forward passes over the 10k test set. The baseline already does a little of this with horizontal-flip TTA, evaluating each test image as-is and mirrored and averaging the two logit vectors. The logic is that the network's prediction on a single view is itself a noisy estimate of the true class posterior — it is slightly sensitive to nuisance transformations it should be invariant to — and averaging its predictions over several label-preserving views cancels that nuisance variance, the same variance-reduction logic as the Lookahead weight averaging but applied to *input views* instead of *weight iterates*. Flip gives two views; the question is whether more views, chosen well, buy more.

I propose **multi-crop test-time augmentation with six weighted views**. The training augmentation tells me which other label-preserving transformations the network is nearly-but-not-exactly invariant to: I trained with horizontal flip and small random *translations*, so the network has been taught to be approximately shift-invariant, and that approximate invariance is exactly the nuisance sensitivity TTA can average away. So I augment the test image with small translations as well as flips — the image, plus a version shifted one pixel up-and-to-the-left and one shifted one pixel down-and-to-the-right. Three translation states (none, up-left, down-right) crossed with two flip states (as-is, mirrored) give six views of each test image. How I combine them matters, because the six views are not equally trustworthy: the *untranslated* image is the cleanest view — it is the actual test image — while the translated ones are slightly degraded by the reflection-padding used to shift them. So I weight the untranslated views more rather than averaging all six uniformly. The two untranslated views get weight 0.25 each, and the four translated views split the remaining half at 0.125 each, which gives the untranslated image half the total weight on its own.

I express that as a hierarchy of averages. `infer_mirror` averages a view with its flip — plain mirror-TTA. `infer_mirror_translate` builds the two translated views by reflection-padding one pixel and cropping the two opposite corners, runs mirror-TTA on each translated view and on the untranslated one, and then blends the untranslated mirror-result 50/50 with the mean of the two translated mirror-results. Working through the weights, the untranslated pair contributes $\tfrac12\cdot\tfrac12 = \tfrac14$ each and each translated view contributes $\tfrac12\cdot\tfrac12\cdot\tfrac12 = \tfrac18$, which is exactly the weighting I want. The other design discipline is the view count, because this is a trade that can go negative: multi-crop inference is a classic ImageNet technique and more crops do keep helping — people have gone to tens or even 144 crops — but every extra view is another full forward pass over the test set added to my timed budget. Six is the point where the accuracy gain still clearly outweighs the inference cost; pushing to dozens would improve accuracy but the inference time would eat the training time I am trying to save. Since flip-TTA already captured the biggest nuisance (left/right) and I am adding the second-biggest (one-pixel shift), the error reduction is modest, but it cashes in as fewer training epochs to hit 94%, so seconds drop even after paying for six-view inference.

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
