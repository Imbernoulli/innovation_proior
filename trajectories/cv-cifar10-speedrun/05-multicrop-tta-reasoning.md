I'm at 12.0 epochs and 4.6 seconds. Every rung so far has worked on *training* — the initialization, the
optimizer, the learning rates. But there is a second place to spend effort that I've been treating as
fixed: *inference*. The rule of the speedrun is that the timer runs from first data access to emitting
test predictions, and arbitrary test-time augmentation is allowed. So inference cost counts against me,
but inference *quality* can be bought, and here is the key leverage: if I can squeeze more accuracy out
of the *same* trained network at evaluation time, then the network doesn't have to be trained as far to
clear 94% — I can shorten the training and let a smarter inference make up the last fraction of a
percent. The trade is epochs of training (expensive) for a few extra forward passes at eval (cheap, and
only over the 10k test set once).

The baseline already does a little of this: horizontal-flip TTA, where each test image is evaluated both
as-is and mirrored, and the two logit vectors are averaged. The logic is that the network's prediction
on a single view is itself a noisy estimate of the true class posterior — the network is slightly
sensitive to nuisance transformations it should be invariant to — and averaging its predictions over
several label-preserving views of the same image cancels that nuisance variance, the same variance-
reduction logic as the Lookahead averaging, but over *input views* instead of *weight iterates*. Flip
gives me two views. The question is whether more views, chosen well, buy more.

What other label-preserving transformations is the network nearly-but-not-exactly invariant to? The
training augmentation tells me: I trained with horizontal flip and small random *translations*. So the
network has been taught to be invariant to small shifts, but only approximately — and that approximate
invariance is exactly the kind of nuisance sensitivity TTA can average away. So I should augment the
test image with small translations too, not just flips. Concretely: take the image, and also a version
shifted one pixel up-and-to-the-left and one shifted one pixel down-and-to-the-right. Three translation
states (none, up-left, down-right), each in two flip states (as-is, mirrored), gives **six views** of
each test image. Run the network on all six and combine.

How to combine matters. The six views are not equally trustworthy. The *untranslated* image is the
cleanest view — it's the actual test image — while the translated ones are slightly degraded by the
reflection-padding I use to shift them. So I shouldn't average all six uniformly; I should weight the
untranslated views more. The scheme: the two untranslated views (as-is + mirrored) get weight 0.25
each, and the four translated views (two translations × two flips) split the remaining half, 0.125 each.
Equivalently: average the flip pair first (mirror-TTA), do that for the un-translated image and for each
translation, then combine the un-translated result and the mean-of-translations 50/50. That gives the
untranslated image half the total weight on its own, which is what I want.

Let me write it as a hierarchy of averages. `infer_mirror` averages a view with its flip; `infer_mirror_
translate` builds the two translated views by reflection-padding one pixel and cropping the two corners,
runs mirror-TTA on each, and blends the untranslated mirror-result with the mean translated mirror-
result 50/50:

```python
def infer_mirror(inputs, net):
    return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

def infer_mirror_translate(inputs, net):
    logits = infer_mirror(inputs, net)
    pad = 1
    padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
    inputs_translate_list = [
        padded_inputs[:, :, 0:32, 0:32],   # up-and-to-the-left by one pixel
        padded_inputs[:, :, 2:34, 2:34],   # down-and-to-the-right by one pixel
    ]
    logits_translate_list = [infer_mirror(t, net) for t in inputs_translate_list]
    logits_translate = torch.stack(logits_translate_list).mean(0)
    return 0.5 * logits + 0.5 * logits_translate
```

I should be disciplined about the view count, because this is a trade and the trade can go negative.
Multi-crop inference is a classic ImageNet technique and more crops do keep helping — people have gone to
tens or even 144 crops — but every extra view is another full forward pass over the test set, which adds
to my timed budget. Six views is the point where the accuracy gain still clearly outweighs the inference
cost; pushing to dozens of crops would improve accuracy but the inference time would eat the training
time I'm trying to save. So: six views, weighted, and stop.

The prediction. The mechanism is real — averaging the network over label-preserving views it's only
approximately invariant to should shave the test error, and crucially it lets me cash that error
reduction in as *fewer training epochs* to hit 94%. So I expect epochs-to-94% to drop from 12.0, with
the seconds following down even after paying for the six-view inference, because the saved training
epochs are worth more than the added eval passes. The size should be modest — flip-TTA was already
capturing the biggest nuisance (left/right), and I'm adding the second-biggest (one-pixel shift), so
diminishing returns. The risk is on the trade's wrong side: if the network is already nearly translation-
invariant, the translated views add little signal while still costing forward passes, and the net could
be a wash. But six cheap views against a saved epoch of training is a favorable bet. The change is the
`tta_level=2` multi-crop inference path; code in the answer.
