I start from the failure mode, not from a new loss. If I take a trained model and push on the forget set
with gradient ascent, I get forgetting pressure, but the sign sends me uphill on an unbounded
cross-entropy objective. If I finetune only on the retain set, I preserve utility, but I do not directly
erase the forgotten examples. If I randomize the forget labels, I get a bounded target, but the update
still runs through the whole network. The recurring problem is that the loss may be about `D_f`, while
the coordinates being moved are all the coordinates.

So I want to separate two questions that previous rules tend to merge. First, which parameters are most
involved in the forget-set behavior at the trained model? Second, once I know those parameters, what
bounded objective should I optimize on them? A derivative gives me a cheap answer to the first question.
For the trained weights `theta_o`, I can differentiate a forgetting loss with respect to every parameter.
For classification that loss is the ordinary CE on `D_f`; for diffusion it is the ordinary denoising MSE
on the forget concept or class. A large absolute derivative means that a small movement of that
coordinate strongly changes the forget loss. A small derivative means that this coordinate is not doing
much for this particular forgetting request.

The sign of this setup needs care. The implementation may compute `-CE` or `-MSE` because it thinks in
gradient-ascent language, while the displayed loss may be positive CE or MSE. That does not change the
mask, because the mask uses absolute values after accumulation. The coordinate score is
`abs(grad_theta ell_f(theta_o; D_f))`, and multiplying the loss by `-1` leaves that score unchanged.

Now I need to turn scores into a hard decision. I define a binary mask `m_S` with one entry per
parameter coordinate:

`m_S = 1(abs(grad_theta ell_f(theta_o; D_f)) >= gamma)`.

The hard threshold `gamma` is not an absolute number I can reuse across architectures. The paper-level
description says the median is an effective practical choice, and the public code implements the same
idea by ranking all absolute scores globally and writing mask files such as `with_0.5.pt`. With
`0.5`, the largest half of scores get mask value `1`, and the rest get `0`, up to integer flooring. The
mask is global, not layerwise. That matters because a layerwise median would force every layer to update
half its weights even if the forget signal is concentrated elsewhere.

The masked model is best written as

`theta_u = m_S * (Delta theta + theta_o) + (1 - m_S) * theta_o`.

This equation tells me that the mask is not pruning the forward pass. The full network still computes
features and predictions. The mask constrains the update: masked-out coordinates remain at their
original values, while masked-in coordinates are allowed to move. In code, the direct way is to multiply
each parameter gradient by the corresponding mask before `optimizer.step()`. The current reference code
adds an important guard: after the step, it restores masked-out coordinates to the saved starting values
and clears their momentum buffers. That guard is not cosmetic. With SGD momentum or weight decay, a zero
gradient is not enough to guarantee exact immobility of a coordinate.

Now I can choose the actual forgetting objective. For classification, random labeling is the useful
bounded core. The mathematical objective draws a label `y' != y` for each forget example and minimizes
CE on `D_f` under those random labels, plus a retain CE term weighted by `alpha > 0`:

`E_{(x,y) in D_f, y' != y} CE(theta_u; x, y') + alpha E_{(x,y) in D_r} CE(theta_u; x, y)`.

The reference implementation is slightly looser than the displayed math. For CIFAR-10 and SVHN it draws
fresh labels with `torch.randint(0, num_classes, target.shape)`, so the original class can occur. For
CIFAR-100 and TinyImageNet it rewrites a copied forget dataset with `np.random.randint` and concatenates
that dataset with the retain dataset. I should not silently "fix" that in a code-faithful artifact: the
paper formula excludes the true label, while the canonical code samples from all classes.

I also need to be precise about the retain term. The formula has an explicit `alpha`; the classifier
code realizes the default recipe without an explicit scalar in `RL.py`. On CIFAR-10 and SVHN it performs
masked forget updates and then masked retain updates in the same epoch. On CIFAR-100 and TinyImageNet it
trains over the union of randomized forget examples and retain examples. Both are implementations of the
same pressure balance, but neither is the single minibatch expression from the old draft.

For diffusion models the random-label idea becomes a pseudo-conditioning idea. In the objective, for a
forget concept `c`, I make the denoiser under `c` imitate the denoiser under another condition `c' != c`
on the same noised input, then add a retain denoising loss:

`E ||epsilon_theta_u(x_t | c') - epsilon_theta_u(x_t | c)||_2^2 + beta ell_MSE(theta_u; D_r)`.

For DDPM, the reference code chooses `c' = (label_to_forget + 1) % 10`, detaches the pseudo output, and
uses `forget_loss + alpha * remain_loss`. For Stable Diffusion class removal, it uses the same next-class
description pattern. For NSFW removal, it uses the nude prompt as the forget prompt and the clothed-person
prompt as both pseudo and retain prompt. The mask is still the same kind of ranked absolute-gradient mask,
but in Stable Diffusion it is built over `model.model.diffusion_model` parameters and applied by stripping
the `model.diffusion_model.` prefix from parameter names.

One tempting shortcut is to compute the mask lazily from the first forget minibatch. I reject that. The
source definition and the reference code both compute the mask in a setup pass over the forget loader at
the original weights. A first-minibatch mask is a possible engineering approximation, but it is not the
canonical method, and it would be a source of hindsight drift in the deliverable.

The resulting method is simple: compute a forget-gradient score at the original model, keep the top
fraction of coordinates as the update region, freeze the rest exactly, and run a bounded random-label or
pseudo-conditioning unlearning objective with a retain term. The important insight is the update
footprint. Forgetting does not have to be made safer only by tuning the loss; it can be made safer by
letting the forget loss move only the coordinates that the forget set actually activates.
