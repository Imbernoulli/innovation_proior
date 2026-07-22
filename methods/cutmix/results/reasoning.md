OK, let me think this through from scratch. I've got a CNN classifier and the thing I keep running into is that it bets everything on one small patch of the object. Look at a class activation map for a trained net on a dog: it's one hot blob over the face. Occlude the face and the prediction falls apart. That's the same fragility that hurts generalization — the network never learned that the body, the legs, the ears are also evidence — and it's why weakly-supervised localization is so bad, because the "where is the dog" signal is just that one blob, not the dog.

So how do people fight this? The trick that's around is regional dropout. Dropout itself (Srivastava and co.) randomly zeroes hidden activations so no single unit can dominate; the natural transplant to the input is to zero a *region* of the image. Cutout (DeVries and Taylor) does exactly that — cut a square out of the image, set it to zero, at a random spot. Random Erasing fills the hole with noise instead. Hide-and-Seek hides a grid of patches. The logic is clean and it works: if I randomly delete the most discriminative patch some fraction of the time, the network is forced to find class evidence in the pixels that remain, so the evidence spreads over the whole object. Accuracy up, localization up.

But stare at what Cutout actually does to a training image. A 32×32 CIFAR image with a, say, 16×16 black square stamped on it. A quarter of the image is now *zero*. Those pixels contribute nothing to the forward pass, nothing to the loss, nothing to the gradient. I've thrown a quarter of every image in the trash. And CNNs are notoriously data-hungry — the whole reason augmentation exists is that we never have enough informative pixels. Deliberately blanking a chunk of every image to get the regularization feels like paying for the regularization in the one currency I can least afford. Can I get the regional-dropout effect *without* wasting the deleted region?

Let me look at the other side of the augmentation world for a contrast: Mixup (Zhang and co.). It refuses to delete anything. It takes two images and forms the per-pixel convex blend x̃ = λx_A + (1−λ)x_B, and — crucially — the matching blend of the one-hot labels ỹ = λy_A + (1−λ)y_B, with λ from a Beta distribution. Every pixel of the output is a real combination of real pixels; nothing is wasted; and the soft label tracks the mixture. So Mixup has exactly the property Cutout lacks: full use of pixels. And it works for classification.

But Mixup has the dual problem. Look at a Mixup image: it's a ghost. Two photographs superimposed at 60/40 opacity, a translucent dog floating over a translucent cat. No real image ever looks like that. And the class activation map on such an input is a mess — the network can't decide where either object is, because neither object is really *there*, both are half-erased everywhere. That diffuseness is fine-ish for a label that says "60% dog," but it's poison for localization: there's no contiguous, sharp object to point at. And notice — Mixup doesn't drop a region at all. There's no occluded patch forcing whole-object attention; it's a global wash. So Mixup gives up the very thing regional dropout is good at.

So I've got two methods that are each other's complement. Cutout: a real deleted region (good for localization, whole-object attention), but the region is wasted (bad for data efficiency). Mixup: every pixel informative (good for efficiency), but no deleted region and the image is unnatural (bad for localization). What I'd want is the intersection: a deleted region that is *also* informative, and an image that stays locally natural. Is there a single operation that lands in that intersection?

Cutout deletes a region and fills it with zeros. The zeros are the only thing wasteful about it — the *region* is exactly the asset I want to keep. So what happens if I fill that region not with zeros, but with the corresponding region *from another training image*? I cut a rectangle out of image A and paste in the same-shaped rectangle of pixels from image B. Let me check this candidate against each of the three things I want, one at a time, because if it fails any of them it's not the intersection I was after. Is there a deleted region of A? Yes — a contiguous rectangle of A is gone, exactly the occlusion Cutout wanted, so the network still has to recognize A from a partial view; the whole-object/localization benefit survives. Are the pasted pixels informative? Yes — they're real pixels of a real image B, not zeros, so nothing is wasted; the data-efficiency virtue of Mixup is recovered. Is the composite locally natural? Yes — and this is the part that beats Mixup: it's a real patch of B sitting on a real patch of A with a sharp boundary, not a translucent overlay. Locally, everywhere, the pixels look like a genuine photograph; there's no ghosting. The activation map can find a clean B-object in the pasted rectangle and a clean A-object in what remains. All three hold, which is what I was hoping for and what neither parent achieves alone.

So the operation is: pick a binary mask M ∈ {0,1}^{W×H} that's 1 on the "keep A" region and 0 on the "paste B" rectangle, and form

  x̃ = M ⊙ x_A + (1 − M) ⊙ x_B.

Now the label. The composite genuinely contains class-A pixels and class-B pixels — it is, in a concrete spatial sense, part A and part B. What should I tell the network it is? If I commit to a single one-hot label — say "it's A because most of it is A" — I'm lying about the B patch and I'm back to a hard target that ignores the construction. Label smoothing already taught me a soft target is fine and useful. And Mixup taught me the honest move: mix the labels in the same proportion the inputs were mixed. So

  ỹ = λ y_A + (1 − λ) y_B,

where λ is the *fraction of the image that is A*. The supervision now matches the pixel composition: if 70% of the image is A, the target is 70% A.

Let me pressure-test that label choice before I commit to it, because it's the one place I could be fooling myself, and I want to know *concretely* what the alternatives cost. Take a sample that is 55% A, 45% B. Under a one-hot label that picks the majority class, the target is pure A: the 45% of the image that is a perfectly clear, un-occluded B-object is being called A, and the gradient pushes the network to read those B-pixels as A-evidence. That's not a soft inaccuracy, it's the wrong sign on the cleanest signal in the image. Now take a fixed 0.5/0.5 label and apply it to a 90% A, 10% B sample: a 10% B-patch is told to account for half the prediction, so a tiny sliver of B is supposed to be worth as much as nearly the whole A-object. Both alternatives drive the target away from the actual pixel ratio; only λ = area(A) keeps target and composition aligned. I can't run a full training comparison in my head, but the area-proportional argument is concrete enough that I'd expect both variants to measurably underperform if I sweep them — that's the first ablation I'd want to actually run.

For this to be coherent, λ has to be exactly the area fraction occupied by A, and the rectangle of B has to occupy fraction 1−λ. So I need to draw a box whose area is a 1−λ fraction of the image. Let me draw λ ~ Beta(α, α), same family Mixup uses — symmetric because A-vs-B is an arbitrary labelling. Then I want a rectangle of area (1−λ)·WH. I have freedom in the shape; the simplest choice is to make the box a scaled copy of the image, so its width and height each scale by the same factor s. Then r_w = W·s and r_h = H·s, and the area fraction is r_w r_h / (WH) = s². Setting s² = 1−λ gives s = √(1−λ):

  r_w = W √(1−λ),  r_h = H √(1−λ).

Let me actually check that this gives the area I claimed, on a real 32×32 grid, because the code will take integer pixel counts and I want to see whether the rounding bites. At λ = 0.75: s = √0.25 = 0.5, so cut_w = cut_h = int(32·0.5) = 16, and the area fraction is 16·16/1024 = 0.25 = 1−λ exactly. Good there. But at λ = 0.5: s = √0.5 ≈ 0.7071, int(32·0.7071) = int(22.6) = 22, area fraction 22·22/1024 = 0.4727 — not 0.5. And at λ = 0.1: s ≈ 0.9487, int(30.36) = 30, area 30·30/1024 = 0.8789, not 0.90. So the continuous identity is exact, but the `int()` truncation already makes the *realized* pasted area a little smaller than the nominal 1−λ even before any clipping. That's a real wrinkle: the nominal λ I drew is not quite the true area split. I'll come back to it — the fix is to compute λ from the box I actually pasted, not the one I asked for.

Place the box uniformly: the center r_x ~ Unif(0, W), r_y ~ Unif(0, H), and the box is the region where M = 0 (filled from B), with M = 1 everywhere else (kept from A). If nothing got truncated or clipped the kept-A area fraction would be 1 − (1−λ) = λ, matching the weight I put on y_A. So the geometry and the label are *designed* to agree; whether they actually agree pixel-for-pixel depends on the rounding I just saw and on edge effects, which I now have to look at.

Before that, two sanity questions on the shape of the construction itself.

Why a contiguous rectangle, and not, say, Mixup's blend or a scatter of pixels from B? Because the localization benefit is the entire point of the exercise, and it requires a *coherent occlusion*. A contiguous patch of B forces the network to recognize the partial A-object from whatever spatial region remains, the way a real occluder would — that's what spreads A's evidence over A's whole visible extent. A scattered or translucent mixture doesn't occlude anything coherently; it just adds noise everywhere, which is the Mixup failure. Contiguity is load-bearing.

Why apply this at the pixel level and not on a feature map? The phenomenon I'm fixing — "the net bets on one discriminative patch, occlude it and it dies" — is a statement about the function of the *input image*: occlusion, decision-boundary behavior near real images, where the object is. If I mix feature maps instead, the network gets to choose a representation that makes the constraint easy to dodge, and the occlusion no longer corresponds to a spatial occlusion of the object. The cleanest place to impose it is the rawest one, the input. (I'd expect feature-level mixing to still help somewhat — it's still a regularizer — but input-level to be best; another thing to settle by ablation rather than by assertion.)

Which Beta? λ ~ Beta(α, α). At α = 1, Beta(1,1) is Uniform(0,1), so the area fraction is uniform — every patch size equally likely, maximally diverse occlusions, from a sliver to nearly the whole image. Small α piles mass at 0 and 1 (almost-no-mix or almost-full-swap); large α concentrates near half-and-half. Uniform is the natural default and it's the one knob; I'll sweep α to see whether α = 1 really is best or whether some concentration helps. The probability of applying the operation is a separate training knob; for the CIFAR scaffold below I use 0.5, while the large ImageNet run can use 1.0.

Now implementation, and there are two things to get right. First, the pairing: I don't need a second data loader to get image B. Within a minibatch the examples are already i.i.d. draws, so I can pair the batch with a shuffled copy of itself — one random permutation of the batch indices gives me a random partner B for every A, free of extra I/O. Same trick Mixup uses.

Second is the edge effect I flagged, and I want to *trace it* rather than hand-wave, because it directly corrupts the label if I ignore it. When I place the box, I have to clip it to the image borders — if the sampled center is near an edge, part of the nominal box falls outside the image. Take a concrete bad case: λ = 0.5 on the 32×32 grid, so the nominal box is 22×22 from above, and suppose the center lands at (2, 2), right in a corner. The box spans cx − cut_w//2 = 2 − 11 = −9 to cx + cut_w//2 = 2 + 11 = 13 in x, and the same in y. Clipping the lower edge to 0 gives x ∈ [0, 13], y ∈ [0, 13] — a 13×13 patch, not 22×22. The pasted-B area is now 13·13/1024 = 0.165, so only 16.5% of the image is actually B, not the nominal 50%. If I kept the drawn λ = 0.5 as the label, I'd be telling the network "this is half B" while the image is 83.5% A. That's a gross miscalibration — exactly the kind of label-vs-pixel mismatch I just argued against in the one-hot case, except now I'd be inflicting it on myself through a coding shortcut.

So the fix follows directly: after pasting, recompute λ from the realized box,

  λ ← 1 − (bbx2 − bbx1)(bby2 − bby1) / (W·H),

i.e. one minus the true pasted-area fraction. Running the corner case through this: λ ← 1 − 0.165 = 0.835, so the label becomes 83.5% A — which is exactly the true surviving-A fraction. And note this same recomputation also absorbs the `int()`-rounding gap I found earlier, since it measures the box that actually got stamped in. This re-adjustment isn't cosmetic; it's what keeps the "label = area fraction" invariant true after both truncation and clipping.

And the loss has a shortcut worth pinning down. My target is the soft label λy_A + (1−λ)y_B. I claim cross-entropy is linear in the target, so CE(p, λy_A + (1−λ)y_B) = λ·CE(p, y_A) + (1−λ)·CE(p, y_B), which would let me avoid ever building the soft vector. Let me verify that on a tiny numeric case rather than trust it. Take a 3-class problem, logits (2.0, −1.0, 0.5); softmax is p = (0.7856, 0.0391, 0.1753). Let y_A be class 0 and y_B be class 2, with λ = 0.7. The right-hand side: CE(p, A) = −log 0.7856 = 0.2413, CE(p, B) = −log 0.1753 = 1.7413, so 0.7·0.2413 + 0.3·1.7413 = 0.6913. The left-hand side, against the genuine soft vector (0.7, 0, 0.3): −[0.7·log 0.7856 + 0·log 0.0391 + 0.3·log 0.1753] = 0.6913. They match to four places — which they must, since CE(p, t) = −Σ_c t_c log p_c is literally linear in t. So I never materialize the soft vector: I compute ordinary integer-label cross-entropy against y_A and against y_B and take the λ-convex combination, and it works with the stock CrossEntropyLoss that wants integer targets.

Everything else — the network, SGD with momentum, weight decay, the step schedule, the underlying crop/flip/normalize — is untouched; the only new code constructs the patched input and evaluates the paired target.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = datasets.CIFAR100(root='~/data', train=True, download=False,
                             transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                           shuffle=True, num_workers=8)

net = build_network()                       # e.g. a ResNet / PyramidNet
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.25, momentum=0.9, weight_decay=1e-4,
                      nesterov=True)
net = net.cuda()
mix_strength = 1.0                          # alpha; 1.0 -> lambda ~ Uniform(0,1)
mix_prob = 0.5                              # official CIFAR example; ImageNet uses 1.0


def rand_bbox(size, lam):
    # box of area (1-lam)*W*H, shape a scaled copy of the image, placed uniformly
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)             # side scale s = sqrt(1-lam) -> area frac s^2 = 1-lam
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)   # clip to borders -> true area may shrink
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def make_training_pairs(x, y, strength=mix_strength):
    if strength <= 0 or np.random.rand() >= mix_prob:
        return x, (y, y, 1.0)               # leave batch unchanged
    lam = np.random.beta(strength, strength)
    index = torch.randperm(x.size(0)).cuda()        # partner B = shuffled batch, no extra I/O
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]   # paste B's patch into A
    # re-adjust lambda to the TRUE pasted-area fraction after border clipping
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, (y_a, y_b, lam)


def compute_loss(criterion, outputs, target_spec):
    y_a, y_b, lam = target_spec
    # CE is linear in the target: loss on lam*y_a+(1-lam)*y_b
    #   == lam*CE(.,y_a) + (1-lam)*CE(.,y_b); no soft-label vector needed
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


def train(epoch):
    net.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, target_spec = make_training_pairs(inputs, targets, mix_strength)
        outputs = net(inputs)
        loss = compute_loss(criterion, outputs, target_spec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
