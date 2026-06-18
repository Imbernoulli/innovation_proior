OK, let me think this through from scratch. I've got a CNN classifier and the thing I keep running into is that it bets everything on one small patch of the object. Look at a class activation map for a trained net on a dog: it's one hot blob over the face. Occlude the face and the prediction falls apart. That's the same fragility that hurts generalization — the network never learned that the body, the legs, the ears are also evidence — and it's why weakly-supervised localization is so bad, because the "where is the dog" signal is just that one blob, not the dog.

So how do people fight this? The trick that's around is regional dropout. Dropout itself (Srivastava and co.) randomly zeroes hidden activations so no single unit can dominate; the natural transplant to the input is to zero a *region* of the image. Cutout (DeVries and Taylor) does exactly that — cut a square out of the image, set it to zero, at a random spot. Random Erasing fills the hole with noise instead. Hide-and-Seek hides a grid of patches. The logic is clean and it works: if I randomly delete the most discriminative patch some fraction of the time, the network is forced to find class evidence in the pixels that remain, so the evidence spreads over the whole object. Accuracy up, localization up.

But stare at what Cutout actually does to a training image. A 32×32 CIFAR image with a, say, 16×16 black square stamped on it. A quarter of the image is now *zero*. Those pixels contribute nothing to the forward pass, nothing to the loss, nothing to the gradient. I've thrown a quarter of every image in the trash. And CNNs are notoriously data-hungry — the whole reason augmentation exists is that we never have enough informative pixels. Deliberately blanking a chunk of every image to get the regularization feels like paying for the regularization in the one currency I can least afford. Can I get the regional-dropout effect *without* wasting the deleted region?

Let me look at the other side of the augmentation world for a contrast: Mixup (Zhang and co.). It refuses to delete anything. It takes two images and forms the per-pixel convex blend x̃ = λx_A + (1−λ)x_B, and — crucially — the matching blend of the one-hot labels ỹ = λy_A + (1−λ)y_B, with λ from a Beta distribution. Every pixel of the output is a real combination of real pixels; nothing is wasted; and the soft label tracks the mixture. So Mixup has exactly the property Cutout lacks: full use of pixels. And it works for classification.

But Mixup has the dual problem. Look at a Mixup image: it's a ghost. Two photographs superimposed at 60/40 opacity, a translucent dog floating over a translucent cat. No real image ever looks like that. And the class activation map on such an input is a mess — the network can't decide where either object is, because neither object is really *there*, both are half-erased everywhere. That diffuseness is fine-ish for a label that says "60% dog," but it's poison for localization: there's no contiguous, sharp object to point at. And notice — Mixup doesn't drop a region at all. There's no occluded patch forcing whole-object attention; it's a global wash. So Mixup gives up the very thing regional dropout is good at.

So I've got two methods that are each other's complement. Cutout: a real deleted region (good for localization, whole-object attention), but the region is wasted (bad for data efficiency). Mixup: every pixel informative (good for efficiency), but no deleted region and the image is unnatural (bad for localization). I want the intersection: a deleted region that is *also* informative, and an image that stays locally natural.

Now the two descriptions almost collide on their own. Cutout deletes a region and fills it with zeros. What if I fill that region not with zeros, but with the corresponding region *from another training image*? I cut a rectangle out of image A and paste in the same-shaped rectangle of pixels from image B. Look at what each desideratum says about this. Is there a deleted region of A? Yes — a contiguous rectangle of A is gone, exactly the occlusion Cutout wanted, so the network still has to recognize A from a partial view; the whole-object/localization benefit survives. Are the pasted pixels informative? Yes — they're real pixels of a real image B, not zeros, so nothing is wasted; the data-efficiency virtue of Mixup is recovered. Is the composite locally natural? Yes — and this is the part that beats Mixup: it's a real patch of B sitting on a real patch of A with a sharp boundary, not a translucent overlay. Locally, everywhere, the pixels look like a genuine photograph; there's no ghosting. The activation map can find a clean B-object in the pasted rectangle and a clean A-object in what remains.

So the operation is: pick a binary mask M ∈ {0,1}^{W×H} that's 1 on the "keep A" region and 0 on the "paste B" rectangle, and form

  x̃ = M ⊙ x_A + (1 − M) ⊙ x_B.

Now the label. The composite genuinely contains class-A pixels and class-B pixels — it is, in a concrete spatial sense, part A and part B. What should I tell the network it is? If I commit to a single one-hot label — say "it's A because most of it is A" — I'm lying about the B patch and I'm back to a hard target that ignores the construction. Label smoothing already taught me a soft target is fine and useful. And Mixup taught me the honest move: mix the labels in the same proportion the inputs were mixed. So

  ỹ = λ y_A + (1 − λ) y_B,

where λ is the *fraction of the image that is A*. The supervision now matches the pixel composition: if 70% of the image is A, the target is 70% A.

For this to be coherent, λ has to be exactly the area fraction occupied by A, and the rectangle of B has to occupy fraction 1−λ. So I need to draw a box whose area is a 1−λ fraction of the image. Let me draw λ ~ Beta(α, α), same family Mixup uses — symmetric because A-vs-B is an arbitrary labelling. Then I want a rectangle of area (1−λ)·WH. I have freedom in the shape; the natural choice is to make the box a scaled copy of the image, so its width and height each scale by the same factor. If r_w = W·s and r_h = H·s, the area fraction is r_w r_h / (WH) = s². I want that to equal 1−λ, so s = √(1−λ):

  r_w = W √(1−λ),  r_h = H √(1−λ).

And place it uniformly: the center r_x ~ Unif(0, W), r_y ~ Unif(0, H). The box is the region where M = 0 (filled from B); everywhere else M = 1 (kept from A). Then the kept-A area fraction is 1 − (1−λ) = λ, which is exactly the weight λ I put on y_A. The geometry and the label agree by construction. Good — that's the whole method in two equations.

Let me pressure-test the label choice before I build it, because it's the one place I could be fooling myself. Suppose I instead used a one-hot label, committing to whichever class owns the larger area. Then a sample that is 55% A, 45% B gets supervised as pure A — the 45% of the image that is a perfectly clear B-object is being called A, and the gradient pushes the network to read B-pixels as A-evidence. That's actively wrong and it throws away the cleanest signal in the image. What about always using 0.5/0.5 regardless of the actual ratio? Then a sample that's 90% A, 10% B is told it's half B, which is also miscalibrated — now a tiny B-patch is supposed to account for half the prediction. The only label that doesn't lie is the area-proportional one. So ỹ = λy_A + (1−λ)y_B with λ = area(A) it is. (I'd want to confirm both the one-hot and the fixed-half variants actually train worse — they should, by this argument.)

Why a contiguous rectangle, and not, say, Mixup's blend or a scatter of pixels from B? Because the localization benefit is the entire point of the exercise, and it requires a *coherent occlusion*. A contiguous patch of B forces the network to recognize the partial A-object from whatever spatial region remains, the way a real occluder would — that's what spreads A's evidence over A's whole visible extent. A scattered or translucent mixture doesn't occlude anything coherently; it just adds noise everywhere, which is the Mixup failure. Contiguity is load-bearing.

Why apply this at the pixel level and not on a feature map? The phenomenon I'm fixing — "the net bets on one discriminative patch, occlude it and it dies" — is a statement about the function of the *input image*: occlusion, decision-boundary behavior near real images, where the object is. If I mix feature maps instead, the network gets to choose a representation that makes the constraint easy to dodge, and the occlusion no longer corresponds to a spatial occlusion of the object. The cleanest place to impose it is the rawest one, the input. (I'd expect feature-level mixing to still help somewhat — it's still a regularizer — but input-level to be best.)

Which Beta? λ ~ Beta(α, α). At α = 1, Beta(1,1) is Uniform(0,1), so the area fraction is uniform — every patch size equally likely, maximally diverse occlusions, from a sliver to nearly the whole image. Small α piles mass at 0 and 1 (almost-no-mix or almost-full-swap); large α concentrates near half-and-half. Uniform is the natural default and it's the one knob; I'll sweep α to confirm but expect α = 1 to be the sweet spot for maximally diverse patches. The probability of applying the operation is a separate training knob; for the CIFAR scaffold below I use 0.5, while the large ImageNet run can use 1.0.

Now implementation, and there are two things to get right. First, the pairing: I don't need a second data loader to get image B. Within a minibatch the examples are already i.i.d. draws, so I can pair the batch with a shuffled copy of itself — one random permutation of the batch indices gives me a random partner B for every A, free of extra I/O. Same trick Mixup uses.

Second — and this is a subtlety the clean math hides — when I actually place the box, I clip it to the image borders. If the sampled center is near an edge, part of the nominal box falls outside the image and gets cut off, so the *actual* pasted rectangle is smaller than the nominal (1−λ)·WH. If I keep using the nominal λ for the label, the label is now miscalibrated against the true pixel ratio. So after pasting, I recompute λ from the realized box:

  λ ← 1 − (bbx2 − bbx1)(bby2 − bby1) / (W·H),

i.e. one minus the true pasted-area fraction. Now the label weight tracks exactly how many pixels of A actually survived. This re-adjustment isn't cosmetic — it's what keeps the "label = area fraction" invariant true after clipping.

And the loss has the same shortcut Mixup exploits. My target is the soft label λy_A + (1−λ)y_B. Cross-entropy is linear in the target, so

  CE(p, λy_A + (1−λ)y_B) = λ·CE(p, y_A) + (1−λ)·CE(p, y_B).

So I never materialize the soft vector — I compute ordinary integer-label cross-entropy against y_A and against y_B and take the λ-convex combination. Works with the stock CrossEntropyLoss that wants integer targets. Everything else — the network, SGD with momentum, weight decay, the step schedule, the underlying crop/flip/normalize — is untouched; the only new code constructs the patched input and evaluates the paired target.

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

The chain: a classifier bets on one discriminative patch, so occlusion and localization both fail; regional dropout (Cutout) fixes that by deleting a region, but the deleted region is wasted zeros; Mixup wastes nothing by blending whole images but its translucent composite is unnatural and has no coherent occlusion, so localization suffers. Fill Cutout's deleted region with a real patch cut from another image instead of zeros — now there is a genuine occluded region (whole-object attention survives), no pixel is wasted (data-efficient), and the composite is locally natural (sharp real patches, no ghost). Set the soft label to the area-proportional mix λy_A + (1−λ)y_B with λ the surviving-A fraction; draw the box as a √(1−λ)-scaled copy of the image placed uniformly, re-adjust λ to the true clipped area, and exploit cross-entropy's linearity to train against the two integer labels with weights λ and 1−λ — a few lines on top of an otherwise unchanged training loop.
