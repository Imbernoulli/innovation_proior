TrivialAugment settled the question I left open and, in doing so, exposed the ceiling of the whole family. On the hard pair it did exactly what I bet on: ResNet-56 on CIFAR-100 came in at $74.71$, *above* RandAugment's $74.35$ — confirming the previous rung was over-regularizing the hardest task with its two fixed-strong composed ops, and that easing to one op at random per-image magnitude, with zero knobs to mis-set, was the right correction. CIFAR-10 ResNet-20 landed $93.36$, a hair *below* RandAugment's $93.51$, as predicted, since single-op-at-random-magnitude is on average gentler and CIFAR-10 had the least headroom; FashionMNIST was the wash, $94.24$ versus $94.43$. But read the two CIFAR results together and the story is sobering: the two best photometric-policy rungs merely *trade* — RandAugment wins CIFAR-10 by $0.15$, TrivialAugment wins CIFAR-100 by $0.36$, neither dominates. I am moving accuracy *around* within the policy family, not reliably *up*. That is the signature of a saturating lever: I have largely exhausted what *diversifying photometric and geometric perturbations* can buy on these tasks.

So the productive question is no longer how to tune the policy's strength — both rungs answered that and it plateaued — but what *kind* of regularization the whole family structurally fails to do. RandAugment and TrivialAugment both *diversify*: they show the network recolored, rotated, sheared, solarized versions of each image, but every variant still contains the *whole object*. The policy moves and recolors content; it never *removes* it. So the network can still find the single most-discriminative region and bet on it — diversity makes that cue appear in many guises but never forces the network to survive *without* it. That is exactly the latch a small CIFAR ResNet falls into with limited capacity and data, and diversifying the cue's appearance does not break it; *deleting* the cue would.

I propose **Cutout**: for each training image, zero a contiguous square region of side $L$ at a uniform-random center. It is information *deletion* rather than diversification — dropout moved to input space and given a spatial prior — and three derivation steps make it work. The first is *why contiguous*. Dropout zeroes hidden activations at random and works in fully-connected layers by averaging over sub-networks, but it limps in convolutional nets for a structural reason: neighboring pixels and activations carry nearly the same information, so zeroing one unit removes nothing — the neighbors pass essentially the same signal forward, and the removal is undone by redundancy. The per-map variants that drop whole feature maps or the max activation attack this but fall *behind* plain dropout once batch normalization is present, and BN is everywhere in these models. The fix follows directly from the diagnosis: do not remove a pixel, remove a whole *contiguous* block. Delete a connected region and there are no surviving neighbors *inside* it to leak the information back; the block is genuinely gone, and contiguous removal forces a *global* understanding of the image to fill the hole — context, not just local features.

The second step is *where to remove it*. The per-map variants operate on feature maps individually, which is their hidden weakness: remove a region from one map and the same content survives in another, so the network still "sees" it and the representation just gets noisy. To make content genuinely disappear I remove it *once, at the input*, before any feature map is computed. Then the hole propagates through every subsequent map; no map anywhere still contains the removed pixels except whatever the network can *infer* from surrounding context — the difference between a noisy-but-complete representation and content that is actually absent. Removing a contiguous square at the input is also, conceptually, closer to *augmentation* than to noise: I am manufacturing plausibly-occluded images, and a net that has only ever seen whole objects is brittle when a key part is hidden. Show it a car with its wheels covered, and it must learn the rest of the image still says "car" — breaking the latch the policy family left intact.

The third step is the set of details that decide whether it works in this harness. The lever is *how much* I remove, not the silhouette, so I use a square with one knob, the side length $L$, at a uniform-random center. A subtlety matters: if every patch were forced fully inside and always size $L$, *every* image would lose exactly an $L\times L$ chunk and the net would overcorrect to a world where everything is always occluded. Instead I *clip* the patch coordinates to the image bounds — pick a random center $(y,x)$, take $y\pm L/2$ and $x\pm L/2$, clamp each to $[0,h]$/$[0,w]$, and zero that possibly-truncated region — so when the center lands near a border only a sliver actually zeros. That gives a mix of heavily- and barely-occluded images for free, which is what I want. The fill value and its placement are equally load-bearing. The mask runs *after* `ToTensor` and *before* `Normalize` and fills with zeros, so a zeroed pixel becomes $(0 - \text{mean})/\text{std}$ after normalization — a small constant near the per-channel dataset mean once the mean is subtracted, *not* a big black bias. Zeroing on the un-normalized $[0,1]$ tensor and then normalizing puts the patch at roughly the dataset mean, so it barely disturbs the batch statistics BN depends on; running the mask *after* `Normalize` would zero already-centered data and place the patch at a value that is not the mean, a worse choice. And there is no test-time rescaling: Cutout is not multiplicative noise whose expectation must be corrected, it is augmentation generating images, so at test time I feed the unmodified image — the fixed test transform is already just `ToTensor` + `Normalize`, which is correct.

The one value I cannot tune is $L$. The validated CIFAR-10 setting is $L=16$, and the harness fixes `length=16` for all three pairs. On CIFAR-10 that is optimal. On CIFAR-100, finer-grained with 100 classes, the guidance is a *smaller* hole ($L=8$, since fine detail matters more relative to context) — but I am given $16$ everywhere, so I expect Cutout to be optimally tuned for CIFAR-10 and slightly *over*-deleting on CIFAR-100, which bounds how much it can win the hard pair. On FashionMNIST the mask runs after the harness's channel-repeat, so the same $16\times16$ hole zeros all three repeated channels of the grayscale silhouette — a strong, content-relevant occlusion the color-muted policies could not deliver. I define the `Cutout` class inside `build_train_transform` and insert it after `ToTensor`, before `Normalize`, keeping the geometric floor underneath; note this rung *drops* the photometric policy entirely — Cutout is a different family, and the comparison I want is deletion-on-the-geometric-floor versus diversification-on-the-geometric-floor, not a stack of both. So I expect Cutout to clear both policy rungs on CIFAR-10 and lift FashionMNIST most reliably, while at least holding CIFAR-100; if so, *information deletion* is a genuinely stronger lever than *information diversification* on these tasks.

```python
# EDITABLE region of custom_augment.py (lines 246-275) -- step 3: Cutout
def build_train_transform(config):
    """Cutout augmentation: random square mask after ToTensor.

    Pipeline: RandomCrop + HFlip + ToTensor + Cutout(1, 16) + Normalize.
    """
    class Cutout:
        def __init__(self, n_holes=1, length=16):
            self.n_holes = n_holes
            self.length = length

        def __call__(self, img):
            h, w = img.size(1), img.size(2)
            mask = torch.ones_like(img)
            for _ in range(self.n_holes):
                y = torch.randint(0, h, (1,)).item()
                x = torch.randint(0, w, (1,)).item()
                y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
                x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
                mask[:, y1:y2, x1:x2] = 0
            return img * mask

    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(config['mean'], config['std']),
    ])
```
