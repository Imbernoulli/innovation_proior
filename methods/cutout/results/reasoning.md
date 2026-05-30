OK, let me think about this from the pain. I've got a big convolutional net that overfits, and the regularizer everyone reaches for — dropout — just doesn't pull its weight in the convolutional layers. In the fully-connected layers it's great: zero out activations at random, and you get this bagging-like averaging over a huge family of sub-networks, co-adaptation broken. But move it into the conv stack and it limps. Why?

Two reasons, and the second one is the interesting one. First, conv layers already have far fewer parameters than FC layers, so they need less regularization to begin with — fine, that's a matter of degree. The second reason is structural: in an image, *neighboring pixels carry almost the same information*. So if dropout zeroes one activation, or one input pixel, the thing it was carrying doesn't actually go away — the neighbors are still active and they pass essentially the same signal forward. The removal gets undone by redundancy. So conv-dropout doesn't average over sub-networks the way FC-dropout does; it just teaches the feature detectors to shrug off a bit of noise. Weaker effect, different mechanism.

People have tried to patch this. SpatialDropout drops a whole feature map at a time instead of single activations — that defeats the within-map neighbor redundancy, sure. Max-drop kills the single most-active unit across maps. Both help a little on their own. But here's the deflating part: once you put batch normalization in the network, both of them fall *behind* plain dropout again. So the patches don't survive contact with the rest of the modern toolbox, which is exactly the regime I care about. I want something that still helps when BN and heavy augmentation are already in place.

So let me not fight the diagnosis — let me use it. The reason pixel-wise removal fails is neighbor redundancy: drop one pixel, the neighbors fill it back in. The fix that follows directly is: don't remove a pixel, remove a whole *contiguous region*. If I delete a connected block, there are no surviving neighbors inside it to leak the information — the block is genuinely gone. That's the move neighbor-redundancy is begging for.

Is there precedent that contiguous removal does something pixel-wise removal can't? Yes — think about the autoencoder side. Denoising autoencoders erase scattered individual pixels and reconstruct; they learn local features. Context encoders erase a big *contiguous* region and reconstruct, and that forces the model to understand the image *globally* — it has to know what kind of thing is in the picture to fill the hole — so it learns higher-level features. Same lesson: a contiguous hole forces global/contextual reasoning in a way scattered pixels don't. Those were self-supervised reconstruction setups, but the principle — contiguous removal forces context — is exactly what I want to import into plain supervised classification.

Now the second design question: at which layer do I remove the region? The per-map dropout variants all operate on feature maps *individually*. That's their hidden weakness. If I drop a region from one feature map, the same content can still be present in another map, so the network as a whole still "sees" it — the representation just gets noisy and inconsistent, which again only teaches noise-robustness. To genuinely make the content disappear, I should remove it once, at the *input*, before any of the feature maps are computed. Then the hole propagates through every subsequent map — there's no map anywhere that still contains the removed pixels, except whatever the network can *infer* from the surrounding context. That's the difference between "noisy redundant representation" and "content actually absent." So: remove at the input layer, and remove a contiguous block.

Put those two together and the operation writes itself: take each training image and zero out a contiguous region of it at a random location. It's dropout, but moved to input space and given a *spatial prior* — instead of independent per-unit Bernoulli masks, I impose that the dropped units form a connected patch, the same way a CNN imposes spatial structure on an MLP. And conceptually it sits closer to data augmentation than to noise: I'm not perturbing activations, I'm manufacturing new, plausibly-occluded images. Which is also the right intuition for *why* it should help downstream — object occlusion is everywhere in real vision (tracking, pose, recognition), and a net that has only seen whole objects is brittle when a key part is hidden. Show it partially-occluded versions and it learns that a car with its wheels covered is still a car: it has to lean on the *rest* of the image, the full context, instead of betting everything on one most-telling feature.

What shape should the region be? I could agonize over rectangles, ellipses, irregular masks. But the thing that actually controls the effect is *how much* I remove, not the silhouette — the amount of context I take away is the lever; the outline barely matters. So I'll use a square and tune one number, the side length L. One knob.

Where do I put it? Pick a center pixel uniformly at random over the image and lay the square around it. And now a subtlety that turns out to matter a lot. If the center is near an edge, the square sticks out past the image border. Do I forbid that — force the whole patch to stay inside — or allow it to hang off the edge? Intuitively "keep it inside" feels cleaner. But think about what "always fully inside, always size L" does: *every* training image loses exactly an L×L chunk, no exceptions. The net never sees a mostly-intact image. If instead I let the square hang off the border, then whenever the center lands near an edge only a sliver actually gets zeroed, so a good fraction of images stay largely visible. That mix — some heavily occluded, some barely touched — is what I want: the model needs clean-ish examples too, or it overcorrects to a world where everything is always occluded. So I deliberately allow the patch to extend past the borders. (Equivalently I could keep the patch fully inside but only apply cutout to, say, half the images so the other half are untouched — same "mix of corrupted and clean" principle, and it works about as well.) This "you need some clean images" point is easy to miss and probably why such a simple trick wasn't already standard.

Two implementation details that keep it honest. I'm filling the hole with *zeros*, so I want the data normalized to roughly zero mean per channel — then a zeroed patch sits at about the dataset mean and barely disturbs the batch statistics that BN depends on, rather than injecting a big black bias. (Filling with zeros on un-normalized data, or with random noise, would distort the running stats and add spurious signal.) And unlike dropout, there's *no* test-time rescaling. Dropout scales activations at test time to match the training-time expectation of a multiplicative mask. But cutout isn't a multiplicative noise whose expectation I need to correct — it's data augmentation, generating images. At test time I just feed the unmodified image, exactly as I would with crops and flips. No correction.

One more thing I should resolve: should the removed region be *targeted* at the most salient part of the object, or just random? The targeted version is tempting — find the maximally-activated region (upsample the top feature map, threshold at its mean to get a mask) and occlude *that*, so the net is forced off its favorite feature, the max-drop instinct but in input space. It does work. But it means storing and processing saliency maps every epoch — real machinery. And when I compare, random fixed-size squares perform *just as well*. So the targeting buys nothing over a uniformly-placed random square, and the random version is trivially simple and free. Take the simple one.

That's the whole method. And it's almost free to compute: it's a per-image transform, so it runs on the CPU during data loading, in parallel with the GPU doing the actual training — the cost hides behind the forward/backward pass. Implementation is a transform appended to the pipeline after normalization: build an all-ones mask, zero a clipped L×L square around a random center, multiply it into the image.

```python
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Cutout(object):
    """Zero out n_holes square patches of side `length` from an image tensor."""

    def __init__(self, n_holes, length):
        self.n_holes = n_holes          # 1 patch is enough
        self.length = length            # the one knob, L; tune per dataset

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)    # center chosen uniformly over the image...
            x = np.random.randint(w)

            # ...and clipped to the border, so a patch near an edge sticks out
            # and only partly lands -> some images stay mostly visible
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.      # contiguous hole: no neighbors inside to leak

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)       # same hole across all channels (drop at input)
        return img * mask                # zeros == ~dataset mean after normalization


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    Cutout(n_holes=1, length=16),        # appended last; no test-time transform / rescale
])
trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=8)

# network, nn.CrossEntropyLoss(), SGD(Nesterov, mom=0.9, wd=5e-4), step-decay,
# standard train loop — all unchanged; cutout runs on CPU during loading, ~free.
```

The chain: dropout under-regularizes conv nets because neighboring pixels are redundant, so pointwise removal is undone by the neighbors, and per-map variants leave the removed content alive in other maps and die under batch norm. Remove the content once at the input, as a contiguous square, and it is genuinely gone from every downstream feature map — the network must recover the class from the surrounding context, which is exactly occlusion-robustness and whole-image reasoning. Place the square at a uniform random center and let it hang off the borders so a mix of heavily- and barely-occluded images is seen; fill with zeros on zero-mean-normalized data so batch statistics are undisturbed; skip any test-time rescaling because this is augmentation, not multiplicative noise; tune the single side-length knob. It stacks on top of batch norm, dropout, and standard augmentation, and runs for free on the CPU during data loading.
