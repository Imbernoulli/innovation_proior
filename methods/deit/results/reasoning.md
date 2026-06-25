The thing I want to settle is whether a pure-attention image classifier is *intrinsically* data-hungry, or whether it just hasn't been trained right. The recent result is striking: take a Transformer, chop an image into 16×16 patches, treat each patch as a token, run the standard NLP transformer stack, read out a class token, and you get ImageNet accuracy that rivals strong convnets. The catch is the demonstration leaned on a private dataset of hundreds of millions of labeled images, and the stated conclusion was that these transformers "do not generalize well when trained on insufficient amounts of data." That conclusion is the thing I distrust. Convnets generalize from ImageNet-1k's ~1.2M images because convolution bakes in priors — locality, translation equivariance — that match how images are built. A transformer has almost none of that prior; self-attention is happy to relate any patch to any other, so it has to *learn* image structure from scratch. So of course it needs more data. But "needs more data" and "needs 300 million images" are different claims, and the 300M figure came bundled with a training recipe tuned for that regime — heavy weight decay, no strong augmentation. Before I accept that the architecture is intrinsically data-hungry, I want to separate the architecture from the recipe: train the *same* network on ImageNet-1k alone and push the recipe as hard as I can, and see where the accuracy actually lands. If there's still a gap after that, the next question is whether I can hand the transformer the spatial prior it's missing without putting convolutions into it — and a teacher convnet is the obvious place such a prior already lives. Let me work through the recipe first, then come back to the teacher.

First let me be clear about the architecture I'm keeping, because I'm not changing it — I'm changing how it's trained. An image at 224² is cut into N = 14×14 = 196 patches of 16×16 pixels; each patch (3·16·16 = 768 numbers) is linearly projected to width D. Self-attention is permutation-invariant, so I add a positional embedding to each patch token. I append a learnable class token to the sequence; it flows through all the blocks, mixing with the patches through self-attention, and its final state is linearly projected to the class logits — that's the readout, replacing a convnet's global pool. Each block is the standard thing: a multi-head self-attention sublayer, softmax(QKᵀ/√d)·V with the √d guarding the softmax from saturating, then a feed-forward network of two linear layers with a GeLU between, widening D→4D and back, both sublayers residual and layer-normed. No batch norm anywhere, which I note now because it'll matter — it means I can shrink the batch size without poisoning normalization statistics.

Now the training recipe, which is where the data-efficiency has to come from. The first thing I'd reach for out of habit on a vision model is SGD with momentum, the way I'd train a ResNet. Let me sanity-check that against AdamW. The gap is not subtle — SGD pretraining lands around 74.5% top-1 where AdamW reaches about 81.8%. Seven points. Transformers just don't train well with plain SGD; the adaptive per-parameter scaling of Adam, with decoupled weight decay, is doing real work here. So AdamW it is, and SGD is off the table for pretraining.

Next, the optimization hyperparameters carried over from the large-scale recipe. The large-scale setup used a heavy weight decay of 0.3. Let me think about why that might be wrong here. Heavy weight decay is a regularizer you can afford — even want — when you have 300M images and overfitting is the enemy. On 1.2M images with the heavy *augmentation* I'm about to pile on, the effective regularization is already strong, and an additional 0.3 weight decay over-constrains the weights and hurts convergence. Drop it to about 0.05. And scale the learning rate with batch size, the standard linear rule, lr = base × (batch/512) — I use 512 as the base denominator. With cosine decay and a warmup of a few epochs (transformers are touchy early in training), 300 epochs. One more touchy thing: initialization. Transformers are sensitive to it — in preliminary tries some initializations simply don't converge — so initialize the weights from a truncated normal, which is the stable choice.

Now the heart of the recipe: augmentation, which is how I substitute "more data" for the data I don't have. The logic is direct — the transformer lacks the convnet's spatial priors, so I have to show it more varied data to teach it the invariances a convnet gets for free. So pile on strong augmentation: a learned augmentation policy (RandAugment — I find it beats AutoAugment here), Mixup (blend two images and their labels), CutMix (paste a patch of one image into another and mix labels proportionally), and random erasing (blank out a random region). Does it help? Look at what happens when I remove pieces. Knock out *both* Mixup and CutMix and accuracy collapses from ~81.8 to ~75.8 — six points gone. These label-mixing augmentations are not optional garnish; they're central. Almost every augmentation I test helps. The one exception is dropout — it actually hurts here, so I cut it. In its place for regularizing depth I use stochastic depth (randomly skip whole residual blocks during training at rate ~0.1), which is known to ease the convergence of deep transformer stacks; dropping blocks is a better-matched regularizer for this architecture than dropping activations.

There's one augmentation that turns out to matter more than I'd expect, so let me single it out: repeated augmentation. The idea is that within a batch, instead of one augmented view per image, you include *several* augmented views of the same images (I use 3 repetitions). It sounds like it would just waste batch capacity on redundant samples, but it gives a significant boost — removing it drops accuracy by several points (to ~76.5). The intuition is that seeing multiple augmentations of the same image in one gradient step produces a more useful averaged gradient for learning the invariances, and it interacts well with the label-mixing augmentations. A practical consequence: with 3 repetitions I only see a third of the distinct images per "epoch," so what I call 300 epochs is really 100 passes over the data, each 3× longer — I keep the 300-epoch label so training time compares directly with non-repeated training. And label smoothing at ε = 0.1 on the true labels, to keep the classifier from getting overconfident.

So far this is recipe, not architecture, and it's already enough to get a convolution-free transformer to convnet-competitive accuracy on ImageNet-1k alone, in a couple of days on one node — which answers the first question: the data-hunger was the recipe, not the architecture. Now the second question, distillation, where I want to actually add something transformer-specific.

The standard tool is soft distillation: train the student to match a teacher's softened output distribution. Concretely, minimize a mix of the ground-truth cross-entropy and the KL divergence between teacher and student softmaxes at temperature τ,

  L = (1−λ)·L_CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ), ψ(Z_t/τ)).

The τ² prefactor is the part I want to make sure I understand before I rely on it, because if I get it wrong the distillation term will be silently mis-weighted against the hard-label term. The claim usually made is that it compensates for the softening. Let me actually differentiate to see what the softening does to the gradient. With p = ψ(Z_s/τ) and q = ψ(Z_t/τ), the temperature-τ distillation cross-entropy is −Σ_j q_j log p_j, and its derivative with respect to student logit Z_s(i) is (1/τ)·(p_i − q_i) — the chain rule pulls a 1/τ out of the Z_s/τ inside the softmax. So the bare softened term carries an explicit 1/τ. I want to check that and see how the *whole* gradient scales, so I take a random pair of 1000-class logit vectors and compute. The analytic (1/τ)(p−q) matches a finite-difference gradient to eight digits at τ = 1, 2, 3, 5, so the 1/τ factor is real. And the magnitude of the full gradient vector falls off faster than 1/τ — ‖∇‖ goes 5.2e-2, 1.1e-2, 4.9e-3, 1.7e-3 as τ goes 1, 2, 3, 5 — because the (p − q) difference itself also flattens as the softmaxes get softer, so the two effects compound to roughly 1/τ². Multiplying the loss by τ² puts it back: ‖τ²·∇‖ comes out 5.2e-2, 4.5e-2, 4.4e-2, 4.3e-2, essentially flat in τ. So the τ² is exactly what keeps the distillation gradient on the same footing as the hard-label gradient as I vary the temperature — good, I can trust the formula. But notice what this cost: the soft loss has two knobs to tune (τ and λ), and it asks the student to imitate a full distribution. Let me ask whether there's something simpler that fits this setting better.

What if I treat the teacher's *decision* — its top-1 prediction — as if it were a ground-truth label? Let y_t = argmax_c Z_t(c) be the teacher's hard call. Then the loss is just two cross-entropies, one against the true label and one against the teacher's label, weighted equally:

  L = ½·L_CE(ψ(Z_s), y) + ½·L_CE(ψ(Z_s), y_t).

This is parameter-free — no τ, no λ to cross-validate — and conceptually the teacher's label y_t plays exactly the same role as the true label y. There's a subtle bonus: because I re-evaluate the teacher on the *augmented* crop the student sees, the hard label y_t can differ from the dataset label when the augmentation has mangled the image (a CutMix'd or heavily-erased crop might genuinely look more like the teacher's call than the original class), so the teacher provides a label that's consistent with what the student is actually looking at. And it can be softened with label smoothing if I want. When I compare, this hard-label distillation comes out ahead of the soft version in my setting, so I'll build on it.

Now the part I actually want to invent, which is *how* the teacher signal enters a transformer. The naive thing is to put both losses on the single class-token output. But the class token is already being pulled toward the true label; asking it to also reproduce the teacher's label is asking one vector to serve two masters. The transformer gives me a cleaner option that a convnet doesn't: I can just add another token. So introduce a second learnable token — a distillation token — alongside the patch tokens and the class token. It's used exactly like the class token: it's prepended to the sequence, it interacts with the patches and with the class token through self-attention at every layer, and its final state is read out by its own linear classifier. The only difference is its target: the class token is supervised by the true label, the distillation token by the teacher's (hard) label. Both tokens are learned by backprop. My hope is that this gives the network two readout pathways — one tuned to the dataset labels, one tuned to the teacher's labels — that share the same patch representation but don't have to compromise with each other. But that's only a hope until I show the two tokens actually diverge; if they don't, I've just paid for a duplicate.

I should check this isn't a trivial reparameterization — maybe two tokens with the same target would do just as well, in which case the "distillation" framing is empty. So run the control: use a second *class* token (same true-label target as the first), initialized randomly and independently. What happens during training is telling — the two class tokens converge to essentially the same vector (cosine similarity 0.999) and produce near-identical outputs, and the second one adds nothing to accuracy. Identical targets make the tokens redundant; the network has no reason to keep them apart. Contrast the real distillation token: it and the class token converge to genuinely *different* vectors — at the input their cosine similarity is about 0.06, and although they grow more aligned as they pass through the layers (since true label and teacher label are usually the same answer), even at the last layer their similarity is about 0.93, still short of 1. They're solving similar-but-not-identical problems, so they stay distinct, and that distinctness is where the gain comes from. The distillation token is doing real work precisely because its target differs.

At test time I have two classifiers — the class head and the distillation head — and a choice of how to predict. I could use either alone. But they encode complementary information (one true-label-tuned, one teacher-tuned), so the natural thing is to fuse them: add the softmax outputs of the two heads and predict the argmax. Late fusion of the two heads is my default readout.

One more design choice about distillation, and it's the one I find most interesting. Which teacher? The obvious instinct is to distill from another transformer of comparable or higher accuracy. But test it against distilling from a *convnet* teacher of similar performance, and the convnet teacher gives the better student. Why would a transformer learn more from a convnet than from a peer transformer? Because distillation transfers more than the label — it transfers the teacher's *inductive bias*. The convnet has the locality/translation-equivariance prior the transformer lacks, and by training the transformer to match a convnet's decisions (especially across all those augmented crops), I'm effectively teaching the transformer to behave as if it had that prior. A transformer teacher has nothing extra to give in that regard. So use a strong convnet as the teacher — that's the choice that most directly addresses the transformer's original weakness.

That leaves resolution. Following the train-low/fine-tune-high idea, I pretrain at 224² and fine-tune at 384², which is faster overall and more accurate under strong augmentation. The patch size stays 16, so going to 384² raises the number of patches N from 196 to a larger grid. The transformer and the classifiers don't care — self-attention handles more tokens unchanged — but there are exactly N positional embeddings, one per patch, and now I need more of them. So I resize the positional-embedding grid by interpolation. My first instinct is bilinear, the default for image resizing, and at first I don't expect any subtlety. But when I think about what bilinear interpolation does to a *vector-valued* grid, something bothers me. Each interpolated position is a convex combination Σ_k w_k a_k of its neighbour embeddings, w_k ≥ 0, Σ w_k = 1 — i.e. a weighted average. For two equal-norm neighbours the algebra is exact: ‖(a+b)/2‖ = r·√((1+cosθ)/2), which equals r only if the two vectors are identical (θ = 0) and is strictly smaller otherwise. These embeddings are 768-dimensional vectors for *distinct* positions, so they are nowhere near collinear, and the average should come out noticeably shorter than its inputs. Let me put numbers on it before deciding it matters: I take four neighbour vectors at the scale of the trained embeddings (norm ≈ 0.55 each) and average them as bilinear would at a cell centre. The interpolated vector lands at norm ≈ 0.27 — roughly *half* the neighbours' norm — and I see the same ~0.5 ratio across repeated draws. That is not a rounding effect; it is a systematic shrink. And the pretrained transformer was tuned to positional vectors of the original magnitude, so feeding it embeddings at half-norm should throw the attention logits off — which is consistent with the sharp accuracy drop I get when I use bilinearly-resized embeddings directly. The fix is to interpolate with bicubic instead, whose negative side-lobe weights let the reconstructed vector overshoot toward its neighbours rather than strictly average them, so it approximately preserves the norm; then fine-tune. During fine-tuning I keep the strong training-time augmentation rather than dampening it, and I fine-tune with the teacher too (both true label and teacher label, the teacher taken at the matching resolution) — using true labels only at fine-tune time throws away the teacher's benefit. AdamW or SGD both work for this short fine-tune.

A nice property falls out of the no-batch-norm point I flagged earlier: since there's no batch-norm whose statistics degrade at small batch sizes, I can train with smaller batches without losing accuracy, which makes the larger models easier to fit on a node.

Let me put it down, grounded in how the distilled transformer and its loss actually get built — the distillation token added to the sequence, the two heads, and the distillation loss that combines the supervised and teacher signals.

```python
import torch
from torch import nn
import torch.nn.functional as F


class DistilledTransformer(nn.Module):
    """Patch-token transformer with BOTH a class token and a distillation token."""
    def __init__(self, img=224, patch=16, in_ch=3, dim=768, depth=12, heads=12,
                 num_classes=1000, drop_path=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        n_patches = (img // patch) ** 2
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))   # the new token
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_patches + 2, dim))  # +2: cls + dist
        dpr = torch.linspace(0, drop_path, depth)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, drop_path=float(dpr[i]))
                                      for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head      = nn.Linear(dim, num_classes)   # supervised by TRUE label
        self.head_dist = nn.Linear(dim, num_classes)   # supervised by TEACHER label
        for p in (self.cls_token, self.dist_token, self.pos_embed):
            nn.init.trunc_normal_(p, std=0.02)         # transformers are init-sensitive

    def forward_features(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)            # (B, N, dim)
        cls  = self.cls_token.expand(x.size(0), -1, -1)
        dist = self.dist_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.pos_embed         # cls, dist, patches
        x = self.norm(self.blocks(x))
        return x[:, 0], x[:, 1]                                       # (class out, distill out)

    def forward(self, x):
        x_cls, x_dist = self.forward_features(x)
        y, y_dist = self.head(x_cls), self.head_dist(x_dist)
        if self.training:
            return y, y_dist                                         # two losses at train time
        return (y.softmax(-1) + y_dist.softmax(-1)) / 2              # LATE FUSION at test


class HardDistillationLoss(nn.Module):
    """L = 1/2 CE(student_cls, true) + 1/2 CE(student_dist, teacher_argmax).
    Teacher is re-evaluated on the SAME augmented input the student sees."""
    def __init__(self, teacher):
        super().__init__()
        self.teacher = teacher.eval()

    def forward(self, inputs, outputs, labels):
        out_cls, out_dist = outputs
        with torch.no_grad():
            teacher_labels = self.teacher(inputs).argmax(dim=1)      # convnet teacher
        base = F.cross_entropy(out_cls, labels)                      # true-label head
        distill = F.cross_entropy(out_dist, teacher_labels)          # teacher-label head
        return 0.5 * base + 0.5 * distill


class SoftDistillationLoss(nn.Module):
    """Alternative: (1-lambda) CE(student,true) + lambda * tau^2 KL(student/tau || teacher/tau)."""
    def __init__(self, teacher, tau=3.0, lam=0.1):
        super().__init__()
        self.teacher, self.tau, self.lam = teacher.eval(), tau, lam

    def forward(self, inputs, outputs, labels):
        out_cls, out_dist = outputs
        with torch.no_grad():
            t = self.teacher(inputs)
        base = F.cross_entropy(out_cls, labels)
        T = self.tau
        kd = F.kl_div(F.log_softmax(out_dist / T, dim=1),
                      F.log_softmax(t / T, dim=1),
                      reduction='sum', log_target=True) * (T * T) / out_dist.numel()
        return (1 - self.lam) * base + self.lam * kd


def resize_pos_embed(pos_embed, old_grid, new_grid):
    """Higher-res fine-tuning: interpolate the patch positional embeddings with BICUBIC
    (bilinear shrinks the vector norm and breaks the pretrained model)."""
    cls_dist = pos_embed[:, :2]                                      # keep cls + dist embeds
    patch_pe = pos_embed[:, 2:]
    B, _, D = patch_pe.shape
    patch_pe = patch_pe.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    patch_pe = F.interpolate(patch_pe, size=(new_grid, new_grid),
                             mode='bicubic', align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    return torch.cat([cls_dist, patch_pe], dim=1)


# Training: AdamW, lr = 5e-4 * batch/512, cosine, weight decay 0.05, warmup 5, 300 epochs,
# RandAugment + Mixup(0.8) + CutMix(1.0) + Erasing(0.25) + repeated augmentation (x3),
# stochastic depth 0.1, label smoothing 0.1, NO dropout, truncated-normal init.
```

The causal chain: a convolution-free patch-token transformer had only been shown to work after pre-training on hundreds of millions of images, with the conclusion that it needs huge data — but that data-hunger comes from the transformer's lack of spatial priors, which I can compensate without adding convolutions. So I train the same architecture on ImageNet-1k alone with AdamW (SGD trains it far worse), a lower weight decay than the large-scale recipe (whose heavy decay hurts here), heavy label-mixing augmentation plus repeated augmentation (manufacturing the missing data — removing Mixup/CutMix or repeated-aug costs many points), stochastic depth instead of dropout, and truncated-normal init — which already reaches convnet-competitive accuracy in a couple of days. Then to distill, I use parameter-free hard-label distillation (treating the teacher's top-1 as a label) routed through a *new distillation token* that sits beside the class token and is supervised by the teacher rather than the true label — verified non-redundant because a duplicate class token collapses into the original while the distillation token stays distinct — fused with the class head at test time, distilled preferentially from a *convnet* teacher whose inductive bias the transformer thereby inherits; finally I fine-tune at higher resolution, resizing the positional embeddings with norm-preserving bicubic interpolation.
