The thing I want to settle is whether a pure-attention image classifier is *intrinsically* data-hungry, or whether it just hasn't been trained right. The recent result is striking: take a Transformer, chop an image into 16×16 patches, treat each patch as a token, run the standard NLP transformer stack, read out a class token, and you get ImageNet accuracy that rivals strong convnets. The catch is the demonstration leaned on a private dataset of hundreds of millions of labeled images, and the stated conclusion was that these transformers "do not generalize well when trained on insufficient amounts of data." That conclusion is the thing I distrust. Convnets generalize from ImageNet-1k's ~1.2M images because convolution bakes in priors — locality, translation equivariance — that match how images are built. A transformer has almost none of that prior; self-attention is happy to relate any patch to any other, so it has to *learn* image structure from scratch. So of course it needs more data. But "needs more data" and "needs 300 million images" are different claims, and the 300M figure came bundled with a training recipe tuned for that regime — heavy weight decay, no strong augmentation. Before I accept that the architecture is intrinsically data-hungry, I want to separate the architecture from the recipe: train the *same* network on ImageNet-1k alone and push the recipe as hard as I can, and see where the accuracy actually lands. If there's still a gap after that, the next question is whether I can hand the transformer the spatial prior it's missing without putting convolutions into it — and a teacher convnet is the obvious place such a prior already lives. Let me work through the recipe first, then come back to the teacher.

First let me be clear about the architecture I'm keeping, because I'm not changing it — I'm changing how it's trained. An image at 224² is cut into N = 14×14 = 196 patches of 16×16 pixels; each patch (3·16·16 = 768 numbers) is linearly projected to width D. Self-attention is permutation-invariant, so I add a positional embedding to each patch token. I append a learnable class token to the sequence; it flows through all the blocks, mixing with the patches through self-attention, and its final state is linearly projected to the class logits — that's the readout, replacing a convnet's global pool. Each block is the standard thing: a multi-head self-attention sublayer, softmax(QKᵀ/√d)·V with the √d guarding the softmax from saturating, then a feed-forward network of two linear layers with a GeLU between, widening D→4D and back, both sublayers residual and layer-normed. No batch norm anywhere, which I note now because it'll matter — it means I can shrink the batch size without poisoning normalization statistics.

Now the training recipe, which is where the data-efficiency has to come from. The first thing I'd reach for out of habit on a vision model is SGD with momentum, the way I'd train a ResNet. But this architecture has no batch norm to keep activations well-scaled, and self-attention's landscape seems less likely to tolerate a single global step size than a convnet's does — which argues for an optimizer with per-parameter adaptivity instead. Rather than trust the argument alone, the way to settle it is a matched-budget swap: pretrain the same architecture on the same data for the same schedule length, changing only SGD-with-momentum versus AdamW's adaptive per-parameter scaling and decoupled weight decay. My prediction is that SGD undertrains this architecture by a wide margin under that matched budget while AdamW closes most of the gap toward convnet-level accuracy; if instead the two land close together, SGD's simplicity would be worth keeping. Whichever optimizer wins that matched-budget comparison is what I use for pretraining, and I expect that to be AdamW.

Next, the optimization hyperparameters carried over from the large-scale recipe. The large-scale setup used a heavy weight decay of 0.3. Let me think about why that might be wrong here. Heavy weight decay is a regularizer you can afford — even want — when you have 300M images and overfitting is the enemy. On 1.2M images with the heavy *augmentation* I'm about to pile on, the effective regularization is already strong, and an additional 0.3 weight decay over-constrains the weights and hurts convergence. Drop it to about 0.05. And scale the learning rate with batch size, the standard linear rule, lr = base × (batch/512) — I use 512 as the base denominator. With cosine decay and a warmup of a few epochs (transformers are touchy early in training), 300 epochs. One more touchy thing: initialization. Transformers are known to be sensitive to initialization scale — a poorly-scaled start risks stalling optimization outright — so initialize the weights from a truncated normal, which keeps the initial activation scale controlled and is the stable choice for this architecture.

Now the heart of the recipe: augmentation, which is how I substitute "more data" for the data I don't have. The logic is direct — the transformer lacks the convnet's spatial priors, so I have to show it more varied data to teach it the invariances a convnet gets for free. So pile on strong augmentation: a learned augmentation policy (RandAugment, which I favor over AutoAugment as the newer and generally stronger of the two automatic policies, though that preference is worth confirming directly rather than assumed), Mixup (blend two images and their labels), CutMix (paste a patch of one image into another and mix labels proportionally), and random erasing (blank out a random region). The way to check whether this stack is earning its keep is systematic knockout at matched training budget: pretrain the full recipe, then pretrain again with one ingredient removed at a time, everything else held fixed. My prediction for the label-mixing pair specifically is that Mixup and CutMix are not optional garnish — since they manufacture the most novel training signal out of existing images — so removing both together should cost more accuracy than any single other ingredient. Dropout is the one place I predict the opposite sign: on a model already under heavy augmentation with no batch norm, dropout should be redundant regularization pressure that just slows convergence, so I plan to cut it in favor of stochastic depth (randomly skip whole residual blocks during training at rate ~0.1), which regularizes at the level of whole residual paths — a better match for a stack of residual transformer blocks than dropping individual activations, and known to ease convergence of deep transformer stacks in general. The knockout ablation is what confirms or overturns each of these calls before I lock in the final recipe.

One more augmentation choice worth singling out rather than folding in silently: repeated augmentation. The idea is that within a batch, instead of one augmented view per image, you include *several* augmented views of the same images (I use 3 repetitions). On its face this looks wasteful — it should just burn batch capacity on redundant samples with no new information. My competing hypothesis is that it isn't wasteful, because the gradient at each step is effectively averaged over several augmentations of the same underlying image, which should be a better-conditioned estimate of the invariance-learning signal than a single noisy view, and should compound with the label-mixing augmentations rather than fight them. The same knockout-ablation test as above — full recipe versus recipe with repeated augmentation removed, matched budget — decides between these: if it's genuinely wasteful, removing it should be free or even help; if my hypothesis is right, removing it should cost real accuracy. I'm predicting the latter, so I keep 3 repetitions in the default recipe, with the caveat that it also changes what an "epoch" means: with 3 repetitions I only see a third of the distinct images per pass, so a 300-epoch run under this scheme is really 100 passes over the full dataset, each 3× as long — I keep the 300-epoch label so training time compares directly against runs without repeated augmentation. And label smoothing at ε = 0.1 on the true labels, to keep the classifier from getting overconfident.

So far all of this is recipe, not architecture — same blocks, same class-token readout, only the optimizer, regularization, and augmentation changed. My prediction is that this recipe alone is enough to bring a convolution-free transformer to convnet-competitive accuracy on ImageNet-1k, trained in a couple of days on one node with no external data. The decision rule for the first question is exactly that comparison: if the recipe-only model lands in the same range as convnets of similar size and speed, the data-hunger was a recipe artifact and not an architectural one; if a large gap remains after pushing the recipe this hard, the architecture itself would be implicated instead. I'll carry that as the working answer to the first question and move to the second: distillation, where I want to actually add something transformer-specific.

The standard tool is soft distillation: train the student to match a teacher's softened output distribution. Concretely, minimize a mix of the ground-truth cross-entropy and the KL divergence between teacher and student softmaxes at temperature τ,

  L = (1−λ)·L_CE(ψ(Z_s), y) + λ·τ²·KL(ψ(Z_s/τ), ψ(Z_t/τ)).

The τ² prefactor is the part I want to make sure I understand before I rely on it, because if I get it wrong the distillation term will be silently mis-weighted against the hard-label term. The claim usually made is that it compensates for the softening. Let me actually differentiate to see what the softening does to the gradient. With p = ψ(Z_s/τ) and q = ψ(Z_t/τ), the temperature-τ distillation cross-entropy is −Σ_j q_j log p_j, and its derivative with respect to student logit Z_s(i) is (1/τ)·(p_i − q_i) — the chain rule pulls a 1/τ out of the Z_s/τ inside the softmax. So the bare softened term carries an explicit 1/τ. I want to check that and see how the *whole* gradient scales, so I take a random pair of 1000-class logit vectors and compute. The analytic (1/τ)(p−q) matches a finite-difference gradient to eight digits at τ = 1, 2, 3, 5, so the 1/τ factor is real. And the magnitude of the full gradient vector falls off faster than 1/τ — ‖∇‖ goes 5.2e-2, 1.1e-2, 4.9e-3, 1.7e-3 as τ goes 1, 2, 3, 5 — because the (p − q) difference itself also flattens as the softmaxes get softer, so the two effects compound to roughly 1/τ². Multiplying the loss by τ² puts it back: ‖τ²·∇‖ comes out 5.2e-2, 4.5e-2, 4.4e-2, 4.3e-2, essentially flat in τ. So the τ² is exactly what keeps the distillation gradient on the same footing as the hard-label gradient as I vary the temperature — good, I can trust the formula. But notice what this cost: the soft loss has two knobs to tune (τ and λ), and it asks the student to imitate a full distribution. Let me ask whether there's something simpler that fits this setting better.

What if I treat the teacher's *decision* — its top-1 prediction — as if it were a ground-truth label? Let y_t = argmax_c Z_t(c) be the teacher's hard call. Then the loss is just two cross-entropies, one against the true label and one against the teacher's label, weighted equally:

  L = ½·L_CE(ψ(Z_s), y) + ½·L_CE(ψ(Z_s), y_t).

This is parameter-free — no τ, no λ to cross-validate — and conceptually the teacher's label y_t plays exactly the same role as the true label y. There's a subtle bonus: because I re-evaluate the teacher on the *augmented* crop the student sees, the hard label y_t can differ from the dataset label when the augmentation has mangled the image (a CutMix'd or heavily-erased crop might genuinely look more like the teacher's call than the original class), so the teacher provides a label that's consistent with what the student is actually looking at. And it can be softened with label smoothing if I want. Whether that crop-consistency advantage actually beats the richer information in a soft distribution isn't obvious a priori — soft distillation keeps more of the teacher's uncertainty, hard distillation keeps a label that's always valid for the exact crop shown — so the two losses need a head-to-head comparison at matched training budget, teacher and student held fixed, only the loss swapped. My prediction is that the crop-consistency property dominates here, because the training-time augmentation is aggressive enough that a target computed once off the un-augmented image is a worse guide than a hard label recomputed for exactly what the student sees; if that's right, hard-label distillation should come out ahead, and that's the comparison that decides which loss I build on.

Now the part I actually want to invent, which is *how* the teacher signal enters a transformer. The naive thing is to put both losses on the single class-token output. But the class token is already being pulled toward the true label; asking it to also reproduce the teacher's label is asking one vector to serve two masters. The transformer gives me a cleaner option that a convnet doesn't: I can just add another token. So introduce a second learnable token — a distillation token — alongside the patch tokens and the class token. It's used exactly like the class token: it's prepended to the sequence, it interacts with the patches and with the class token through self-attention at every layer, and its final state is read out by its own linear classifier. The only difference is its target: the class token is supervised by the true label, the distillation token by the teacher's (hard) label. Both tokens are learned by backprop. My hope is that this gives the network two readout pathways — one tuned to the dataset labels, one tuned to the teacher's labels — that share the same patch representation but don't have to compromise with each other. But that's only a hope until I show the two tokens actually diverge; if they don't, I've just paid for a duplicate.

I should check this isn't a trivial reparameterization — maybe two tokens with the same target would do just as well, in which case the "distillation" framing is empty and I've just added a duplicate. The control that isolates this is a second *class* token instead of a distillation token: same architecture, same true-label target as the first class token, initialized randomly and independently, everything else held fixed. My prediction, if identical targets make tokens redundant, is that the two class tokens should converge toward each other during training — nothing in the objective pulls them apart — ending up highly similar and adding no accuracy over a single class token. The real distillation token gets the opposite prediction: because its target (the teacher's label) genuinely differs from the class token's target (the true label) even though the two frequently agree, I expect it to converge toward a distinct vector, not a copy — most different early in the network where the two objectives are least entangled, and only partially converging as depth mixes the two signals together, since a lot of what predicts the true label also predicts the teacher's label. The decision rule is straightforward: measure token similarity and accuracy at the end of training. If the distillation token behaves like the class-token control — converges to near-identical, adds nothing — the whole distillation-token design is unjustified and I should fall back to a single readout with a mixed loss. If it stays measurably distinct while the duplicate class token collapses, that's the evidence the distillation token is doing something the duplicate doesn't, and the design is worth keeping.

At test time I have two classifiers — the class head and the distillation head — and a choice of how to predict. I could use either alone. But if the two heads really do stay distinct rather than collapsing into copies of each other, as the token-similarity test above is designed to check, then they should be tuned to genuinely different signals (true label versus teacher label) and ought to be complementary rather than redundant — in which case fusing them should beat either alone. So my default readout adds the softmax outputs of the two heads and predicts the argmax, and I'll confirm the fused prediction actually beats each individual head once training and the token-similarity check are both in.

One more design choice about distillation, and it's the one I find most interesting. Which teacher? The obvious instinct is to distill from another transformer of comparable or higher accuracy — match the student's architecture family, on the theory that a teacher closer in kind gives cleaner signal. But there's a competing hypothesis: distillation transfers more than the argmax label, it transfers something about the teacher's decision boundary, and if the transformer's actual weakness is a missing spatial prior, then the teacher whose decisions most directly encode that prior should be the more useful one to imitate, regardless of architectural kinship. A convnet's decisions are shaped by locality and translation equivariance in a way a transformer teacher's aren't. The way to tell these apart is a matched comparison — two teachers of similar standalone accuracy, one a transformer, one a convnet, same student architecture and training recipe, only the teacher swapped — and see which student ends up stronger. If the architectural-kinship hypothesis is right, the transformer teacher should win or tie; if the inductive-bias-transfer hypothesis is right, the convnet teacher should win despite being a different kind of model. I expect the second: transferring a prior the student structurally lacks should matter more than matching architecture family, so I'm planning to use a strong convnet as the teacher, with this comparison as the check that confirms or overturns that plan before I commit to it for the full training run.

That leaves resolution. Following the train-low/fine-tune-high idea, I pretrain at 224² and fine-tune at 384², which is faster overall and more accurate under strong augmentation. The patch size stays 16, so going to 384² raises the number of patches N from 196 to a larger grid. The transformer and the classifiers don't care — self-attention handles more tokens unchanged — but there are exactly N positional embeddings, one per patch, and now I need more of them. So I resize the positional-embedding grid by interpolation. My first instinct is bilinear, the default for image resizing, and at first I don't expect any subtlety. But when I think about what bilinear interpolation does to a *vector-valued* grid, something bothers me. Each interpolated position is a convex combination Σ_k w_k a_k of its neighbour embeddings, w_k ≥ 0, Σ w_k = 1 — i.e. a weighted average. For two equal-norm neighbours the algebra is exact: ‖(a+b)/2‖ = r·√((1+cosθ)/2), which equals r only if the two vectors are identical (θ = 0) and is strictly smaller otherwise. These embeddings are 768-dimensional vectors for *distinct* positions, so they are nowhere near collinear, and the average should come out noticeably shorter than its inputs. Let me put numbers on it before deciding it matters: I take four neighbour vectors at the scale of the trained embeddings (norm ≈ 0.55 each) and average them as bilinear would at a cell centre. The interpolated vector lands at norm ≈ 0.27 — roughly *half* the neighbours' norm — and I see the same ~0.5 ratio across repeated draws. That is not a rounding effect; it is a systematic shrink. And the pretrained transformer was tuned to positional vectors of the original magnitude, so feeding it embeddings at roughly half that magnitude should throw the attention logits off — my prediction is that resizing this way costs real accuracy at fine-tuning time, sharply enough to be visible rather than a rounding-level effect, which is worth checking directly before trusting either interpolation method. The fix is to interpolate with bicubic instead, whose negative side-lobe weights let the reconstructed vector overshoot toward its neighbours rather than strictly average them, so it should approximately preserve the norm; then fine-tune. During fine-tuning I plan to keep the strong training-time augmentation rather than dampening it, and to keep fine-tuning with the teacher too (both true label and teacher label, the teacher taken at the matching resolution) rather than switching to true labels alone — dropping the teacher partway through would discard a signal the model has been relying on for no clear gain, so the default is to keep it unless a direct comparison says otherwise. For this short fine-tune, already close to a good optimum, I don't expect optimizer choice to matter much, so I default to AdamW again for consistency with pretraining.

A nice property falls out of the no-batch-norm point I flagged earlier: batch norm's statistics degrade at small batch sizes, but there's no batch norm here for that to happen to, so the batch size should be free to shrink without the accuracy penalty that would hit a batch-normed convnet — which makes the larger models easier to fit on a node.

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
    (bilinear shrinks the vector norm, which is predicted to hurt the pretrained model)."""
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
