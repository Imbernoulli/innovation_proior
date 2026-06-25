Let me start from the thing that actually bugs me. In language, the single move that made pretraining both simple and scalable was masked language modeling: hide some tokens, predict them from context, done. BERT does this and the representations transfer everywhere. Vision Transformers are now architecturally almost identical to language Transformers — patches in, self-attention, out — so why don't we have the same recipe for them? The self-supervised methods that work best on ViTs right now operate on the *whole image*: take two augmented crops, push their global representations to agree. That's powerful, but it throws away exactly the thing MLM exploits — the internal token structure. A ViT sees an image as a sequence of patch tokens, and I'm only ever supervising the global summary of that sequence. That feels like leaving the BERT idea on the table.

So let me try to literally port MLM. Mask some patches, predict them. Immediately I hit the question that language gets for free and vision doesn't: predict them *as what*? In BERT the answer is "predict which WordPiece token it was," and that works because the tokenizer already carved language into a finite vocabulary of meaningful units. The masked-prediction task is well-posed because the target is a meaningful discrete label. For an image patch there is no vocabulary. I could predict the raw pixels of the masked patch — that's the obvious thing — but I already know how that tends to go: regressing pixels burns the model's capacity on high-frequency texture and short-range correlations, and the features you get are weak when you freeze them and evaluate. The whole point of the BERT framing is that predicting a *meaningful label* is better than predicting raw signal. So the real problem isn't masking; masking is easy. The real problem is: what is the visual tokenizer? What turns a patch into a meaningful target?

Let me think about what a visual tokenizer would need. In language the tokens carry semantics because of word-frequency statistics — meaning falls out of the corpus. Pixels don't do that; a 16×16 patch is continuous and genuinely ambiguous about what it "is." When *does* semantic structure emerge from images without labels? The honest empirical answer from the view-agreement line of work — instance discrimination, MoCo, BYOL, SwAV, DINO — is: it emerges by *bootstrapping*. You take two distorted views of one image, force their representations to agree, prevent collapse, and over training the network's outputs become semantically organized. There's no shortcut to "meaning from a single patch"; meaning comes from this self-referential agreement process. That's a discouraging fact at first because it seems to say: to get a semantic tokenizer, I have to first run a whole representation-learning procedure to *make* the tokenizer, then freeze it, then run masked modeling against it. A two-stage pipeline.

And that's exactly the shape of the one prior MIM method that works. Let me look hard at it. BEiT takes a discrete VAE — the one trained for DALL-E on a big external dataset — and uses it as the tokenizer. The dVAE maps each of the 196 patches of a 224×224 image to a discrete id out of a vocabulary of 8192. Then BEiT blockwise-masks about 40% of the patches, runs a ViT on the corrupted image, and predicts the original visual-token id at each masked position with a softmax cross-entropy. Writing the objective as a distillation makes the structure clear: you're minimizing, over masked positions,

  −Σ_i m_i · P_φ(x_i)ᵀ log P_θ(x̂_i),

where φ is the frozen dVAE producing a one-hot target P_φ(x_i) over the 8192 classes, m is the mask, and θ is the model being trained on the corrupted image x̂. This is just knowledge distillation: distill from a fixed teacher φ into the model θ.

Now let me poke at where this falls short, because the gaps are going to tell me what to build. First, is the dVAE actually a *semantic* tokenizer? If I take the dVAE's patch tokens and treat them as features and evaluate their quality directly, they're nearly useless for semantics — they encode low-level texture, a handful of percent in k-NN. So BEiT's targets carry detail, not meaning. That's a problem, because the whole reason I wanted to leave pixel regression was to get *meaningful* targets, and a low-level tokenizer half-defeats that. Second, the tokenizer is offline: fixed architecture, trained on an extra dataset, frozen. So it can't adapt to whatever data I actually train on, and I've inherited a separate large-scale training stage that has nothing to do with my task. Third — and this nags at me — the target is a one-hot id. A patch is ambiguous; forcing it to be exactly one of 8192 categories throws away the fact that it might be 60% "fur," 30% "ear," 10% "shadow." Hard discretization seems wrong for a continuous, ambiguous signal.

So I have two challenges sharpened: the tokenizer should be (a) semantically meaningful and (b) jointly trainable in one stage, ideally adapting to my data. Let me hold both and look at the other side of the field — the view-agreement methods — and see if anything there is shaped like a tokenizer.

DINO is the cleanest one to stare at. It's self-distillation with no labels, on the global image. Two augmented views u, v. A student network and a teacher network share the same architecture: backbone f plus a projection head h, and they emit a K-dimensional distribution from the `[CLS]` token. The teacher's parameters are an exponential moving average of the student's — the teacher is literally a slow, time-averaged copy of the student, no separate training. The loss is the cross-entropy from teacher to student, symmetrized over the two views:

  L_[CLS] = −P_θ′^[CLS](v)ᵀ log P_θ^[CLS](u),

with θ′ the EMA teacher and θ the student. The thing I have to understand before I reuse any of this is *why it doesn't collapse*, because a "match your own slow copy" objective has an obvious trivial solution: output the same constant distribution for every image, and the loss is minimized with zero information learned. DINO does two opposing things to the teacher's output before the softmax to prevent that — it *centers* (subtracts an EMA of the teacher's batch-mean output, c ← m·c + (1−m)·mean(teacher out)) and it *sharpens* (divides by a small teacher temperature τ_t). I want to see for myself that these two actually pull against each other rather than just take it on faith, so let me run a tiny numerical case with K=4 output dimensions. Suppose one output dimension is running away — across three images the teacher logits are

  [4.0, 0.1, 0.0, −0.1], [3.6, 0.0, 0.2, 0.0], [4.2, −0.1, 0.1, 0.0].

Sharpen with τ_t = 0.04 and no centering: every row softmaxes to essentially [1, 0, 0, 0]. So the teacher target is one-hot on dimension 0 for *every* image — and if the student learns to reproduce that, all images map to the same output. That's the collapse, made concrete: sharpening alone drives straight into it. Now center first — subtract the batch-mean logit vector [3.933, 0.0, 0.1, −0.033] — then sharpen with the same τ_t = 0.04. The three rows become

  [0.298, 0.686, 0.005, 0.011], [0.000, 0.065, 0.787, 0.149], [0.996, 0.000, 0.001, 0.003].

Three *different* peaked distributions: centering removed the common runaway component and let the small per-image differences in dimensions 1–3 decide the peak, and sharpening kept each target confident. To check the other force in isolation, drop the sharpening (τ_t = 1) and keep only centering: the same three rows give entropies 1.383, 1.374, 1.377 against a uniform maximum of ln 4 = 1.386 — i.e. centering by itself flattens everything to near-uniform, which is the opposite failure (no signal). So the two genuinely oppose: centering alone → near-uniform mush, sharpening alone → one-hot collapse, and only together do I get distinct, confident per-image targets. That's the property I'll have to preserve in anything I build on top of this. Student temperature is 0.1, teacher temperature is warmed from 0.04 up to 0.07. The head is a 3-layer MLP, hidden 2048, with an ℓ₂-normalized bottleneck and a weight-normalized final layer to a big K, no batch norm.

Now let me write BEiT's loss and DINO's loss next to each other and just look at them.

  BEiT:  −Σ_i m_i · P_φ(x_i)ᵀ log P_θ(x̂_i)        teacher = frozen φ, on patches
  DINO:  −          P_θ′^[CLS](v)ᵀ log P_θ^[CLS](u)  teacher = EMA θ′, on [CLS]

These are the *same* object. Both are a cross-entropy from a teacher distribution to a student distribution. The only structural difference is where the teacher distribution comes from — a pre-fixed φ in one case, the online EMA θ′ in the other — and which token it's read off of: patches versus the `[CLS]` token.

That's the hinge. What if I take BEiT's masked-patch objective but replace the frozen dVAE teacher φ with DINO's online EMA teacher θ′, applied to the patch tokens? Then the tokenizer is no longer an offline artifact — it's the teacher network, a time-averaged copy of the very model I'm training, evaluated on the *clean* (unmasked) image. The student sees the masked image and has to recover, at each masked patch, the distribution the teacher assigns to that same patch in the clean image. The tokenizer becomes *online*: it co-evolves with the model, needs no separate stage, and its "domain knowledge" is whatever data I'm actually training on. Both challenges dissolve at once — and the semantics of this tokenizer come from exactly the bootstrapping process I was earlier afraid I'd have to run as a separate stage. I don't run it separately; it's the same process, sharing the same network.

Let me write the masked objective in this online form. Two views u, v; blockwise-mask each into û, v̂ (I'll keep BEiT's blockwise masking — masking contiguous blocks, not scattered single patches, so the model can't cheat with one-pixel-away neighbors). The student processes the masked view and outputs a patch distribution P_θ^patch(û_i); the teacher processes the *clean* view and outputs P_θ′^patch(u_i). Recover masked patches to the teacher's clean-image outputs:

  L_MIM = −Σ_{i=1}^N m_i · P_θ′^patch(u_i)ᵀ log P_θ^patch(û_i),

summed only over masked positions (the m_i factor), symmetrized by adding the matching (v̂, v) term. So the teacher — backbone f_t plus the patch projection head — *is* the visual tokenizer, generating an online token distribution for every masked patch.

But wait. I need to check that this online tokenizer is actually semantic, because the whole motivation was that BEiT's was not. The teacher is just an EMA of the student. If the only thing I ever train on is "recover masked patches from their own EMA targets," is there anything pinning those targets to *meaning*? Let me sanity-check the degenerate case: train MIM alone, no view-agreement, and look at the frozen features. They're nearly useless — single-digit k-NN, ~30% linear. That's the warning I needed. Pure masked self-distillation on patches gives the model nothing that forces the patch distribution to be semantically organized; it can satisfy "match my own slow copy" with representations that mean nothing globally. The bootstrapping that makes DINO's outputs semantic was driven by the *cross-view `[CLS]` agreement*, and I've thrown that away.

So I have to keep the `[CLS]` self-distillation. It is not an auxiliary nicety — it's the thing that injects semantics into the shared network, and therefore into the online tokenizer. I'll keep DINO's loss verbatim, with one change: the student's input is now the masked view, since the student always sees masked images here:

  L_[CLS] = −P_θ′^[CLS](v)ᵀ log P_θ^[CLS](û),

cross-view, symmetrized. The teacher reads `[CLS]` from the clean view, the student from the masked view. Now the same network is simultaneously learning a globally-semantic `[CLS]` representation (via cross-view agreement) and serving its patch outputs as the MIM tokenizer — and because they're the same network, the semantics earned on `[CLS]` leak into the patch tokens.

That last clause suggests a concrete design question: should the `[CLS]` token and the patch tokens go through the *same* projection head, or separate heads? If semantics acquired on `[CLS]` is what I'm relying on to make the patch tokenizer meaningful, then routing patches through the same head — same weights — is the most direct way to share that earned semantics. Let me reason about the alternative: separate heads means the patch head only ever learns from the MIM loss, which I just argued is semantically weak on its own. A shared head lets the patch path ride on the `[CLS]` path's semantics. I'll share the entire head, h^[CLS] = h^patch, for both student and teacher. (There's a subtlety I notice: the student's patch head only ever sees *masked* tokens during training, whose distribution differs from natural patches — so one could imagine sharing the head only on the teacher side. But the simplest fully-shared choice is the one I'll commit to; the gain from being clever about partial sharing is not there.)

Next, the target form. BEiT used a one-hot id; I argued that's wrong for ambiguous patches. With an online teacher I'm free to use the teacher's full softmax distribution as the target instead of its argmax. Let me make sure that matters and isn't just aesthetic. A patch genuinely can be a mix — part of an ear, part of fur, part of background — and a soft target lets the student match that mixture, whereas hardmax forces a single category and discards the ambiguity. There's also a continuity point I shouldn't lose: the teacher is moving every step, so its argmax for a given patch can flip between neighboring token ids across consecutive steps, giving the student a jittery one-hot target; the full softmax changes smoothly even when the argmax flips, which should make the moving-teacher target less noisy. So I use the teacher's softmax distribution (sharpened) as the supervisory signal, not the one-hot token id. For the `[CLS]` stream I already have DINO's centering+sharpening machinery; I'll apply centering+sharpening to the patch stream too, but with *its own* center and *its own* temperature, because patch tokens don't cluster into clean semantic prototypes the way `[CLS]` does — they vary more widely — so the patch distribution deserves separately tuned smoothing. I'd expect the centering to matter less for patches than for `[CLS]` (the per-patch collapse pressure is weaker, since 196 patches per image already spread mass across dimensions where a single `[CLS]` does not), and the sharpening to be the part doing most of the work — but that's a hypothesis I'd want to confirm by ablating the patch center, not something I can settle from the math alone, so I keep both knobs and tune them separately.

Let me make the centering+sharpening explicit as a single shared subroutine, because both streams use it. Given a student logit s and teacher logit t, with a center c and temperatures (τ_s, τ_t): detach the teacher (it's a target, no gradient flows into it — gradients only update the student, and the teacher follows by EMA); form the student probability softmax(s/τ_s); form the teacher target softmax((t − c)/τ_t), i.e. center then sharpen; return the cross-entropy −Σ t·log s. The teacher's center is updated as an EMA of the teacher's batch-mean output. The `[CLS]` stream gets a center C of shape [1, K] and the patch stream a center C′ of shape [1, 1, K] (per output channel, averaged over batch and over the patch axis).

How big should the output dimension K be? DINO used a very large K (tens of thousands) for the single `[CLS]` distribution. But here the cost arithmetic is different, and worth doing explicitly: in DINO each image produces one K-vector, whereas here *every* patch produces one, so for N = 196 patches the teacher-target tensor is 196× larger per image. Concretely the patch-logit activation per image is N·K = 196·K floats; pushing K from 8192 to 65536 takes that from ~1.6M to ~12.8M floats per image, an 8× blow-up in the dominant memory term, multiplied again by batch size. So a large K is genuinely expensive here in a way it is not for `[CLS]`, and it's only worth paying if accuracy rises with K. I'd want to sweep K and look at frozen-feature accuracy; my expectation is that the patch vocabulary saturates well before tens of thousands, because each patch only needs enough categories to cover the distinct local structures, far fewer than the distinct *images* a `[CLS]` codebook must separate. Pending that sweep I set K = 8192 — matching BEiT's dVAE vocabulary, large enough to be unlikely to bottleneck, and cheap enough per patch.

How do I weight the two losses? I have L_[CLS] and L_MIM, and the question is whether they need a relative coefficient. There's a reason to suspect they don't: both are cross-entropies over a K-way softmax with the same student temperature and the same head, so they're already on the same numerical scale — neither is a raw-pixel MSE that would dwarf the other. Per-patch the MIM term is averaged over masked positions, per-image the `[CLS]` term is a single cross-entropy, so they live in the same range by construction. That makes λ1 = λ2 = 1 the natural starting point rather than an assumption I'm sneaking in; I'd only reach for a coefficient if a sweep showed one stream getting swamped, and I don't expect it given the matched scales. So I start at one-to-one and treat any tuning as a fallback I probably won't need.

Now the masking ratio. How much should I mask? In BERT the canonical answer is 15%; BEiT pushed to ~40% precisely so the model can't solve masked patches from immediately adjacent pixels. I don't have a first-principles value, so I'd sweep it and watch frozen-feature accuracy; I'd expect a fairly wide tolerant band rather than a sharp optimum, because the blockwise masking already forces non-local prediction across a range of ratios. One thing I do want to build in deliberately: instead of fixing the ratio, sample it from a range. That's a free form of augmentation — the teacher's clean-image outputs get matched against student masked-image outputs at many different masking levels, so the student sees a harder, more varied reconstruction task without my having to pick the single "right" ratio. So I'll center around 0.3 with a ±0.2 spread and let the per-image draw vary.

Then I try to bolt on multi-crop, because every strong view-agreement method (SwAV, DINO) leans on it: 2 global crops at 224² plus 10 local crops at 96², where local crops go only to the student. And here I hit a wall. Naively masking the global crops and running multi-crop makes training *unstable* — I watch the clustering quality (NMI) during training and it dips. Let me figure out why before patching it. The student is being fed masked global crops and *non-masked* local crops; the teacher only ever sees clean images. So the student's input distribution is a mismatch — some inputs masked, some not — and the agreement targets are computed from clean images throughout. The obvious fix is to also mask the local crops so everything's consistent. I try it: it doesn't fix the instability, and actually a local 96² crop is so small that an individual patch in it has almost no meaningful content to predict — masking there is just noise. (Confirmed indirectly: restricting the local crops to be less tiny mitigates the drop.) So masking-everything is the wrong patch.

The right way to read the instability: the problem is the *mismatch* between masked global crops and non-masked crops in the same batch, not masking per se. So let me make the masking itself stochastic at the image level. With probability one-half, set the masking ratio to 0 — that image is clean; with probability one-half, mask at a nonzero ratio (drawn from the 0.3 ± 0.2 range) and apply MIM to both global crops. For this to be coherent the zero-mask image has to genuinely contribute *nothing* to the patch loss while still acting like a normal DINO image for `[CLS]`, so let me trace the loss code on a tiny case to be sure the bookkeeping actually does that. Take N = 4 patches and per-patch cross-entropies [0.5, 2.0, 1.0, 3.0]. A masked image with mask = [0,1,0,1] gives per-image patch loss (0.5·0 + 2.0·1 + 1.0·0 + 3.0·1) / max(2, 1) = 5.0/2 = 2.5, which is exactly the mean over the two masked patches — good, unmasked positions drop out and the average is over masked count, not over all N. Now a clean image, mask = [0,0,0,0]: numerator is 0, and the denominator `mask.sum().clamp(min=1)` = max(0,1) = 1, so the patch loss is 0/1 = 0 — no contribution and no division-by-zero. So the clamp isn't cosmetic; it's what lets a zero-mask image sit in the same batch and contribute exactly zero to MIM while its `[CLS]` term proceeds normally. Let me also trace the view-pairing while I'm here, for 2 global crops and no local crops: the loop runs teacher global q against every student crop v, taking the patch (MIM) branch when v == q and the `[CLS]` branch otherwise, which gives same-view pairs {(0,0),(1,1)} → 2 MIM terms and cross-view pairs {(0,1),(1,0)} → 2 `[CLS]` terms. Add 10 local crops and the count becomes 2 MIM terms (still only on the global same-view pairs, since the teacher never sees locals) and 22 `[CLS]` terms — MIM stays anchored to the global crops and multi-crop only enriches the `[CLS]` agreement, which is exactly the division of labor I want. So "random MIM" interleaves clean and masked images across the batch, removing the systematic mismatch: the clean passes keep the `[CLS]` bootstrapping healthy while the masked passes drive the local objective. Multi-crop with random MIM it is.

Let me also nail the optimization recipe, inheriting what the view-agreement methods established: AdamW, batch size 1024, learning rate linearly warmed for the first 10 epochs to a base value that scales with batch size, lr = 5e−4 × batch_size / 256, then cosine-annealed; weight decay cosine-annealed up over training. The teacher EMA momentum follows a cosine schedule from 0.996 toward 1 — slow at first, slower later, so the tokenizer is a stable target. Student temperature 0.1; teacher temperatures warmed up from low values, separately for the `[CLS]` and patch streams. The projection head is the DINO head: 3-layer MLP, hidden 2048, GELU, ℓ₂-normalized bottleneck, weight-normalized last layer to K = 8192, batch-norm-free. ViT-S/16 or B/16 backbone, 224 input, patch 16, so N = 196 patch tokens; a learnable `[MASK]` embedding replaces masked patch embeddings before the transformer.

Let me put the whole thing down as code, mirroring how I'd actually wire it.

```python
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class iBOTHead(nn.Module):
    def __init__(self, in_dim, out_dim, patch_out_dim=8192, nlayers=3,
                 hidden_dim=2048, bottleneck_dim=256, norm_last_layer=True,
                 shared_head=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        if shared_head:
            self.last_layer2 = self.last_layer
        else:
            self.last_layer2 = nn.utils.weight_norm(nn.Linear(bottleneck_dim, patch_out_dim, bias=False))
            self.last_layer2.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer2.weight_g.requires_grad = False

    def forward(self, x):                       # x: [n, 1+N, in_dim]
        if x.ndim == 2:
            x = F.normalize(self.mlp(x), dim=-1, p=2)
            return self.last_layer(x)
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)         # L2-normalized bottleneck
        cls = self.last_layer(x[:, 0])          # [n, K]      -> [CLS] target/logit
        patch = self.last_layer2(x[:, 1:])      # [n, N, K]   -> per-patch logits
        return cls, patch

class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops,
                 warmup_teacher_temp=0.04, teacher_temp=0.07,
                 warmup_teacher_patch_temp=0.04, teacher_patch_temp=0.07,
                 warmup_epochs=30, nepochs=800,
                 student_temp=0.1, center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.ngcrops, self.nlcrops = ngcrops, nlcrops
        self.ncrops = ngcrops + nlcrops
        self.cm, self.cm2 = center_momentum, center_momentum2
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.register_buffer("center",  torch.zeros(1, out_dim))        # [CLS] center
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))  # patch center
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
            np.ones(nepochs - warmup_epochs) * teacher_temp,
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_patch_temp, teacher_patch_temp, warmup_epochs),
            np.ones(nepochs - warmup_epochs) * teacher_patch_temp,
        ))

    def forward(self, student_out, teacher_out, student_local_cls, student_mask, epoch):
        s_cls, s_patch = student_out                # student sees MASKED views
        t_cls, t_patch = teacher_out                # teacher sees CLEAN views (the tokenizer)
        if student_local_cls is not None:
            s_cls = torch.cat([s_cls, student_local_cls])
        s_cls   = (s_cls   / self.student_temp).chunk(self.ncrops)
        s_patch = (s_patch / self.student_temp).chunk(self.ngcrops)
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        # teacher: center then sharpen -> soft target (NOT one-hot), detached
        t_cls_c   = F.softmax((t_cls   - self.center)  / temp,  dim=-1).detach().chunk(self.ngcrops)
        t_patch_c = F.softmax((t_patch - self.center2) / temp2, dim=-1).detach().chunk(self.ngcrops)

        L_cls = L_mim = n1 = n2 = 0
        for q in range(len(t_cls_c)):
            for v in range(len(s_cls)):
                if v == q:                          # same view -> masked-patch reconstruction
                    l = torch.sum(-t_patch_c[q] * F.log_softmax(s_patch[v], dim=-1), dim=-1)
                    m = student_mask[v].flatten(-2, -1).float()        # only masked positions
                    l = torch.sum(l * m, dim=-1) / m.sum(dim=-1).clamp(min=1.0)
                    L_mim += l.mean(); n2 += 1
                else:                               # cross-view -> [CLS] self-distillation
                    l = torch.sum(-t_cls_c[q] * F.log_softmax(s_cls[v], dim=-1), dim=-1)
                    L_cls += l.mean(); n1 += 1
        L_cls = L_cls / n1 * self.lambda1
        L_mim = L_mim / n2 * self.lambda2
        self.update_center(t_cls, t_patch)
        return {"cls": L_cls, "patch": L_mim, "loss": L_cls + L_mim}

    @torch.no_grad()
    def update_center(self, t_cls, t_patch):        # EMA of teacher batch-mean
        cls_center = torch.sum(t_cls, dim=0, keepdim=True)
        patch_center = torch.sum(t_patch.mean(1), dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cls_center)
            dist.all_reduce(patch_center)
            world = dist.get_world_size()
        else:
            world = 1
        cls_center = cls_center / (len(t_cls) * world)
        patch_center = patch_center / (len(t_patch) * world)
        self.center = self.center * self.cm + cls_center * (1 - self.cm)
        self.center2 = self.center2 * self.cm2 + patch_center * (1 - self.cm2)

def sample_mask(H, W, ratio, shape="block", log_aspect=(np.log(0.3), np.log(1 / 0.3))):
    high = ratio * H * W
    if shape == "rand":
        mask = np.hstack([np.zeros(H * W - int(high)), np.ones(int(high))]).astype(bool)
        np.random.shuffle(mask)
        return mask.reshape(H, W)

    mask = np.zeros((H, W), dtype=bool)
    mask_count = 0
    while mask_count < high:
        max_mask_patches = high - mask_count
        delta = 0
        for _ in range(10):
            low = (min(H, W) // 3) ** 2
            target_area = np.random.uniform(low, max_mask_patches)
            aspect_ratio = np.exp(np.random.uniform(*log_aspect))
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                top = np.random.randint(0, H - h + 1)
                left = np.random.randint(0, W - w + 1)
                num_masked = mask[top: top + h, left: left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if not mask[i, j]:
                                mask[i, j] = True
                                delta += 1
            if delta > 0:
                break
        if delta == 0:
            break
        mask_count += delta
    return mask

def train_step(images, masks, student, teacher, loss_fn, opt, ema_m, epoch, n_global):
    # teacher: CLEAN global crops -> the online tokenizer
    t_out = teacher(images[:n_global])
    # student: MASKED global crops
    s_out = student(images[:n_global], mask=masks[:n_global])
    # local crops go only to the student, no masking, [CLS] only (DINO multi-crop)
    student.backbone.masked_im_modeling = False
    s_local_cls = student(images[n_global:])[0] if len(images) > n_global else None
    student.backbone.masked_im_modeling = True

    loss = loss_fn(s_out, t_out, s_local_cls, masks, epoch)["loss"]
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                           # teacher = EMA of student
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(ema_m).add_((1 - ema_m) * ps.detach().data)
```

So the causal chain: I wanted BERT's masked-token recipe for ViTs, and the only hard part was the visual tokenizer — the target for a masked patch — because patch semantics don't fall out of pixels but only emerge through view-agreement bootstrapping, which looked like it forced a separate offline tokenizer stage (BEiT's frozen dVAE, low-level and non-adaptive). Writing BEiT's masked-patch loss next to DINO's `[CLS]` self-distillation showed they are the same cross-entropy-to-a-teacher, differing only in whether the teacher is a fixed φ or an online EMA θ′; swapping in the online teacher on the patch tokens turns the tokenizer into a self-distilling, jointly-trained, domain-adaptive module. Keeping DINO's cross-view `[CLS]` agreement is what makes that online tokenizer semantic (MIM alone collapses to meaningless features); sharing the projection head pipes that semantics into the patch path; using the teacher's soft distribution instead of a one-hot id respects patch ambiguity; summing the two losses one-to-one, masking ~0.3±0.2 in contiguous blocks, and making the masking stochastic per image ("random MIM") so masked and clean crops coexist makes multi-crop stable. The teacher, backbone plus shared patch head, evaluated on the clean image, *is* the online tokenizer.
