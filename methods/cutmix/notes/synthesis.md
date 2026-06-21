# CutMix synthesis (Phase 1.5)

arXiv 1905.04899 (verified). Authors Yun et al. 2019. Canonical impl: clovaai/CutMix-PyTorch (train.py).

## Pain point / research question
Regional-dropout regularizers (Cutout, Random Erasing, Hide-and-Seek) improve generalization & localization by forcing the CNN to attend to the *whole* object instead of only the most discriminative part. They do so by deleting a contiguous region of the input — zeroing it (Cutout) or filling with random noise (Random Erasing). But the deleted region is *uninformative*: those pixels contribute nothing to the forward pass / loss. CNNs are data-hungry, so wasting a chunk of every training image is inefficient. Question: how to keep regional-dropout's "attend to whole object" benefit while making the deleted region carry useful signal.

## Load-bearing ancestors (from .bbl / intro)
- **Dropout (Srivastava 2014):** randomly zero hidden activations -> co-adaptation broken; the "feature removal" template at the activation level.
- **Cutout (DeVries & Taylor 2017, 1708.04552):** zero out a single fixed-size square region of the *input* image at a random location. Acts like input-space dropout of a contiguous block; forces reliance on whole-object context, mild occlusion robustness. Limitation: deleted pixels = 0 information; reduces informative pixel proportion.
- **Random Erasing (Zhong 2017):** same idea but fills region with random values. Same information-loss problem.
- **Hide-and-Seek (Singh 2017):** grid of patches randomly hidden -> localization. Same deletion idea.
- **DropBlock (Ghiasi 2018):** regional dropout in *feature* space.
- **Mixup (Zhang 2017, 1710.09412):** x̃ = λx_A+(1−λ)x_B, ỹ = λy_A+(1−λ)y_B, λ~Beta(α,α). Uses ALL pixels (no deletion) and mixes labels. Limitation: pixel-wise blend produces locally ambiguous / unnatural images (ghosting); CAM shows model confused; hurts localization & detection even when it helps classification.
- **Label smoothing (Szegedy 2016):** soft targets; relevant because the mixed label ỹ is a soft target.

## The derivation (insight-before-method)
Two desiderata in tension: (1) keep regional dropout (delete a region -> attend to whole object), (2) every pixel must be informative (Mixup's virtue). Cutout deletes a region but fills with nothing. Mixup fills every pixel but the fill is a global ghosted blend, not natural, and there's no "region" being dropped.
Resolution: instead of filling the deleted region with zeros, fill it with a **patch cut from another training image**. Now: there IS a deleted region (regional-dropout benefit retained), and the region is filled with real pixels from image B (informative, like Mixup). The composite looks locally natural (a real patch on a real image, sharp boundary, no ghosting) unlike Mixup's translucent overlay.

Then the label: the composite contains class-A pixels and class-B pixels. Following Mixup's label-mixing principle, set ỹ = λ y_A + (1−λ) y_B, where λ is the **area fraction** occupied by image A. This couples the label to the actual pixel composition.

## The formulas (verified against 3-Method.tex)
- x̃ = M ⊙ x_A + (1 − M) ⊙ x_B,  M ∈ {0,1}^{W×H} binary mask.
- ỹ = λ y_A + (1 − λ) y_B.
- λ ~ Beta(α, α); ALL experiments α = 1 -> λ ~ Uniform(0,1).
- Box B = (r_x, r_y, r_w, r_h): r_x ~ Unif(0,W), r_y ~ Unif(0,H), r_w = W√(1−λ), r_h = H√(1−λ).
  -> cropped area ratio (r_w r_h)/(WH) = 1−λ. The box is the region SET TO 0 in M (filled from x_B).
  -> so fraction from A = 1 − (1−λ) = λ. Label weight λ on A matches its kept-area fraction. CONSISTENT.
- Aspect ratio of box ∝ image (both scaled by √(1−λ)), so box is a rescaled copy of image shape.

## Implementation (verified against train.py + supp algorithm)
- Per-minibatch: shuffle minibatch along batch axis (rand_index = randperm) -> partner B = batch[rand_index]. One loader, no extra I/O (same trick as mixup).
- lam = np.random.beta(beta, beta).  (args.beta = 1 for the headline runs)
- rand_bbox(size, lam): W=size[2], H=size[3]; cut_rat=sqrt(1−lam); cut_w=int(W*cut_rat); cut_h=int(H*cut_rat); cx=randint(W); cy=randint(H); bbx1=clip(cx−cut_w//2,0,W) etc.
- input[:,:,bbx1:bbx2,bby1:bby2] = input[rand_index,:,bbx1:bbx2,bby1:bby2]  (paste B's patch into A).
- **lam re-adjusted**: lam = 1 − ((bbx2−bbx1)*(bby2−bby1))/(W*H). NECESSARY because clipping the box at image borders changes the true pasted area away from the nominal 1−λ; the label weight must track the *actual* pasted area, not the nominal one.
- loss = criterion(output,target_a)*lam + criterion(output,target_b)*(1−lam). Uses linearity of CE in target -> never materialize soft label (same trick as mixup).
- cutmix_prob: apply CutMix with some probability r per iteration. Official README examples use CIFAR-100 `--cutmix_prob 0.5` and ImageNet `--cutmix_prob 1.0`; both use `--beta 1.0`.

## Design-decision -> why table
- **Fill region with a real patch (vs zeros/noise):** zeros waste pixels (Cutout's flaw); real patch keeps all pixels informative AND keeps a "deleted region" semantics. Alternatives rejected by ablation: none beat it.
- **Rectangular contiguous patch (vs Mixup's global blend):** global blend is unnatural/ambiguous (ghosting) -> CAM confusion -> hurts localization. A sharp rectangular paste is locally natural; both halves are real images. The "attend to whole object from partial view" benefit needs a contiguous occlusion.
- **Label = area-proportional mix λ (vs one-hot dominant label):** ablation "One-hot CutMix" (commit to larger-area class) degrades; "Complete-label" (always 0.5/0.5) degrades. Area-proportional is the honest target matching pixel composition and is required for the gradient to see both classes proportionally.
- **λ ~ Beta(α,α), α=1 (uniform):** ablation over α∈{0.1,..,4}; all beat baseline, best at α=1. Uniform = maximally diverse patch sizes.
- **Box aspect ∝ image (r_w=W√(1−λ), r_h=H√(1−λ)):** makes area exactly 1−λ with a single λ, box shape mirrors image. Ablation: "Fixed-size" and "Center Gaussian" variants degrade -> uniform location + area-scaled box is best.
- **Re-adjust λ to actual pasted area:** border clipping changes true area; label must match true area or it's miscalibrated.
- **Apply at input level (vs feature level):** ablation right plot — input level best; feature-level CutMix at most layers still beats baseline but input is best (occlusion semantics live in pixel space).
- **Shuffle within minibatch:** cheap random cross-image pairing, no second loader.
- **cutmix_prob differs by training recipe:** CIFAR reference example mixes on half the iterations; ImageNet reference example mixes every iteration. The probability knob controls the clean-vs-augmented batch mix without changing the image/label formula.

## Evaluation settings (pre-method facts)
CIFAR-10/100 (32×32, 50k/10k), ImageNet-2012 (1.28M, 1000 cls, top-1/5). Backbones: ResNet-50/101, PyramidNet-200, WideResNet, ResNeXt. SGD+momentum 0.9, weight decay, step LR. Augment underneath: pad-crop, flip, normalize. Also WSOL (CUB200, ImageNet-loc) via CAM, transfer to Pascal VOC detection / MS-COCO captioning, robustness (occlusion, adversarial FGSM), OOD detection. (Outcomes excluded.)

## Unsourced facts: none — all formulas verified against 3-Method.tex and supp1_Algorithm.tex; all code against train.py.
