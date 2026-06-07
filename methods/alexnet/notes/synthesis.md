# AlexNet — synthesis (grounded in NeurIPS 2012 paper + torchvision)

## Source
- Primary: Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS 2012. NO arXiv. PDF read in full (9 pp), proceedings.neurips.cc/.../c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.
- Canonical code: torchvision/models/alexnet.py — note this is the SINGLE-stream "one weird trick" (Krizhevsky 2014) variant: channels 64/192/384/256/256, classifier 4096/4096/1000, dropout 0.5 in first two FC, AdaptiveAvgPool2d((6,6)). Reproduces ~56.5% top-1 / 79.1% top-5 (simplified recipe). The ORIGINAL paper is two-GPU with 96/256/384/384/256.

## Pain point / research question (in-frame, 2012)
- Object recognition in realistic settings needs huge labeled data + high-capacity models. Small datasets (NORB, Caltech-101/256, CIFAR-10/100, tens of thousands) let simple methods nearly solve MNIST (<0.3%), but real-world variability demands models with large learning capacity.
- ImageNet now exists: 15M labeled images, 22k categories; ILSVRC subset ~1.2M train, 50k val, 150k test, 1000 classes. Metrics top-1, top-5 error.
- The capacity needed is so large the task cannot be specified by even ImageNet, so the model must encode strong priors. CNNs (LeCun) do: capacity controlled by depth/breadth; strong, mostly-correct assumptions — stationarity of statistics + locality of pixel dependencies — so far fewer connections/params than equivalent fully-connected nets, easier to train, theoretical best only slightly worse.
- Obstacle: CNNs prohibitively expensive at high resolution/large scale. GPUs + optimized 2D conv now make it feasible.
- Two sub-problems to solve to make this work: (1) train fast enough (saturating tanh/sigmoid too slow), (2) prevent overfitting of a 60M-param model even with 1.2M images.

## Architecture (ORIGINAL, two-GPU; Sec 3.5)
- Input 224×224×3 (random crops from 256×256). Down-sampled images to 256×256 (rescale shorter side to 256, central crop), subtract per-pixel mean over training set. Raw centered RGB.
- 8 weight layers: 5 conv + 3 FC.
- conv1: 96 kernels 11×11×3, stride 4. → ReLU → LRN → max-pool(3,2 overlapping).
- conv2: 256 kernels 5×5×48 (48 because input split across 2 GPUs). → ReLU → LRN → max-pool.
- conv3: 384 kernels 3×3×256 (connected to all GPU maps of layer2). → ReLU. (no pool/norm)
- conv4: 384 kernels 3×3×192. → ReLU.
- conv5: 256 kernels 3×3×192. → ReLU → max-pool.
- FC6: 4096. FC7: 4096. FC8: 1000-way softmax.
- ReLU after every conv and FC.
- Response-norm layers follow conv1 and conv2. Max-pool follows both LRN layers and conv5.
- Neurons per layer: 253,440–186,624–64,896–64,896–43,264–4096–4096–1000. 60M params, 650k neurons.

## The novel/unusual features (Sec 3, ordered by importance)
### 3.1 ReLU f(x)=max(0,x) — non-saturating
- Standard: f=tanh(x) or sigmoid (1+e^-x)^-1, both saturating → slow gradient descent.
- ReLU non-saturating → several× faster training. Fig 1: 4-layer CNN on CIFAR-10 reaches 25% training error 6× faster with ReLU than tanh. Large nets on large data infeasible with saturating units.
- Prior alt: Jarrett et al. f=|tanh(x)| with contrast norm + local avg pooling, good on Caltech-101, but their concern was preventing overfitting, different effect from ReLU's faster *fitting*.

### 3.2 Two-GPU training
- GTX 580 has 3GB → limits net size. 1.2M examples make nets too big for one GPU. Spread net across two GPUs.
- Cross-GPU parallelization: put half the kernels (neurons) on each GPU; GPUs communicate only in certain layers. E.g. layer-3 kernels take input from all layer-2 kernel maps; layer-4 kernels take input only from layer-3 maps on the same GPU. Connectivity pattern chosen by cross-validation to tune communication to acceptable fraction of compute.
- Similar to columnar CNN of Cireşan et al., except columns not independent.
- Reduces top-1/top-5 by 1.7%/1.2% vs net with half as many kernels per conv layer on one GPU. Two-GPU net slightly faster to train.

### 3.3 Local Response Normalization (LRN)
- ReLUs don't need input normalization to avoid saturation; but local normalization aids generalization.
- b^i_{x,y} = a^i_{x,y} / (k + α Σ_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} (a^j_{x,y})^2)^β
- Sum over n adjacent kernel maps at same spatial position; N total kernels. Constants k=2, n=5, α=1e-4, β=0.75, set on validation set.
- Applied after ReLU in certain layers (conv1, conv2).
- Form of lateral inhibition (inspired by real neurons): competition for big activities among outputs of different kernels. "Brightness normalization" (no mean subtracted).
- Reduces top-1/top-5 by 1.4%/1.2%. CIFAR-10 4-layer: 13%→11% with norm.

### 3.4 Overlapping pooling
- Pooling grid spaced s apart, each summarizing z×z neighborhood. s=z → traditional non-overlapping. s<z → overlapping.
- Use s=2, z=3. Reduces top-1/top-5 by 0.4%/0.3% vs s=2,z=2. Slightly harder to overfit.

## Overfitting countermeasures (Sec 4)
### 4.1 Data augmentation (both label-preserving, CPU-generated while GPU trains previous batch → "free")
- (a) Translations + horizontal reflections: extract random 224×224 patches (+ their reflections) from 256×256 images → ×2048 training set. At test: 5 crops (4 corners + center) + reflections = 10 patches, average softmax. Without this scheme, substantial overfitting forcing smaller nets.
- (b) RGB intensity / PCA color augmentation: PCA on RGB pixel values over the training set. To each pixel I_{xy}=[I^R,I^G,I^B]^T add [p1,p2,p3][α1λ1, α2λ2, α3λ3]^T, where p_i,λ_i are eigenvectors/eigenvalues of 3×3 RGB covariance, α_i ~ N(0,0.1) drawn once per image per epoch (re-drawn each presentation). Captures invariance of object identity to illumination intensity/color. Reduces top-1 by >1%.

### 4.2 Dropout
- Combining many models reduces test error but too expensive for big nets. Dropout = efficient model combination at ~2× training cost.
- Set each hidden neuron's output to 0 with prob 0.5. Dropped neurons don't contribute to forward pass or backprop. Each input → samples a different architecture, all sharing weights.
- Reduces complex co-adaptations: a neuron can't rely on particular other neurons being present → forced to learn more robust features useful with many random subsets.
- Test: use all neurons, multiply outputs by 0.5 — reasonable approximation to the geometric mean of the predictive distributions of exponentially many dropout nets.
- Used in first two FC layers. Without dropout: substantial overfitting. Dropout ~doubles iterations to converge.

## Training details (Sec 5)
- SGD, batch 128, momentum 0.9, weight decay 0.0005. Weight decay here not merely a regularizer — it reduces training error too (important for the model to learn).
- Update rule:
  v_{i+1} := 0.9·v_i − 0.0005·ε·w_i − ε·⟨∂L/∂w |_{w_i}⟩_{D_i}
  w_{i+1} := w_i + v_{i+1}
  (i = iter index, v = momentum, ε = lr, ⟨⟩_{D_i} = avg over batch D_i of gradient at w_i)
- Init: weights ~ N(0, 0.01). Biases = 1 in conv2,4,5 and FC layers (gives ReLUs positive inputs, accelerates early learning); biases = 0 in remaining layers (conv1, conv3).
- Equal lr all layers, manually adjusted: lr divided by 10 when validation error stopped improving; initialized at 0.01, reduced 3× before termination.
- ~90 epochs over 1.2M images, 5–6 days on two GTX 580 3GB GPUs.

## Objective
- Net maximizes multinomial logistic regression objective = maximize average over training cases of log-prob of correct label under prediction (softmax). I.e. cross-entropy / NLL.

## Depth matters
- Removing any single conv layer (each <1% of params) → ~2% worse top-1. So depth important, not just params.

## Load-bearing ancestors (for context / baselines)
- LeCun et al. 1990 — CNNs, backprop-trained digit recognition. Capacity controlled by depth/breadth; stationarity + locality priors.
- Nair & Hinton 2010 — ReLU (rectified linear units improve RBMs).
- Hinton et al. 2012 (arXiv:1207.0580) — dropout (preventing co-adaptation of feature detectors).
- Jarrett et al. 2009 — |tanh| + contrast norm + local avg pool; best multi-stage architecture for object recognition (ICCV).
- Cireşan et al. 2011/2012 — high-performance GPU CNNs, multi-column / columnar CNNs.
- LeCun, Bottou et al.; Turaga et al.; Lee et al. — CNN lineage.
- Deng et al. 2009 — ImageNet. Berg/Deng/Fei-Fei 2010 — ILSVRC.
- Prior ILSVRC SoTA: sparse coding (47.1/28.2 ILSVRC-2010), SIFT+FV (Sánchez & Perronnin 2011; 45.7/25.7). These are the baselines AlexNet beats; do NOT report AlexNet's own wins in context.
- Russell et al. (LabelMe), Griffin (Caltech-256), Fei-Fei (Caltech-101).

## Design-decision → why table
| Decision | Why this, not alternative |
|---|---|
| ReLU not tanh/sigmoid | saturating units → tiny gradients in saturation → slow GD; ReLU non-saturating gradient = 1 for x>0 → ~6× faster to a training-error target; at this scale (large net, large data, days of training) speed is decisive — couldn't run the experiment with tanh. |
| Train on 2 GPUs, half kernels each, communicate only some layers | 3GB GPU can't hold the net; splitting kernels halves per-GPU memory; restricting cross-GPU connectivity keeps communication an acceptable fraction of compute (tunable). vs single smaller net: 2-GPU bigger net is 1.7/1.2% better. |
| LRN after ReLU on conv1,conv2; k=2,n=5,α=1e-4,β=0.75 | ReLU is unbounded → one kernel can fire arbitrarily large; lateral inhibition makes kernels compete at each location, big activities normalized by neighbors → contrast, better generalization; ~1.4/1.2% gain. Constants picked on val set. "Brightness" norm (no mean subtraction) because activations are post-ReLU ≥0. |
| Overlapping pooling s=2,z=3 | overlapping windows give more pooling units summarizing richer neighborhoods; empirically harder to overfit, 0.4/0.3% better than s=z=2. |
| Depth = 5 conv + 3 FC | removing any one conv layer (<1% params) loses ~2% top-1 → depth not param count carries it. |
| Translation+reflection crops ×2048 | cheap label-preserving expansion; without it 60M params overfit 1.2M images badly, would force a smaller net. Test 10-crop averaging reduces variance. |
| PCA color jitter, α~N(0,0.1) per image | object identity invariant to illumination intensity/color; injects that invariance; >1% top-1. |
| Dropout 0.5 in first 2 FC, ×0.5 at test | FC layers hold most of the 60M params → where overfitting lives; dropout = cheap ensemble of 2^n shared-weight nets preventing co-adaptation; test ×0.5 ≈ geometric mean of the ensemble. Only FC (conv layers already param-light + weight-sharing regularized). |
| weight decay 5e-4 | not just regularization — it lowered *training* error here; needed for the model to learn at all. |
| bias init = 1 in conv2,4,5,FC; 0 elsewhere | positive bias feeds ReLUs positive inputs early → non-zero gradients → accelerates early learning; conv1/conv3 left at 0. |
| weights ~ N(0,0.01) | small symmetric-breaking init. |
| global mean-subtraction only, raw RGB | minimal preprocessing; let the net learn features. |
| lr 0.01, ÷10 on val plateau, momentum 0.9 | standard SGD; manual step decay when val error stalls. |

## Notes on code grounding
- The original is 2-GPU with grouped convs (groups=2 in conv2,4,5). The widely-used canonical single-stream torchvision form folds the two towers into one stream with channels 64/192/384/256/256 and no LRN; I will present the in-frame reasoning landing on the *original* two-tower architecture (since that's the discovery), and give working code in the single-stream torchvision style (the canonical implementation), with LRN shown as in the paper. This is the standard reconciliation; flag clearly that channel counts in modern code differ from the paper's original.
