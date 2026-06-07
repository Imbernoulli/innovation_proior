# DenseNet — synthesis notes (Phase 1.5)

## Pain point at the time (2015–2016)
CNNs went from LeNet (5 layers) → VGG (19) → and only "last year" Highway/ResNet crossed 100 layers.
As depth grows, the input signal and the gradient pass through many transformations and can
"wash out" by the time they reach the other end — vanishing/exploding signal. The whole subfield
(Highway, ResNet, stochastic depth, FractalNet, deep supervision) is converging on one trick:
**create short paths from early layers to late layers**. The research question: find the
*simplest* connectivity that maximizes information + gradient flow, and ideally also fixes the
parameter waste of just stacking ever-wider/deeper plain nets.

## Load-bearing ancestors (verified against primary text + sources)

### Highway Networks (Srivastava et al. 2015, arXiv:1505.00387)
- First to train >100 layers end-to-end. Formula: **y = H(x)·T(x) + x·C(x)**, T transform gate,
  C carry gate, usually C = 1−T (both data-dependent, learned via sigmoid).
- Gates let info pass "unimpeded" through highways. Limitation: gates are **learned and add
  parameters**; the path is only open when the gate decides to open it. Not a guaranteed clean path.

### ResNet (He et al. 2015, arXiv:1512.03385)
- Degradation problem: deeper *plain* nets have **higher training error** — an optimization
  failure, not overfitting (a shallower net + identity-copied extra layers is a strictly
  available solution the optimizer can't find).
- Fix: residual learning. Make layers fit F(x)=H(x)−x; output **x_ℓ = H_ℓ(x_{ℓ−1}) + x_{ℓ−1}**.
  Identity shortcut adds **zero parameters**, gradient flows directly through identity.
- Limitation the paper reacts against: the identity and H output are combined by **summation**.
  Summation can "impede" information flow — features get merged additively, and each residual
  unit still carries its own full-width weights (state is rewritten every block).

### Identity Mappings / pre-activation ResNet (He et al. 2016, arXiv:1603.05027)
- Reorder block to **BN–ReLU–Conv** so the shortcut is a *clean* identity all the way through.
- Derivation: with clean identity, x_L = x_ℓ + Σ_{i=ℓ}^{L−1} F(x_i). Then
  ∂L/∂x_ℓ = ∂L/∂x_L · (1 + ∂/∂x_ℓ Σ F). The "1" term means the gradient at x_L reaches x_ℓ
  **undiminished** regardless of depth → no vanishing through the skip path.
- DenseNet borrows the BN-ReLU-Conv composite H exactly ("Motivated by [identity-mappings]").

### Stochastic Depth (Huang et al. 2016, arXiv:1603.09382)
- Randomly **drop** whole residual blocks during training (replace F with identity), survival
  prob p_ℓ decaying linearly with depth. Trains a 1202-layer ResNet successfully.
- Two payoffs that directly seed DenseNet:
  (1) **Many layers are redundant** — you can drop them and accuracy *improves*. So a deep net's
      per-layer dedicated full-width weights are largely wasted.
  (2) Dropping a block creates a **direct connection** between its neighbors → the surviving
      network has short paths between non-adjacent layers. DenseNet = make those direct
      connections *permanent and deterministic*.
- "Our paper was partly inspired by that observation."

### Inception / GoogLeNet (Szegedy et al. 2015) — the concatenation precedent + bottleneck
- Inception module **concatenates** filter outputs of different sizes → increases feature
  diversity, reduces correlation. So concatenation (not summation) as a combine op is already
  on the table; DenseNet pushes it across *layers* not just within a module.
- Also the **1×1 bottleneck** trick: a 1×1 conv before the expensive 3×3 conv to cut the number
  of input channels and thus FLOPs/params. DenseNet reuses this verbatim (the "-B" variant).

## The core idea and why it falls out
If the field's lesson is "short paths help" and stochastic depth's lesson is "dedicated
full-width per-layer weights are redundant," the limit case is: connect **every** layer to
**every** later layer directly, and make each layer add only a *tiny* amount of new information.

Two combine choices for "give layer ℓ everything before it":
- **Sum** (ResNet style): x_ℓ = H_ℓ(x_{ℓ−1}) + x_{ℓ−1}. Requires equal channel count; merges
  features so the identity and new features can interfere/cancel; only connects ℓ−1 → ℓ.
- **Concatenate** (chosen): **x_ℓ = H_ℓ([x_0, x_1, …, x_{ℓ−1}])**. Every preceding feature-map
  is preserved verbatim and individually accessible; no information is overwritten. This gives
  ℓ inputs to layer ℓ and L(L+1)/2 total connections in an L-layer net.

Concatenation is what makes "explicitly separate added info from preserved info" possible:
preserved info is never summed away, only appended.

## Growth rate k — the key efficiency knob (with arithmetic)
Each H_ℓ produces exactly **k** new feature-maps (k small, e.g. 12). Layer ℓ then sees
**k₀ + k·(ℓ−1)** input channels (k₀ = channels entering the block). k is the *growth rate*.

Why small k works: the concatenated stack is a "global state / collective knowledge." Every
layer can read the whole state, so each layer only needs to *contribute* k new maps rather than
re-derive/re-carry the whole representation. No need to replicate state from layer to layer
(unlike plain nets / ResNets, where each layer's full width is re-learned every block).

**Parameter arithmetic (concat vs constant-width):**
A 3×3 conv at layer ℓ costs ≈ (in_channels)·k·3·3 = 9k·(k₀ + k(ℓ−1)) params.
Summing over ℓ = 1..L: total ≈ 9k·[L·k₀ + k·L(L−1)/2] = O(k²L²).
With *small* k this O(k²L²) is tiny. Contrast a constant-width net of width W: each layer is
9·W·W and L layers give 9·W²·L = O(W²L). To get comparable representational reach W must be
large (e.g. 256–512), so W²L ≫ k²L² when k≈12. Equivalent accuracy at ~1/3 the ResNet params
(and 0.8M-param DenseNet ≈ 10.2M-param 1001-layer ResNet) is the consequence. The efficiency
is *because* features are reused instead of re-learned: there are no redundant carried copies.

## Down-sampling: dense blocks + transition layers
Concatenation needs equal spatial size, but conv nets must down-sample. So split the net into
several **dense blocks** (dense connectivity only inside a block, fixed spatial size), separated
by **transition layers** = BN → 1×1 conv → 2×2 average pool. The 1×1 conv adjusts channel count;
the avg-pool halves spatial size. CIFAR: 3 blocks at 32×32, 16×16, 8×8. ImageNet: 4 blocks +
a stem (7×7 stride-2 conv, 3×3 stride-2 maxpool) and a global-avg-pool + softmax head.

## Bottleneck (-B): 1×1 conv inside H
Layer ℓ produces only k maps but *consumes* k₀+k(ℓ−1) — input grows linearly while output is
fixed at k. So later layers have huge inputs. Insert a 1×1 conv first to squeeze the big input
down to a fixed **4k** channels before the 3×3 conv:
H = BN-ReLU-Conv(1×1, →4k) - BN-ReLU-Conv(3×3, →k). The 3×3 now always sees 4k inputs instead
of k₀+k(ℓ−1), so its cost stops growing with ℓ. 4k chosen empirically as a good
capacity/efficiency point (enough mixing, still cheap).

## Compression (-C): θ at transitions
A block ending with m maps → transition outputs ⌊θ·m⌋ maps, 0<θ≤1, θ=0.5 used. Halves the
channel count entering the next block, further shrinking the model. Justified by the feature-reuse
heatmap: transition-layer outputs get the *lowest* average weight from later layers (most
redundant), so they are the safest to compress. Both → **DenseNet-BC**.

## Implicit deep supervision (why training is easy)
Each layer connects directly to the final classifier through at most 2–3 transition layers, so
every layer receives gradient ~directly from the loss — like Deeply-Supervised Nets (auxiliary
classifiers at every layer) but with a **single** shared loss, far simpler. Also short input→layer
paths. Plus a regularizing effect (less overfitting on small data) from feature reuse.

## Connection to stochastic depth (discussion)
In stochastic depth, if all intermediate residual layers between two pooling layers are dropped,
the two surviving layers become directly connected — same *connectivity pattern* DenseNet builds
deterministically. So DenseNet is, loosely, the "always-on" version of the connectivity that
stochastic depth creates randomly.

## Motivating/diagnostic findings (pre-method, allowed in context)
- Deeper plain nets → higher *training* error (ResNet degradation).
- Stochastic depth: dropping random residual layers improves accuracy → deep nets are redundant.
- Inception/Szegedy: strong correlation among same-layer features; can aggregate channels without
  hurting accuracy → motivation to increase feature diversity, reduce correlation.
- Feature-reuse heatmap (DenseNet's own diagnostic on a *trained net*, about how features are
  used — this is an analysis of the architecture's behavior, used to justify compression; treat
  as method-internal, keep out of context.md; it can appear as "what I'd want to verify").

## Design-decision → why table
- Concatenate not sum → preserve every feature verbatim, individually accessible; separate added
  vs preserved info; enables small-k. Sum forces equal width and can cancel/merge features.
- Connect ALL preceding layers (not just k-nearest) → maximum short paths; (later verified
  long-range connections are necessary — removing them hurts).
- Small growth rate k (≈12) → each layer adds little to shared state; O(k²L²) params, big savings.
- H = BN-ReLU-Conv (pre-activation) → clean gradient path, borrowed from identity-mappings.
- Dense blocks + transition layers → reconcile concat (needs equal size) with required downsampling.
- Transition = 1×1 conv + 2×2 avg pool → cheap channel adjust + spatial halving; avg (not max)
  pool standard for transitions.
- Bottleneck 1×1 → 4k → cap the 3×3 conv's input at 4k so cost doesn't grow with depth.
- Compression θ=0.5 → halve channels at transitions; transition outputs are most redundant.
- Stem for ImageNet (7×7 s2 conv + 3×3 s2 maxpool) → standard ImageNet downsampling, matches ResNet.
- Global avg pool + single softmax → standard head; single loss gives implicit deep supervision.
- num_init_features = 2k for BC / 16 for basic → enough channels entering block 1.

## Canonical implementation
torchvision `densenet.py` (downloaded to code/torchvision_densenet.py). Implements DenseNet-BC:
_DenseLayer (BN-ReLU-Conv1x1→bn_size·k, BN-ReLU-Conv3x3→k, optional dropout, optional
checkpointing), _DenseBlock (keeps a growing list of feature tensors, cats them as input to each
layer, returns full cat), _Transition (BN-ReLU-Conv1x1 halving channels + 2x2 avgpool),
DenseNet (7x7 stem + maxpool, 4 blocks with //2 compression at transitions, final BN, GAP, FC).
bn_size=4 (=4k bottleneck), default compression //2 (θ=0.5), block_config (6,12,24,16)=DenseNet-121.
This is the grounding for the Phase-2 code.
