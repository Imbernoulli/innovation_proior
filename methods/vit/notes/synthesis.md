# ViT — Synthesis Notes (Phase 1.5)

## The pain point / research question
Computer vision is dominated by CNNs (LeNet → AlexNet → ResNet). Convolution bakes in strong image-specific
inductive biases: locality (a unit sees a small neighborhood), 2D neighborhood structure, and translation
equivariance (a conv kernel applied everywhere ⇒ shift the input, shift the output). These priors make CNNs
sample-efficient on mid-sized data. Meanwhile in NLP, the Transformer (Vaswani 2017) — pure self-attention, almost
no domain prior — has become dominant precisely *because* it scales: pre-train on huge corpora, fine-tune; 100B+
params; no sign of saturation. The question: can we drop the convolutional inductive bias entirely and run a
**standard** Transformer on images, and does the NLP scaling story (data beats prior) carry over to vision?

Key tension: in-built priors help when data is scarce but cap the ceiling; a low-prior, high-capacity model
should underperform at small scale yet keep improving with more data. Sun 2017 ("revisiting unreasonable
effectiveness of data") already showed CNN performance grows logarithmically with dataset size up to JFT-300M.
BiT (Kolesnikov 2020) showed large supervised pre-training (ImageNet-21k, JFT-300M) + simple transfer beats
clever tricks. So the substrate (huge labeled datasets + transfer recipe) already exists; what's missing is a
*architecture that scales like a Transformer* applied to pixels.

## Load-bearing ancestors

### 1. The Transformer / scaled dot-product attention (Vaswani et al. 2017)
- Self-attention: from a sequence z ∈ R^{N×D}, project to q,k,v via U_qkv ∈ R^{D×3D_h}. A = softmax(qkᵀ/√D_h),
  SA(z)=Av. Cost O(N²·D_h) — quadratic in sequence length.
- **Why /√D_h:** with q,k entries ~ iid mean-0 var-1, the dot product q·k has variance D_h. Without scaling, as
  D_h grows the logits have large magnitude ⇒ softmax saturates ⇒ near-one-hot ⇒ vanishing gradient through
  softmax ⇒ training stalls. Dividing by √D_h restores logit variance ≈1, keeping softmax in its high-gradient
  regime. This is THE derivation to live out.
- **Multi-head (MSA):** run k heads in parallel, each in a D_h=D/k subspace, concat, project by U_msa ∈ R^{kD_h×D}.
  One softmax averages; multiple heads let different heads attend to different relations/positions simultaneously
  (one global, one local, etc.). Setting D_h=D/k keeps compute & params constant vs single head of width D.
- Encoder block = MSA + position-wise MLP (FFN), each wrapped with residual + LayerNorm. FFN: two linear layers
  with a nonlinearity, hidden width ~4×D (gives per-token capacity/mixing across channels; attention mixes across
  tokens, FFN mixes across features).
- Attention is permutation-equivariant ⇒ it has NO notion of order ⇒ must inject position info explicitly
  (Vaswani used fixed sinusoids; BERT used learned position embeddings).

### 2. BERT (Devlin 2019) — pretrain/finetune + [CLS] token + learned pos emb
- Established: pretrain a Transformer encoder on huge data, fine-tune on small downstream task.
- **[CLS] token:** prepend one learnable token; its final-layer state is a pooled sequence representation fed to a
  classifier head. Avoids picking/pooling over content tokens — gives a content-free slot that attention fills.
- Base/Large configs (L=12 D=768 H=12; L=24 D=1024 H=16) — ViT directly reuses these, plus a Huge.
- Learned 1D position embeddings, init normal(stddev=0.02) (ViT copies this init).

### 3. Pre-LN placement (Wang 2019; Xiong 2020 analysis)
- Post-LN (LN after residual, as in original Transformer) has large gradients near the output at init ⇒ unstable
  with large LR ⇒ needs careful warmup. Pre-LN (LN *inside* the residual branch, before MSA/MLP) gives
  well-behaved gradients at init, an unimpeded residual path ⇒ trains stably with larger LR, less warmup
  sensitivity. ViT uses Pre-LN: z'=MSA(LN(z))+z, z=MLP(LN(z'))+z'.

### 4. GELU (Hendrycks) — FFN nonlinearity
- Smooth gating x·Φ(x); standard in BERT/GPT Transformers, slightly better than ReLU in these large pretrained
  models. ViT keeps the NLP-standard GELU rather than swapping in something image-specific.

### 5. Cordonnier et al. 2020 — self-attention CAN express convolution
- Theorem: an MSA layer with N_h heads + a (relative) positional encoding can express ANY convolution of kernel
  √N_h × √N_h. They extract 2×2 patches and apply full SA. ViT is "most related" to this but the difference is the
  thesis: with enough pre-training data, the vanilla Transformer LEARNS whatever spatial structure it needs
  (including conv-like local heads) rather than having it hard-wired or proven-expressible at small scale. The
  2×2 patch restricts to tiny images; ViT uses 16×16 patches → medium-res images, tractable sequence length.

### 6. Prior attention-in-vision attempts (the "撞墙" of the field)
- Local attention / stand-alone self-attention (Parmar 2018; Ramachandran 2019; Hu 2019; Zhao 2020): restrict
  attention to local neighborhoods to replace conv. Theoretically efficient but need specialized/custom attention
  patterns ⇒ not efficient on TPU/GPU accelerators ⇒ don't scale.
- Sparse Transformers (Child 2019), axial attention (Ho 2019; Wang 2020 Axial-DeepLab), block attention
  (Weissenborn 2019): approximate global attention. Same problem: bespoke kernels, hard to scale on hardware.
- iGPT (Chen 2020): Transformer on raw pixels (after lowering res/color), generative/unsupervised, linear-probe
  72% ImageNet. Shows Transformers can model pixels but pixel-level is expensive and accuracy lagged.
- The common failure: either keep the CNN (attention as add-on) or build a custom attention that fights the
  hardware. Nobody had just run the *unmodified, hardware-friendly* Transformer at scale on image patches.

## The central design move: patches as tokens
- Naive: each pixel a token. N=H·W, attention O(N²)=O((HW)²) — bi-quadratic in resolution — intractable for
  224×224 (50k pixels). This is the wall.
- Fix: cut the image into a grid of P×P patches. x∈R^{H×W×C} → N=HW/P² flattened patches x_p∈R^{N·(P²·C)}.
  P=16 on 224 → 14×14=196 tokens. Sequence length now NLP-scale; attention O(N²) is fine. Bigger P ⇒ shorter
  sequence ⇒ cheaper but coarser; N ∝ 1/P².
- **Patch embedding:** flatten each patch (P²·C dims) and apply ONE trainable linear map E ∈ R^{(P²·C)×D} to get D.
  Equivalent to a Conv2d with kernel=stride=P (non-overlapping). This is the ONLY place spatial cutting happens.

## Equations to derive inline (main + appendix)
- z0 = [x_class; x¹_p E; …; x^N_p E] + E_pos,  E∈R^{(P²C)×D}, E_pos∈R^{(N+1)×D}
- z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}
- z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ
- y = LN(z_L^0)
- Appendix SA: [q,k,v]=zU_qkv; A=softmax(qkᵀ/√D_h); SA(z)=Av; MSA=[SA_1;…;SA_k]U_msa, D_h=D/k.

## Design decisions → why (with rejected alternatives)
- **Patchify (P=16)** — needed to escape O((HW)²); P trades cost vs granularity; N∝1/P². 2×2 (Cordonnier) too small
  for medium res; 16 keeps ~196 tokens. Hybrid alt: feed CNN feature-map patches (P=1 on feature map) — helps at
  small compute, gap vanishes at scale.
- **Single linear patch projection (not a conv stem)** — keep it "as standard a Transformer as possible," minimal
  image prior; the only injected 2D structure is the patch cut.
- **[CLS] token vs GAP** — inherited from BERT to stay close to NLP Transformer; appendix shows GAP works *equally
  well* but needs a different (lower) learning rate — so the choice is about staying standard, not accuracy. This is
  a key "the paper says it doesn't matter much, just LR" nuance to capture honestly.
- **Learned 1D pos emb vs 2D/relative/none** — appendix ablation: NO pos emb is much worse (0.61 vs 0.64 5-shot);
  among 1D/2D/relative, ~no difference. Reason: encoder operates on patches (14×14) not pixels (224×224); spatial
  relations at this coarse resolution are easy to learn from a flat learned table. So pick the simplest (1D). The
  network *learns* 2D topology anyway (row/col + distance structure visible in learned pos emb similarity).
- **/√D_h scaling** — variance/softmax-saturation argument above.
- **Multi-head, D_h=D/k** — different heads capture different (global/local) relations; keep compute constant.
- **Pre-LN + residuals** — stable training at large LR / scale (Xiong analysis).
- **GELU, FFN 4×** — NLP-standard; channel-mixing capacity complementing token-mixing attention.
- **Pretrain head = MLP w/ one hidden layer + tanh; finetune head = single linear, zero-init** — richer head while
  representation forms; at transfer, swap to a fresh zero-init linear D×K (zero-init = start from no-op, robust).
- **Higher-res fine-tuning + 2D interpolation of pos emb** — keep P fixed, longer sequence; pretrained pos emb table
  no longer matches grid ⇒ bilinearly interpolate by 2D location. Train-test resolution discrepancy (Touvron 2019),
  BiT practice.
- **Adam (not SGD) even for the ResNet baselines; high weight decay 0.1** — found better for transfer at scale
  (appendix SGD-vs-Adam ablation on ResNets); strong regularization essential when training from scratch on small
  data.
- **Class-token MLP head with tanh hidden (pre_logits/representation_size)** — confirmed in official JAX code.

## Inductive-bias / data-scale thesis (the heart)
- ViT has *much less* image-specific inductive bias than CNNs: only the patch cut and the finetune pos-emb interp
  inject 2D structure. MLP layers are local+translation-equivariant; self-attention is global. All spatial relations
  must be learned from data (pos emb start position-agnostic).
- Prediction (the bet): at small data (ImageNet alone) ViT underperforms comparable ResNets by a few points — the
  missing prior hurts. As data grows (ImageNet-21k → JFT-300M, 14M→300M imgs) ViT catches and overtakes — "large
  scale training trumps inductive bias." Diagnostic ablations on JFT subsets (9M/30M/90M/300M) show the crossover.
- These motivating/diagnostic facts (CNN priors help small-data; data-scaling curves; the crossover intuition) are
  pre-method context; the *proposed-method win numbers* (88.55% etc.) are out of scope.

## Canonical implementation (grounding for code)
- Official: google-research/vision_transformer (JAX/Flax) `models_vit.py`:
  patch embed = nn.Conv(features=hidden_size, kernel=patches.size, strides=patches.size, padding=VALID); reshape to
  (n, h*w, c); prepend zero-init cls token; AddPositionEmbs init normal(0.02); Encoder1DBlock = Pre-LN MSA + Pre-LN
  MlpBlock (gelu), residuals; final encoder_norm LN; classifier 'token' → x[:,0]; pre_logits Dense+tanh if
  representation_size; head Dense zero-init.
- Widely-used PyTorch: lucidrains/vit-pytorch `vit.py` — Rearrange patches → LayerNorm→Linear→LayerNorm; cls token
  + pos_embedding as nn.Parameter(randn); Pre-LN Attention(scale=dim_head**-0.5)+FeedForward(GELU); final norm;
  pool 'cls'→x[:,0]; mlp_head Linear. Final code will mirror PyTorch structure (clearest), faithful to official.
