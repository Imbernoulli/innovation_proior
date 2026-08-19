# lion-program-search — rung table (working notes, not part of the narrative)

Research question: discover (not hand-design) a first-order optimizer update rule that
generalizes from cheap proxy tasks to SOTA-scale training, with memory <= AdamW's two buffers.
All measured numbers below are from methods/lion/src/release.tex, Table `tab:multi`
(lines 941-952), "The performance of various optimizers to train ViT-S/16 and ViT-B/16 on
ImageNet (with RandAug and Mixup)". Metrics: ImageNet / ReaL / V2 top-1 accuracy, three runs
averaged. All entrants get their own lr/weight-decay tuned (release.tex L931-932).

## Given (pre-existing, goes in 00-initial-context.md, not a rung)
AdamW — decoupled weight decay, coupled second moment (Adam's usual m/sqrt(v)).
- ViT-S/16: ImageNet 78.89, ReaL 84.61, V2 66.73 (L946-948)
- ViT-B/16: ImageNet 80.12, ReaL 85.46, V2 68.14 (L950-952)
This is the bar every rung must clear.

## Rung 1 — priorauto: PowerSign_e / AddSign_1 (Bello et al. 2017, restricted expression-tree
search over fixed operands {g, m} — reuse an EXISTING automated-discovery result before
building new search machinery)
Formulas (verified via WebFetch of ar5iv.labs.arxiv.org/html/1709.07417, arXiv:1709.07417,
"Neural Optimizer Search with Reinforcement Learning", Bello et al. 2017 — the paper is not
locally cached, so this is the one piece of grounding fetched fresh rather than found on disk;
recorded below and in methods/lion/refs/):
  PowerSign (alpha=e, f(t)=1 default): update = e^{sign(g)*sign(m)} * g
  AddSign   (alpha=1, f(t)=1 default): update = (1 + sign(g)*sign(m)) * g
  m = bias-corrected EMA of the gradient.
Motivation: context.md's Background already names these as the output of a restricted,
fixed-operand/bounded-tree search (Neural Optimizer Search) — the natural first check before
investing in a new, more flexible search space is whether that prior automated result already
clears AdamW under THIS eval protocol (ViT-S/16 & ViT-B/16, ImageNet, RandAug+Mixup).
Measured (release.tex L946, L950, cols 6-7 of tab:multi):
- ViT-S/16: PowerSign 77.36 / 83.39 / 65.17;  AddSign 77.37 / 83.36 / 64.52
- ViT-B/16: PowerSign 78.95 / 84.76 / 67.46;  AddSign 78.50 / 84.49 / 65.95
Result: BOTH below AdamW on every column at both scales. Restricted-tree automated discovery
does not clear the hand-designed bar here -> motivates a genuinely more flexible program
representation (arbitrary imperative program over buffers, not a fixed-operand expression tree)
and running a new search rather than reusing a prior automated result.

## Rung 2 — single_beta: Ablation_0.9 / Ablation_0.99 — `m = interp(g, m, beta); update = sign(m)`
(release.tex L960-964, exact ablation definition + name)
Motivation: run the flexible program-search machinery (regularized evolution, warm-started at
AdamW, abstract-execution pruning) instead of a restricted tree. Supporting real numbers for
*why* directed search is needed at all, cited in this rung's reasoning as background computation
(not itself feedback-with-a-metric-table, so quoted inline, not as a separate rung):
  - random search over 2M programs on the proxy task: best still significantly worse than AdamW
    (release.tex L418).
  - regularized evolution (warm-start=AdamW, pop=1000, tournament=2) significantly outperforms
    both an AdamW-hyperparameter-tuning baseline and a random-search baseline, each given 4x more
    compute (release.tex L427, L451).
  - abstract-execution pruning: 69.8+-1.9% of statements end up redundant (removal makes programs
    ~3x shorter), cache hit rate 89.1+-0.6% (~10x search-cost reduction) (release.tex L464).
  - proxy task: 3-layer/96-hidden/3-head ViT, 10% ImageNet, 30k steps, batch 64, 64x64 images,
    patch 16 (release.tex L1324-1325; already given in methods/lion/results/context.md's Code
    framework / Evaluation settings, so treated as PRE-EXISTING harness detail, not new here).
The raw discovered program (post-pruning; release.tex Program lst:raw / lst:p2, transcribed in
methods/lion/notes/synthesis.md) reduces, on first read, to "sign of an accumulating buffer" —
the same shape as signSGD-momentum, just with an unfamiliar constant. Rung-2 proposal: test that
literal simplest reading — ONE interpolation constant used both to update the buffer and to form
the signed step — at the two natural candidate values suggested by the raw program's own
constants (beta~0.9, matching ordinary momentum; beta~0.99, matching the buffer's slower
constant), rather than assuming two constants are required.
Measured (release.tex L946, L950, cols 8-9 of tab:multi):
- ViT-S/16: Ablation_0.9 78.23 / 84.28 / 66.13;  Ablation_0.99 78.19 / 84.17 / 65.96
- ViT-B/16: Ablation_0.9 79.54 / 85.10 / 68.07;  Ablation_0.99 79.90 / 85.36 / 68.20
Result: both beat PowerSign/AddSign (rung 1) at every column/scale -> program search did find a
real improvement over the restricted-tree result. But both still lose to AdamW itself on
ImageNet at both scales (78.23/78.19 < 78.89; 79.54/79.90 < 80.12) and mostly on ReaL/V2 too
(only Ablation_0.99's V2 B/16 68.20 edges AdamW's 68.14). A single constant does not clear the
bar.

## Rung 3 (final) — lion: decoupled two-constant sign-momentum
  c_t = beta1*m_{t-1} + (1-beta1)*g_t ;  theta_t = theta_{t-1} - eta*(sign(c_t) + lambda*theta_{t-1})
  m_t = beta2*m_{t-1} + (1-beta2)*g_t   [[beta1=0.9, beta2=0.99]]
Motivation: rung 2's failure is diagnosable -- a single EMA constant cannot simultaneously hold a
long memory (favors beta near 1) and stay reactive to the fresh gradient in the step itself
(favors beta well below 1); the raw discovered program never actually used one constant -- its
two chained `interp` calls use DIFFERENT constants (~0.9 and ~1.1; exact appendix pseudocode
constants 0.8999999761581421 and 1.109133005142212, release.tex L1370-1373) -- substituting one
into the other (verified symbolically and numerically, methods/lion/notes/sources.md
"Verification scratch") shows the second buffer is exactly a single persistent EMA of the raw
gradient at beta2~0.99 (~0.99822 at full precision), decoupled from what gets signed, which is a
beta1~0.9 blend of that buffer with the fresh gradient. Also strips: `cosh` (release.tex L527,
provably dead -- overwritten every iteration before use), `arcsin`/`clip` on the incoming
gradient (release.tex L528, ablated with no quality drop), and the `m*m; sqrt; m/abs_m` chain
(exactly sign(m) since sqrt(m^2)=|m|).
Measured (release.tex L946, L950, rightmost col of tab:multi, boldface = best):
- ViT-S/16: Lion 79.46 / 85.25 / 67.68
- ViT-B/16: Lion 80.77 / 86.15 / 69.19
Result: beats AdamW, PowerSign/AddSign, and both single-constant ablations on every column at
both scales. Endpoint of the ladder -- the published method.

## Not used as separate rungs (why)
- Raw discovered program (pre-simplification) vs the algebraically-simplified/pruned form: no
  distinct measured numbers exist for these -- redundant-statement removal and the v-fold are
  exact identities (verified above), and arcsin/clip removal is reported only qualitatively ("no
  quality drop", release.tex L528) with no separate accuracy row. Folded into rung 3's reasoning
  as on-page derivation, not given its own feedback file.
- RAdam / AdaBelief / AMSGrad (release.tex L946-952, cols 3-5): real numbers exist but they are
  not part of context.md's established background/baselines and are not tested by the narrator
  at any point (nobody proposes trying them) -- omitted to avoid inventing an untested rung.
