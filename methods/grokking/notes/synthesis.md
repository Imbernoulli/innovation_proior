# Grokking synthesis

arXiv 2201.02177 (verified). Power, Burda, Edwards, Babuschkin, Misra. OpenAI. (ICLR 2021 workshop /
arXiv 2022). Empirical-first phenomenon paper.

## Pain / research question
Generalization of overparameterized nets defies classical learning theory. Want a clean, fast,
controllable TESTBED to study generalization decoupled from training-set performance — where
memorization vs. true generalization can be cleanly separated and studied in detail on a single GPU.

## The setup (the "method")
Binary operation tables a∘b=c. a,b,c are DISCRETE ABSTRACT SYMBOLS with NO internal structure — the
network never sees numbers in decimal or permutations in line notation; it sees distinct tokens and
must infer all properties from how symbols interact. Training on a fraction of all equations = filling
blanks in the op table (like Sudoku). Each equation tokenized as: <x> <op> <y> <=> <x∘y>, each of the
5 positions a separate token. Train on a random fraction (train_pct) of all p² (or |S5|²) equations;
rest is validation.
Operations (p=97 prime): x+y, x−y, x/y mod p; x²+y², x²+xy+y², x²+xy+y²+x, x³+xy, x³+xy²+y mod 97;
mixed [x/y if y odd else x−y]; group S5: x·y, x·y·x⁻¹, x·y·x.

Model: standard DECODER-ONLY transformer (Vaswani-style, causal attention masking), loss & accuracy
computed only on the ANSWER token (the c after =). 2 layers, width (d_model) 128, 4 attention heads,
≈4×10⁵ non-embedding params.

Optimizer (main): AdamW, lr 1e-3, weight decay 1, β1=0.9, β2=0.98, linear LR warmup over first 10
updates, no LR anneal (chose simplicity over slight gains), minibatch 512 or half of train set
(whichever smaller), budget 1e5 gradient updates. For the dramatic grokking figure (modular division):
Adam, NO weight decay, budget 1e6 to emphasize how late generalization begins. Learning-time-curve
experiments: budget 5e5. 3 seeds (7 for learning-time).

## The phenomenon: GROKKING (delayed generalization)
Train acc reaches ~100% in <10³ updates (memorizes fast). Validation acc stays at chance until ~10⁵
steps, then SUDDENLY climbs to perfect generalization — generalization happens ~1000× LATER than
memorization, "well past the point of overfitting." Loss curves show validation loss rising
(overfitting) then a SECOND DESCENT (validation loss falls again far later) — but unlike deep double
descent, here it's purely along the TRAINING-TIME axis, far past first interpolation (tens of
thousands of epochs later), and accuracy is monotone (no accuracy peak), so distinct from
model-wise/epoch-wise double descent.

## Key findings (in-scope: about the phenomenon / existing-model behavior)
1. Late generalization (grokking) for a range of models/optimizers/dataset sizes, most pronounced near
   the MINIMAL dataset size for which the net generalizes within budget. Larger train fractions →
   train/val curves track each other (less dramatic).
2. Learning-time curves: converged generalization stays 100% over a range of train sizes, but the
   OPTIMIZATION TIME to reach 99% val acc grows FAST (≈exponentially) as dataset size decreases. Near
   25–30% data for S5, a 1% drop in train data → 40–50% increase in median steps-to-generalize.
   Steps to 99% TRAIN acc stay ~10³–10⁴ regardless.
3. Symmetry: operations symmetric in operands (x+y, x·y, x²+y², x²+xy+y²) need LESS data than
   non-symmetric counterparts (x−y, x/y, x²+xy+y²+x) — partly architecture-dependent (transformer can
   learn symmetric fn by ignoring positional embedding).
4. Equivalence: x+y mod (p−1) and x·y mod p are indistinguishable to the net (abstract symbols; every
   nonzero residue mod prime = power of primitive root), so x−y and x/y take ≈ same data. ✓ observed.
5. Some ops (x³+xy²+y mod 97) NEVER generalize within budget at any % up to 95% — net just memorizes,
   data effectively random to it.

## Ablations / tricks (the design rationale — in-frame)
Tried: full-batch GD, SGD, large/small LR, residual dropout, weight decay, gradient noise (all on S5).
- WEIGHT DECAY: the standout. More than HALVES the data needed vs most other interventions. Weight
  decay toward ORIGIN slightly better than toward INITIALIZATION → the prior "≈zero weights suit small
  algorithmic tasks" explains part but not all of its benefit.
- Noise (minibatch gradient noise, Gaussian weight noise before/after gradient) helps generalization —
  consistent with noise pushing optimization toward FLATTER minima that generalize better.
- LR must be tuned within ~1 order of magnitude for generalization to happen.
- Some generalization happens even with full-batch optimizers and no weight/activation noise at high
  data %.

## Mechanistic hints (in-scope discussion)
- Embedding visualization (t-SNE of output/unembedding layer): for modular addition → circular
  topology ("number line" by adding 8); for S5 → clusters = cosets of a subgroup / conjugates.
  Structure clearer WITH weight decay. The net recovers real structure of the math object.
- Outliers experiment: inject k random-label outliers. Net always reaches 100% train acc (interpolates
  regardless of k); generalization range shrinks as k grows but small k (≤1000) barely hurts → capacity
  far exceeds what's needed to memorize labels, so generalization needs a non-trivial explanation
  (not "couldn't memorize").
- Sharpness: Spearman corr −0.795 (p<1.4e-5) between val acc and sharpness φ of the minimum on S5 →
  grokking may only happen once params reach FLATTER regions of the loss landscape.

## Design-decision → why
- Abstract symbols (no internal structure): forces the net to learn the operation's structure purely
  from symbol interactions; cleanly separates "memorized the table" from "found the pattern." If you
  fed decimal digits the net could exploit surface form.
- Binary op tables / fill-the-blanks: a finite, fully-specified task where train and val are disjoint
  slots of the SAME table — generalization = predicting unseen slots, exactly decoupled from train.
- Small algorithmic dataset, single GPU: fast, reproducible testbed; effects more pronounced than on
  natural data.
- Decoder-only transformer, loss on answer token only: standard seq model; only the c-prediction is
  the supervised target (the question tokens are context).
- 2 layers / 128 / 4 heads: small enough to train to 1e6 steps cheaply, big enough to grok.
- AdamW + weight decay 1: weight decay is THE intervention that most improves data efficiency.
- Long optimization budget (1e5–1e6): generalization happens far past memorization, so you must train
  orders of magnitude longer than to fit train — short budgets miss grokking entirely.
- Train fraction near minimal: grokking most dramatic near the smallest train % that still generalizes.

## Canonical implementation (openai/grok, verified)
- grok/data.py: ops dict incl. mod-97 polynomials, S5 products; tokens = [EOS, =] + render(NUMS) +
  permutations(range(5)); equations rendered as token sequence; splits by train_pct.
- grok/transformer.py: decoder-only Transformer, default n_layers, n_heads, d_model (paper experiments
  override to 2/4/128), MultiHeadAttention with d_key=d_model/heads, FFN d_ff=multiplier*d_model,
  LayerNorm, causal mask, position encodings, optional weight_noise. AdamW + LambdaLR warmup.
- Loss/accuracy on the answer portion of the equation only.
