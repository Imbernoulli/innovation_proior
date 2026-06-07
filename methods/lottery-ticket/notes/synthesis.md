# Lottery Ticket Hypothesis synthesis

arXiv 1803.03635 (verified). Frankle & Carbin, MIT. ICLR 2019 (best paper).

## Empirical-first framing
Motivating finding (Figure 1, in-scope context): take a trained-then-pruned-style SPARSE architecture
and train it FROM SCRATCH with a fresh random init — i.e. randomly sample a sparse subnetwork at a
given sparsity and train it. Result: the sparser the network, the SLOWER it learns and the LOWER its
final test accuracy. (Random sampling models the unstructured pruning of LeCun et al / Han et al.)
Contemporary experience (quoted from Han, Li et al.): "training a pruned model from scratch performs
worse than retraining a pruned model"; "better to retain the weights from initial training for
surviving connections than re-initialize." So conventional wisdom: pruned architectures are hard to
train from the start; you need the big network first.

## The pain / research question
Pruning (LeCun "Optimal Brain Damage" 1990, Hassibi "Optimal Brain Surgeon", Han et al. 2015
magnitude pruning) removes >90% of weights post-training without hurting accuracy → smaller/cheaper
inference. Question: if a 10%-size net can REPRESENT the function, why not just TRAIN that small net
and save training cost too? Answer from the field: you can't — the pruned architecture trained from
scratch underperforms. The puzzle: a small trainable net provably exists (we just pruned to it), yet
SGD from a fresh random init can't reach it.

## The central experiment (the discovery)
The key realization: maybe it's not the ARCHITECTURE alone that fails — maybe the INITIALIZATION
matters. When you prune-then-finetune, the surviving weights keep values that were set by the
original training. What if you keep the surviving connections AND their ORIGINAL random init values?

Central experiment:
1. Randomly initialize f(x; θ₀), θ₀ ~ D_θ.
2. Train for j iterations → θ_j.
3. Prune p% of θ_j (smallest-magnitude weights) → mask m ∈ {0,1}^{|θ|}.
4. RESET surviving weights to their values in θ₀ → the "winning ticket" f(x; m ⊙ θ₀).
The unique step vs all prior pruning: reset to the ORIGINAL init θ₀ (not fine-tune, not re-init
random). Then train f(x; m⊙θ₀) in isolation (mask fixed).

## The Lottery Ticket Hypothesis (formal)
A randomly-initialized dense net contains a subnetwork (a "winning ticket") that, initialized at its
original init and trained in isolation, matches the test accuracy of the full net in ≤ the same #
iterations. Formally: f(x;θ₀); SGD reaches min val loss l at iter j, test acc a. Training f(x; m⊙θ₀)
reaches min val loss l' at iter j', test acc a'. LTH: ∃ m with
  j' ≤ j (commensurate training time), a' ≥ a (commensurate accuracy), ‖m‖₀ ≪ |θ| (fewer params).
"Won the initialization lottery" = combination of weights+connections capable of learning.

## Iterative magnitude pruning (the procedure that works)
One-shot: train once, prune p%, reset. Works but coarse.
ITERATIVE: repeat over n rounds; each round prune p^{1/n}% of the weights that SURVIVED the previous
round, train, reset surviving to θ₀, repeat. Finds winning tickets at SMALLER sizes than one-shot.
(Note p^{1/n} so after n rounds total surviving ≈ (1−p^{1/n})^n... actually paper says each round
prunes p^{1/n}% — e.g. 20% per round; after k rounds fraction remaining = 0.8^k.)
Layer-wise heuristic (Han et al. style): within each layer remove the lowest-magnitude fraction.
Connections to OUTPUT layer pruned at HALF the rate of the rest.
Notation: P_m = ‖m‖₀/|θ| = fraction of weights REMAINING (sparsity of mask). P_m=25% ⇒ 75% pruned.

## Controls (what isolates the claim)
- Random REINITIALIZATION: keep mask m (structure) but draw fresh θ'₀ ~ D_θ. Winning tickets then
  perform far worse / learn slower → structure alone insufficient; the SPECIFIC init matters.
- This is the control that turns "we found a small trainable net" into "the init is the active
  ingredient."

## Deeper-net subtlety: warmup
On Resnet-18, VGG-19 (CIFAR10), iterative pruning FAILS to find winning tickets at the standard
(higher) learning rates UNLESS trained with learning-rate WARMUP. At high LR the reset-to-init ticket
isn't found; warmup (ramp LR up from 0) is needed. (Limitation noted; why is open.)

## Architectures (Figure / Table)
- Lenet-300-100 (MNIST): FC 300,100,10. Adam 1.2e-3, 50K iters, batch 60. Prune fc 20%/round.
- Conv-2/4/6 (CIFAR10, VGG-style): 2/4/6 conv layers (3x3), maxpool every 2 conv, then FC 256,256,10.
  Adam 2e-4/3e-4/3e-4. Prune conv 10-15%, fc 20%.
- Resnet-18, VGG-19 (CIFAR10): SGD momentum 0.9, LR 0.1→0.01→0.001. Prune conv 20%, fc 0%. Need warmup.
- All inits: Gaussian Glorot (Xavier).

## Why the init matters (discussion, in-frame design rationale)
- Not because init weights are already near final: Appendix shows winning-ticket weights MOVE FARTHER
  than other weights during training. So it's not "already trained."
- Conjecture: the init lands in a region of the loss landscape amenable to the optimizer/dataset/model.
- Structure encodes an inductive bias customized to the task (found via heavy use of training data).
- Generalization: test acc rises then falls as pruned ("Occam's Hill") — overparam net overfits, too
  pruned has too little capacity; winning tickets generalize better (smaller train-test gap).
- Lottery Ticket Conjecture: SGD seeks out and trains a well-initialized subnetwork; dense nets train
  easier because they contain MORE candidate subnetworks (more lottery tickets) → higher chance one wins.

## Canonical implementation (google-research/lottery-ticket-hypothesis, verified)
prune_by_percent_once:
  sorted_weights = np.sort(np.abs(final_weight[mask == 1]))   # surviving weights by |magnitude|
  cutoff_index = np.round(percent * sorted_weights.size).astype(int)
  cutoff = sorted_weights[cutoff_index]
  return np.where(np.abs(final_weight) <= cutoff, 0, mask)    # prune below cutoff
Experiment loop: experiment() repeats {train_model → prune_masks(prune_by_percent, PRUNE_PERCENTS) →
reset to presets(θ₀)} for `iterations` rounds. Lenet OPTIMIZER Adam, masks applied as m⊙θ.
Mask applied by multiplying weights by m every forward pass (model_fc applies masks to weights).

## Design-decision → why
- Reset to θ₀ (not re-init, not fine-tune): the central hypothesis is that the ORIGINAL init is the
  active ingredient; resetting tests exactly that. Fine-tuning would conflate; re-init is the control.
- Magnitude pruning: cheap, proven (Han et al.); small-magnitude weights contribute least.
- Layer-wise (vs global): prune a fixed fraction within each layer so no layer is wiped out.
- Outputs pruned at half rate: output layer is small/sensitive; aggressive pruning there hurts.
- Iterative over one-shot: gradually removing weights and re-finding the ticket each round finds
  smaller tickets; one big cut removes weights that would've mattered.
- p^{1/n} per round: geometric schedule so n rounds compose to target sparsity p.
- Unstructured (per-weight) pruning: finds sparser tickets than structured; (downside: not
  hardware-friendly — a limitation).
- Warmup on deep nets: high LR early destroys the reset-to-init ticket's trainability; warmup rescues.
