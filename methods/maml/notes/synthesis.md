# MAML — synthesis notes (Phase 1.5)

## Pain point
Few-shot learning: adapt to a new task from K examples (K=1,5) without overfitting. Deep nets need lots of data. Want a mechanism that is general across task forms (classification, regression, RL) and model architectures.

## Goal / first-principles object
Learn an INITIALIZATION θ such that, after a few gradient steps of the ordinary learner on a new task's small support set, the model generalizes well on that task's query set. Meta-train across a distribution of tasks p(T).

## Load-bearing ancestors (and gaps)
- **Pretraining + fine-tuning** (Donahue/DeCAF 2014): pretrain on big data, fine-tune. Gap: features are optimized for the source objective, NOT for being a good *starting point for fast adaptation*. No reason a few steps suffice; can need lots of data + steps. "Pretrain on all tasks" averages contradictory targets → in regression collapses to the mean output. In RL, pretraining can be worse than random init (Parisotto actor-mimic).
- **Metric-based meta-learning**: Siamese nets (Koch 2015), Matching Networks (Vinyals 2016), Prototypical Networks (Snell 2017). Learn an embedding + nonparametric comparison (nearest neighbor / attention-weighted). Gap: tied to classification; no obvious extension to RL/regression; adaptation is "compare in embedding space," not a general learning procedure.
- **Optimization-as-a-model / learned optimizers**: Schmidhuber 1987 (self-referential), Bengio&Bengio 1990/92 (learn the update rule), Hochreiter 2001 (LSTM learns to learn), Andrychowicz 2016 (learning to learn by gradient descent by gradient descent), Ravi&Larochelle 2017 (LSTM meta-learner that ALSO learns the initialization). Gap: the learner/optimizer is a separate parametric model (RNN) — extra parameters, architectural constraints, harder to scale, and you must learn the optimizer itself.
- **Memory-augmented / recurrent meta-learners**: MANN (Santoro 2016), RL^2 (Duan 2016), Wang 2016. Ingest the dataset into a recurrent net. Gap: require a recurrent architecture; black-box adaptation; can't simply keep fine-tuning at test time.
- **Context-vector adaptation** (Rei 2015): adapt only a learned input-concatenated context vector by gradient. Gap: less flexible than adapting all parameters; sub-par on hard tasks.

## The idea
Don't learn an optimizer or a metric. Use the SAME ordinary gradient-descent learner at test time. Just choose where it STARTS so that a step or two suffices. Treat the post-adaptation test loss as the training signal for θ. Bilevel.

## Key derivation (the heart)
Inner (adaptation), one step:
  θ'_i = θ − α ∇_θ L_{T_i}(f_θ)    [computed on support set]
Meta-objective (query sets):
  min_θ Σ_i L_{T_i}(f_{θ'_i})
Meta-update:
  θ ← θ − β ∇_θ Σ_i L_{T_i}(f_{θ'_i})

Meta-gradient for one task, chain rule (θ' is a function of θ):
  ∇_θ L_{T_i}(f_{θ'_i}) = (dθ'_i/dθ)^T ∇_{θ'} L_{T_i}(f_{θ'_i})
where θ'_i = θ − α ∇_θ L^{tr}_{T_i}(f_θ), so the Jacobian
  dθ'_i/dθ = I − α ∇²_θ L^{tr}_{T_i}(f_θ)   (Hessian of the inner/support loss)
Hence
  ∇_θ L_{T_i}(f_{θ'_i}) = (I − α ∇²_θ L^{tr}_{T_i}(f_θ)) ∇_{θ'} L^{ts}_{T_i}(f_{θ'_i}).
This is a Hessian-vector product (no need to materialize the Hessian). Note support loss in the Hessian, query loss in the post-update gradient.

Multi-step: product of (I − α H_k) Jacobians over the inner steps — backprop through the unrolled inner loop.

FOMAML: drop the second-order term, treat dθ'/dθ ≈ I:
  ∇_θ L ≈ ∇_{θ'} L^{ts}_{T_i}(f_{θ'_i})  — just evaluate the query gradient AT θ'_i and apply it to θ.
Works nearly as well empirically (47.70→48.07 on MiniImagenet 1-shot in their numbers; ~33% compute saving). Why: ReLU nets are locally near-linear (Goodfellow 2015 "linear" cite) → ∇²L ≈ 0 → (I − αH) ≈ I. Crucial: still evaluate the gradient at the POST-update params, not at θ.

## Model-agnostic
Nothing about the derivation assumes the model form — only that f_θ is differentiable in θ and trained by GD. So same algorithm for MSE regression, cross-entropy classification, and policy-gradient RL (inner and meta gradients both via REINFORCE; meta-optimizer TRPO; finite-diff HVP to avoid third derivatives).

## Train vs test split (important detail)
Support set (D / "a") used for the inner adaptation gradient; query set (D' / "b") used for the meta-loss. Using different samples for the two is what forces the initialization to GENERALIZE rather than memorize, and matches test-time conditions.

## Code structure (cbfinn/maml, TF1)
- weights = dict of tf.Variables (manual, so we can functionally apply them).
- forward(inp, weights): functional forward using a given weight dict.
- inner: grads = tf.gradients(loss_a, weights); fast_weights = weights - update_lr*grads.
- repeat for num_updates-1 more steps on support.
- meta loss = loss_func(forward(inputb, fast_weights), labelb).
- FLAGS.stop_grad → tf.stop_gradient(grad) on inner grads ⇒ FOMAML.
- meta optimizer: Adam.compute_gradients(total_losses2[-1]) then apply; miniimagenet clips grads to [-10,10].
- support = "a"/inputa, query = "b"/inputb.
Modern equivalent: PyTorch with functional_call / higher / learn2learn — clone params, take create_graph=True inner step (second order) or detach (first order). I'll write a clean PyTorch functional version for answer.md/reasoning, faithful to this logic.

## Design choices → why
- One inner step (often): cheap, and matches the "make a step or two suffice" goal; multi-step is a straightforward unroll.
- Fixed α small (0.01 sine, 0.1 RL, 0.4 omniglot): hyperparam; can be meta-learned but fixed works.
- Separate support/query samples: prevents memorization, enforces generalization (the meta-signal IS held-out error).
- Adam as meta-optimizer (supervised), TRPO (RL): stabilize the outer loop; finite-diff HVP in RL to avoid 3rd derivatives.
- Grad clip on miniimagenet: deeper conv net → meta-grads can blow up.
- FOMAML: speed; justified by local linearity of ReLU nets.
- No extra params: the whole point vs LSTM meta-learner / metric nets.
