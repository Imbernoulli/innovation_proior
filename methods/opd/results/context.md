## Research question

We have a large, expensive teacher model that is very good at multi-step reasoning — say
competition math — and a much smaller student we can actually afford to serve. The goal is to
transfer the teacher's *behavior* into the student so the student solves the same problems, at
a fraction of the inference cost. The catch that makes this hard is specific to
**auto-regressive generation**: a language model predicts one token at a time, each token
conditioned on the tokens it already produced, so the distribution of *partial sequences* the
model sees is determined by the model itself. A training procedure that only ever shows the
student states that *the teacher* visits leaves it untrained on the states *the student* will
actually find itself in at inference. Because each next-token prediction depends on the
previous ones, a single early divergence pushes the student into a region of sequence space it
was never trained on, and the error compounds the rest of the way out — the classic
train-inference mismatch of imitation learning (Pomerleau 1991; Ross & Bagnell 2011). On top
of that the student has *less capacity than the teacher*: it cannot represent every nuance of
the teacher's next-token distribution, so the transfer objective has to decide what the
student should do when it cannot match the teacher exactly. The precise problem is to specify
a per-token transfer signal that (1) is computed on the *student's own* generated
trajectories, so the states match those at inference; (2) is *dense* — informative at every
token, not just at the end of a solution; (3) degrades gracefully under capacity mismatch,
keeping the student coherent rather than smeared across behaviors it cannot execute; (4) needs
no separately trained reward model that could be gamed; and (5) is cheap enough per step to
run inside an ordinary fine-tuning loop. Each existing recipe achieves some of these and gives
up on others.

## Background

By this time the dominant way to make a small model good at a generative task is **knowledge
distillation** (Hinton et al. 2015; Sanh et al. 2019): train the student to imitate the
teacher's token-level output distribution. For auto-regressive models there are two standard
flavors. **Supervised KD** has the teacher assign token-level probabilities over a *fixed*
dataset of sequences and trains the student to match them; **sequence-level KD** (Kim & Rush
2016) generates a fixed set of high-probability sequences from the teacher and runs ordinary
maximum-likelihood fine-tuning on them. Both minimize, in effect, the **forward KL** between
teacher and student, `D_KL(p_T || p_S) = Σ_v p_T(v) log(p_T(v)/p_S(v))`. Forward KL is the
maximum-likelihood objective, and it is **mass-covering**: the `p_T(v)` weight forces the
student to put probability *somewhere* on every token the teacher finds plausible. The two
load-bearing facts about it, both knowable before any new method:

- **Forward KL is mass-covering / zero-avoiding.** Wherever `p_T(v) > 0`, the term
  `p_T(v) log(p_T(v)/p_S(v))` blows up as `p_S(v) → 0`, so the student is pushed to cover the
  full support of the teacher. When the student has far less capacity than the teacher and
  cannot represent all the teacher's modes, covering them all means smearing probability mass
  across regions the student cannot actually execute, including the teacher's long-tail
  low-probability tokens — which at free-run generation produces incoherent or hallucinated
  text (Huszár 2015; Malinin & Gales 2019). KL's asymmetry is therefore not a technicality:
  reversing the arguments changes the optimization pressure from covering support to avoiding
  mass where the teacher assigns little probability, a tradeoff often illustrated by fitting a
  single Gaussian to a Gaussian mixture. Which pressure belongs in a practical distillation loop
  is still tied to the data-collection and optimization choices.

- **Off-policy training causes compounding error.** Whether the fixed sequences come from
  ground truth or from the teacher, they are not drawn from the student's own evolving
  distribution. The student learns in contexts the *teacher* frequents, not the ones *it* will
  visit; an early mistake the teacher would never make drops the student into states absent
  from training, and the auto-regressive dependence carries that divergence forward — known in
  the imitation-learning literature as exposure bias (Ranzato et al. 2015; Bengio et al. 2015)
  and as the distribution-shift failure of behavior cloning that on-policy data collection
  (DAgger; Ross et al. 2011) was designed to fix.

A second background line is **reinforcement learning fine-tuning**. Here the student samples
its own trajectories (on-policy, so the state distribution is correct) and is updated by a
**policy gradient** toward a scalar reward — in reasoning, typically a verifier's pass/fail on
the final answer (REINFORCE; PPO, Schulman et al. 2017). RL fixes the state-mismatch problem
of off-policy KD, but its signal is **sparse**: one scalar per episode regardless of how many
tokens the episode contains. Informally this is `O(1)` bits of feedback per episode, whereas a
per-token target distribution carries `O(N)` bits for an `N`-token sequence — so RL spends a
lot of rollouts on credit assignment, and it needs either a verifiable answer or a *learned*
reward model, which can be exploited (reward hacking). RLHF/RLAIF additionally regularize the
learned policy back toward the initial model with a KL penalty, establishing that a per-token
KL term and a policy-gradient update sit naturally in the same loop.

The conceptual bridge between these lines is **distillation-as-imitation-learning with an
interactive expert**: the teacher is an expert that can label *any* state, including the
student's own, so on-policy imitation (roll out with the student, query the expert on those
states, update) applies directly — combining on-policy data collection with a per-token target
instead of a sparse reward.

## Baselines

These are the prior methods a new transfer recipe would be measured against and would react to.

**Supervised KD / SeqKD (forward KL on a fixed dataset)** — Hinton et al. 2015; Sanh et al.
2019; Kim & Rush 2016. Train the student to match the teacher's per-token distribution on a
fixed set of sequences (ground-truth, or sampled once from the teacher):
`L_SD = E_{(x,y)} [ (1/L_y) Σ_n D_KL(p_T(·|y_<n,x) || p_S(·|y_<n,x)) ]`. SeqKD is the special
case where the divergence is replaced by plain negative log-likelihood on teacher samples. The
training signal is rich (full token-level teacher distribution) and the loop is a stable
supervised one. **Gap:** it is off-policy — the conditioning prefixes `y_<n` come from the
fixed dataset, never from the student — so it leaves the student untrained on the states its
own generations reach, and the auto-regressive dependence turns an early divergence into a
cascading one; and the forward-KL objective is mass-covering, so a low-capacity student is
pushed to spread mass over the teacher's full support and generates incoherently at free run.

**Imitation-style mixed-data KD** — ImitKD (Lin et al. 2020), f-distill (Wen et al. 2023).
These recognize the imitation-learning connection and *mix* student-generated sequences with
the fixed dataset (a partial on-policy fraction), keeping a forward-KL (ImitKD) or
total-variation (f-distill) token-level objective. **Gap:** they only ever go *partway*
on-policy and stay with mass-covering token objectives, so the train-inference mismatch is
reduced but not removed, and the capacity-mismatch smearing of forward KL remains.

**Sequence-level reverse KL via policy gradient** — MiniLLM (Gu et al. 2023). Frame KD as RL:
minimize the *sequence-level* reverse KL `L(θ) = D_KL(p_S || p_T)` with `y ~ p_S`, whose
gradient by the policy-gradient theorem is
`∇L = -E_{y~p_S} Σ_t (R_t - 1) ∇ log p_S(y_t|y_<t,x)`, where the per-step "reward" is
`r_t = log( p_T(y_t|·) / p_S(y_t|·) )` and `R_t = Σ_{t'≥t} r_{t'}` is its accumulation to the
end of the sequence. This is on-policy, dense at the token level, and mode-seeking — exactly
the three properties one wants. **Gap:** the estimator inherits the pathologies of long-horizon
policy gradients. `R_t` sums a future random reward, so the variance is high; small students
discover degenerate sequences (repeated phrases) that the teacher scores highly — reward
hacking; and `R_t` mechanically favors short sequences, biasing the student toward empty
responses. To use it at all MiniLLM bolts on a battery of stabilizers — a single-step
decomposition, teacher-mixed sampling, length normalization, importance weights, and PPO-style
clipping — which add hyperparameters and move the procedure away from a simple supervised loop.

**Generalized KD (GKD)** — Agarwal et al. 2023 (ICLR 2024). Unify the above along two axes: a
choice of divergence and a *student-data fraction* `λ`. The objective is
`L_GKD = (1-λ) E_{(x,y)~data}[ D(p_T||p_S)(y|x) ] + λ E_{x, y~p_S}[ D(p_T||p_S)(y|x) ]`, with no
backpropagation through the sampling. For the divergence it uses the **generalized
Jensen–Shannon** family `D_JSD(β)(P||Q) = β·D_KL(P || M) + (1-β)·D_KL(Q || M)`,
`M = βP + (1-β)Q`, which interpolates forward KL (`β → 0`, since
`lim_{β→0} D_JSD(β)/β = D_KL(P||Q)`, Huszár 2015) and reverse KL (`β → 1`). Crucially, because
the teacher's *log-probabilities* are available, GKD computes the per-token divergence
**analytically by summing over the vocabulary** rather than by Monte-Carlo policy gradient,
which is lower-variance and "closer to supervised training" — and so it needs none of MiniLLM's
stabilizers. GKD reports that the best divergence is **task-dependent**, and that pushing the
student-data fraction up (more on-policy) consistently helps once it is a substantial fraction.
**Gap:** GKD is presented and tuned around classic encoder-decoder tasks (summarization,
translation, grade-school arithmetic with small T5 students) and treats the divergence and `λ`
as open knobs to be swept per task. It leaves unsettled which point in that design space should
be chosen when the teacher is a strong reasoner and the student is a small base model.

## Evaluation settings

The natural yardsticks for transferring math reasoning into a small student, all pre-existing:

- **GSM8K** (Cobbe et al. 2021) — 1319 grade-school multi-step word problems; exact-answer
  accuracy, final answer extracted and checked, greedy decoding (temperature 0, n=1).
- **MATH-500** — a 500-problem subset of competition-style problems; exact-answer accuracy,
  greedy decoding.
- **AMC23** — the 40 problems of the 2023 American Mathematics Competition; a small set, so
  scored as **avg@8** (8 independent samples per problem at temperature 0.6, top-p 0.95,
  averaged) to reduce variance, the small-eval protocol used for AIME/AMC-style sets.
- Answers are extracted from `\boxed{...}` and graded with a symbolic checker (`math-verify`),
  so credit reflects mathematical correctness, not surface match.
- **Models and data**: a small base student and a math-tuned larger teacher; training prompts a
  subset of an open math-reasoning corpus (problem statements with reference solutions). The
  student is a *base* model, so a minimal `Question: … Answer:` template is used.
- **Training harness**: a generic student-teacher trainer can either reuse dataset completions
  or sample completions from the student, controlled by a student-data fraction `λ`. For any
  prompt-completion sequence it runs the student and the frozen teacher over the same tokens,
  marks prompt/padding positions with `-100`, and calls one editable loss function on the two
  logit tensors before the optimizer step. The harness exposes decoding temperature, `λ`, and a
  divergence interpolation knob; bf16, gradient checkpointing, and gradient accumulation are
  standard systems settings.

## Code framework

The transfer signal plugs into a generic student-teacher trainer that already exists. The
trainer owns everything except the loss: it supplies prompt-completion tokens from the data
source chosen by its mixing rule, runs a forward pass of *both* the student and the frozen
teacher over the same tokens to produce two logit tensors of shape `[B, T, V]`, marks the
completion positions in `labels` (with `-100` on prompt/padding), and calls one function to turn
the two logit tensors into a scalar loss before backpropagating and stepping the optimizer.
What is *not* settled is that function: the generic per-token teacher-student divergence to
optimize. That is the single empty slot.

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(
    student_logits: torch.Tensor,    # [B, T, V] student logits over the tokens
    teacher_logits: torch.Tensor,    # [B, T, V] frozen-teacher logits over the same tokens
    labels: torch.Tensor = None,     # [B, T]; -100 on prompt/padding, token id on completion
    beta: float = 0.5,               # optional divergence knob, if the loss uses one
    temperature: float = 1.0,        # softmax temperature applied to logits
    reduction: str = "batchmean",    # "batchmean" / "sum" / "mean" / "none"
    step: int = 0,                   # current training step (for curriculum losses)
    total_steps: int = 0,            # total planned steps
    lmbda: float = 0.5,              # data-source mixing fraction the trainer applies upstream
) -> torch.Tensor:
    # The trainer has already produced student_logits and teacher_logits over the same
    # (prompt + completion) tokens; labels marks which positions are completion tokens.
    # TODO: the transfer loss we will define here. Turn the two logit tensors into a
    #       signal over the completion positions, mask out labels == -100, and reduce
    #       to a scalar.
    pass


# existing student-teacher training loop the loss plugs into (read-only)
def train(student, teacher, prompts, optimizer, sample_completion, total_steps,
          temperature=1.0, lmbda=0.5):
    for step in range(total_steps):
        batch = next(prompts)
        # trainer chooses a prompt-completion source using lmbda, then runs both models
        # over the same prompt+completion tokens
        input_ids, labels = sample_completion(student, batch, lmbda=lmbda)
        student_logits = student(input_ids).logits            # [B, T, V]
        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits        # [B, T, V], teacher frozen
        loss = compute_distill_loss(                           # the one empty slot
            student_logits, teacher_logits, labels,
            temperature=temperature, reduction="batchmean",
            step=step, total_steps=total_steps, lmbda=lmbda,
        )
        optimizer.zero_grad()
        loss.backward()                                       # grad flows only through student_logits
        optimizer.step()
```

The trainer supplies the two logit tensors and the completion mask each step;
`compute_distill_loss` is where the per-token transfer objective will live.
