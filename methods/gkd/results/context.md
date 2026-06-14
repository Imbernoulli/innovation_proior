# Context: compressing autoregressive language models by distillation (circa 2022-2023)

## Research question

Large autoregressive language models keep getting better mainly by scaling parameters and
data, but a model with billions of parameters is expensive to serve: every generated token is
a full forward pass, and the memory footprint alone can rule out deployment. The practical need
is to take a large, capable **teacher** model and produce a much smaller **student** that
retains as much of the teacher's task ability as possible, so that inference is cheap while
quality stays high. The teacher is available to query — we can read its full next-token
probability distribution at any context, not just sample from it.

The precise difficulty is specific to *autoregressive generation*. A language model defines a
distribution over sequences by a product of next-token conditionals,
`p(y|x) = Π_n p(y_n | y_{<n}, x)`, and at inference it generates one token at a time, each
conditioned on the tokens it has already produced. So at deployment the student walks through a
sequence of partial-sequence contexts *that it generated itself*. Any training procedure that
only ever shows the student "good" contexts — contexts drawn from a fixed corpus or from the
teacher — leaves behavior on other reachable contexts weakly constrained. At the same time, the
student is not merely data-limited; it is capacity-limited, so an objective that is harmless when
the student can match the teacher exactly can behave badly when the student cannot represent all
of the teacher's mass. The open problem is to compress the teacher without turning this sequential
prediction problem into an unstable, high-variance reinforcement-learning problem.

## Background

By this time, knowledge distillation is the standard tool for compressing neural networks. The
original idea (Buciluǎ et al. 2006; Hinton, Vinyals & Dean 2015) is that a trained teacher's
output distribution carries more information than the hard label: the relative probabilities it
assigns to the *wrong* classes encode a similarity structure ("this 2 looks a bit like a 7")
that a hard one-hot target throws away. Hinton et al. formalize this with a softmax temperature
`T`, `q_i = exp(z_i/T) / Σ_j exp(z_j/T)`, training the student to match the teacher's softened
distribution; in the high-temperature limit the gradient `∂C/∂z_i ≈ (z_i − v_i)/(N T²)` reduces
distillation to matching logits, and the per-example gradients are scaled by `T²` so the soft-
and hard-target contributions stay balanced. The soft targets give a rich, low-variance signal,
so a student can be trained on much less data and at a higher learning rate than from scratch.

Two facts about this landscape are load-bearing.

First, **the divergence direction matters when the student is too small to match the teacher.**
The default distillation objective is the forward KL from teacher to student,
`KL(P‖Q) = Σ_c P(c) log(P(c)/Q(c))` with `P` the teacher and `Q` the student. KL is not
symmetric, and the two directions behave very differently under capacity mismatch. This is
well studied outside distillation: maximum likelihood is exactly forward KL and is
*mass-covering* — wherever the teacher puts probability, the term blows up unless the student
also puts probability there, so an underpowered student spreads its mass to cover every mode,
including low-probability tokens. The reverse KL `KL(Q‖P)` is *zero-forcing / mode-seeking* —
the student is penalized for putting mass where the teacher has none, so it concentrates on a
single high-probability region. Huszár (2015) makes the connection precise for generative
sequence models and introduces a **generalized Jensen-Shannon divergence**, a one-parameter
family interpolating between the two directions that is, moreover, *bounded* even when the two
distributions have disjoint support (plain KL can be infinite). He shows the small-coefficient
limit of this family recovers forward KL, so the family genuinely contains both behaviors and
everything between.

Second, **autoregressive generation has a well-documented failure mode under fixed-data
training**, called exposure bias (Ranzato et al. 2015; Bengio et al. 2015). At training time
the model only ever conditions on prefixes drawn from real (or teacher) sequences; at test time
it conditions on its own prefixes. Once it emits one unlikely token, it lands in a context it
never saw in training, tends not to recover, and the error cascades through the rest of the
generation. The same phenomenon is the central object of *imitation learning*: a policy trained
only on an expert's state distribution incurs errors that compound over the horizon. The reduction
analysis (Ross & Bagnell 2010; Ross, Gordon & Bagnell 2011) makes this quantitative — if a
behaviorally-cloned policy makes mistakes with probability `ε` under the expert's state
distribution, its expected cost over a horizon `T` can grow like `T²ε`, because the first
mistake moves it off the expert's distribution and every subsequent step is then out of
distribution. That quadratic, rather than linear, dependence on the horizon is precisely the
cascading-error picture, now with a bound attached to it.

## Baselines

These are the prior methods a new distillation approach would be measured against and would
react to.

**Supervised KD (Hinton et al. 2015; Sanh et al. 2019, DistilBERT).** Train the student to match
the teacher's token-level distribution on a *fixed* dataset of sequences — either ground-truth
outputs labeled by the teacher, or any corpus the teacher can score. The per-token objective is
the forward KL (equivalently cross-entropy against the soft teacher distribution),
`L_SD(θ) = E_{(x,y)~(X,Y)}[ KL(p_T ‖ p_S^θ)(y|x) ]`, averaged over the tokens of each sequence.
This is the workhorse and gives a rich signal from the full teacher distribution. **Gap:** the
dataset is fixed and off-policy, so the prefixes the student trains on are not the prefixes it
generates at inference — the train-inference mismatch above is left untouched. And the forward
KL is mode-covering, so a small student is pushed to put mass on teacher-unlikely tokens, which
shows up as degraded, sometimes hallucinated, generations.

**Sequence-Level KD (Kim & Rush 2016).** The teacher's distribution over whole sequences is an
exponentially large object; SeqKD approximates it by its *mode*, found with beam search, and
trains the student by plain negative log-likelihood on the teacher's beam outputs `ŷ`:
`L_SeqKD ≈ −log p_S(ŷ|x)`. This is behavioral cloning on teacher-generated data and is simple
and effective for translation. **Gap:** the training set is still fixed and generated *by the
teacher*, not by the student, so it does not match the student's own inference distribution; it
collapses the teacher to a single mode, discarding the token-level distribution the teacher
could otherwise provide; and as behavioral cloning it inherits the `T²ε` compounding-error
bound.

**DAgger (Ross, Gordon & Bagnell 2011).** The general fix for compounding errors in sequential
prediction. Forward training trains a separate policy `π_t` for each step `t` on the state
distribution induced by `π_1, …, π_{t-1}`, replacing the quadratic dependence with a bound
linear in `T` when the cost-to-go penalty is controlled — but it needs `T` distinct policies,
impractical when the horizon is large or undefined. DAgger instead keeps one
policy and fixes the *data*: at iteration `i` it rolls out a mixture
`π_i = β_i π* + (1−β_i) π̂_i` of the expert and the current learner, collects the visited
states, has the expert label them, **aggregates** them into a growing dataset, and retrains.
It schedules `β_i → 0` (e.g. `β_i = p^{i-1}`, or `β_i = I(i=1)` so only the first iteration uses
the expert), and reduces to no-regret online learning, giving
`J(π̂) ≤ J(π*) + O(uTε_N) + O(1)`. **Gap:** it is generic imitation learning with a 0-1 / oracle-action loss, stated
for control problems, not autoregressive language models; it learns from the expert's *chosen
actions*, not from a full probability distribution, even when (as with a queryable teacher) the
whole distribution is available; and aggregating an ever-growing dataset is awkward at the scale
of LM training.

**ImitKD (Lin et al. 2020).** Brings the imitation-learning view to autoregressive distillation:
the student is the learner, the teacher is the interactive oracle. It samples training prefixes
from a mixture of a fixed dataset and the student's own generations (DAgger-style, but with data
*replacement* per batch rather than aggregation), under an exponential schedule `β_i = r^{i/I}`,
and applies a token-level forward-KL / oracle-token loss. **Gap:** it keeps the forward KL at
the token level and largely uses the oracle's top action, so much of the teacher's available
log-probability information is unused; its schedule keeps fixed-data training in the loop, leaving
open what happens when the training distribution is allowed to track the learner more aggressively;
it is demonstrated only at modest scale; and its setup is separate from the reward-optimized
fine-tuning pipelines used for large language models.

**Sequence-level reverse-KL via policy gradient (MiniLLM, Gu et al. 2023, concurrent).** Frames
distillation as RL with the sequence-level reverse KL between teacher and student as the reward,
optimized by policy gradient — directly backpropagating through the student's sampling. **Gap:**
backpropagating through sampling is high-variance and needs a battery of stabilizing tricks
(variance reduction, reward-hacking mitigation, length-bias correction); it is restricted to
reverse KL and is far from the simplicity of supervised training.

## Evaluation settings

The natural yardsticks already in use for autoregressive distillation, defined before any new
method:

- **Abstractive summarization** on XSum (news article → one-sentence summary), scored by ROUGE-2
  of predicted summaries against references on the validation split; diversity of generations
  quantified by Self-BLEU across samples (100 = deterministic, 0 = maximally diverse).
- **Machine translation** on WMT14 English→German, scored by BLEU on the validation split with
  beam-search decoding.
- **Arithmetic reasoning** on GSM8K grade-school word problems, with few-shot chain-of-thought
  prompting (the first four CoT exemplars), scored by exact-match accuracy of the final answer
  (verified with an external calculator). For a reasoning-focused setup the analogous yardsticks
  are competition-math benchmarks (e.g. MATH-style problem sets and small contest sets), graded
  by `\boxed{}`-extracted final answers, with greedy decoding for the large sets and an
  averaged multi-sample protocol (avg@k at a modest temperature) for the small ones.
- Protocol: students of several sizes (e.g. ~77M / 250M / 800M) distilled from a single larger
  teacher (~3B); the student is first supervised-fine-tuned so it can already generate adequate
  sequences; evaluation by greedy sampling or by temperature sampling, with configurations
  selected per task on validation data.

## Code framework

The standard distillation loop already has the required mechanics. It draws a batch of inputs
(and optionally reference outputs), runs the student and the frozen teacher forward over the same
tokens to get two logit tensors `[B, T, V]`, and feeds them, with a token mask, to a scalar loss
whose output is backpropagated into the student. The scaffold below leaves the data assembly and
the distributional discrepancy as generic empty slots.

```python
import torch
import torch.nn.functional as F


def prepare_batch(student, teacher, batch, gen_config):
    """Assemble the (prompt + completion) token ids the batch will train on, and the
    label mask (-100 on prompt/padding)."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]                 # -100 on prompt & padding
    # TODO: choose the completion tokens for this training batch.
    return input_ids, attention_mask, labels


def distill_loss(student_logits, teacher_logits, labels=None,
                 temperature=1.0, reduction="batchmean"):
    """Map the two distributions over the completion tokens to a scalar.
    student_logits, teacher_logits: [B, T, V] over the same tokens.
    labels: [B, T], -100 on positions to ignore."""
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    # TODO: define the token-level discrepancy.
    per_token = ...                           # [B, T]

    if labels is not None:
        mask = labels != -100
        per_token = per_token[mask]

    if reduction == "batchmean":
        if labels is not None:
            return per_token.sum() / mask.sum()
        return per_token.sum() / per_token.size(0)
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token


# existing distillation training loop the loss plugs into
def train(student, teacher, data_loader, optimizer, gen_config):
    teacher.eval()
    for batch in data_loader:
        input_ids, attention_mask, labels = prepare_batch(student, teacher, batch, gen_config)
        student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        # align logits to the completion tokens (shift by one), then:
        loss = distill_loss(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

The two `# TODO`s are the only moving parts the new distillation rule has to specify.
