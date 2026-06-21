# Context: post-training LLMs for verifiable reasoning with demonstrations and rollouts

## Research question

We have a pretrained language-model policy `π_θ` and want to post-train it to solve
verifiable reasoning tasks — mathematics problems where a rule-based checker can score any
candidate solution as correct (1) or incorrect (0). Two kinds of training data are available
for every prompt `q`: *online* data, the model's own sampled rollouts `τ ~ π_θ(·|q)` scored
by the verifier; and *offline* data, a high-quality demonstration trajectory `τ★` for the
same prompt produced by a stronger teacher or a human. These two data sources are consumed by
two training paradigms — reinforcement learning on the scored rollouts, and supervised
fine-tuning on the demonstrations.

The question is how to train from these two signals inside one post-training pass on a mixed
dataset that pairs each prompt with both a demonstration and on-policy rollouts.

## Background

By this time, reinforcement learning with verifiable rewards (RLVR) is the engine behind the
reasoning gains of frontier models (o1-style systems, DeepSeek-R1, Kimi). The dominant
on-policy recipe samples a group of rollouts per prompt, scores them, and pushes up the
likelihood of the better ones. Supervised fine-tuning (SFT) on teacher demonstrations is the
classical way to distill a skill quickly. The load-bearing concepts of the time:

- **The policy-gradient frame.** Post-training is policy optimization: a trajectory `τ` is a
  sequence of token-actions, `π_θ(τ|q)` is the policy, and a generic gradient ascent step on an
  expected-reward objective is `E[Â · ∇log π_θ]` for some advantage/credit term `Â`. The
  score-function identity `∇_θ E_{τ~π_θ}[f(τ)] = E_{τ~π_θ}[f(τ) ∇_θ log π_θ(τ)]`, with
  `E_{τ~π_θ}[∇log π_θ] = 0`, is the basic tool, together with `∇log π_θ = (1/π_θ)∇π_θ`. A second
  tool is the change-of-measure (importance-reweighting) identity: for any sampling density
  `s(τ)` positive wherever `π_θ` is,
  `E_{τ~π_θ}[f ∇log π_θ] = E_{τ~s}[(π_θ/s) f ∇log π_θ] = E_{τ~s}[(1/s) f ∇π_θ]`, which lets a
  gradient defined as an expectation under one distribution be estimated from samples of another.

- **Generalized Advantage Estimation (Schulman et al. 2015).** GAE established the idea that
  there is a *family* of estimators of the same underlying policy gradient, indexed by knobs
  that trade bias against variance; one chooses an instance from the family rather than a single
  fixed formula.

- **Trust regions and clipping (TRPO; PPO, Schulman et al. 2015, 2017).** Conservative updates
  that keep the new policy close to the rollout policy stabilize RL; PPO's clipped surrogate is
  the standard practical realization, and its piecewise objective acts as a stop-gradient that
  switches off the update on samples that have moved too far. Several methods modify this
  stabilization device (DAPO, CISPO, GSPO, Clip-Cov).

- **Behavior cloning / SFT as next-token NLL.** SFT minimizes
  `−Σ_t log π_θ(τ★_t | q, τ★_{<t})` on demonstrations. The corresponding
  log-likelihood ascent direction is `Σ_t ∇log π_θ = Σ_t (1/π_θ)∇π_θ`; equivalently,
  descent on the NLL follows the negative loss gradient. Demonstration tokens the model assigns
  low probability get pulled up hardest.

Several contemporaneous empirical findings about these systems:

- **"Zero-RL" behavior across model families (SimpleRL-Zoo, Zeng et al. 2025).** Applying RL
  directly to a base model relies on the model sampling reward-bearing rollouts. RL-zero works
  for the Qwen family but largely fails for Llama, while SFT helps both.
- **SFT memorizes, RL generalizes (Chu et al. 2025).** On rule-based textual and visual
  reasoning variants, RL with outcome rewards generalizes out-of-distribution while SFT tends
  to memorize the demonstrations; SFT is commonly applied first because it stabilizes the
  output format that subsequent RL builds on.
- **RLVR and the capability boundary (Yue et al. 2025).** RLVR lifts Pass@1 but not large-`k`
  Pass@k: it sharpens what the base model can already do rather than adding reasoning the model
  does not sample on its own.
- **Order and combination (Fu et al. 2025).** The standard `SFT → RL` pipeline is multi-stage,
  and `SFT → RL` differs markedly from `RL → SFT` on the same model.
- **Group-relative advantage on uniform groups.** When every rollout in a prompt's group
  receives the same reward, the group-relative advantage is identically zero for that prompt.
  Per-question rollout accuracy can be tracked across training to see which prompts the model
  solves over time.

## Baselines

These are the prior methods a new procedure would be measured against.

**Supervised fine-tuning / behavior cloning (Wei et al. 2021; Torabi et al. 2018).** Treat
the demonstration set `D_SFT = {(q, τ★)}` as the target and minimize token-level
cross-entropy `L_SFT(θ) = −Σ_i Σ_t log π_θ(τ★_{i,t} | q_i, τ★_{i,<t})`. It fits the teacher
distribution and raises competence on prompts the model could not solve. The log-likelihood
ascent direction is `Σ_t (1/π_θ)∇π_θ`.

**GRPO (Shao et al. 2024, DeepSeekMath).** A value-function-free on-policy RL method. For each
prompt sample a group of `G` rollouts from the rollout policy `π_{θ_old}`, score them with the
verifier `R(τ) ∈ {0,1}`, and form a group-relative advantage by standardizing within the group:
`Â_{i} = (R(τ_i) − mean_{k}{R(τ_k)}) / std_{k}{R(τ_k)}`. Optimize the clipped surrogate
`L = −(1/G)Σ_i Σ_t min(r_{i,t} Â_i, clip(r_{i,t}, 1−ε, 1+ε) Â_i)` with the per-token ratio
`r_{i,t} = π_θ / π_{θ_old}`, the baseline being the group mean (no critic). It sharpens
reasoning the model can already partly do, and is cheap (no value network). It is purely
on-policy, drawing its signal from what the rollout policy samples; when all `G` rollouts in a
group get the same score the within-group advantage is zero for that prompt.

**LUFFY (Yan et al. 2025).** Augment on-policy RLVR with off-policy reasoning traces from a
stronger policy. Concatenate the off-policy demonstrations with the on-policy rollouts and
compute one group-normalized advantage over the combined set
(`Â = (R − mean_{G_on ∪ G_off})/std`), with `π_ref ≡ 1` for the off-policy traces since the
behavior policy that produced them is unavailable. To stop the model from rigidly imitating the
off-policy traces and collapsing its entropy, it adds *policy shaping* via regularized
importance sampling, amplifying the learning signal on low-probability but crucial tokens.
This lets the model imitate the strong traces when its own rollouts fail and explore when they
succeed, and it trains weak models with on-policy RLVR alongside the off-policy traces. The
balance between imitation and exploration is governed by a preset mixing/shaping mechanism; the
off-policy traces are consumed as an RL signal through an importance ratio with reference policy
`π_ref ≡ 1`, and they are kept in the on-policy advantage group with combined statistics.

**SRFT (Fu et al. 2025).** A single-stage method that combines SFT-style imitation with an
offline-RL objective derived from the GRPO loss by setting `π_ref ≡ 1` and dropping clipping.
It folds offline and online learning into one stage rather than a pipeline, using a fixed recipe
to weight the two terms; the `π_ref ≡ 1` offline-RL term reduces the importance ratio to a
rejection-sampling form.

## Evaluation settings

The natural yardsticks already in use, which would be the testbed:

- **Backbones:** Qwen2.5-Math-1.5B and Qwen2.5-Math-7B (Yang et al. 2024), and Llama-3.1-8B
  (Grattafiori et al. 2024) — spanning families where on-policy RL behaves very differently
  (Qwen amenable to zero-RL, Llama not) and scales where bootstrapping matters more.
- **In-distribution math reasoning benchmarks:** AIME 2024, AIME 2025, AMC (AMC12 2022/2023),
  MATH-500 (Hendrycks et al. 2021), Minerva (Lewkowycz et al. 2022), OlympiadBench
  (He et al. 2024).
- **Out-of-distribution suites** (for the 7B backbone): ARC-c (Clark et al. 2018) and
  GPQA-Diamond (Rein et al. 2024) — to probe whether gains transfer beyond math.
- **Metrics / protocol:** rule-based answer verification giving binary correctness. Pass@1
  under non-zero-temperature sampling (avg@`k` over many samples for the small benchmarks,
  e.g. avg@32 for AIME/AMC), temperature 0.6, top-p 0.95, max generation length 8,192 tokens.
  Pass@k up to large `k` (e.g. 1024, via bootstrap over many generations) as an explicit probe
  of the model's *exploration capacity* / capability boundary.
- **Training:** GRPO-style on-policy optimization; group/rollout size on the order of
  8 responses per prompt at temperature 1.0; constant learning rate ~5e-6 with AdamW; a single
  pass over a mixed dataset that pairs each prompt with both a demonstration and on-policy
  rollouts.
- **Diagnostic instrumentation:** per-question rollout-accuracy grids across training steps
  (to see which prompts the model can/can't solve over time), token-level entropy, average
  response length, and online/offline data composition in mixed-data runs.

## Code framework

A single-pass training loop already exists. For each prompt it can draw a group of on-policy
rollouts, verify them, compute a group-relative RL advantage and an RL surrogate loss, and it
also has access to a demonstration trajectory per prompt against which a supervised
(next-token) loss can be computed. The optimizer (AdamW), the batch construction machinery, and
the GRPO advantage/clip primitives already exist. What is not settled is the missing update
module that assembles the available online and demonstration records into one actor batch and one
actor objective.

```python
import torch


def grpo_group_advantage(rewards):
    """Group-relative advantage over a prompt's on-policy rollouts (already provided).
    rewards: tensor of per-rollout verifier scores in {0,1}. Returns standardized advantages."""
    mean = rewards.mean()
    std = rewards.std()
    if std == 0:
        std = torch.tensor(1.0)            # degenerate group: every rollout scored the same
    return (rewards - mean) / (std + 1e-6)


def rl_surrogate_loss(log_prob, old_log_prob, advantages, eos_mask, clip=0.2):
    """Clipped on-policy surrogate (GRPO/PPO style), already provided."""
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
    return -masked_mean(torch.min(unclipped, clipped), eos_mask)


def sft_loss(log_prob, eos_mask):
    """Token-level negative log-likelihood on a demonstration (already provided)."""
    return masked_mean(-log_prob, eos_mask)


class PostTrainer:
    """Single-pass post-training loop. For each prompt it has both verifier-scored
    model rollouts and an offline demonstration target available."""

    def train_step(self, batch):
        total_loss = 0.0
        for prompt in batch:
            rollouts = self.sample_rollouts(prompt)          # τ_i ~ π_θ(·|q)
            rewards = self.verify(rollouts)                  # R(τ_i) ∈ {0,1}
            demo = prompt.demonstration                      # τ★ (offline target)

            # TODO: the update module we will design.
            actor_batch = self.prepare_actor_batch(prompt, rollouts, rewards, demo)
            loss_q = self.actor_loss(actor_batch)
            total_loss = total_loss + loss_q

        total_loss.backward()
        self.optimizer.step()                                # AdamW, lr ~5e-6
        self.optimizer.zero_grad()

    def prepare_actor_batch(self, prompt, rollouts, rewards, demo):
        # TODO: represent the online and demonstration signals for the actor.
        pass

    def actor_loss(self, actor_batch):
        # TODO: assemble an objective from existing RL and supervised primitives.
        pass
```

The empty update slot is the only missing piece; the supervised loss, the on-policy clipped
loss, and the group-relative advantage primitive are already available.
