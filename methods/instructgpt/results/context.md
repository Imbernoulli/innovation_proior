# Context: aligning a pretrained language model with what users actually want

## Research question

We have large pretrained language models that are astonishingly capable at continuing text, yet routinely do the wrong thing when a person actually tries to *use* them: they fabricate facts, ignore explicit constraints in the instruction, emit toxic or biased text, or simply answer a different question than the one asked. The root cause is structural, not a bug to be patched. The model was trained to maximize the likelihood of the next token on a random slice of the internet. That objective is a *proxy* for what we want, and it is the wrong proxy: "predict what some webpage would say next" is not "do what this user is asking, helpfully and safely." The two objectives diverge, and the divergence does not shrink as the model gets bigger — a far larger model continues to misbehave in the same ways. Capability and alignment-with-intent are decoupled.

The problem is: given (a) a pretrained language model, (b) a distribution of real instructions/prompts that users send, and (c) the ability to pay a team of humans to provide feedback, produce a model whose *default* behavior is to follow the user's intent — explicit intent (do the task) and implicit intent (be truthful, not toxic). Two hard constraints make this nontrivial. First, "follow intent helpfully" cannot be written down as a loss function; it lives only in human judgment, which is expensive and slow. Second, whatever we do must not destroy the broad capabilities the model acquired during pretraining — a fix that makes the model obedient but stupid on standard tasks (an "alignment tax") is a poor trade and creates a bad incentive to deploy the unaligned-but-capable model instead.

## Background

**Pretrained language models and their misalignment.** The base models here (Brown et al. 2020, GPT-3) are decoder-only transformers trained by next-token prediction on internet-scale text. They exhibit in-context learning: prefix the input with a few examples or an instruction and they will often perform the task zero- or few-shot. But this is brittle — the user has to discover a prompt that works — and the underlying objective never rewarded "being helpful," so unhelpful, untruthful, and toxic continuations are all perfectly likely under the model. The observation that scaling parameters does not by itself produce intent-following behavior is the motivating fact for everything that follows: if scale alone fixed it, there would be nothing to do.

**Learning a reward from human feedback, instead of hand-specifying it.** For tasks where the objective is easy to *judge* but hard to *write down*, a line of work learns the reward function from human comparisons. Christiano et al. (2017), "Deep reinforcement learning from human preferences," established the template on robotics and Atari: repeatedly show a human two short behavior clips, ask which is better, fit a reward model to those comparisons, and optimize the agent against the learned reward with RL — interleaving so the reward model is queried for new comparisons on the agent's current behavior. The key insight they established is that humans give *more reliable* signal as relative comparisons ("A is better than B") than as absolute scores, and that a learned reward model lets a handful of expensive human judgments be amortized over millions of cheap RL queries.

**The Bradley-Terry comparison model.** The mathematical object underneath "fit a reward to comparisons" is the Bradley-Terry model (Bradley & Terry 1952) for paired comparisons. Each item $i$ carries a latent positive strength $s_i$, and the probability it beats item $j$ is $P(i \succ j) = s_i/(s_i + s_j)$. Writing $s_i = \exp(r_i)$ turns this into $P(i \succ j) = \sigma(r_i - r_j)$ where $\sigma$ is the logistic sigmoid: the *difference* of latent scores is the log-odds of preference. Fitting it by maximum likelihood is identical to logistic regression, and the absolute level of the scores is unidentifiable — only differences carry meaning.

**Reinforcement learning machinery.** Optimizing a policy against a scalar reward when you can only sample uses policy-gradient methods. Vanilla REINFORCE — $\nabla_\theta J = \mathbb{E}[R \, \nabla_\theta \log \pi_\theta]$ — is unbiased but high-variance, and an unconstrained gradient step can move the policy so far that it collapses. Two tools address this. Generalized Advantage Estimation (Schulman et al. 2016) replaces the raw return with a variance-reduced advantage $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, trading a little bias for much less variance via a learned value baseline. Proximal Policy Optimization (Schulman et al. 2017) constrains the update with a clipped surrogate: with importance ratio $\rho_t = \pi_\theta/\pi_{\text{old}}$, it maximizes $\min(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t)$, which removes the incentive to move the policy far in any single update — a cheap trust region, far simpler to run than second-order constrained methods.

**Porting reward-modeling to language, and the over-optimization problem.** Ziegler et al. (2019), "Fine-tuning language models from human preferences," carried the comparison-reward-model + RL approach to language models on stylistic continuation and summarization, and flagged that running RL against a learned reward on a language model is unstable unless the policy is held near a trusted starting distribution. Stiennon et al. (2020), "Learning to summarize from human feedback," carried this further on the summarization task and studied *reward over-optimization* directly: because the reward model is only an approximation of human judgment, accurate near the data it was trained on, a policy given free rein will drift into regions where the model's score is high but real human preference is low (reward hacking). They observed empirically that this decoupling worsens the harder the policy is pushed away from the data the reward model was fit on. These two papers leave a clear next problem: move from a single narrow task to the broad, open-ended distribution of real user instructions while preserving the capabilities learned during pretraining.

**Diagnostic finding on fine-tuning and capability loss.** A recurring, knowable-in-advance phenomenon when you fine-tune a pretrained model hard on a narrow objective is regression on tasks outside that objective — the model forgets or degrades general capabilities. When a policy is pushed to maximize a learned preference reward, it measurably loses ground on standard public NLP benchmarks (reading comprehension, QA, translation) relative to the pretrained model. This is the alignment-tax phenomenon, a fact about what hard fine-tuning does, and any serious alignment method has to confront it.

## Baselines

**Supervised fine-tuning on demonstrations (SFT / behavior cloning).** Collect human-written ideal responses to prompts and fine-tune the pretrained model to maximize their log-likelihood. Core math: ordinary cross-entropy next-token loss, restricted to the demonstration set. This is the natural first thing to try and it genuinely teaches the model the *form* of instruction-following. Its gap: it is capped by imitation. There is exactly one demonstration per prompt and no signal about *why* one response is better than another, so for open-ended tasks (where many responses are acceptable and quality is graded) the model can only mimic, never learn to discriminate or to exceed the demonstrator. Demonstrations are also expensive, and validation loss overfits quickly even as true downstream quality keeps improving — a sign that the supervised loss is the wrong target.

**Few-shot / prompted base model.** Leave the pretrained model untouched and engineer a prefix (a few worked examples, or a hand-tuned instruction preamble) that coaxes it into instruction-following mode. Core idea: exploit in-context learning, no training. Its gap: brittle and shallow — the burden is on the user to find a working prompt, the effect does not persist or generalize, and the model's underlying tendencies (hallucination, toxicity) are unchanged.

**Cross-task instruction tuning on public NLP datasets (FLAN, Wei et al. 2021; T0, Sanh et al. 2021).** Assemble many existing NLP datasets, reformat each as a natural-language instruction plus input/output, and supervised-fine-tune on the union; this improves zero-shot performance on *held-out* NLP tasks. Core idea: instruction-following as multi-task supervised learning over academic tasks. Its gap relative to the goal here: these dataset collections are dominated by tasks that are easy to score automatically (classification, QA), which is *not* how people actually use the models — real prompts are heavily open-ended generation and brainstorming. So the alignment signal is mismatched in distribution; the resulting model under-serves the real user prompt distribution even though it does well on academic held-out tasks. This baseline isolates the lesson that *which distribution you align on* matters as much as the act of instruction-tuning.

**Optimize human judgment directly with RL.** Skip the reward model: sample a response, ask a human if it's good, use that as reward, policy-gradient. Its gap is fatal in practice — RL needs millions of episodes and you cannot keep a human in that loop; the human throughput is many orders of magnitude too low. This baseline isolates the throughput wall: the objective one actually cares about is the one that cannot be queried at the rate RL demands.

## Evaluation settings

The natural yardstick is human preference on the *real* prompt distribution: held-out prompts from the same source as training (real user submissions plus labeler-written prompts spanning generation, open-/closed-domain QA, brainstorming, chat, rewriting, summarization, classification, extraction), with labelers judging which model's output they prefer against a fixed reference policy, plus an absolute 1–7 Likert quality rating and a battery of binary metadata flags (follows the instruction, appropriate for a customer assistant, hallucinates on closed-domain input, contains sexual/violent content, denigrates a protected class, etc.). A separate pool of held-out labelers who produced none of the training data tests whether preferences generalize beyond the people who were trained on.

Alongside human evaluation, automatic evaluation on public datasets serves two purposes: safety-oriented sets — TruthfulQA (truthfulness), RealToxicityPrompts scored by a toxicity classifier (toxicity), Winogender and CrowS-Pairs (bias) — and capability-oriented zero-shot sets that probe whether fine-tuning eroded general ability: SQuADv2, DROP, QuAC, HellaSwag, RACE, SST, RTE/WSC, WMT15 Fr→En translation, and CNN/DailyMail and Reddit TL;DR summarization. Generation tasks are sampled at temperature 0 truncated at the first newline; multiple-choice answers are scored by lowest average per-token log-probability. Bias is reported as the entropy of the model's choice distribution.

## Code framework

The available primitives are standard and established: a pretrained decoder-only transformer, a tokenizer, an Adam optimizer, an autoregressive sampling routine, and the textbook building blocks of next-token cross-entropy, the Bradley-Terry/logistic pairwise comparison loss, generalized advantage estimation, and the PPO clipped-surrogate update — each usable on its own. What objects are built from these primitives, how human feedback is turned into a training signal, and how the pieces are wired into a fine-tuning procedure is left open.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

base_model = PretrainedCausalLM(...)
tokenizer = Tokenizer(...)
optimizer = torch.optim.Adam(base_model.parameters(), betas=(0.9, 0.95))

def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)

def gather_token_logprobs(logits, input_ids):
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    labels = input_ids[:, 1:]
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

# The fine-tuning procedure itself is to be built.
# TODO
```
