# Constitutional AI (RLAIF)

## Problem

Train a helpful *and* harmless assistant **without any human feedback labels for harm** — the only human input being a short written list of principles (a "constitution") — and do it *non-evasively*, so the model engages with sensitive requests and explains its objections rather than stonewalling. Standard RLHF gets harmlessness from tens of thousands of human harm-preference labels, which is expensive, opaque, costly to re-steer, and (because crowdworkers reward refusals) trains evasive models.

## Key idea

Large LMs can already *recognize* harm (chain-of-thought reasoning improves this), so push human supervision upstream to a written constitution and let the model generate the harm signal itself. Two stages:

1. **Supervised (Critique → Revision → SFT):** the model critiques and revises its own harmful responses, guided by sampled principles, and is finetuned on the revisions → **SL-CAI**.
2. **RL from AI Feedback (RLAIF):** a feedback model labels which of two responses is more harmless per a principle; these AI labels (mixed with human *helpfulness* labels) train a preference model, against which SL-CAI is RL-finetuned → **RL-CAI**.

## Method

**Supervised stage.** For each red-team prompt, sample a (usually harmful) response from a helpful RLHF model. Append a randomly-drawn principle's *critique request*, sample a critique; append its *revision request*, sample a revision. The (prompt, revision) pair is formatted like the original, so the critique→revision loop iterates (≈4 revisions). Use ~16 principles, sampled per step — diversity aids later RL exploration — and few-shot exemplars to keep the model from confusing critique vs revision turns; sample at $T=1$. Finetune a *pretrained* model on all revisions plus helpfulness samples (to retain helpfulness): one epoch, constant LR = 0.5 × pretraining LR, batch size 1024. Critiques help smaller models and add transparency, so they are kept even though large models revise nearly as well directly.

**RL stage.** SL-CAI generates response pairs. A feedback model is asked, as multiple choice, which response is more harmless given a sampled principle; the normalized log-probabilities of "(A)"/"(B)" become soft preference targets. Optionally use chain-of-thought ("Let's think step by step"), which is more accurate but overconfident — so **clamp** the CoT-derived probabilities to ~40–60% for calibration (otherwise RL drifts to extreme outputs). Mix these AI harmlessness comparisons with human helpfulness comparisons, train a preference model (the same RLHF PM procedure), and RL-finetune SL-CAI against it (standard PPO with a KL penalty). Use SL-CAI for both generation and RL initialization so the PM matches the policy distribution. No human labels for harm appear anywhere.

## Code

```python
PRINCIPLES = [ ... ]  # ~16 short written rules; each has a critique_request + revision_request

# Supervised stage: critique -> revision -> SL-CAI
def make_harmless_sft_data(helpful_model, red_team_prompts, principles, n_revisions=4):
    data = []
    for prompt in red_team_prompts:
        resp = sample(helpful_model, prompt, T=1.0)              # usually harmful
        for _ in range(n_revisions):
            p = random.choice(principles)                         # diversity
            critique = sample(helpful_model, FEWSHOT + prompt + resp + p.critique_request, T=1.0)
            revision = sample(helpful_model, FEWSHOT + prompt + resp + critique + p.revision_request, T=1.0)
            data.append((prompt, revision)); resp = revision      # iterate
    return data

# RL stage: AI feedback -> PM -> RLAIF
def ai_label(fb, prompt, a, b, principle, use_cot=False):
    if use_cot:
        cot = sample(fb, COT_FEWSHOT + prompt + principle + a + b
                     + "\nAssistant: Let's think step-by-step:", T=1.0)
        ctx = prompt + principle + a + b + cot + "\nThe answer is:"
    else:
        ctx = FEWSHOT + prompt + principle + a + b + "\nThe answer is:"
    p = softmax([logprob(fb, ctx, " (A)"), logprob(fb, ctx, " (B)")])[0]
    return min(0.6, max(0.4, p)) if use_cot else p               # clamp overconfident CoT

def build_assistant():
    sft = make_harmless_sft_data(helpful_rlhf, red_team_prompts, PRINCIPLES)
    sl_cai = finetune(pretrained, sft + helpfulness_samples,
                      lr=0.5 * PRETRAIN_LR, epochs=1, batch_size=1024)
    prefs = [((prompt, a, b), ai_label(feedback_model, prompt, a, b,
                                       random.choice(PRINCIPLES), use_cot=True))
             for prompt in red_team_prompts
             for a, b in [(sample(sl_cai, prompt, T=1.0), sample(sl_cai, prompt, T=1.0))]]
    pm = train_preference_model(prefs + human_helpfulness_comparisons)
    return rlhf(sl_cai, all_prompts, pm, ref_model=sl_cai)        # RL-CAI
```
