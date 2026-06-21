The problem is to train a helpful and harmless language-model assistant without any human feedback labels for harmlessness. Standard RLHF for harmlessness requires tens of thousands of human preference comparisons, which is expensive, opaque, and hard to iterate. Worse, because crowdworkers tend to reward refusals when asked which response to a harmful prompt is better, RLHF produces evasive assistants that stonewall controversial questions, trading away helpfulness for harmlessness. The goal is to obtain harmlessness from only a short written list of natural-language principles — a constitution — while keeping the model engaged and explanatory rather than evasive.

The crucial observation is that large pretrained language models already recognize harm well, and chain-of-thought reasoning makes their judgments even more accurate, approaching the quality of preference models trained on large human-labeled datasets. That means the model itself can generate the harmlessness supervision, provided a human specifies the principles. The solution is to push human supervision upstream: instead of labeling harmful examples, humans write a short constitution, and the model supplies its own harmlessness labels via self-critique and pairwise AI feedback.

The method is Constitutional AI (CAI). It has two stages: SL-CAI, a supervised fine-tuning stage, and RL-CAI, a reinforcement-learning stage. SL-CAI first puts the model on distribution by turning harmful responses into harmless, engaged ones. Start from a helpful RLHF model and a set of red-team prompts that elicit harmful behavior. For each prompt, sample a harmful response. Then append a critique instruction sampled from the constitution, asking the model to identify what is harmful, unethical, or dangerous about its own response, and sample the critique. Then append a revision instruction asking the model to rewrite the response to remove the harmful content, and sample the revision. Because the revised response is formatted like the original response, this critique-revision loop can be iterated several times, and revisions from every step are kept as training targets. Randomly sampling different constitutional principles at each step gives diversity, which improves exploration for the later RL stage, and few-shot exemplars keep the model from confusing critique and revision turns. A pretrained model — not the helpful RLHF model — is then fine-tuned on these harmless revisions mixed with helpfulness samples, so helpfulness is preserved. This yields SL-CAI, an already mostly harmless assistant.

RL-CAI then refines this model with reinforcement learning using AI-generated harmlessness labels. SL-CAI generates pairs of responses to red-team prompts. A feedback model is shown the prompt, a sampled constitutional principle, and the two candidate responses as a multiple-choice question asking which is more harmless. The normalized log-probabilities of the "(A)" and "(B)" tokens give a soft preference label. Optionally, the feedback model first reasons step by step before answering; this raises accuracy but tends to be overconfident, so the resulting probability is clamped to a middle band around 40–60% to avoid extreme PM targets that would destabilize RL. These AI harmlessness comparisons are mixed with existing human helpfulness comparisons to train a preference model, and SL-CAI is fine-tuned against it with standard PPO plus a KL penalty. Using SL-CAI for both response generation and RL initialization keeps the preference model matched to the policy distribution. The final model is RL-CAI: harmless, non-evasive, and trained without any human labels for harm.

```python
PRINCIPLES = [
    # ~16 short written rules; each has a critique_request and revision_request
    {
        "critique_request": "Identify specific ways the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.",
        "revision_request": "Rewrite the assistant response to remove all harmful content.",
    },
    # ... additional principles sampled per step for diversity
]


def make_harmless_sft_data(helpful_model, red_team_prompts, principles, n_revisions=4):
    """Generate supervised critique->revision data for SL-CAI."""
    data = []
    for prompt in red_team_prompts:
        resp = sample(helpful_model, prompt, T=1.0)  # usually harmful
        for _ in range(n_revisions):
            p = random.choice(principles)
            critique = sample(
                helpful_model,
                FEWSHOT + prompt + resp + p["critique_request"],
                T=1.0,
            )
            revision = sample(
                helpful_model,
                FEWSHOT + prompt + resp + critique + p["revision_request"],
                T=1.0,
            )
            data.append((prompt, revision))  # keep every revision in the chain
            resp = revision  # iterate critique -> revision
    return data


def ai_label(feedback_model, prompt, response_a, response_b, principle, use_cot=False):
    """Return P(response A is more harmless than response B) under the principle."""
    if use_cot:
        cot = sample(
            feedback_model,
            COT_FEWSHOT + prompt + principle + response_a + response_b
            + "\nAssistant: Let's think step-by-step:",
            T=1.0,
        )
        ctx = prompt + principle + response_a + response_b + cot + "\nThe answer is:"
    else:
        ctx = FEWSHOT + prompt + principle + response_a + response_b + "\nThe answer is:"

    p_a = logprob(feedback_model, ctx, " (A)")
    p_b = logprob(feedback_model, ctx, " (B)")
    prob_a = softmax([p_a, p_b])[0]

    if use_cot:
        # Clamp overconfident chain-of-thought judgments for calibration.
        prob_a = min(0.6, max(0.4, prob_a))
    return prob_a


def make_harmless_preference_data(sl_cai, red_team_prompts, principles, use_cot=True):
    """Generate AI-labeled harmlessness preference comparisons for RL-CAI."""
    prefs = []
    for prompt in red_team_prompts:
        a = sample(sl_cai, prompt, T=1.0)
        b = sample(sl_cai, prompt, T=1.0)
        principle = random.choice(principles)
        label = ai_label(feedback_model, prompt, a, b, principle, use_cot=use_cot)
        prefs.append(((prompt, a, b), label))
    return prefs


def build_assistant():
    # Stage 1: supervised Constitutional AI (SL-CAI)
    sft_data = make_harmless_sft_data(helpful_rlhf, red_team_prompts, PRINCIPLES)
    sl_cai = finetune(
        pretrained,
        sft_data + helpfulness_samples,  # mix to retain helpfulness
        lr=0.5 * PRETRAIN_LR,
        epochs=1,
        batch_size=1024,
    )

    # Stage 2: reinforcement learning from AI feedback (RL-CAI)
    prefs = make_harmless_preference_data(sl_cai, red_team_prompts, PRINCIPLES)
    pm = train_preference_model(prefs + human_helpfulness_comparisons)
    rl_cai = rlhf(sl_cai, all_prompts, pm, ref_model=sl_cai)
    return rl_cai
```
