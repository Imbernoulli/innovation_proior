The setting is training a large language-model assistant to be helpful *and* harmless, around 2022. The established alignment recipe is reinforcement learning from human feedback (RLHF). The aim is an assistant that is helpful, honest, and harmless (HHH). The standard way of getting harmlessness is to collect tens of thousands of human preference labels on which of two responses is less harmful.

## Research question

Can an assistant be made harmless using only a short, explicit list of natural-language principles as the human input, rather than human feedback labels for harm — while staying helpful and non-evasive? Several considerations frame this:

1. **Scaling supervision.** As models approach or exceed human ability on some tasks, AI can help supervise AI, with humans providing a small amount of high-quality, legible oversight rather than exhaustive labels.
2. **Evasiveness.** When harmlessness is taught by rewarding human-preferred responses to harmful prompts, crowdworkers tend to reward refusals, so the assistant learns to be evasive — it stonewalls controversial questions ("I can't help with that") instead of engaging and explaining its objection.
3. **Transparency.** Encoding the goals as a short list of written principles would make the training objective legible, in contrast to a model trained on tens of thousands of private preference labels.
4. **Iteration cost.** Changing the harmlessness objective under RLHF means collecting a fresh batch of human labels; an objective living in a few sentences could be re-steered without new data collection.

## Background

- **RLHF** (Christiano et al. 2017; Stiennon et al. 2020; Bai et al. 2022 "Training a Helpful and Harmless Assistant"; Ouyang et al. 2022 InstructGPT). Pipeline: collect human preference comparisons (which of two model responses is better), train a **preference model (PM)** that scores responses, then fine-tune the policy with reinforcement learning (PPO-style) against the PM reward, with a KL penalty to a reference model to keep the policy from drifting. At RL time the immediate reward comes from a learned PM, not a human in the loop; training that PM uses tens of thousands of human labels. The Bai et al. (2022) HH assistant trains separate helpfulness and harmlessness preference data and exhibits a helpfulness/harmlessness tradeoff.
- **Helpful/harmless tension and Elo evaluation.** Prior work measures helpfulness and harmlessness by crowdworker preference comparisons converted to Elo scores; helpful-only models are more helpful but more harmful, HH models are more harmless but more evasive. Evasiveness was *rewarded* by crowdworkers as a response to harmful inputs.
- **Red teaming.** Crowdworkers write adversarial prompts designed to bait the assistant into harmful content (Ganguli et al. 2022, "Red Teaming Language Models"), producing datasets of harmful prompts used to elicit and then measure harmful behavior.
- **In-context capabilities of large LMs.** Large models can follow natural-language instructions and respond to instructions about a piece of text. Their answer probabilities are often well-calibrated when a question is posed as a multiple-choice question (Kadavath et al. 2022 — language models' answer probabilities are calibrated).
- **Chain-of-thought reasoning** (Nye et al. 2021 scratchpads; Wei et al. 2022; Kojima et al. 2022 "Let's think step by step"): prompting a model to reason step by step before answering improves performance on judgment tasks and makes the reasoning legible.
- **Diagnostic finding.** Posed as multiple-choice HHH-identification questions (438 binary comparisons), pretrained/large LMs already identify the more harmless/helpful/honest response with high accuracy, and chain-of-thought *raises* this accuracy — trending toward parity with human-feedback-trained preference models as scale grows.

## Baselines

- **Helpful RLHF.** RLHF trained only on helpfulness comparisons. Very helpful, will comply with pernicious requests. Used here as the *starting* assistant.
- **HH RLHF** (Bai et al. 2022). RLHF on both helpfulness and harmlessness human comparisons. More harmless than helpful-only, more evasive and less helpful, and requires the full harmlessness human-label collection.
- **Other instruction/feedback-trained assistants** (InstructGPT, LaMDA, Sparrow): RLHF-family systems with rule- or human-feedback-based safety, depending on human supervision for safety behavior.

## Evaluation settings

- **Models:** a series of pretrained LMs up to 52B parameters, pretrained as in Bai et al. (2022); helpful RLHF and HH RLHF policies and preference models trained as points of comparison.
- **Data sources (pre-existing):** red-team harmful prompts (42,496 human-written from Ganguli et al. 2022) augmented by few-shot model generation; human-written helpfulness prompts; the helpfulness human-preference comparison dataset.
- **Metrics:** helpfulness and harmlessness **Elo scores** from crowdworker A/B comparisons in open-ended conversation (crowdworkers instructed to prefer non-evasive responses among equally harmless ones); preference-model scores on red-team prompts; multiple-choice accuracy on the 438 HHH-identification comparisons; absolute harmfulness scores from red-team conversations.
- **Protocol knobs (a priori):** sampling temperature, number of revisions, number of written principles, whether to use chain-of-thought for AI judgments, and the RL/PM training hyperparameters inherited from prior RLHF work.

## Code framework

The RLHF machinery already exists: a PM trainer and a PPO-style RL loop against a PM reward. What remains open is how to obtain a harmlessness objective when the human harm labels are off the table.

```python
# Existing RLHF primitives (from prior work) ----------------------------------
def train_preference_model(comparisons):
    """Train a PM that scores a response higher when it's the preferred one."""
    ...
    return pm

def rlhf(policy_init, prompts, pm, ref_model):
    """PPO-style RL: maximize pm(response) with a KL penalty to ref_model."""
    ...
    return policy

def sample(model, context, T=1.0):
    ...

def finetune(model, data, **kw):
    ...

# A 'helpful-only' RLHF assistant is assumed already trained.
helpful_rlhf = rlhf(pretrained, helpfulness_prompts, helpful_pm, pretrained)

# Available inputs: the written principles, the red-team prompts, the helpful
# assistant, and the human *helpfulness* comparisons. No human harm labels.
PRINCIPLES = [...]  # a short written list of natural-language rules (the input)

def build_assistant():
    # TODO: obtain a harmless-and-helpful assistant from the inputs above,
    #       without any human-labeled harm data.
    raise NotImplementedError
```
