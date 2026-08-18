# Source-value fix — awac (track B_selfaccount)

Self-account already on file: `refs/self_accounts/bair_awac_blog_2020.html` (Ashvin Nair and Abhishek Gupta, BAIR Blog, 2020-09-10, https://bair.berkeley.edu/blog/2020/09/10/awac/). Already catalogued in `notes/source_matrix.md`; this note records the specific load-bearing quote used in the svfix pass.

## Quote used (data-efficiency requirement, why bootstrapped Q not AWR's Monte-Carlo return)

> "Consider the robot on the right, trying to reach the goal state with prior trajectory $\tau_1$ and $\tau_2$. On-policy methods cannot effectively use this data, but off-policy algorithms that do dynamic programming can, by effectively "stitching" $\tau_1$ and $\tau_2$ together with the use of a value function or model."

This is the authors' own concrete mechanism for *why* an off-policy bootstrapped critic beats the on-policy / Monte-Carlo-return alternatives (DAPG, AWR, MARWIL) on data efficiency — not just "off-policy is faster" but *trajectory stitching*: Bellman backup composes value across transitions regardless of which logged trajectory they came from, while a Monte-Carlo or TD(λ) return is stuck scoring each trajectory as it was played. The primary paper (`refs/primary/arxiv_source/main.tex`) reports the same HalfCheetah efficiency comparison and numbers but does not use this stitching illustration — it is unique to the self-account (Figure 2 caption/body).

## Where used
- `results/reasoning.md`, paragraph beginning "First, efficiency." — replaces the previous bare assertion ("off-policy actor-critic that bootstraps Q ... reuses the whole buffer") with the stitching mechanism as the actual reason for the observed order-of-magnitude speed gap.
- `results/reasoning.md`, critic paragraph ("That leaves the critic...") — the AWR contrast (bootstrapped Q^π vs AWR's Monte-Carlo/TD(λ) V^{π_β}) now names stitching explicitly, tying the two uses of the same mechanism together.

## Also checked, not separately re-grounded
- Primary appendix (`main.tex` line ~371) and main text (line ~206-207) disagree on which KL direction ("forward" vs "reverse") is the one that samples from the buffer — an inconsistency internal to the paper's own terminology across sections, not something the self-account speaks to. `reasoning.md`'s derivation (KL(π*‖π_θ), sampling from π_β, called "forward") matches the paper's main-text terminology (line 206-207) and its own math; left as is since it is internally consistent and correct.
- Rest of the self-account (three-challenges narrative, BAIR blog "Challenges" section) closely mirrors `main.tex` intro/results content already load-bearing in `reasoning.md`'s requirement-two and requirement-three paragraphs; no additional non-primary content found there beyond the stitching illustration.
