## Decision Problem

A learner faces `n` possible decisions over `T` rounds. On each round it must commit to a decision, or to a probability distribution over decisions, before seeing what the round will cost. After the commitment, an adversary reveals a full cost vector `m(t)`, where `m_i(t)` is the cost decision `i` would have paid on that round. Costs are bounded, and the adversary may choose them adaptively after seeing the learner's current distribution.

The learner cannot compete with the best decision separately on each round, since that would require knowing the cost vector in advance. The meaningful benchmark is the best single fixed decision in hindsight:

`min_i sum_t m_i(t)`.

The goal is to make the learner's cumulative expected cost close to that hindsight benchmark, with a gap that grows sublinearly in `T`.

## What The Learner Can See

This is a full-information setting: after each round, the whole vector of decision costs is revealed, not only the cost of the sampled decision. The learner can therefore revise its opinion of every decision after every round.

The only numerical promise is boundedness. No stochastic model is assumed. The cost sequence can be correlated, adaptive, and adversarial. Any proof has to work from bounded costs and from the algebra of the learner's own distribution; it cannot rely on independence or concentration around a hidden distribution.

## Existing Baselines

The simplest expert-advice baseline predicts with the majority of experts. It fails because a majority can be wrong on every round while one minority expert remains correct.

A deterministic weighted-majority baseline improves on majority voting by maintaining weights on experts and reducing the influence of experts that have erred. A total-weight analysis proves a mistake bound within roughly a factor of two of the best expert, plus a logarithmic term. The factor two is the obstruction: committing deterministically to one side lets an adversary exploit the majority boundary.

Freund and Schapire's Hedge framework generalizes expert advice to online allocation and proves loss bounds through a total-weight potential. This shows that randomized weighted allocation is not tied to binary prediction. Its standard bound charges a second-order term tied to the learner's own mixed losses, so reductions that need a separate certificate for each comparator still leave a gap: the analysis must keep the comparator-side error term clean.

## Analysis Tools

The available proof tools are elementary but precise. Prior weighted-expert analyses suggest tracking an aggregate mass, because it records how much evidence remains on the current pool. A useful proof would need to relate that aggregate both to the learner's expected cost and to a fixed comparator without losing the deterministic factor-two slack.

The useful inequalities are `1 + x <= exp(x)`, logarithm estimates for small `eta`, and separate handling of positive and negative bounded coordinates. A parallel viewpoint measures progress by relative entropy: if the learner pays more than a comparator distribution, the update should make that comparator less surprising under the learner's state.

## Evaluation Surface

The basic evaluation is regret against the best fixed decision as a function of `T`, `n`, and the learning-rate parameter. The desired form is a bound that becomes `O(sqrt(T log n))` under unit-bounded costs.

The same guarantee is also judged by whether it can drive reductions. The relevant test beds are repeated play in zero-sum games, feasibility linear programs with a one-constraint oracle, greedy-style set cover, and boosting with a changing example distribution. A useful method should explain why these settings share one regret calculation rather than a collection of unrelated tricks.
