## Episodic Learning With A Late Reward

Consider a finite-horizon tabular Markov decision process with `S` states, `A` actions, horizon `H`, unknown transition kernels `P_h(.|s,a)`, and rewards in `[0,1]`. In the usual problem, the reward is present during interaction, and the agent is judged by the expected gap between the optimal value and the value of its returned policy from the initial-state distribution.

The statistical difficulty is exploration. Random action noise can take exponential time to reach informative corners of an MDP, while algorithms such as E3, R-MAX, UCBVI, UCB-Q, and EULER deliberately direct the agent toward uncertain or promising parts of the environment. These methods learn efficiently when a single reward objective is fixed.

Many uses do not have a single final reward at data-collection time. A reward may be tuned by trial and error, shaped to produce a desired behavior, or changed while sweeping constraints and Lagrange multipliers. Re-running a reward-aware exploration algorithm for every candidate reward spends fresh environmental samples each time.

## What A Reusable Dataset Must Support

Batch reinforcement learning starts from a logged dataset and then computes a policy without further interaction. Its guarantees typically use a distribution-shift condition: the logged state-action distribution must cover the occupancy distribution of policies the planner might output. Under such coverage, small empirical Bellman errors imply small value error; the policy stays away from places where the transition model is poorly estimated.

The relevant identity is the value-difference or simulation lemma. When two MDPs share a reward and policy but differ in transitions, their value gap telescopes into a sum of transition-estimation errors applied to future value functions, weighted by the policy's occupancy. A logged dataset is useful to the extent that it estimates transitions where later policies may go.

If the reward is revealed after data collection, the later policy is also unknown during data collection. The dataset is then judged against the family of possible occupancies rather than the single occupancy of one-reward learning.

## Known-Set Optimism

E3 and R-MAX give the classical finite-sample template for tabular exploration. They maintain a set of known states or state-action entries, estimate the transition model there, and use optimism to decide whether to exploit the known model or collect more information. The pigeonhole principle limits how often genuinely new information can be obtained before more entries become known.

The template is built around a reward-aware notion of success. R-MAX, for example, initializes unknown entries optimistically so the optimal policy in the current model either achieves high payoff or exposes unknown dynamics. The mechanism is an explore-or-exploit compromise for one payoff problem.

## Occupancy Geometry

An MDP is not a bandit. Some states cannot be selected directly, and some are reachable only with tiny probability no matter what policy is used. A data-collection goal asking for uniform accuracy everywhere is a strong requirement: a rare branch can make a state essentially impossible to sample often.

At the same time, a later adversarial reward can put all value behind a transition that the dataset barely sampled. The relevant tension is between these two facts: impossible-to-reach regions and regions a policy can reach with meaningful probability.

Coverage is thus a geometric property of the transition system rather than a property of a particular reward. The object at issue is a data distribution whose visitation probabilities serve later planning tasks while not demanding samples from unreachable parts of the MDP.

## Evaluation Surface

The guarantee under study is a two-phase protocol. During interaction, the learner observes states, actions, and next states but uses no task reward. After interaction ends, a reward function is supplied, and the learner returns a policy using only the logged trajectories and the reward table.

The benchmark is the number of exploration episodes needed for the guarantee to hold with high probability for arbitrary later rewards, including a sequence of adaptively chosen rewards. The natural comparison is fixed-reward tabular RL, whose minimax sample complexity has one factor of `S` in the leading state dependence.

The implementation slot, stated before the method is known: collect a transition dataset, form empirical transition estimates, and call a known-MDP planner such as value iteration or a controlled approximate solver. The question is how the first phase chooses the data-collection policies without seeing the reward that will later define the task.
