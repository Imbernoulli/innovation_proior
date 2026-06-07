# Step 1 — vanilla PPO, no intrinsic bonus

I start with the most honest baseline: change nothing. Leave the intrinsic-bonus module empty, mix
in zero intrinsic advantage, and let PPO optimize the clipped extrinsic reward alone. This is not a
method I expect to win; it is the measurement that defines the problem. If the no-bonus agent
already solved these games, there would be nothing to explore for.

The reasoning is short because there is no novelty machinery to derive — the contribution here is
*absence*. PPO ascends the clipped surrogate on advantages built from the environment reward, and
explores only through the entropy of its own action distribution. On a game with reasonably
reachable reward that undirected jitter is enough to find the first payoffs and bootstrap; on a
genuinely sparse game the advantage is zero almost everywhere, so there is no gradient signal to
climb and the agent wanders without direction. I expect a split outcome along exactly that line:
something on the easier game, little to nothing on the hard ones — and on a game with large
*negative* rewards available, undirected wandering can be actively harmful, because the agent can
blunder into the penalties as easily as avoid them.

The full PPO derivation — the clipped-surrogate objective, GAE, the parameter-sharing and entropy
details of the fixed loop — is the standalone trace at `methods/ppo/`. Here it is the floor I
measure against. What I am really collecting from this step is the *shape* of the failure: which
games move, which don't, and how unstable the result is across seeds. That shape is what tells me
what an intrinsic bonus has to supply.
