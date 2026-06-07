# Step 3 — RND (random network distillation)

ICM's feedback points straight at its successor. Curiosity rescued Tutankham and made a high-variance
leap on Frostbite, but on Private Eye it scored a flat zero on every seed — never one reward. The
reason isn't that ICM explores badly; it's *where* its bonus comes from. The forward-prediction error
is large only while the controllable dynamics near the agent are unmastered, and it *decays toward
zero* the moment the forward model has learned those local transitions. On Private Eye the local
dynamics at the start are quickly learnable, so the bonus collapses before the agent has crossed the
long reward-free gap — the drive runs out at the first mastered region. And even where ICM did win
(Frostbite), the win came on one seed in three: a jackpot, not a dependable signal.

So two things are wrong, and they share a root. First, the novelty signal decays too fast and too
*locally* — it tracks "have I learned the dynamics right here," when what a long-horizon game needs
is "is this state still unfamiliar *overall*." Second, ICM is doing a lot of work to get there: it
maintains an encoder, an inverse model, and a forward model, three coupled networks whose interaction
is exactly what makes its behavior seed-sensitive. If I want a more *stable* and more *global* novelty
detector, I should look for a signal that is (a) about global familiarity rather than local
predictability, and (b) far simpler, so there's less to go unstable across seeds.

Here is the reframe. ICM's noisy-TV problem came from predicting a target that is *stochastic* — the
next observation is partly random, so the error never fully decays. What if I deliberately pick a
prediction target that is *deterministic* and *inside my model's reach*, so the only thing that can
keep the error high is having seen too little data near this state — i.e. pure novelty? Take a second
network, initialize it randomly, and *freeze* it; its output on any observation is a fixed,
arbitrary, deterministic embedding. Train a predictor to match that frozen target on the states the
agent visits. On states seen often, gradient descent has pulled the predictor onto the target, so the
error is small; on globally novel states it hasn't, so the error is high. The distillation error
$\|\hat f(s)-f(s)\|^2$ *is* a global-novelty signal — it depends on how much relevant data the
predictor has seen, nothing else — and it is exactly two forward passes, no inverse model, no forward
model, no encoder to co-train. That directly answers both of ICM's problems: it's a global, slowly-
decaying familiarity measure, and it is dramatically simpler, so there's far less to make it
seed-fragile. The full argument — why a deterministic in-class target removes the noisy-TV trap, the
randomized-prior reading of the error as uncertainty, the two-value-head / episodic-vs-non-episodic
return treatment, and the observation-normalization that a frozen target requires — is the standalone
trace at `methods/rnd/`.

The delta from step 2: swap the inverse/forward curiosity module for a frozen-random-target +
trained-predictor pair, emit the distillation error as $i_t$, normalize it by a running estimate of
the intrinsic-return std (the error's raw scale drifts), and whiten the observations into the
target/predictor (a frozen random net can't adapt to the input scale on its own). My expectation,
reading ICM's shape: the global novelty signal should finally give the agent a reason to *keep* moving
past the first mastered region — so Private Eye is the game I'm watching. If a non-decaying global
bonus can get even one seed to cross the gap and register a real Private Eye return, that's the
breakthrough ICM couldn't reach. Tutankham and Frostbite I expect to stay roughly where curiosity put
them, perhaps trading a little Frostbite peak for more stability.
