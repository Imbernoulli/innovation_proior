# Step 2 — ICM (curiosity in a controllable-state space)

The no-bonus run told me exactly what's missing. Where the environment reward is sparse, the
advantage is zero almost everywhere, so the agent's only exploration is the entropy jitter of its
own policy — and that jitter is a coin flip. On Frostbite the coin lands right (the first reward is
reachable, all three seeds got traction). On Tutankham it lands right on one seed out of three and
flatlines at literally zero on the other two; on Private Eye it not only fails to find reward but
wanders into the game's penalties, dragging the mean *negative*. The diagnosis is sharp: I don't
have a learning problem, I have a *signal* problem. There is no gradient to climb because the agent
never manufactures anything to be advantaged over. So the next move is forced — I have to synthesize
a reward from the agent's own experience, an intrinsic bonus $i_t$ added to the (mostly zero)
extrinsic reward, large in states the agent hasn't mastered, so that "go somewhere new" becomes
something the policy gradient can actually ascend.

What should that bonus measure? The cheapest honest signal is *prediction error*: build a model of
how the world responds to the agent, and reward the agent where the model is wrong, because being
wrong means the transition is unfamiliar. The naive version — predict the next *pixels* — is a trap.
Picture any source of unpredictable-but-irrelevant variation: flickering background, a distractor
that moves on its own. Pixel-prediction error there is irreducibly high and never decays, so the
agent parks itself farming that error forever instead of exploring. Pixels mix three things — what
the agent controls, what affects it, and irrelevant noise — and the noise dominates the error budget
precisely because it is unpredictable.

So the question isn't "prediction error or not," it's "prediction error in *what space*." If I had a
feature map $\phi(s)$ that kept only the action-relevant content and discarded the noise, then a
forward model in $\phi$-space would respond only to the parts of the world I care about. I can't get
$\phi$ by training it to make forward prediction easy — that collapses $\phi$ to a constant (perfectly
predictable, perfectly useless). The fix is to anchor $\phi$ with an *inverse* task: from
$\phi(s_t),\phi(s_{t+1})$ recover the action $a_t$. To name the action, $\phi$ must encode what the
action changed and has no reason to encode noise the action can't affect — and it can't collapse,
because a constant $\phi$ destroys the action information. Then train a forward model in that anchored
space and let its error be the bonus: high on transitions whose *controllable* consequences are not
yet predicted, decaying as the agent masters them. This is curiosity that is blind to distractors by
construction. The full derivation of the inverse/forward decomposition, the loss weighting, and the
architecture is the standalone trace at `methods/icm/`.

The delta from step 1 is therefore: replace the empty bonus module with one that learns a
controllable-state encoder by inverse dynamics, predicts the next controllable state with a forward
model, and emits the forward error as $i_t$; mix the intrinsic advantage in alongside the extrinsic
one. My expectation, reading the step-1 shape: this should rescue the games where the *first* reward
is hard to reach by random jitter but the dynamics are learnable — I'd expect Tutankham to stabilize
(every seed should at least find the reward now that "explore the controllable world" is rewarded)
and Frostbite to potentially jump, because curiosity keeps pushing the agent into new controllable
configurations. Private Eye is the open question: its reward is not just far but *deceptive*, and a
curiosity bonus that decays as the agent masters local dynamics may run out of drive before it
crosses the long reward-free gap.
