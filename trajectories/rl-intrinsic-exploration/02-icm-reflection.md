The no-bonus run told me exactly what's missing. Where the environment reward is sparse, the
advantage is zero almost everywhere, so the agent's only exploration is the entropy jitter of its own
policy — and that jitter is a coin flip. On Frostbite the coin lands right (the first reward is
reachable, all three seeds got traction). On Tutankham it lands right on one seed out of three and
flatlines at literally zero on the other two; on Private Eye it not only fails to find reward but
wanders into the game's penalties, dragging the mean *negative*. The diagnosis is sharp: I don't have
a learning problem, I have a *signal* problem. There is no gradient to climb because the agent never
manufactures anything to be advantaged over.

So the next move is forced — I have to synthesize a reward from the agent's own experience, an
intrinsic bonus added to the (mostly zero) extrinsic reward, large in states the agent hasn't
mastered, so that "go somewhere new" becomes something the policy gradient can actually ascend. The
cheapest honest such signal is *prediction error* in a learned, action-relevant feature space —
curiosity. Here is how that comes together.