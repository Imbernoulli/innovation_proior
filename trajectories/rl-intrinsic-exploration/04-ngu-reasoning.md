# Step 4 (finale) — NGU (Never Give Up)

RND's feedback hands me a clean, specific failure to fix. RND was the breakthrough — the only
baseline to get real Private Eye return (seed 456 at 2756) — but its mean there was still negative in
`auc`, two of three seeds couldn't sustain it (seed 42 a flat zero, worst auc of the run), and the
whole thing was fragile. And the *reason* is not "the signal is too local" — that was ICM's problem,
which RND already fixed by going global. RND's problem is the opposite end: its novelty is purely
**lifelong**. The distillation error on a region only ever decreases across training, and it has *no
within-episode memory whatsoever*. So once a region's global novelty has worn off, RND gives the
agent no reason to walk back through it — and at the start of every fresh episode, a globally-mastered
state looks exactly as stale as it did at the end of the last episode. There is nothing that resets.

Stare at what Private Eye actually demands and the gap is obvious. To extend its reach the agent has
to re-traverse cleared ground, episode after episode, because the only path to the next undiscovered
area runs through the old one. A lifelong-only bonus has, by construction, stopped paying for that
traversal — so the agent that found 2756 on a lucky seed can't *reliably* reproduce the journey,
because nothing pulls it back through the early rooms once they're globally familiar. Every novelty
bonus I've tried — ICM's forward error, RND's distillation error — shares this: it is a one-shot
frontier-pusher that vanishes, and vanishing is exactly wrong for a game that needs *persistent*,
*repeated* exploration.

So the next move is forced by the diagnosis: I need a novelty that *resets every episode*. Watch a
human explorer — within one attempt they avoid backtracking onto squares they've already stood on
this run, but they happily re-enter, on a fresh attempt, a room they cleared a hundred attempts ago,
because clearing it again is the price of getting deeper. That's two timescales. *Within* an episode,
novelty should be sharp and reset-able: a state I've visited this episode is stale, one I haven't is
fresh, and that judgment snaps back to "everything fresh" the instant a new episode starts. *Across*
episodes, novelty should be slow — a room cleared ten thousand times really is mastered. RND is purely
the slow kind and has none of the fast kind. That missing fast timescale is the whole fix.

The within-episode part I build as a *count*: keep a per-episode memory of the controllable-state
embeddings seen so far this episode (emptied at every episode boundary), and at each step turn "how
near am I to anything I've seen this episode" into a pseudo-count via a $k$-nearest-neighbor kernel
sum, with the episodic bonus $1/\sqrt{\text{count}}$ in the spirit of count-based exploration
(Strehl & Littman 2008). Reset each episode, so the agent is driven to cover new ground *every*
episode — it never gives up. I keep the embedding action-relevant (the inverse-dynamics controllable
state from step 2's ICM) so uncontrollable flicker can't manufacture fake novelty. Then I *keep RND*,
not as the bonus but as a slow **lifelong modulator** that down-weights regions seen across many
episodes — and I combine them *multiplicatively* with a floor at 1, so the lifelong factor can only
*amplify* the episodic drive and never kill it (and a cap so a spike can't explode it). As everything
is globally mastered the modulator goes to 1 and the method degrades gracefully to pure episodic
novelty — the part that should never decay. The full derivation — the kNN pseudo-count with its
scale-free kernel, the cluster floor and saturation cap that close the camping exploits, the
multiplicative-with-floor argument, and the UVFA family that separates an exploit policy from the
exploratory ones so dense games aren't sacrificed — is the standalone trace at `methods/ngu/`.

The delta from step 3 is therefore: add a per-episode episodic-memory module on top of an
inverse-dynamics controllable-state embedding, compute the within-episode kNN pseudo-count bonus,
and multiply it by RND's normalized lifelong factor — $i_t = r^{\text{episodic}}_t \cdot
\min\{\max\{\alpha_t,1\}, L\}$. Where RND alone gave the agent no reason to re-cross cleared ground,
the episodic reset now pays it to re-explore on *every* episode, while the RND factor it already had
keeps down-weighting the genuinely-mastered. My expectation, reading RND's shape: the seed that died
on Private Eye (42, a flat zero) is the one to watch — a persistent, reset-each-episode drive is
precisely what a dead seed lacked, so if the episodic timescale is the right diagnosis, the worst
seed should stop being "found nothing," the negative `auc` should lift toward positive (less time
spent in the red, more sustained progress), and the Private Eye result should become *reliable* rather
than a lucky-seed jackpot — while Tutankham and Frostbite, where the bonus simply matters less once
reward is reachable, should hold roughly steady.
