ICM's feedback points straight at its successor. Curiosity rescued Tutankham (all three seeds now
find and hold the reward, variance collapsed) and made a high-variance leap on Frostbite — but on
Private Eye it scored a flat zero on every seed, never one reward. The reason isn't that ICM explores
badly; it's *where* its bonus comes from. The forward-prediction error is large only while the
controllable dynamics near the agent are unmastered, and it *decays toward zero* the moment the
forward model has learned those local transitions. On Private Eye the local dynamics at the start are
quickly learnable, so the bonus collapses before the agent has crossed the long reward-free gap — the
drive runs out at the first mastered region. And even where ICM won (Frostbite), the win came on one
seed in three: a jackpot, not a dependable signal.

So two things are wrong and they share a root: the novelty signal decays too fast and too *locally*
(it tracks "have I learned the dynamics right here," not "is this state still unfamiliar overall"),
and ICM needs three coupled networks whose interaction is exactly what makes it seed-sensitive. I want
a novelty detector that is *global* (about familiarity, not local predictability) and far *simpler*,
so there's less to go unstable across seeds. That points at a deliberately different kind of
prediction target — deterministic and inside the model's reach, so the only thing keeping its error
high is having seen too little data near this state. Here is that idea worked out.