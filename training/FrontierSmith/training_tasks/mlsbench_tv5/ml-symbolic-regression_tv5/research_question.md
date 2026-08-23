The driver in this task is not negotiable: a fixed population, a fixed
number of generations, one expression collected at the end. Read that as a
hard evaluation meter — on the order of twenty-five thousand fitness calls —
and design the search as an anytime algorithm that could show a defensible
answer at every reading of the meter, not only at the final one. The
question is one of spend: given that every candidate evaluated is budget
irreversibly gone, what schedule of exploration and exploitation extracts
the most held-out accuracy per call?

The levers under study are allocation levers. Front-load diversity while the
meter is full: random immigrants, aggressive structural moves, broad parent
pools. Taper deliberately as it empties: narrow selection toward the
incumbent frontier, shift from structure-changing edits toward conservative
ones, and stop paying for evaluations that cannot change the outcome —
duplicate genotypes, offspring identical to their parents, candidates
dominated on sight. The per-generation feedback the harness already prints
(best fitness, average fitness, best size) is sufficient instrumentation:
an anytime schedule should produce an early, steep drop in best-so-far
fitness and a tail that keeps creeping instead of flatlining, and that
trajectory is the evidence to point at.

Scoring is unchanged — the final expression's held-out R2 — but the
argument is about the path. Show that the schedule beats a uniform,
schedule-free configuration of the same operators at the same meter, and
show where the gains arrive: early generations, late refinement, or waste
avoided in between. A method that merely ties the uniform baseline at the
buzzer has not answered the question this variant asks.
