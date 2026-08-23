Dominated volume is a geometric quantity with sharply uneven stakes. The scorer fixes a
reference point just beyond the objective ranges; every member of the final set then owns an
exclusive region — volume nothing else dominates — and those regions are far from equal. The
objective-wise extreme members pin slabs that no interior point can replace: lose the low-f1
end of the front and a rectangle of score vanishes wholesale, while an equally sized lapse in
the interior costs only a sliver. This variant asks for an algorithm that keeps its books in
exactly those terms.

Concretely, survival credit should be proportional to exclusive volume contribution rather
than to distance-based density. In two objectives the contribution of an interior member is a
closed-form rectangle determined by its sorted neighbours and can be ranked exactly; in three
it must be proxied, and choosing that proxy well is part of the work. The per-objective
minimisers deserve seats before any ranking begins — they are the only members whose loss is
unrecoverable — and mating pressure should keep pushing them outward, since extending the
front's reach converts directly into volume that interior refinement can never manufacture.

The trap is written into the scoreboard itself: evenness is also reported, and a strategy that
hoards boundary mass while its interior thins will read beautifully on volume and terribly on
dispersion. Reference-front distance likewise punishes a boundary-heavy set that stops
tracking the true surface. So the emphasis must be graded, never absolute — anchor the
extremes, weight credit by contribution, and still lay the interior down evenly enough that
the other two numbers hold. All of it under the standing blindness: no problem names, no
per-instance settings, one implementation for two and three objectives at whatever budget the
spec hands over.

What must survive scrutiny is the credit assignment: which members earned their seats through
volume, what the extremes were worth over a run, and how the graded emphasis kept boundary
gains from becoming interior losses.
