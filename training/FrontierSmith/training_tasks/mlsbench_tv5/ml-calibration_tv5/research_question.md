Each of the four classifier-dataset pairs delivers its probabilities with a
different, mutually incompatible geometry. The random forest emits vote
ratios carrying exact zeros and ones —
infinitely wrong under log-loss whenever such a certainty misses. The neural
network produces smooth softmax outputs stretched toward overconfidence. The
boosted trees yield scores whose distortion follows staged additive fitting.
And the SVM arrives pre-squashed — a sigmoid was already fitted to its
margins upstream — so a second aggressive re-mapping tends to overcorrect.
Add two class regimes, ten-way with plentiful per-class data and binary with
little, and any mapping carrying one family's assumptions breaks on another
family's outputs.

The objective here is a calibrator that is agnostic to output geometry. No
detection-and-branching per classifier, no per-setting configuration: one
method whose internal representation absorbs all four shapes. That forces
specific design decisions the scaffold only gestures at — which transform
domain makes a vote ratio, a softmax, and a squashed margin commensurable;
how boundary mass at exact zero and one is handled without distorting the
interior; when per-class corrections are affordable and when classes must
share statistical strength; and how the same machinery degrades from ten
classes to two.

Uniformity is the claim to defend on the unchanged protocol: the method
improves — or at minimum never worsens — ECE, Brier and NLL on every one of
the four classifier-dataset pairs, whereas each reference baseline fails on
at least one of them. The scaffold clips inputs away from the boundary, fits
an independent logistic correction per class in the logit domain,
renormalises, and records a geometry signature it never reads; its per-class
independence and fixed clipping are exactly the assumptions to interrogate.
