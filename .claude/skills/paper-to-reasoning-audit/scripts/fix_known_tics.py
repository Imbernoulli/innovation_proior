#!/usr/bin/env python3
"""Precise one-off fixes for the 11 meta-tics the lint surfaced. Idempotent."""
import os
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (relpath, old, new). graphsage 'the paper abstract' is a dataset reference -> intentionally NOT fixed.
FIXES = [
    ("beit/results/answer.md", "## Code (faithful to the canonical implementation)", "## Code"),
    ("s4/results/answer.md", "## Code (faithful to the canonical implementation)", "## Code"),
    ("vqgan/results/answer.md", "## Code (faithful to the canonical implementation)", "## Code"),
    ("methods/spacy/results/answer.md", "## Code (faithful to the implementation)", "## Code"),
    ("classifier-free-guidance/results/context.md",
     "cosine-style schedule (known)", "cosine-style schedule"),
    ("classifier-free-guidance/results/context.md",
     "variance interpolation (known)", "variance interpolation"),
    ("dueling-dqn/results/context.md",
     'evaluation protocols. (Settings only.)', 'evaluation protocols.'),
    ("rectified-flow/results/answer.md",
     "Velocity-field training (faithful to the canonical implementation: linear interpolation, target $x_1-x_0$, plain L2).",
     "Velocity-field training: linear interpolation, target $x_1-x_0$, plain L2."),
    ("methods/adam-slow-sde/results/answer.md",
     "No official repository accompanies this analysis; the code below is grounded",
     "The code below is grounded"),
    ("biggan/results/answer.md",
     "The paper also characterizes large-scale training instability:",
     "Large-scale training is also characterized by an instability:"),
    ("chebnet/results/context.md",
     "a bottleneck the authors themselves identified.",
     "a well-known bottleneck of this approach."),
]
for rel, old, new in FIXES:
    p = os.path.join(base, rel)
    with open(p, encoding="utf-8") as f:
        s = f.read()
    if old in s:
        f_new = s.replace(old, new)
        with open(p, "w", encoding="utf-8") as f:
            f.write(f_new)
        print(f"FIXED  {rel}  ({s.count(old)}x)")
    elif new in s:
        print(f"already {rel}")
    else:
        print(f"!! NOT FOUND in {rel}: {old[:60]!r}")
