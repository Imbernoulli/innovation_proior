# Changelog

## 2026-08-18 — svfix(W3_primary_only)

**Error found:** the decisive step in `reasoning.md` (nonvanishing of \(\zeta\) on
\(\operatorname{Re}s=1\)) was attributed to Hadamard's 1896 article and presented the
now-standard textbook argument — the trigonometric identity \(3+4\cos u+\cos 2u\ge0\)
applied to \(|\zeta(\sigma)^3\zeta(\sigma+it)^4\zeta(\sigma+2it)|\) — as if that were
Hadamard's own proof. It is not. `refs/hadamard-1896-bsmf.txt` (Numdam OCR of the
primary article, §§2–5) shows Hadamard's actual 1896 argument is a different
mechanism: he tracks the real-variable sum \(S(s)=\sum_p p^{-s}\) and its cosine-weighted
analogue \(P(s)=\sum_p\cos(t\log p)p^{-s}\), shows a hypothetical zero at \(1+it\) forces
\(P(s)/S(s)\to-1\) (hence weight concentration on primes with \(t\log p\) near an odd
multiple of \(\pi\)), then doubles the frequency to \(2t\) and shows that same
concentration would force a *pole* of \(\zeta\) at \(1+2it\) — impossible, since \(\zeta\)'s
only singularity on \(\operatorname{Re}s\ge1\) is the simple pole at \(s=1\). The
3+4cos-identity proof (found instead in `refs/prime-number-theorem-wikipedia.html`,
"Non-vanishing on Re(s)=1" section) is a later, unattributed-in-that-source simplification,
not what is in the primary paper.

**Fix:** rewrote the decisive-step paragraphs of `results/reasoning.md` to run through
Hadamard's actual concentration-then-doubling mechanism, grounded in
`refs/hadamard-1896-bsmf.txt`. Propagated the same correction to
`results/train_answer.md` (paragraph deriving the zero-free line, and the one-line
summary in the closing theorem-recap paragraph) so the discovery write-up matches what
`reasoning.md` now actually derives. `results/answer.md` and `results/context.md` did not
name the specific mechanism and needed no change. The landing (PNT statement, Tauberian
bridge, final asymptotics) is unchanged — only the justification of the zero-free-line
step was wrong.
