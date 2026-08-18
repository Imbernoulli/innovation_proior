# Source-value fix — marcus-electron-transfer (track B_selfaccount)

Self-account already on file and already load-bearing for most of the trace: `refs/marcus-nobel-lecture-1992.txt` (Rudolph A. Marcus, Nobel Lecture, "Electron Transfer Reactions in Chemistry: Theory and Experiment," 1992). Catalogued in `notes/synthesis.md`. This entry records the specific quote used in this svfix pass to additionally ground the equal-energy decisive step (the audit flagged this step as sourced only to primary fn.6 + Eqs.33/38).

## Quote used

> "These two potential energy surfaces will intersect if the electronic coupling which leads to electron transfer is neglected. For a system with N coordinates this intersection occurs on an (N-1) dimensional surface, which then constitutes in our approximation the transition state of the reaction... In classical terms, the transfer at S occurs at fixed positions and momenta of the atoms, and so the Franck-Condon principle is satisfied. Since U_r equals U_p at S, energy is also conserved."
> (`refs/marcus-nobel-lecture-1992.txt`, lines ~294-309.)

## Where used
- `results/reasoning.md`, new paragraph inserted after the "wrong environment... nonequilibrium polarization" paragraph — gives the physical crossing-surfaces picture (Marcus's own later restatement) as a second, geometric route to the same E_X = E_X* condition that the primary paper reaches via the wavefunction eigenstate argument (φ_X + cφ_X*). Framed as the scientist holding both views together, matching the trace's established pattern of cross-checking a derivation by an independent route (already used a few paragraphs later for the free-energy functional).

## Broken download removed
- `refs/marcus-reflections-2020.pdf` was not actually a PDF: it was a Cloudflare "Just a moment..." challenge HTML page (5.7KB) saved with a `.pdf` extension. It was never cited in `notes/synthesis.md`'s source list and not load-bearing anywhere. Deleted per the broken-download rule. Not re-fetched: the Nobel lecture and 1964 Annual Review self-account/retrospective already on file are sufficient and the URL for this specific "reflections 2020" item was not recorded anywhere in the method's notes to retry.
