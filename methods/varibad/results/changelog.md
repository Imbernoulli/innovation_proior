# VariBAD changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, the decisive "what should the decoder reconstruct" step: replaced the
  hand-wave ("if I literally apply a vanilla VAE here I'll get something underwhelming") with the
  documented failure accounting for each decoding target, grounded in the authors' own extended
  JMLR treatment §6.1 / Figs 6–7 (`refs/varibad_jmlr_v22.pdf`, quotes in `notes/sources.md`), which
  the ICLR primary does not contain — the primary only asserts "decoding not only the past but also
  the future is important". The trace now runs through both real holes: decoding the past only is
  minimised by predicting no reward everywhere because unvisited cells carry no loss term at all;
  decoding the future only leaves spurious rewards on already-visited cells because those are
  unlikely to be revisited and so are never penalised; next-step-only penalises too narrow a slice.
  The union (past + future) is what closes both, with a falsifiable belief plot as the check.
- No factual errors found; the ELBO, decoder factorisation, landing and code are unchanged.
