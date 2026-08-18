# Sources — encodec (svfix, track D_candidate)

## Search log
Queries/venues tried before landing on the grounding source below:
1. `grep -ril` for balancer/gradient/EMA terms across methods/encodec/{refs,src,notes} — none existed yet (track D: only the primary `src/*.tex` was on disk).
2. `grep -i encodec SELF_ACCOUNT_SOURCES.md` — no hit.
3. OpenReview `api.openreview.net/notes/search` with terms "High Fidelity Neural Audio Compression", "EnCodec high fidelity neural audio", and `group=TMLR` filter — surfaced only a dblp bibliographic mirror note (id `WiI8lSEaCr`), no TMLR review/rebuttal notes indexed under this term; rate-limited (429) repeatedly.
4. `huggingface.co/papers/2210.13438` — abstract page only, no discussion comments from authors.
5. Alexandre Défossez's personal site `ai.honu.io` → downloaded his PhD manuscript ("Optimization of Fast Deep Learning Network for Audio Analysis and Synthesis") — predates EnCodec (no "encodec" mentions anywhere in the text); searched for "balanc"/"gradient" — no coverage of the multi-loss gradient-balancer idea.
6. `gh api` search of GitHub issues on `facebookresearch/encodec`: issue #18 "Real-world Balancer usage question" (adefossez reply: "The balancer is only used for losses that are defined with respect to the output of the model. Other losses contributions must be computed in a separate backward." — corroborates the commitment-loss exception already in reasoning.md, but not new grounding for the "wall"), issue #47, PR #67, issue #38 (third-party speculation only, not from adefossez) — none document the decisive failure mode itself.
7. **`git log`/`git show` on a local clone of `facebookresearch/encodec`** — found commit `30838f8` ("adding balancer for reference", authored by Alexandre Defossez, the paper's first author, 2022-10-26) which adds `encodec/balancer.py` "for reference" (per the same commit's README diff: "For reference, we also provide the code for our novel MS-STFT discriminator and the balancer."). This file contains the author's own `test()` function — a de-facto lab-notebook demonstration, in code, of exactly the failure the balancer fixes.

## Load-bearing source

**Type:** self-account (author-authored demonstration code in the official release repo, not the paper text itself — the paper only states the balancer's final form, not this worked failure case).
**Title/commit:** `encodec/balancer.py`, commit `30838f83b744b096261a03e26beca07dcf0269c4`, "adding balancer for reference", Alexandre Defossez, 2022-10-26.
**URL:** https://github.com/facebookresearch/encodec/commit/30838f83b744b096261a03e26beca07dcf0269c4
**Local path:** `methods/encodec/refs/encodec_balancer_ref_commit_30838f8.py` (+ `.meta.txt` for commit metadata)

Quoted passage (the author's own unit test, verbatim from the file):

```python
def test():
    from torch.nn import functional as F
    x = torch.zeros(1, requires_grad=True)
    one = torch.ones_like(x)
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}

    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=False)
    balancer.backward(losses, x)
    assert torch.allclose(x.grad, torch.tensor(99.)), x.grad

    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}
    x.grad = None
    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=True)
    balancer.backward({'1': loss_1, '2': loss_2}, x)
    assert torch.allclose(x.grad, torch.tensor(0.)), x.grad
```

**Why this grounds the decisive step:** the rewritten "wall" passage in reasoning.md needs a *concrete, checkable* demonstration that equal nominal weights (λ₁=λ₂=1) do not mean equal contribution once two losses differ in raw gradient scale — the exact mechanism that makes fixed-λ training destabilize when the discriminator gradient spikes. This test is the author's own numeric proof of precisely that: `l1 = |x-1|` has gradient −1 at x=0; `l2 = 100·|x+1|` has gradient +100 at the same point (standing in for a spiking discriminator loss next to a steady reconstruction loss); with equal weights and no rescaling the combined gradient is 99 (i.e. ~99% loss_2, loss_1's signal essentially erased) — exactly asserted as `99.` in the author's own test. With the balancer's rescaling turned on, the same pair collapses to `0.` — matching the reasoning.md derivation `g̃_i = R·(λ_i/Σλ_j)·(g_i/⟨‖g_i‖⟩_β)` term-for-term (λ₁=λ₂=1, Σλ=2, R=1 — the same `total_norm=1.` default in this file — gives g̃_1=-0.5, g̃_2=+0.5, sum 0). This is not a name-drop: the numbers in the rewritten passage are recomputed from and cross-checked against this file's own assertions.

## Errors corrected
None found in this pass; the RVQ/balancer math in reasoning.md was already numerically self-consistent. No changes to answer.md/train_answer.md/code were needed.

## Repair-pass re-verification (2026-08-18)
Prior pass rejected with reason "verifier unavailable" (no specific defect list). Re-ran the full verifier checklist independently:
- Re-fetched the live commit (`curl https://github.com/facebookresearch/encodec/commit/30838f83b744b096261a03e26beca07dcf0269c4.patch`) and diffed its `test()` body against the local `.py` copy byte-for-byte — identical; the saved source is not a fabrication.
- Re-derived the numbers by hand: `g_1 = ∂|x-1|/∂x|_{x=0} = -1`, `g_2 = ∂(100|x+1|)/∂x|_{x=0} = 100`; unweighted sum `-1+100=99` matches the source's `assert torch.allclose(x.grad, torch.tensor(99.))`. Rebalanced: `g̃_1 = 1·(1/2)·(-1/1) = -0.5`, `g̃_2 = 1·(1/2)·(100/100) = 0.5`, sum `0`, matching the source's second assert (`0.`). Both numbers in the reasoning.md passage trace to these asserts, not a name-drop.
- `python3 tools/lint_inframe.py | grep methods/encodec/` → no hits; `grep -n -i "the paper\|the authors\|arxiv\|et al\."` → no hits; `grep -n "Wait\|Alternatively\|Hmm"` → no hits.
- Checked answer.md / train_answer.md balancer sections against the rewritten reasoning.md step — consistent, no conflicting numbers, no edits needed there.
- `git status --short methods/encodec/` clean before this edit; diff scope confirmed limited to this method.
No defect found; this pass only adds the re-verification record.
