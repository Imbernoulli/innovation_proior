**Problem (from step 3).** Karras edged the aggregate (gmean 7.752) but only because DIODE's gain
(15.0 → 8.77) barely outweighed an ImageNet *regression* (6.07 → 12.03). Its monotone `ρ = 7` warp
starves the high-`t` region — fine for DIODE's smooth VP bridge, ruinous for ImageNet's I2SB inpainting
bridge, whose hard mask must be resolved at high `t`. Every schedule so far is monotone — a single
directional bet — but both ends of the trajectory are hard: high `t` establishes conditioning structure,
low `t` commits image content.

**Key idea.** Stop betting on one end. Use a symmetric cosine warp that resolves *both* endpoints densely
and spends the coarse steps in the bend-free middle:
`t_i = t_max + (t_min − t_max)·(1 − cos(π·i/n))/2`. Derived from four conditions on the warp `g(u)`:
`g(0) = 0`, `g(1) = 1`, `g'(0) = 0`, `g'(1) = 0` (zero slope at both ends → nodes bunch there). The
half-cosine is the unique parameter-free S-curve satisfying all four — no exponent, no period to tune.

**Why it works.** Small slope near `u = 0` and `u = 1` puts dense nodes at `t_max` (conditioning) and
`t_min` (content); large slope at the middle spends coarse steps where the bridge is nearly straight. For
`n = 4` the schedule is `{1.0, 0.854, 0.500, 0.147, 1e-4}` — steps `{0.146, 0.354, 0.354, 0.147}`, small
at both ends. It returns the high-`t` node (0.854) that karras starved, recovering the mask resolution.
The geometric-mean aggregate punishes the worst workload hardest, so a schedule that is *good at both ends
everywhere* beats one *excellent at low `t` on two workloads and broken on the third*.

**Scaffold edit / hyperparameters.** None beyond `n`. `g(0) = 0`, `g(1) = 1` are exact, so — like uniform
— **no terminal pin is needed**; terminal equals `t_min` for free. Strictly decreasing (`g` strictly
increasing), length `n + 1`, on `device`, all by construction. (Note: at `n = 2` the lone interior node
sits at `g(0.5) = 0.5`, so cosine ≡ uniform on DIODE — the symmetric shape only acts at NFE = 5, where
there are two interior nodes.)

**What to watch.** ImageNet should move most: karras's 12.025 was high-`t` starvation, so the dense high-`t`
node should drop it sharply, back below uniform's 6.070. edges2handbags should hold near karras's 4.416.
DIODE is schedule-neutral (cosine ≡ uniform at `n = 2`), so no curve-driven claim there. The aggregate
claim: cosine clears karras's 7.752 *comfortably*, driven almost entirely by fixing the ImageNet
factor-of-two, with DIODE schedule-unchanged and edges roughly flat.

```python
def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    import math
    ramp = torch.linspace(0, 1, n + 1)
    cosine_ramp = (1 - torch.cos(ramp * math.pi)) / 2
    sigmas = t_max + (t_min - t_max) * cosine_ramp
    return sigmas.to(device)
```
