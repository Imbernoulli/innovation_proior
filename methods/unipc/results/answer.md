# UniPC, distilled

UniPC (Zhao et al., NeurIPS 2023, arXiv:2302.04867) is a training-free **unified predictor-corrector**
framework for fast diffusion sampling. It adds a *corrector* (UniC) on top of a multistep *predictor*
(UniP) that share one analytical update of *arbitrary order*, obtained by solving a small linear system
in the half-log-SNR ratios. Because the corrector reuses the network evaluation the *next* step already
takes, it raises the realized order by one at **no extra NFE** — which is what gives UniPC its edge over
DPM-Solver++(kM) in the extreme few-step regime (5–10 steps).

## Problem it solves

Fast guided sampling of a pre-trained diffusion model at very small NFE. Existing fast solvers
(DPM-Solver, DPM-Solver++, DEIS) are *predictor-only*: each step commits to an extrapolated estimate
with its leading truncation error uncorrected, and order alone gives diminishing — and under large
guidance, destabilizing — returns. A predictor-corrector pair buys a cheap extra order, but correctors
were absent from fast diffusion solvers because each order is a separate hand-derivation.

## Key idea

**Free corrector via multistep reuse.** In a multistep loop the predictor's base evaluation at step
`i+1` is the network applied at the point step `i` predicted — exactly what a corrector for step `i`
needs. So correct the previous step with the current evaluation: zero extra network calls.

**One update at arbitrary order.** Both UniP and UniC are the same data-prediction step
`x_t = (sigma_t/sigma_{s0}) x - alpha_t h_phi_1 m0 - alpha_t B_h * (correction)`, where the correction
is a linear combination `sum rho_k D1_k` of scaled finite differences `D1_k = (m_k - m0)/r_k` of past
data predictions, with ratios `r_k = (lambda_{s_k} - lambda_{s0})/h`, `h = lambda_t - lambda_{s0}`,
`hh = -h` (data-prediction convention), `h_phi_1 = e^{hh} - 1`. The coefficients `rho` solve `R rho = b`
with `R_i = r^{i-1}` and `b_i` the `phi`-derived sequence (`h_phi_k * i!/B_h`, advanced by
`h_phi_k <- h_phi_k/hh - 1/(i+1)!`) — matching the exact integral's Taylor expansion to the available
order, for *any* order.

- **Predictor (UniP)**, order `p`: reduced solve `rho_p = solve(R[:-1,:-1], b[:-1])` (order 2 →
  `rho_p = 0.5`); `x_t = x_base - alpha_t B_h (rho_p @ D1s)`.
- **Corrector (UniC)**, order `p`: after evaluating `m_t` at the predicted point, full solve
  `rho_c = solve(R, b)` (order 1 → `rho_c = 0.5`); `x_t = x_base - alpha_t B_h (rho_c[:-1] @ D1s +
  rho_c[-1] (m_t - m0))`.

**The `B(h)` knob.** Free scalar on the correction: `bh1` → `B = hh`, `bh2` → `B = e^{hh} - 1`. `bh2`
tracks the exponential weight better at large `h`; default for tight budgets.

## Why it works

- **Corrector = +1 order, +0 NFE.** Adams–Moulton-style refinement using an evaluation the loop already
  makes; the realized order is one above the predictor at the same call count.
- **Unified linear-system form.** Predictor and corrector at any order from one template (reduced vs
  full solve), so "add a corrector" and "raise the order" are one change — which is why a corrector is
  finally practical here.
- **Data prediction + `expm1` + Karras/uniform-`lambda` grid:** guidance stability, numerically honest
  small-`h` factors, budget spent where truncation error lives. Latent models: thresholding off.

## Final algorithm (per step, one network call)

```
for i = 0 .. N-1:
    m_t = predict(x, sigma_i)                       # the only network call this step
    if i > 0:                                        # correct the previous predictor step (free)
        order_c = min(max_order, len(m_list))
        x = UniC(x_prev, s_prev, sigma_i, m_list, lam_list, m_t, order_c)
    push m_t, lam(sigma_i) to history
    if i == N-1: x = m_t; break                      # final: return clean prediction
    order_p = min(max_order, len(m_list)); if lower_order_final: order_p = min(order_p, N-i)
    x_prev, s_prev = x, sigma_i
    x = UniP(x, sigma_i, sigma_{i+1}, m_list, lam_list, order_p)
return x
```

## Working code

Faithful to the canonical UniPC (`solver_type="bh2"`, `predict_x0=True`); `predict(x, sigma)` returns
the guided data prediction `x_theta`.

```python
import torch


class UniPCBH:
    def __init__(self, ns, predict, solver_type="bh2", predict_x0=True):
        self.ns, self.predict = ns, predict
        self.solver_type, self.predict_x0 = solver_type, predict_x0

    def _R_b(self, rks, hh, B_h, order):
        h_phi_1 = torch.expm1(hh)
        R, b, h_phi_k, fact = [], [], h_phi_1 / hh - 1, 1
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * fact / B_h)
            fact *= i + 1
            h_phi_k = h_phi_k / hh - 1 / fact
        return torch.stack(R), torch.stack(b), h_phi_1

    def _bh(self, hh):
        return hh if self.solver_type == "bh1" else torch.expm1(hh)

    def _ratios(self, x, s0, m_list, lam_list, lam_s0, h, order):
        rks, D1s = [], []
        for i in range(1, order):
            rk = (lam_list[-(i + 1)] - lam_s0) / h
            rks.append(rk)
            D1s.append((m_list[-(i + 1)] - m_list[-1]) / rk)
        rks.append(torch.ones((), device=x.device))
        return torch.stack(rks), D1s

    def predictor(self, x, s0, t, m_list, lam_list, order):
        ns = self.ns
        lam_s0, lam_t = ns.lam(s0), ns.lam(t)
        h = lam_t - lam_s0
        hh = -h if self.predict_x0 else h
        B_h, m0 = self._bh(hh), m_list[-1]
        rks, D1s = self._ratios(x, s0, m_list, lam_list, lam_s0, h, order)
        R, b, h_phi_1 = self._R_b(rks, hh, B_h, order)
        x_base = (ns.sigma(t) / ns.sigma(s0)) * x - ns.alpha(t) * h_phi_1 * m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = (torch.tensor([0.5], dtype=x.dtype, device=x.device) if order == 2
                      else torch.linalg.solve(R[:-1, :-1], b[:-1]))
            pred_res = torch.einsum("k,bk...->b...", rhos_p, D1s)
        else:
            pred_res = 0
        return x_base - ns.alpha(t) * B_h * pred_res

    def corrector(self, x_prev, s0, t, m_list, lam_list, model_t, order):
        ns = self.ns
        lam_s0, lam_t = ns.lam(s0), ns.lam(t)
        h = lam_t - lam_s0
        hh = -h if self.predict_x0 else h
        B_h, m0 = self._bh(hh), m_list[-1]
        rks, D1s = self._ratios(x_prev, s0, m_list, lam_list, lam_s0, h, order)
        R, b, h_phi_1 = self._R_b(rks, hh, B_h, order)
        x_base = (ns.sigma(t) / ns.sigma(s0)) * x_prev - ns.alpha(t) * h_phi_1 * m0
        rhos_c = (torch.tensor([0.5], dtype=x_prev.dtype, device=x_prev.device) if order == 1
                  else torch.linalg.solve(R, b))
        D1_t = model_t - m0
        if D1s:
            D1s = torch.stack(D1s, dim=1)
            corr_res = torch.einsum("k,bk...->b...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        return x_base - ns.alpha(t) * B_h * (corr_res + rhos_c[-1] * D1_t)

    @torch.no_grad()
    def sample(self, x, sigmas, max_order=3, lower_order_final=True):
        ns, N = self.ns, len(sigmas) - 1
        m_list, lam_list, x_prev, s_prev = [], [], None, None
        for i in range(N):
            sigma = sigmas[i]
            model_t = self.predict(x, sigma)                       # only call this step
            if x_prev is not None:
                order_c = min(max_order, len(m_list))      # past points available for the corrector
                x = self.corrector(x_prev, s_prev, sigma, m_list, lam_list, model_t, order_c)
            m_list.append(model_t)
            lam_list.append(ns.lam(sigma))
            if i == N - 1:
                x = m_list[-1]
                break
            order_p = min(max_order, len(m_list))
            if lower_order_final:
                order_p = min(order_p, N - i)
            x_prev, s_prev = x, sigma
            x = self.predictor(x, sigma, sigmas[i + 1], m_list, lam_list, order_p)
        return x
```

## Relation to prior methods

- **DPM-Solver++(kM)** is the predictor-only multistep solver UniP generalizes; UniPC adds the free
  corrector (UniC) for +1 realized order at the same NFE.
- **DDIM** is the first-order base step (`order = 1`).
- **DEIS** is a noise-face multistep predictor; UniPC's data-face predictor-corrector is more stable
  under guidance and corrects each step.
- The grid is the **Karras/EDM** power schedule (`rho = 7`) or uniform-`lambda`.
