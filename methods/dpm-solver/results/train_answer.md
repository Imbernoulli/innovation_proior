Sampling a trained diffusion probabilistic model means turning an initial noise vector into a clean sample by repeatedly querying a noise-prediction network. The standard ancestral sampler follows the reverse-time SDE and needs hundreds or thousands of sequential network evaluations because each step injects new Brownian noise and is only a first-order correction. Generic adaptive ODE solvers do better on the deterministic probability-flow ODE, but they still need on the order of sixty evaluations and fall apart when pushed toward the ten-evaluation regime. The reason is that they treat the whole right-hand side of the ODE as an opaque black box, wasting error budget on a part that can be solved exactly and becoming unstable when the step size grows.

The better approach is to exploit the structure of the probability-flow ODE. It is semi-linear: the term involving the data state is linear in the state and schedule-dependent, while the remaining term is the neural network. Solving the linear part exactly with variation of constants and changing the integration variable from time to the half-log-SNR leaves a single exponentially-weighted integral of the network output. Taylor-expanding the network in this log-SNR coordinate and integrating term by term gives a family of solvers, DPM-Solver, with orders one, two, and three. The first-order member is exactly DDIM, which explains why DDIM already outperformed plain Euler discretizations, and the higher-order members add one or two inexpensive intermediate network evaluations per step to model how the prediction changes across wider steps. The time grid is spaced uniformly in the half-log-SNR rather than in raw time, which is the natural coordinate once the schedule-dependent terms are handled analytically. With third-order steps where the budget allows and lower-order fallback steps to hit an exact evaluation count, DPM-Solver reaches high sample quality in roughly ten to twenty network calls without retraining the model.

```python
import torch


class NoiseSchedule:
    """Variance-preserving schedule with closed-form lambda and inverse lambda."""
    def __init__(self, schedule="linear", beta_0=0.1, beta_1=20.0):
        self.schedule = schedule
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def marginal_log_mean_coeff(self, t):
        # log(alpha_t) for a continuous VP schedule.
        if self.schedule == "linear":
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        raise NotImplementedError

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_lambda(self, t):
        return self.marginal_log_mean_coeff(t) - torch.log(self.marginal_std(t))

    def inverse_lambda(self, lam):
        # Closed-form inverse for the linear VP schedule.
        a = 0.25 * (self.beta_1 - self.beta_0)
        b = 0.5 * self.beta_0
        # alpha = exp(-a t^2 - b t); lambda = log(alpha / sqrt(1-alpha^2)).
        # Solve numerically with Newton's method for robustness.
        t = torch.ones_like(lam) * 0.5
        for _ in range(8):
            alpha = torch.exp(-a * t ** 2 - b * t)
            f = torch.log(alpha / torch.sqrt(1 - alpha ** 2)) - lam
            dalpha = -alpha * (2 * a * t + b)
            df = (dalpha / alpha) + alpha * dalpha / (1 - alpha ** 2)
            t = t - f / df
        return torch.clamp(t, 0.0, 1.0)


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule):
        self.model_fn = model_fn              # noise prediction epsilon_theta(x, t)
        self.noise_schedule = noise_schedule  # provides alpha, sigma, lambda, inverse_lambda

    def get_time_steps(self, t_T, t_0, N, device):
        lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T, device=device))
        lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0, device=device))
        lambdas = torch.linspace(lambda_T.item(), lambda_0.item(), N + 1, device=device)
        return self.noise_schedule.inverse_lambda(lambdas)

    def get_orders_and_timesteps(self, steps, order, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
        elif order == 2:
            K = steps // 2 if steps % 2 == 0 else steps // 2 + 1
            orders = [2] * K if steps % 2 == 0 else [2] * (K - 1) + [1]
        elif order == 1:
            K, orders = steps, [1] * steps
        else:
            raise ValueError("order must be 1, 2, or 3")
        timesteps = self.get_time_steps(t_T, t_0, K, device)
        return timesteps, orders

    def first_order_update(self, x, s, t, model_s=None):
        ns = self.noise_schedule
        lambda_s = ns.marginal_lambda(s)
        lambda_t = ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        if model_s is None:
            model_s = self.model_fn(x, s)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)
        phi_1 = torch.expm1(h)
        return torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s

    def second_order_update(self, x, s, t, r1=0.5, model_s=None):
        ns = self.noise_schedule
        lambda_s = ns.marginal_lambda(s)
        lambda_t = ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        s1 = ns.inverse_lambda(lambda_s + r1 * h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s1 = ns.marginal_std(s1)
        sigma_t = ns.marginal_std(t)
        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)
        x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model_fn(x_s1, s1)
        return (torch.exp(log_alpha_t - log_alpha_s) * x
                - sigma_t * phi_1 * model_s
                - (0.5 / r1) * sigma_t * phi_1 * (model_s1 - model_s))

    def third_order_update(self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None):
        ns = self.noise_schedule
        lambda_s = ns.marginal_lambda(s)
        lambda_t = ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        s1 = ns.inverse_lambda(lambda_s + r1 * h)
        s2 = ns.inverse_lambda(lambda_s + r2 * h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_s2 = ns.marginal_log_mean_coeff(s2)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s1 = ns.marginal_std(s1)
        sigma_s2 = ns.marginal_std(s2)
        sigma_t = ns.marginal_std(t)
        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = phi_12 / (r2 * h) - 1.0
        phi_2 = phi_1 / h - 1.0
        x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model_fn(x_s1, s1)
        x_s2 = (torch.exp(log_alpha_s2 - log_alpha_s) * x
                - sigma_s2 * phi_12 * model_s
                - (r2 / r1) * sigma_s2 * phi_22 * (model_s1 - model_s))
        model_s2 = self.model_fn(x_s2, s2)
        return (torch.exp(log_alpha_t - log_alpha_s) * x
                - sigma_t * phi_1 * model_s
                - (1.0 / r2) * sigma_t * phi_2 * (model_s2 - model_s))

    def sample(self, x_T, t_T=1.0, t_0=0.001, steps=20, order=3):
        device = x_T.device
        timesteps, orders = self.get_orders_and_timesteps(steps, order, t_T, t_0, device)
        x = x_T
        model_prev = None
        for i in range(len(orders)):
            s = timesteps[i]
            t = timesteps[i + 1]
            if orders[i] == 1:
                x = self.first_order_update(x, s, t, model_s=model_prev)
                model_prev = None
            elif orders[i] == 2:
                x = self.second_order_update(x, s, t, model_s=model_prev)
                model_prev = None
            elif orders[i] == 3:
                x = self.third_order_update(x, s, t, model_s=model_prev)
                model_prev = None
            # Pass the just-computed prediction to the next step for multistep reuse.
            # Here we recompute it; a multistep variant would cache model_s instead.
        return x
```
