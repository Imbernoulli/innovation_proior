Sampling a trained diffusion probabilistic model means turning an initial noise vector into a clean sample by repeatedly querying a noise-prediction network. The standard ancestral sampler follows the reverse-time SDE and needs hundreds or thousands of sequential network evaluations because each step injects new Brownian noise and is only a first-order correction. Generic adaptive ODE solvers do better on the deterministic probability-flow ODE, but they still need on the order of sixty evaluations and fall apart when pushed toward the ten-evaluation regime. The reason is that they treat the whole right-hand side of the ODE as an opaque black box, wasting error budget on a part that can be solved exactly and becoming unstable when the step size grows.

The better approach is to exploit the structure of the probability-flow ODE. It is semi-linear: the term involving the data state is linear in the state and schedule-dependent, while the remaining term is the neural network. Solving the linear part exactly with variation of constants and changing the integration variable from time to the half-log-SNR leaves a single exponentially-weighted integral of the network output. Taylor-expanding the network in this log-SNR coordinate and integrating term by term gives a family of solvers, DPM-Solver, with orders one, two, and three. The first-order member is exactly DDIM, which explains why DDIM already outperformed plain Euler discretizations, and the higher-order members add one or two inexpensive intermediate network evaluations per step to model how the prediction changes across wider steps. The time grid is spaced uniformly in the half-log-SNR rather than in raw time, which is the natural coordinate once the schedule-dependent terms are handled analytically. With third-order steps where the budget allows and lower-order fallback steps to hit an exact evaluation count, DPM-Solver reaches high sample quality in roughly ten to twenty network calls without retraining the model.

```python
import torch


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule, algorithm_type="dpmsolver"):
        assert algorithm_type == "dpmsolver"
        self.model_fn = model_fn              # noise prediction epsilon_theta(x, t)
        self.noise_schedule = noise_schedule  # marginal_log_mean_coeff/std/lambda/inverse_lambda

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            lambdas = torch.linspace(lambda_T.item(), lambda_0.item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(lambdas)
        if skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        if skip_type == "time_quadratic":
            return torch.linspace(t_T ** 0.5, t_0 ** 0.5, N + 1).pow(2).to(device)
        raise ValueError("skip_type must be logSNR, time_uniform, or time_quadratic")

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
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

        if skip_type == "logSNR":
            timesteps = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            base = self.get_time_steps(skip_type, t_T, t_0, steps, device)
            timesteps = base[torch.cumsum(torch.tensor([0] + orders), 0).to(device)]
        return timesteps, orders

    def dpm_solver_first_update(self, x, s, t, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)
        if model_s is None:
            model_s = self.model_fn(x, s)
        phi_1 = torch.expm1(h)
        return torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        s1 = ns.inverse_lambda(lambda_s + r1 * h)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)
        if model_s is None:
            model_s = self.model_fn(x, s)
        phi_11, phi_1 = torch.expm1(r1 * h), torch.expm1(h)
        x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
        model_s1 = self.model_fn(x_s1, s1)
        return (torch.exp(log_alpha_t - log_alpha_s) * x
                - sigma_t * phi_1 * model_s
                - (0.5 / r1) * sigma_t * phi_1 * (model_s1 - model_s))

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        s1 = ns.inverse_lambda(lambda_s + r1 * h)
        s2 = ns.inverse_lambda(lambda_s + r2 * h)
        log_alpha_s = ns.marginal_log_mean_coeff(s)
        log_alpha_s1 = ns.marginal_log_mean_coeff(s1)
        log_alpha_s2 = ns.marginal_log_mean_coeff(s2)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        if model_s is None:
            model_s = self.model_fn(x, s)
        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
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
```
