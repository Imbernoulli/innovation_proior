The problem is to minimize a finite-sum objective f(w) = sum_k p_k F_k(w), with p_k = n_k / n, over a network of devices that are not allowed to move their raw data off-device. Two features make this hard. First, the local data distributions D_k are non-identically distributed, so the local risks F_k differ from one another and from the global objective; the minimizer of any single F_k can be far from the minimizer of f. Second, the devices themselves are heterogeneous in compute, memory, battery, and connectivity, so they cannot all be expected to perform the same amount of work in a fixed round. Communication is the bottleneck, which pushes the design toward doing many local steps per round and involving only a small fraction of devices each round.

The standard recipe is FedAvg: broadcast the current model w^t, let each selected device run several epochs of local SGD on its own F_k, and average the returned models weighted by sample count. That averaging only makes sense when the local solutions stay in a common basin, which is true when everyone starts from the same w^t and the local data are similar. Under non-IID data, however, more local epochs drive each device toward its own local optimum, so the very knob that reduces communication rounds amplifies client drift and can make the averaged model diverge. FedAvg also handles systems heterogeneity crudely: a device that cannot finish its quota is dropped, wasting its partial work and biasing the effective sampling distribution if stragglers have distinctive data. What is needed is a way to cap drift directly and to let devices contribute partial solutions instead of being excluded.

The method is FedProx. It keeps the FedAvg server-side aggregation unchanged but changes the local objective on each device. Instead of minimizing F_k(w), device k approximately minimizes

h_k(w; w^t) = F_k(w) + (mu/2) ||w - w^t||^2.

The added term is a quadratic spring anchored at the broadcast model w^t. Its gradient is mu(w - w^t), so a local SGD step on h_k becomes w <- w - eta (grad F_k(w) + mu (w - w^t)) — ordinary local SGD plus a restoring force that pulls the parameters back toward w^t. Because the spring acts on the objective itself, it tethers drift regardless of which local solver is used or how many steps the device takes; setting mu = 0 recovers FedAvg exactly. The same quadratic also convexifies the local subproblem: if F_k has negative curvature bounded below by -L_ I, then the Hessian of h_k is bounded below by (mu - L_) I, so choosing mu > L_ makes h_k strongly convex even when F_k is non-convex. That strong convexity gives a clean displacement bound, ||argmin h_k - w^t|| <= (1/(mu - L_)) ||grad F_k(w^t)||, which is the workhorse of the convergence analysis.

FedProx also handles variable device effort without dropping stragglers. A returned model w_k is called gamma-inexact for h_k if ||grad h_k(w_k; w^t)|| <= gamma ||grad F_k(w^t)||, with gamma in [0,1]. Gamma equals zero for an exact local minimizer and is close to one for a barely-started solve. A slow device simply returns a solution with larger gamma; its partial work is aggregated into the server average rather than discarded. The number of local epochs is just a proxy for gamma, so the formalism covers per-device, per-round variation in compute. This is a lightweight client-side change: the server still computes a sample-weighted mean of the returned models, and no full global gradient is needed, unlike DANE-style gradient-correction schemes that fail under low participation.

The convergence story is as follows. Under L-smooth F_k, a negative-curvature bound L_, mu > L_, and a bounded dissimilarity E_k ||grad F_k(w)||^2 <= B^2 ||grad f(w)||^2 measuring statistical heterogeneity, one round of FedProx decreases the global objective in expectation by rho ||grad f(w^t)||^2 for an explicit rho that depends on mu, L, L_, B, gamma, and the number of sampled devices K per round. Two qualitative constraints appear: gamma B < 1, meaning very sloppy local solves are only safe when the network is relatively homogeneous, and B / sqrt(K) < 1, meaning higher heterogeneity requires more devices per round. Telescoping the per-round decrease gives convergence to an epsilon-approximate stationary point in O(Delta / (rho epsilon)) rounds, where Delta = f(w^0) - f^*. In the convex case with exact solves, choosing mu proportional to L B^2 recovers the SGD complexity up to constants, showing that FedProx matches distributed SGD asymptotically while being much more robust to heterogeneous clients.

```python
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.training import optimizer
from .fedbase import BaseFedarated


class PerturbedGradientDescent(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
        self._lr_t = None
        self._mu_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="prox_mu")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")
        var_update = state_ops.assign_sub(
            var, lr_t * (grad + mu_t * (var - vstar))
        )
        return control_flow_ops.group(var_update)

    def set_params(self, global_params, client):
        with client.graph.as_default():
            for variable, value in zip(tf.trainable_variables(), global_params):
                self.get_slot(variable, "vstar").load(value, client.sess)


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        self.inner_opt = PerturbedGradientDescent(
            params["learning_rate"], params["mu"]
        )
        super(Server, self).__init__(params, learner, dataset)

    def train_round(self, round_num):
        _, selected_clients = self.select_clients(
            round_num, num_clients=self.clients_per_round
        )
        np.random.seed(round_num)
        active_clients = np.random.choice(
            selected_clients,
            round(self.clients_per_round * (1 - self.drop_percent)),
            replace=False,
        )
        self.inner_opt.set_params(self.latest_model, self.client_model)

        client_solutions = []
        for client in selected_clients.tolist():
            client.set_params(self.latest_model)
            if client in active_clients:
                solution, stats = client.solve_inner(
                    num_epochs=self.num_epochs, batch_size=self.batch_size
                )
            else:
                partial_epochs = np.random.randint(low=1, high=self.num_epochs)
                solution, stats = client.solve_inner(
                    num_epochs=partial_epochs, batch_size=self.batch_size
                )
            client_solutions.append(solution)
            self.metrics.update(rnd=round_num, cid=client.id, stats=stats)

        self.latest_model = self.aggregate(client_solutions)
        self.client_model.set_params(self.latest_model)

    def aggregate(self, weighted_solutions):
        total_weight = 0.0
        base = [0] * len(weighted_solutions[0][1])
        for weight, solution in weighted_solutions:
            total_weight += weight
            for i, value in enumerate(solution):
                base[i] += weight * value.astype(np.float64)
        return [value / total_weight for value in base]
```
