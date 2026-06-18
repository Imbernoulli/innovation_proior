I start with the uncomfortable observation, not with an architecture. The usual answer to multivariate traffic forecasting is to make every layer spatial and temporal: graph convolution for the road network or a learned dependency graph, plus a recurrent or convolutional sequence module. That makes sense if the problem is inherently "compute messages over topology at every step." But the comparisons are getting more elaborate than the accuracy gaps they are chasing. So I want a sharper question: what does the graph-and-sequence stack give the regressor that a plain shared regressor lacks?

Take the simplest shared regressor and look at what it is forced to identify. It receives a recent value trace for a variable and maps that trace to a future trace. If sensor `i` and sensor `j` happen to have almost the same last `P` values, then the shared regressor sees almost the same input and must make almost the same prediction. That is not a capacity theorem; it is a representation problem. The two samples can have different futures because they are different variables in different parts of the system. The value trace alone does not say which one it is. The same problem happens in time. A twelve-step trace at morning buildup and a similar trace at evening recovery can ask for different continuations, but a regressor that only sees the trace has no clean way to know which phase it is in.

Now the graph models look different. A graph convolution does not merely mix neighbors; it gives a variable a context that depends on that variable's place in the system. A sequence module does not merely add depth; it gives the computation temporal context. So perhaps the graph is not the sacred object here. Perhaps the load-bearing thing is distinguishability: the model needs to know which variable the sample is from and which periodic time state it is anchored at. If I can supply that directly, I can test whether the heavy machinery was one route to this information rather than the necessary mechanism.

The most direct spatial marker is a learned table with one vector per variable. Let `E in R^{N x D}`. When I forecast variable `i`, I concatenate row `E_i` to its representation. Then two identical value traces from two different variables are no longer identical inputs to the downstream regressor. This is deliberately less structured than a graph. It does not tell the model who is upstream or downstream. It only gives each variable its own learned coordinate, and gradient descent can arrange those coordinates so variables with similar forecasting behavior land near each other.

The temporal marker can be just as direct because the preprocessing already gives periodic covariates. A five-minute traffic dataset has `N_d = 288` time slots per day and `N_w = 7` days per week. I can keep two learned tables, `T^TiD in R^{N_d x D}` and `T^DiW in R^{7 x D}`. At the forecast origin I look up the current time-of-day row and the current day-of-week row, then concatenate them to every node's representation. A raw normalized scalar would force the model to learn the periodic regimes through a continuous coordinate; a table lets slot 84 and slot 228 have completely different signatures if the data demand it.

The value trace still needs an embedding. For node `i`, let `X^i_{t-P:t} in R^P` be the past value sequence in a clean notation. Map it by one shared fully connected layer: `H^i_t = FC_embedding(X^i_{t-P:t})`, with `H^i_t in R^D`. The shared layer is important: I am not giving every node a separate forecaster. I am giving one forecaster a way to tell otherwise-colliding samples apart.

Now concatenate the four pieces:

`Z^i_t = H^i_t || E_i || T^TiD_t || T^DiW_t`.

In the symmetric-width case with all three tables enabled, each term has width `D`, so `Z^i_t in R^{4D}`. I should not hard-code `4D` in the actual module, because the branches may be optional and their widths may differ: `embed_dim + node_dim * I_node + temp_dim_tid * I_tid + temp_dim_diw * I_diw`. But the clean derivation is exactly the four-way concatenation.

The encoder can now be a plain residual MLP. For layer `l`, preserve the width and compute

`(Z^i_t)^{l+1} = FC^l_2(sigma(FC^l_1((Z^i_t)^l))) + (Z^i_t)^l`.

There is no graph convolution hidden here, no attention, and no recurrence. The residual connection is just an optimization choice: each block learns a correction while keeping the representation dimension fixed. The forecast head is another fully connected layer:

`Y_hat^i_{t:t+F} = FC_regression((Z^i_t)^L)`,

which gives the `F` future values for node `i`.

The indexing must be exact because an off-by-one silently changes the method. In the generated data, time-of-day values are `k / steps_per_day` for integer `k`, never `1.0`, so `floor(tod * steps_per_day)` lands in `0..steps_per_day-1`. Day-of-week is generated as `d / 7`, so `floor(dow * 7)` lands in `0..6`. The forecast is anchored at the last observed step, so I use `history_data[:, -1, :, 1]` for the time-of-day index and `history_data[:, -1, :, 2]` for the day-of-week index. I keep these as `[B, N]` index tensors, even though the preprocessing tiles the same time value across nodes; the lookup therefore returns `[B, N, D]` temporal embeddings and transposes them into `[B, D, N, 1]`.

The history embedding shape also needs careful bookkeeping. The runner supplies `history_data` as `[B, P, N, C]`. I select the first `input_dim` forward channels, transpose to `[B, N, P, input_dim]`, flatten each node's history to length `P * input_dim`, and apply a `1x1` convolution from `P * input_dim` to `embed_dim`. That is just a shared linear map over each node's flattened history, implemented as convolution for tensor convenience.

The loss has no hidden sign issue: it is absolute error, averaged over nodes and forecast steps, `1/(N F) sum_i sum_j |Y_hat^i_j - Y^i_j|`. A benchmark harness may mask missing values before averaging, but the objective direction is the same and lower is better. There is no diffusion normalization constant, no Laplacian sign convention, and no recurrence state to reconcile. All constants are table sizes, widths, and horizon lengths.

So the method I arrive at is a direct identity-augmented MLP: shared value-history embedding, learned per-node table, learned time-of-day table, learned day-of-week table, residual MLP encoder, linear regression head. The claim is falsifiable in exactly the right way. Remove the node table and similar value histories from different variables collide again. Remove the temporal tables and similar value histories at different periodic phases collide again. If the full model works, the lesson is that a simple regressor can compete once the samples are made distinguishable; topology is useful context, but not the only way to supply that context.
