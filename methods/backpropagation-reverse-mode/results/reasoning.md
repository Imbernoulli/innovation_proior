I start with the thing that makes hidden units hard. If a network has only input units connected straight to output units, the error at an output can tell me directly how to change the connecting weights. But as soon as I insert hidden units, the training data no longer tells me what those hidden units should do. The only quantity I can compare with the desired answer is the final output, and yet I need a direction for every weight below it. I need the scalar error at the top to assign credit to all the internal computations that helped create it.

The most literal route is to perturb one weight, run the whole network again, and see whether the error goes up or down. For weight w_i I can use (E(w + h e_i) - E(w)) / h. That is tempting because it treats the network as a black box, but the cost is wrong in exactly the case I care about. If there are P weights, one gradient step costs P extra forward evaluations, and the step size h adds its own numerical compromise. Making h small reduces truncation error but increases cancellation. This is not a learning rule for a large network; it is a diagnostic.

I can also try to write the derivative formulas out symbolically. For one or two layers, that is tolerable. But a network is a composition with shared intermediate values. If I differentiate the full expression as an expression, products duplicate their subexpressions, nested products duplicate them again, and the derivative expression can become much larger than the computation I started with. Worse, every change in the architecture asks me to do the algebra again. I do not want a giant formula. I want a procedure that follows the computation itself.

Forward sensitivity is closer. While I evaluate every unit, I can carry along the derivative of that unit with respect to one selected weight. Addition, multiplication, and the nonlinearity all have local derivative rules, so the chain rule pushes this sensitivity forward cleanly. But this still has the wrong asymmetry. One forward sensitivity sweep answers one question: how does the output change if this one input direction changes? I have many weights and one scalar error. Carrying one direction at a time makes the cost scale with the number of weights.

So I should stop looking at the computation from the weights outward. The output is one scalar. Every intermediate value matters only through its effect on that scalar. Let the computation create intermediate values v_1, v_2, ..., v_N, with the loss L as the final scalar. For any intermediate v_i, the quantity I actually need is not "how does v_i change when one particular weight changes?" It is "how does L change when v_i changes?" Write that as vbar_i = dL/dv_i. If I know vbar for a child value v_j that directly used v_i, then the contribution passing through that child is vbar_j * dv_j/dv_i. If v_i feeds several later values, the effects add:

  vbar_i = sum over children j of vbar_j * dv_j/dv_i.

This is just the chain rule, but the direction has flipped. The reason the flip matters is topological: when I sweep the operation list backward, every child of v_i has already received its total effect on L, so I can add the local contribution into v_i and move on. I initialize the final scalar with Lbar = dL/dL = 1. Then every operation sends its adjoint to its parents. An addition z = a + b sends zbar to both a and b. A multiplication z = a b sends zbar * b to a and zbar * a to b. A nonlinearity z = phi(a) sends zbar * phi'(a) to a. Shared subexpressions are no longer duplicated; their incoming contributions are accumulated in one number. The derivative computation becomes dynamic programming over the graph.

Now the neural-network formulas fall out as a special case. For an output unit j, the loss contribution is 1/2 (y_j - d_j)^2, so dE/dy_j = y_j - d_j. If y_j = phi(x_j), then the sensitivity to the unit's total input is

  delta_j = dE/dx_j = (y_j - d_j) phi'(x_j).

The total input is x_j = sum_i w_ij y_i + b_j. A weight w_ij appears locally as w_ij y_i, so

  dE/dw_ij = delta_j y_i.

The lower unit output y_i affects the loss through every outgoing connection to higher units, so

  dE/dy_i = sum_j delta_j w_ij,

and if y_i = phi(x_i), then

  delta_i = dE/dx_i = phi'(x_i) sum_j w_ij delta_j.

That is the same backward accumulation in layer notation. Start with the output errors, convert them to deltas by the activation derivative at the already-computed forward activations, move one layer down, sum the weighted deltas from the layer above, and compute weight gradients as "receiving unit delta times sending unit activity." Every edge is used once in the forward computation and once in the backward computation. I do not perturb each weight. I do not build a symbolic derivative expression. I reuse the values the forward pass already produced and send one scalar sensitivity backward through each local operation.

The storage requirement is now obvious too. The local derivative at a node usually depends on the forward value at that node: phi'(x_i) needs x_i or y_i, and d(w_ij y_i)/dw_ij needs y_i. So the forward pass must leave behind the activations or an operation tape. If I unroll an iterative network through time, the backward pass needs the history of unit states at each time step; if the same weight is reused at multiple unrolled positions, the gradient for the actual tied weight is the sum of the gradients for all its occurrences. That is the price of the backward sweep: memory for the path that the computation took. The payoff is that one scalar loss gives me all parameter derivatives in one reverse traversal.

This also clarifies why the method belongs to the broader computation graph, not just to a particular neural-network picture. A layered net is only one kind of directed acyclic computation once it is evaluated. The rule is: run the program forward, record the intermediate values and dependency edges, set the output adjoint to one, traverse operations in reverse topological order, and for each operation add output_adjoint times the local partial derivative into each input adjoint. If the original program costs C elementary operations, the backward pass touches the same local operations in reverse and costs only a small constant multiple of C. For a function from many parameters to one loss, that changes the gradient cost from "one forward-like sweep per parameter" to "one forward pass plus one backward pass."

The learning rule is then plain gradient descent in weight space. Accumulate dE/dw over one case or over a batch of cases, and change each weight by -eta dE/dw. A momentum term can update a velocity rather than the position directly, and small random initial weights break symmetry so hidden units do not all receive identical signals. Those are optimization details. The core computation is the adjoint sweep: one scalar error at the end, local chain-rule messages moving backward, all partial derivatives collected at the leaves. I can now write the method as a small graph algorithm: topologically order the values from the forward evaluation, zero every sensitivity, set the loss sensitivity to one, visit the values in reverse order, and add each child's sensitivity times each local partial into the parent; the adjoints sitting on the parameter leaves are exactly the gradient components I need.
