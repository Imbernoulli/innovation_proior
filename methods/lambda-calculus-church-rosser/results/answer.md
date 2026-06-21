# Church-Rosser theorem，提炼版

Church-Rosser theorem 的核心结论是：lambda calculus 的 beta-reduction 具有 confluence。若同一个 term `M` 可以分别归约到 `N1` 和 `N2`，那么存在某个 `P`，使得 `N1` 和 `N2` 都能继续归约到 `P`。

这就是 diamond property 的思想：归约路径可以分叉，但分叉后仍可汇合。

最重要的后果是 normal form 的策略无关性。若某个 lambda term 存在 normal form，那么它至多只有一个 normal form。因此不同归约顺序可能走出不同中间项、不同步数，甚至某些策略可能发散；但只要成功到达 normal form，就不会得到另一个不兼容的答案。

这个定理的独特洞察不只是“先算哪里都差不多”，而是把计算从具体执行顺序提升到了重写系统的全局几何：

```text
term        = 归约图中的节点
reduction   = 节点之间的有向边
strategy    = 在图中选择一条路径
confluence  = 分叉路径下游仍可汇合
normal form = 若存在，则是唯一的不可再归约终点
```

因此，Church-Rosser theorem 把局部执行的不确定性和全局语义的确定性分开。解释器可以选择不同 redex，证明可以采用不同 reduction sequence，但 lambda calculus 的等式意义不被这些选择撕裂。它说明计算不是只有一条时间线，而是一个受 confluence 约束的归约空间。
