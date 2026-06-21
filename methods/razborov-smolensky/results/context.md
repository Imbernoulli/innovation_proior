## 问题位置
Razborov-Smolensky 方法属于 1980 年代小深度电路下界路线的下一步：在 Ajtai/Furst-Saxe-Sipser/Yao/Hastad 用随机限制证明 `PARITY` 不在 `AC0` 之后，它把目标扩展到带模计数门的 `AC0[p]`。核心问题是：一个常深、可多项式大小、允许 `MOD_p` 门的电路，能否计算不同模数的计数函数 `MOD_q`。

## 电路模型
`AC0[p]` 指常深、多项式大小、无界扇入的 `AND`、`OR`、`NOT`、`MOD_p` 门电路，其中 `p` 通常取素数。Razborov 与 Smolensky 的典型结论是：若 `p` 与 `q` 是不同素数，则 `MOD_q` 不属于 `AC0[p]`；更定量地，深度为 `d` 的 `AC0[p]` 电路要计算 `MOD_q` 需要指数型大小下界，常见讲义形式为 `2^{Omega(n^{1/(2d)})}`。
