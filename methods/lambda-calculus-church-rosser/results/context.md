## Research question

Lambda calculus 中，beta-reduction 把一次函数应用看成一次代换。一个 term 往往含有多个可归约位置，执行顺序因此是不唯一的。若不同顺序产生不同的最终 term，lambda calculus 就只能描述某种具体求值策略，而不能作为稳定的计算等式理论。问题在于：从同一个 term 出发的多条归约路径，它们最终的结果是否能够协调？

## Background

把 lambda calculus 看成抽象重写系统时，term 是节点，单步归约是有向边。一次计算不再只是线性轨迹，而是在一个由归约关系生成的图中行走。

confluence 的图像是 diamond property 的推广：若 `M ->* N1` 且 `M ->* N2`，则存在某个 `P`，使得 `N1 ->* P` 且 `N2 ->* P`。这里 `->*` 是零步或多步归约。

## Evaluation setting

评价对象不是实验指标，而是概念解释是否抓住定理的结构作用。最低要求是说清三点：第一，beta-reduction 的路径分叉来自 redex 选择；第二，confluence 的定义及其对归约路径的约束；第三，normal form 与归约顺序之间的关系。
