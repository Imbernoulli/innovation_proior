## 问题定位

Mirror symmetry 的关键不只是"两个 Calabi-Yau 看起来不同却物理等价"。它真正改变方法论的地方，是把一个 Calabi-Yau 上的 complex moduli 与另一个镜像 Calabi-Yau 上的 complexified Kahler moduli 对调。这个对调让物理中的 string duality 不再只是解释同一理论的两种语言，而变成一种生成数学预测的计算机器。

需要解释的核心问题是：为什么这种对偶能把枚举几何计数，转译为周期积分和 variation of Hodge structures。

## 物理与几何设置

在 topological string 的语言中，A-model 主要看 symplectic/Kahler 数据，B-model 主要看 complex structure 数据。对一个镜像对 `(X, X^vee)`，mirror symmetry 断言 `X` 的 A-model 与 `X^vee` 的 B-model 等价；于是 `X` 上的 complexified Kahler moduli 被 `X^vee` 上的 complex moduli 重新表达。

这正是"镜像"的含义：不是找一个形状相似的空间，而是找一个使两类变形参数互换的对偶空间。原来在 `X` 上属于 Kahler 几何和曲线计数的问题，在 `X^vee` 上变成复结构族、全纯体形式及其周期的问题。

## 枚举几何

枚举几何中的典型任务，是数 Calabi-Yau 三维流形上的有理曲线，或更现代地说，计算 Gromov-Witten invariants。quintic threefold 是一个标准研究对象，涉及任意次数有理曲线计数，以及量子修正和 multiple cover 贡献。

## 镜像转译

镜像侧的 B-model 把计数问题转成 period calculation。给定镜像 Calabi-Yau 的复结构族，取全纯三形式 `Omega`，研究它在三维同调循环上的积分 `int_gamma Omega`。这些周期随复结构参数变化，形成 variation of Hodge structures，并满足 Picard-Fuchs 微分方程。

计算路线因此变为：解 Picard-Fuchs 方程得到周期，选取 large complex structure limit 附近的规范坐标，用 mirror map 把镜像复结构参数翻译回原空间的 Kahler 参数，再从 Yukawa coupling 或 prepotential 的级数展开中读出枚举不变量。曲线计数被改写为一套微分方程、单值化和级数展开问题。

## 方法论意义

这是一种思维突破，因为它没有在原来的枚举空间中寻找更强的局部计数技巧，而是利用对偶几何改变"什么是可计算对象"。A-model 中的枚举几何量，在 B-model 中对应由复结构的 Hodge 理论控制，可由周期和 Picard-Fuchs 系统计算。

所以 mirror symmetry 的独特洞察，是把 string duality 当成数学翻译器：同一个物理量，在一边表现为枚举几何不变量，在另一边表现为可计算的 Hodge-theoretic 数据。它让物理直觉能够预测具体整数和生成函数。
