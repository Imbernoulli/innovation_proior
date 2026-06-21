Zig-zag product 的核心答案是：不要一次性显式写出一个“像随机图”的大图，而是把扩展性拆成两个可组合的来源。

给定一个大的 `D`-正则图 `G` 和一个小的 `d`-正则图 `H`，其中 `H` 的顶点数等于 `G` 的度数 `D`，zig-zag product 把 `G` 的每个顶点替换成一个 `H` 的小云。新图的顶点是 `(v,a)`：`v` 是大图顶点，`a` 是大图的一条边标签。一步行走是：

1. 在当前云内沿 `H` 走一步，打散边标签。
2. 用打散后的标签在 `G` 中走一步，完成全局移动。
3. 在目标云内再沿 `H` 走一步，修复到达后的局部分布。

于是新图大约继承 `G` 的规模，继承 `H` 控制的低度数，并且继承二者组合后的扩展性。这个结构的独特洞察是：大图负责全局扩展，小图负责局部扩展；局部混合让全局边选择不再被少数标签支配，全局移动又把质量送到远处。两者不是简单叠加，而是互相补位。

这突破了“随机图存在但显式构造困难”的障碍。随机正则图说明常数度扩展图大量存在，但随机证明没有给出一个可递归生成、可本地寻址、可逐步验证的邻接规则。Zig-zag product 把问题改写成一个稳定的构造循环：

```text
当前常数度扩展图
  -> 图平方：增强全局扩展，但度数升高
  -> zig-zag：用固定小扩展图恢复低度数，同时保留足够扩展
  -> 更大的常数度扩展图
```

只要一开始硬编码一个固定大小的小扩展图，每轮都能确定性地产生更大的图。邻居查询只是若干次 rotation map 调用，因此是显式的；扩展性由乘积定理递归保证，因此不需要在指数大的候选空间中搜索随机图。

一句话概括：zig-zag product 把“随机全局连通性”分解成“可迭代的全局放大 + 可复用的局部混合”，从而在保持低度数的同时构造出任意大的显式扩展图。

## Code illustration

```python
def rotation_map_G(v, a):
    """Return (w, b): G-edge (v,a) lands at w with back-label b."""
    # Example: explicit D-regular rule on Z_N (D must match |H|)
    w = (v + (2 * a + 1)) % G_N
    b = a
    return w, b

def rotation_map_H(a, i):
    """Return (a_prime, i_back): H-edge (a,i) lands at a_prime with back-label i_back."""
    # Example: explicit d-regular rule on Z_D
    a_prime = (a + i) % H_D
    i_back = (-i) % H_D
    return a_prime, i_back

def zig_zag_rotation_map(rot_g, rot_h, vertex, edge_label_pair):
    v, a = vertex
    i, j = edge_label_pair

    a_prime, i_back = rot_h(a, i)   # zig inside source cloud
    w, b_prime = rot_g(v, a_prime)  # global step in G
    b, j_back = rot_h(b_prime, j)   # zag inside destination cloud

    return (w, b), (j_back, i_back)
```
