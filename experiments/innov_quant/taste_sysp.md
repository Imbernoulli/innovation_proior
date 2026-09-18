# 加不加 system prompt:2026 的 idea / taste / research-judgment(只两条 RL 之后的 lora 臂)

差 = **不加提示(`nosyspp`) − 加 2026 提示(`y26pp`)**。
两边采样逐字相同(pp=1.5),**唯一变量是有没有系统提示**;
日志已核:`nosyspp` 打印 `system message: (none)`。
裸 tag 的老产出同时还差 presence_penalty,**不能当这个对照用**(§34)。

## 0. 数据在不在(应为 idea 480 / ideav2 360 / judge3 1300 题)

| 臂 | family | 加提示 `y26pp` | 不加 `nosyspp` |
|---|---|---:|---:|
| 9B RL lora | idea | 480 | 作业还在跑 |
| 9B RL lora | ideav2 | 360 | 作业还在跑 |
| 9B RL lora | judge3 | 1300 | 作业还在跑 |
| 4B RL lora | idea | 480 | 作业还在跑 |
| 4B RL lora | ideav2 | 360 | 作业还在跑 |
| 4B RL lora | judge3 | 1300 | 作业还在跑 |

## 9B RL lora:不加提示 − 加 2026 提示

> 还没有可比的格子(作业在跑),跳过。

## 4B RL lora:不加提示 − 加 2026 提示

> 还没有可比的格子(作业在跑),跳过。

## 一眼表:Stouffer Z(正 = **不加**提示更好)

| 口径 | 9B RL lora | 4B RL lora |
|---|---:|---:|
| research judgment(judge3) | — | — |
| taste(idea+ideav2) | — | — |
| 两者合并 | — | — |

★ = 该格 p < 0.05。**负号 = 那句 `It is now year 2026. You are a good researcher.` 在帮忙**;正号 = 它在拖后腿。

