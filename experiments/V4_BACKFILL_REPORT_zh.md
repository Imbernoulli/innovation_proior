# data_v4 分段回填 —— 全量 dry-run 报告 + 20 单元 pilot（2026-08-18）

对应 `COLLEAGUE_PROPOSALS_REVIEW_zh.md` §2.5 / §5.4 / §7-1。
工具：`tools/v4_backfill.py`（`--dry-run` / `--apply` / `--show <unit>`）。
基线：`23ff22f29`（07-21 audit-edit 落地前的最后一棵树）。人类侧（statement / context / train_answer / verify）**一律保持 HEAD，本轮一个字都没动**。

---

## 0. 结论速查

| 项 | 数 |
|---|---|
| 扫描单元（OLD 与 HEAD 都有 `reasoning.md`） | **346**（HEAD 多出 `cpv4b-graph-bfs-boundary`，基线无，跳过） |
| OLD 段落总数（去代码块） | 7,589 → PRESENT 4,603 / PARTIAL 1,341 / **MISSING 1,645** |
| MISSING 分类 | **BACKFILL 695** / OTHER 934 / KEEP-OUT 16 |
| 实际计划回填 | **673 段，219 个单元**（另 22 段被一致性守卫丢弃，见 §3） |
| 其中逐字原文 | **543 段（80.7%）**；最小改写 130 段（只删口癖 / `:`→`.`） |
| 体量 | 中位 6,790 → **9,102 字符**（1.70k → **2.28k tok**）；**>32k tok 的单元 0 个** |
| 自查标记密度（broad） | 中位 0.78 → **1.32**/千 tok（OLD 天花板 2.11） |
| 三道口癖闸 | `deliberately` 81 文件 / `convinced myself` 5 / `Causal recap` 50 / top-1 开头 0.9% —— **回填前后逐项零变化** |

**一句话**：误删是真的、可以逐段还原，而且还原后不碰任何一条红线；但**"中位回到 4.4k tok、自查密度 ≥2/千 tok"这两个验收指标，靠这次回填达不到**——原因在 §5，不是工具保守，是基线本身就没有那么多东西。

---

## 1. 做法（全部可复算）

**切段**：按空行切块，围栏感知（``` 块永远自成一块，内容从不被读取、从不被写回）。单独成行的 markdown 标题（`## Final solver`）与其后一段合并，因为它是那段的标题而不是一个段落。

**判"还在不在"**：07-21 那一刀不是纯删除——299/346 个文件被改写过，其中 100 个只是被换了措辞。所以逐字 diff 会把"被改写但内容还在"的段落误判成"被删"，于是用两个信号：

- `g_cov`：OLD 段落的 4-gram 有多少比例出现在 HEAD 全文里（逐字存活）；
- `c_cov`：OLD 段落的**具体值指纹**（反引号内联代码 + 数字，如 `S = 0`、`{2, 4}`、`10^6`）有多少比例出现在 HEAD 全文里。改写会换措辞，但不会换掉一次手工回溯里的那些数。

判定（阈值都写在 `tools/v4_backfill.py` 顶部，可调）：

```
PRESENT : g_cov>=0.45  或  (具体值>=12 且 c_cov>=0.60)  或  (具体值>=3 且 c_cov>=0.70)
PARTIAL : g_cov>=0.25  或  (具体值>=12 且 c_cov>=0.38)  或  (具体值>=3 且 c_cov>=0.45)
MISSING : 其余
```

**默认只回填 MISSING**。PARTIAL（1,341 段）是"HEAD 里压缩保留了一部分"的段落，整段贴回会和现有文字重复，所以默认关闭，用 `--include-partial` 才打开（打开后计划插入 1,508 段、中位 3.0k tok）。

**分类**（先 KEEP-OUT，再四类回填，最后 OTHER）：

| 类 | 判据 | MISSING 计数 |
|---|---|---|
| KEEP-OUT opener | 段首 `Reading the problem/objective/...` | 6 |
| KEEP-OUT recap | 标题 `Causal recap` | 5 |
| KEEP-OUT tic-only | 含 `deliberately`/`convinced myself` 且**全段没有任何验证内容** | 5 |
| **BACKFILL edge** | 标题/段首含 `Edge case` / `Corner case` / `the corners` | **38** |
| **BACKFILL retrace** | 标题含 trace/re-verify/sanity-check/by hand …；**或**段内 ≥6 个含数字的内联代码 + 走查动词（`trace/walk/run/step/expected/matches/correct/prints`） | **482** |
| **BACKFILL oracle** | oracle / differential test / brute force / stress test / cross-check / self-verification / zero mismatches | **100** |
| **BACKFILL ship** | 标题 `Final solution/solver/program` 或段内 `what I ship` / `I submit` | **53** |
| OTHER | 其余（候选方案罗列、推导、bug 诊断） | 934 |

**落位**：每段回填到它在 OLD 里**最近的、在 HEAD 中存活的前置锚点**之后；锚点映射强制单调（OLD 顺序即 HEAD 顺序）。ship 段若其后 OLD 只剩代码块和（永不回填的）causal recap，则落到文末。

**口癖处理（不是整段丢弃）**：
- 含 `convinced myself` 的**句子**整句删（那句话本身就是口癖：自我表扬，不是检查）；
- `deliberately` **就地删词**，句子保留（`**Edge cases, deliberately, because this is where this kind of code dies.**` → `**Edge cases, because this is where this kind of code dies.**`）；
- 段尾因为引出一个我们**不**回填的代码围栏而悬着的 `:`，改成 `.`（只动标点，不加词）。

**硬约束（脚本内自检 + 事后复核，全部 0 违例）**：不新增任何句子（`tools/make_commit_coda.py` 全程未运行）、不回填任何代码围栏、opener/recap 双重拦截、工件 regex 零命中、不产生半截句（22 段以小写延续词开头、其 OLD 前驱是代码块的碎片被直接丢弃）。

---

## 2. 每类段落计数（346 单元全量）

四个回填家族在 OLD 里的**总量与去向**（这一栏比 review §2.3 的"标记文件数"更严格：标记词被换掉但内容还在，算 PRESENT，不算误删）：

| 家族 | OLD 段总数 | PRESENT（改写后仍在） | PARTIAL（部分保留） | **MISSING（真删）** | 计划回填 |
|---|---|---|---|---|---|
| edge 边界枚举 | 306 | 143 | 118 | **45** | 38 |
| retrace 手工回溯 | 1,766 | 744 | 531 | **491** | 482 |
| oracle 差分/独立预言 | 632 | 398 | 133 | **101** | 100 |
| ship 交卷决定 | 347 | 231 | 58 | **58** | 53 |
| 合计 | 3,051 | 1,516 | 840 | **695** | **673** |

**与 review §2.3 表的差异，必须摆在桌面上**：review 用"文件里是否出现 `Edge cases` / `Final solution` 字样"计数，得出边界 −79%、交卷 −65%。按内容判定，真删的比例低得多（edge 45/306 = 15%，ship 58/347 = 17%）——因为 07-21 那一刀在很多文件里把 `**Edge cases.**` 那段**改写并压缩进正文**了（例如 `fcs-p2-01`：HEAD 的"The corners fall to the same two mechanisms. `S = 0`…"逐个覆盖了 OLD 那七条边界）。**"标记消失" ≠ "内容消失"，这是本次回填规模比 review 预期小的第一原因。**

回填后标记文件数（仅供与 review 的表对齐）：

| 标记（按文件） | OLD | HEAD | 回填后 |
|---|---|---|---|
| `Edge case` | 288 | 86 | **121** |
| `Final solution`/`Final solver`/`what I ship` | 334 | 108 | **158** |
| `oracle` | 183 | 167 | **172** |
| `re-trace` | 193 | 28 | **98** |

---

## 3. 体量投影（32k 预算）

| | 中位 | 均值 | p90 | 最大 | 合计 |
|---|---|---|---|---|---|
| OLD `23ff22f29` | 17,466 字符 / **4.37k tok** | 18,986 | 25,885 | 46,335 | 6.57M |
| HEAD | 6,790 / **1.70k tok** | 11,243 | 24,843 | 40,374 | 3.89M |
| **回填后** | **9,102 / 2.28k tok** | 12,609 | 24,843 | 40,374 | 4.36M |

- **超 32k tok 的单元：0 个；超 16k tok 的单元：0 个。** 最大单元 40,374 字符 ≈ 10.1k tok（`ale-49`，且它一段都没回填——增长全部来自别的单元）。离 32k 渲染预算还差 3 倍以上，`§2.13(b)` 的截断损失通道不会被触发。
- 全量回填只让 data_v4 总字符从 3.89M 回到 4.36M（OLD 是 6.57M）。

---

## 4. 三个被回填单元的 before/after

### 4.1 `fcs-p2-18`（19,879 → 5,449 → **10,939**，+9 段：edge×1 / oracle×1 / retrace×7）

HEAD 只留下了"贪心为什么在原理上不对"的断言；被删掉的是**把它做实的那几次手工走查**。

> **HEAD 锚点段尾**：…`= 1 + min(dp[i+1][j], dp[i][j-1])`, a `min` over two interior costs that depend on the entire span `s[i+1..j-1]` the greedy has not looked at. Whenever those two costs differ, a fixed left-or-right rule is wrong on some input.
>
> **↓ 回填（oracle，逐字）**：**Stress-testing the greedy before committing.** The danger spot is the mismatch branch: "insert to match the left end, advance `i`" versus "insert to match the right end, retreat `j`." A natural implementation has to pick one rule and stick with it — say, always mirror the left character onto the right. Let me hunt for an `s` where that fixed rule, or indeed *any* fixed local rule, is forced into a worse global count than the optimum.
>
> **↓ 回填（retrace，逐字）**：Now run the greedy with the "mirror-left-onto-right, advance `i`" rule. `i=0, j=4`: `s[0]='a'`, `s[4]='a'` — match, step inward to `i=1, j=3`. `s[1]='b'`, `s[3]='d'` — mismatch. … The greedy counted `2`. Here it happens to *match* the optimum.

后一段的价值恰恰在最后一句：**这次走查没抓到 bug**，下一段才构造出真正的反例。这是"真自查"和"表演自查"的分界，被 07-21 一起删掉了。

### 4.2 `cpv4-strings-hash-negzero`（18,238 → 4,729 → **9,691**，+6 段 retrace）

> **HEAD 锚点段尾**：…so a symbol of value `0` becomes an ordinary digit (`10^9+1`) rather than vanishing from a prefix hash. Now `-3` maps to `999999998` and `5` to `1000000006` — distinct, collision gone.
>
> **↓ 回填（retrace，删了一个 `deliberately`）**：I trace the all-zero overlap corner `a = [0, 0, 0, 0, 0]`, `n = 5`, whose true answer is `4` (the length-4 block of zeros starts at index 0 and at index 1 — distinct starts, overlapping, allowed). The search starts `lo=0, hi=5`. First `mid = 2`: `hasDup(2)` scans windows `[0,0]` at i=0..3, the first repeat appears immediately, returns true; `ans=2, lo=3`. `mid = (3+5)/2 = 4`: …

原文是 `I deliberately trace the all-zero overlap corner …`，只删掉那一个副词，其余逐字。

### 4.3 `fcs-p2-21`（20,337 → 6,380 → **7,779**，+3 段：oracle×1 / retrace×1 / ship×1）

OLD 的交卷段是两句"我说服了我自己"+ 一句真交卷。前者删掉，后者留下：

> **OLD**：**Final solution.** I convinced myself the idea is right by breaking the tempting "tallest-face greedy" with a concrete `17`-vs-`23` instance and by proving that strict base nesting implies strict area shrink…, and I convinced myself the *code* is right by tracing the strict-comparison bug on a cube to a precise cause, fixing it, and differential-testing against an independent longest-path oracle. That is what I ship — one self-contained file, the simple `O(m^2)` ordered DP I can defend rather than the greedy I broke**:**
>
> **↓ 回填后**：**Final solution.** That is what I ship — one self-contained file, the simple `O(m^2)` ordered DP I can defend rather than the greedy I broke**.**

（结尾 `:` 原本引出 C++ 代码块；代码块不回填，所以 `:`→`.`，不加任何词。）

---

## 5. 自查标记密度：**验收指标达不到，且不是这次回填的错**

两套词表，同口径（tok = 字符/4）：

| 词表 | OLD `23ff22f29` | HEAD | **回填后** | 硬指标 |
|---|---|---|---|---|
| broad（工具 `RE_SELFCHECK`：Wait/Hmm/verify/actually/double-check/sanity-check/let me test/re-trace/counterexample/mismatch/cross-check） | 中位 2.11 | 0.78 | **1.32** | — |
| strict（`SFT_DATA_FULL_FORENSICS_zh.md:77` 原词表：Wait/Hmm/verify/Actually/But wait/double-check/sanity check/Let me test/recheck） | 中位 **0.54** | 0.00 | **0.46** | ≥2/千 tok |

**关键事实：整档回滚到 `23ff22f29` 也只有 0.54/千 tok。** `DATA_RECOMMENDATIONS_zh.md:20` 的 ≥2/千 tok 是拿 base 自然思考（4.93）和 wave2-cp 拒绝采样 rollout（7.45）当参照定的，而 data_v4 从第一天起就是**写好的光滑叙述**（FOR §2.4 的原话："创新数据的 think 密度 ≈ 0"）。分段回填能把它从 0.00 拉回基线的 0.46——**恢复被误删的部分，做不到、也不可能做到把它变成 rollout 数据**。

→ **建议**：把 ≥2/千 tok 这条指标从 data_v4 回填的验收里摘出去，交给 §7-2 的 wave3 切片和 `tools/hardcp_rollout.py` 的真 rollout。data_v4 回填的验收应该改成三条可达的：(i) 四家族 MISSING 段清零；(ii) 三条口癖闸零回升；(iii) 中位 tok 回到 2.3k 且 >32k 单元为 0。

同理，"中位回到 ~4.4k tok"也达不到（实际 2.28k）。差额在**934 个 MISSING 的 OTHER 段**（候选方案罗列 / 推导 / bug 诊断）——review §2.5 只授权了四类，这 934 段没在授权里。如果要 4.4k，就要扩授权范围，而那正是 review §2.4 警告过的"无差别恢复"。**这个决定留给人。**

---

## 6. 闸门投影（全量回填后 vs HEAD）

| 闸 | HEAD | 回填后（346 全量投影） | 判定 |
|---|---|---|---|
| top-1 开头占比 | **0.9%**（3/346） | **0.9%**（3/346） | ✅ 远低于 5% |
| `deliberately` | 81 文件 / 100 处 | **81 / 100** | ✅ 零回升 |
| `convinced myself` | 5 文件 | **5** | ✅ 零回升 |
| `Causal recap` | 50 文件 | **50** | ✅ 零回升 |
| `Reading the problem` | 14 文件 | **14** | ✅ 零回升 |
| 工件 regex `getenv(\|ALE_BASELINE\|<model_answer>\|// ale-\d+` | 4 文件（全在代码围栏内，历史遗留） | **4** | ✅ 零新增（代码围栏从不回填） |
| `tools/lint_inframe.py` | data_v4 命中 0 | **0** | ⚠️ 见下 |

⚠️ `lint_inframe.py` 的扫描根是 `<root>/<slug>/results/*.md`，**data_v4 的单元没有 `results/` 子目录，所以它从来就没扫过 data_v4**。"grep data_v4 无新增命中"这条闸对本次改动是恒真的，不构成保护。它的 `C_rsn_header`（reasoning.md 正文里出现 markdown 标题）这一条如果扩到 data_v4，会命中 38 个本来就用 `##` 分节的单元（ale-* 为主）——与回填无关，是既有形态。

---

## 7. Pilot：删得最多的 20 个单元

选法：按 `OLD − HEAD` 字符差排序，取前 20 个**有回填候选**的单元。前 24 名里有 3 个候选为 0，跳过：`ale-49`（删的 35,750 字符里绝大部分是一个巨大的 `// ale-49:` C++ 围栏，代码不回填）、`fcs-gr-04`、`fcs-nt-04`（被删段落全部判为 PARTIAL / OTHER）。

| 单元 | HEAD → 回填后 | 段数 |
|---|---|---|
| `ale-v2-07` | 10,269 → 11,010 | +1 ship |
| `fcs-v2-gx-01` | 7,790 → 10,961 | +5 oracle×2 retrace×3 |
| `ale-02` | 7,758 → 7,826 | +1 ship |
| `fcs-tr-05` | 8,064 → 9,706 | +3 edge×1 oracle×1 retrace×1 |
| `fcs-v2-gr-03` | 7,685 → 8,755 | +2 |
| `fcs-v2-st-02` | 6,439 → 7,849 | +2 retrace×2 |
| `fcs-v2-tr-02` | 7,755 → 9,687 | +2 |
| `fcs-p2-11` | 6,622 → 9,136 | +3 retrace×3 |
| `fcs-p2-18` | 5,449 → 10,939 | +9 |
| `fcs-gr-07` | 7,624 → 11,170 | +2 retrace×2 |
| `fcs-p2-14` | 6,888 → 8,981 | +3 |
| `fcs-gr-02` | 5,968 → 8,315 | +2 retrace×2 |
| `fcs-p2-21` | 6,380 → 7,779 | +3 |
| `fcs-p2-05` | 5,697 → 7,669 | +3 retrace×3 |
| `fcs-v2-tr-01` | 7,277 → 7,943 | +2 |
| `fcs-ds-06` | 7,237 → 8,345 | +1 retrace×1 |
| `cpv4-dp-interval-negzero` | 7,654 → 10,437 | +2 |
| `fcs-p2-13` | 5,605 → 8,544 | +4 |
| `fcs-v2-st-01` | 7,115 → 9,456 | +3 |
| `cpv4-strings-hash-negzero` | 4,729 → 9,691 | +6 retrace×6 |

合计 **59 段**。pilot 内自查密度（broad）中位 0.65 → **1.63**。每个单元单独 commit（`v4-backfill: <unit> — +N segments (…)`）。

---

## 8. 复算方式

```bash
python3 tools/v4_backfill.py --dry-run                 # 本报告 §0/§2/§3/§5/§6 的所有数
python3 tools/v4_backfill.py --show fcs-p2-18          # 逐段判定 + 落位计划
python3 tools/v4_backfill.py --dry-run --include-partial   # 把 PARTIAL 也算上的上界
python3 tools/v4_backfill.py --apply --units fcs-p2-18     # 只写 data_v4/<unit>/reasoning.md
```

基线 blob 一律用 `git show 23ff22f29:<path>` 读，不落盘、不改工作树。token 用 `字符/4` 估（与 review 同口径）。
