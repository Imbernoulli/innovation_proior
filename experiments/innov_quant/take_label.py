"""从 subagent 的 jsonl 转写里取最后一条 assistant 文本,解析成标注 JSON 存盘。

slug 不靠我记,从第一条 user 消息里那句「完整读这个文件」后面的路径反解出来 ——
记错 slug 会把标注挂到别的题上,而且解盲后一声不响。
"""
import json, re, sys, os

S = os.path.dirname(os.path.abspath(__file__))
LAB = f"{S}/explore_labels"
os.makedirs(LAB, exist_ok=True)

for path in sys.argv[1:]:
    slug, last = None, None
    for ln in open(path):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        m = r.get("message") or {}
        c = m.get("content")
        txt = c if isinstance(c, str) else " ".join(
            b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else ""
        if slug is None and r.get("type") == "user":
            g = re.search(r"explore_blind/([A-Za-z0-9_.-]+)\.md", txt)
            if g:
                slug = g.group(1)
        if r.get("type") == "assistant" and txt.strip():
            last = txt
    if not slug or not last:
        print(f"[bad] {path}: slug={slug} last={'有' if last else '无'}"); continue
    s = last.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    s = s[i:j + 1] if i >= 0 and j > i else s
    d, fix = None, 0
    # 回包偶尔少最后一两个右花括号(H 块自己是闭合的,只是外层没收口)。
    # 只补右花括号,别的一概不改:补不出合法 JSON 就当这题没标。
    for k in range(3):
        try:
            d = json.loads(s + "}" * k); fix = k; break
        except Exception as e:
            err = e
    if d is None:
        print(f"[bad] {slug}: JSON 解析失败 {err}"); continue
    if fix:
        print(f"[warn] {slug}: 回包少了 {fix} 个右花括号,已补")
    if sorted(d) != list("ABCDEFGH"):
        print(f"[bad] {slug}: 块 {sorted(d)}"); continue
    json.dump(d, open(f"{LAB}/{slug}.json", "w"), ensure_ascii=False, indent=1)
    print(f"[ok] {slug}")
