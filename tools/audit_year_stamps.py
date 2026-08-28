#!/usr/bin/env python3
"""Find rows whose `It is now year YYYY.` stamp is earlier than the work they cite.

Three detectors, all chosen so a hit cannot be an ordinary English word or a bare
numeric value. A previous attempt keyed on method names like "clip" and produced
143 false positives out of 188 (every one was gradient clipping or a clip() call),
so nothing ambiguous is allowed in here.

  1. explicit citations   "(Author, 2018)" / "Author et al., 2018" / "Author 2018)"
  2. library names        causallearn, PyTorch, JAX, ... - never ordinary words
  3. dataset/model names  CIFAR-10, MobileNetV2, ... - hyphen/case-distinctive

A row is flagged when the newest thing it cites postdates its own stamp by more
than --slack years.
"""
import argparse, json, re, sys, collections

CITE = re.compile(r"""
    (?: \(\s*[A-Z][A-Za-z'\-]+ (?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-z'\-]+)? [^()]{0,40}?,\s*(19[5-9]\d|20[0-2]\d) \s*\)
      | \b[A-Z][A-Za-z'\-]+\s+et\s+al\.?,?\s*\(?(19[5-9]\d|20[0-2]\d)\)?
      | \b[A-Z][A-Za-z'\-]+\s+(?:&|and)\s+[A-Z][A-Za-z'\-]+,?\s*\(?(19[5-9]\d|20[0-2]\d)\)?
      | \barXiv[:\s]*(?:(\d{2})(\d{2})\.\d{4,5})
    )""", re.X)

LIB = {  # name -> first public release year; all case-sensitive, none is an English word
    "causallearn": 2021, "causal-learn": 2021, "PyTorch": 2016, "TensorFlow": 2015,
    "JAX": 2018, "Keras": 2015, "Theano": 2008, "scikit-learn": 2010, "sklearn": 2010,
    "HuggingFace": 2018, "Transformers": 2019, "vLLM": 2023, "DeepSpeed": 2020,
    "Optuna": 2018, "LightGBM": 2016, "XGBoost": 2014, "CatBoost": 2017,
    "Weights & Biases": 2018, "wandb": 2018, "Gymnasium": 2022, "Stable-Baselines3": 2020,
    "Flash-Attention": 2022, "FlashAttention": 2022, "bitsandbytes": 2021, "PEFT": 2022,
    "einops": 2020, "Numba": 2012, "CuPy": 2017, "Triton": 2021, "causallearn": 2021,
}
NAMED = {  # dataset / model / benchmark names
    "CIFAR-10": 2009, "CIFAR-100": 2009, "ImageNet": 2009, "MS-COCO": 2014, "COCO": 2014,
    "MobileNetV2": 2018, "MobileNetV3": 2019, "ResNet-20": 2015, "ResNet-50": 2015,
    "VGG-16": 2014, "EfficientNet": 2019, "DenseNet": 2016, "Inception-v3": 2015,
    "AlphaFold": 2021, "AlphaGo": 2016, "MuZero": 2019, "D4RL": 2020, "AntMaze": 2020,
    "MuJoCo": 2012, "Atari": 2013, "GSM8K": 2021, "MATH-500": 2023, "MMLU": 2020,
    "HellaSwag": 2019, "SuperGLUE": 2019, "GLUE": 2018, "SQuAD": 2016, "LAMBADA": 2016,
    "ISPRS Vaihingen": 2012, "Vaihingen": 2012, "Potsdam": 2012,
    "LLaMA": 2023, "Llama-2": 2023, "Llama-3": 2024, "Mistral-7B": 2023, "Qwen2": 2024,
    "GPT-3": 2020, "GPT-4": 2023, "BERT": 2018, "RoBERTa": 2019, "T5": 2019,
    "Stable Diffusion": 2022, "DALL-E": 2021, "Midjourney": 2022,
}


def cited_years(text):
    out = []
    for m in CITE.finditer(text):
        for g in m.groups():
            if not g:
                continue
            if len(g) == 2:                       # arXiv YYMM
                y = 2000 + int(g)
                if 1991 <= y <= 2026:
                    out.append(("arxiv", y))
                break
            y = int(g)
            if 1950 <= y <= 2026:
                out.append(("cite", y))
            break
    for name, y in LIB.items():
        if re.search(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", text):
            out.append((f"lib:{name}", y))
    for name, y in NAMED.items():
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", text):
            out.append((f"name:{name}", y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/v2_multisetting_4b/innovation_v2_timeonly.jsonl")
    ap.add_argument("--slack", type=int, default=0, help="years of tolerance")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    YR = re.compile(r"year (\d{4})")
    flagged, n, nostamp = [], 0, 0
    for i, line in enumerate(open(a.src)):
        if not line.strip():
            continue
        r = json.loads(line)
        m = YR.search(r.get("system") or "")
        if not m:
            nostamp += 1; continue
        n += 1
        stamp = int(m.group(1))
        body = " ".join(t["value"] for t in r["conversations"])
        ev = [(k, y) for k, y in cited_years(body) if y > stamp + a.slack]
        if ev:
            newest = max(y for _, y in ev)
            flagged.append({"row": f"{i:05d}", "stamp": stamp, "newest": newest,
                            "gap": newest - stamp,
                            "evidence": sorted({k for k, y in ev if y == newest})[:6]})
    flagged.sort(key=lambda d: -d["gap"])
    print(f"有年份戳的行 {n}（无戳 {nostamp}），命中 {len(flagged)} = {len(flagged)/max(1,n)*100:.1f}%")
    print(f"\n{'行':>7}{'戳':>7}{'最新引用':>9}{'差':>5}  证据")
    for d in flagged[:30]:
        print(f"{d['row']:>7}{d['stamp']:7d}{d['newest']:9d}{d['gap']:5d}  {', '.join(d['evidence'])}")
    if a.out:
        json.dump(flagged, open(a.out, "w"), indent=2)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
