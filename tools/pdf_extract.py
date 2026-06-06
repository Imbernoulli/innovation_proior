#!/usr/bin/env python3
"""
pdf_extract.py — one-command text extraction for paper-to-reasoning Phase 1.

Bundles the recipe from the anthropics/skills `pdf` skill so a subagent never has
to re-derive how to get text out of a PDF. Strategy, in order, per page:

  1. pdfplumber  — text-layer extraction with layout (best for normal PDFs)
  2. pdftotext -layout (poppler) — fast fallback for the whole file
  3. OCR via pdf2image + pytesseract — for IMAGE-ONLY scans (e.g. old Optik/
     pre-internet papers) where 1 and 2 return little/nothing

A page is treated as "scanned" (→ OCR) when text-layer extraction yields fewer
than --min-chars characters. OCR is attempted only if --ocr is set OR auto-OCR
triggers on a near-empty text layer (default on).

Usage:
  python tools/pdf_extract.py PAPER.pdf                 # all pages, auto-fallback
  python tools/pdf_extract.py PAPER.pdf --pages 3-12    # a page range (1-based)
  python tools/pdf_extract.py PAPER.pdf --pages 5       # a single page
  python tools/pdf_extract.py PAPER.pdf --no-ocr        # disable OCR fallback
  python tools/pdf_extract.py PAPER.pdf --ocr --dpi 300 # force OCR at higher dpi

Notes:
  * For a PDF that already reads fine, prefer the native Read tool with a pages=
    range — it renders pages visually and handles equations. Use THIS script when
    you want machine-readable *text* (to grep / chunk / quote) or to OCR a scan.
  * Optional deps (install only what you need):
        pip install pdfplumber           # step 1
        pip install pdf2image pytesseract # step 3 (also: brew/apt poppler, tesseract)
    poppler's `pdftotext` (step 2) is usually already present.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_pages(spec: str | None, total: int) -> list[int]:
    """Return a sorted list of 0-based page indices from a 1-based 'A-B' / 'N' spec."""
    if not spec:
        return list(range(total))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a) if a else 1
            hi = int(b) if b else total
        else:
            lo = hi = int(part)
        for p in range(lo, hi + 1):
            if 1 <= p <= total:
                out.add(p - 1)
    return sorted(out)


def page_count(pdf: Path) -> int:
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf)) as doc:
            return len(doc.pages)
    except Exception:
        pass
    # fall back to pdfinfo if available
    if shutil.which("pdfinfo"):
        try:
            out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
            for line in out.stdout.splitlines():
                if line.lower().startswith("pages:"):
                    return int(line.split(":", 1)[1].strip())
        except Exception:
            pass
    return 0


def extract_plumber(pdf: Path, pages: list[int]) -> dict[int, str]:
    """page index -> text via pdfplumber (empty dict if lib missing)."""
    try:
        import pdfplumber  # type: ignore
    except Exception:
        return {}
    res: dict[int, str] = {}
    with pdfplumber.open(str(pdf)) as doc:
        for i in pages:
            if i < len(doc.pages):
                res[i] = doc.pages[i].extract_text() or ""
    return res


def extract_pdftotext(pdf: Path, pages: list[int]) -> dict[int, str]:
    """page index -> text via `pdftotext -layout` (empty if tool missing)."""
    if not shutil.which("pdftotext"):
        return {}
    res: dict[int, str] = {}
    for i in pages:
        p = i + 1
        try:
            out = subprocess.run(
                ["pdftotext", "-layout", "-f", str(p), "-l", str(p), str(pdf), "-"],
                capture_output=True, text=True,
            )
            res[i] = out.stdout or ""
        except Exception:
            res[i] = ""
    return res


def extract_ocr(pdf: Path, pages: list[int], dpi: int) -> dict[int, str]:
    """page index -> OCR text via pdf2image + pytesseract."""
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        sys.stderr.write(
            "[pdf_extract] OCR requested but deps missing: "
            "pip install pdf2image pytesseract (and install poppler + tesseract)\n"
        )
        return {}
    res: dict[int, str] = {}
    for i in pages:
        p = i + 1
        try:
            imgs = convert_from_path(str(pdf), dpi=dpi, first_page=p, last_page=p)
            res[i] = pytesseract.image_to_string(imgs[0]) if imgs else ""
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[pdf_extract] OCR failed on page {p}: {e}\n")
            res[i] = ""
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract text from a PDF with auto-fallback to OCR.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", default=None, help="1-based range, e.g. '3-12' or '5' or '1,4-6'")
    ap.add_argument("--min-chars", type=int, default=40,
                    help="below this many chars a page is treated as scanned (OCR)")
    ap.add_argument("--ocr", action="store_true", help="force OCR on all requested pages")
    ap.add_argument("--no-ocr", action="store_true", help="never OCR, even on empty pages")
    ap.add_argument("--dpi", type=int, default=200, help="OCR rasterization DPI")
    args = ap.parse_args()

    if not args.pdf.exists():
        sys.stderr.write(f"[pdf_extract] no such file: {args.pdf}\n")
        return 2

    total = page_count(args.pdf)
    if total == 0:
        sys.stderr.write("[pdf_extract] could not determine page count; install pdfplumber or poppler\n")
        total = 10_000  # let downstream tools clamp
    pages = parse_pages(args.pages, total)
    if not pages:
        sys.stderr.write("[pdf_extract] no pages selected\n")
        return 2

    if args.ocr:
        text = extract_ocr(args.pdf, pages, args.dpi)
        used = {i: "ocr" for i in pages}
    else:
        text = extract_plumber(args.pdf, pages)
        used = {i: "pdfplumber" for i in text}
        # fill any pages pdfplumber missed (lib absent) with pdftotext
        missing = [i for i in pages if i not in text]
        if missing:
            for i, t in extract_pdftotext(args.pdf, missing).items():
                text[i], used[i] = t, "pdftotext"
        # auto-OCR the near-empty pages (scans), unless disabled
        if not args.no_ocr:
            scanned = [i for i in pages if len((text.get(i) or "").strip()) < args.min_chars]
            if scanned:
                for i, t in extract_ocr(args.pdf, scanned, args.dpi).items():
                    if len(t.strip()) > len((text.get(i) or "").strip()):
                        text[i], used[i] = t, "ocr"

    for i in pages:
        body = (text.get(i) or "").rstrip()
        sys.stdout.write(f"\n===== page {i + 1} [{used.get(i, 'none')}] =====\n{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
