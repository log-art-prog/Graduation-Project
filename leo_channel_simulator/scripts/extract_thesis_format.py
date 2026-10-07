"""提取学长论文（陈万埼 2026）的版式规格，为 LaTeX 模板复刻提供参数。

只读分析：页面尺寸/边距估算、字体字号统计、章节与图表编号风格、参考文献样式。
运行：python scripts/extract_thesis_format.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = Path(r"c:\Users\龙贝\Desktop\leo_channel_simulator\2022113396-陈万埼-贾敏教授-基于CGAN的NGSO星座星地信道建模方法与仿真-毕业论文(3).pdf")

doc = fitz.open(PDF)
print(f"总页数: {len(doc)}")

# ---------- 1. 页面尺寸 ----------
p0 = doc[0]
print(f"页面尺寸: {p0.rect.width:.1f} x {p0.rect.height:.1f} pt "
      f"({p0.rect.width/72*25.4:.0f} x {p0.rect.height/72*25.4:.0f} mm)")

# ---------- 2. 正文字体字号统计（取正文区样本页） ----------
font_stats: Counter = Counter()
for pno in range(2, min(len(doc), 12)):  # 跳过封面/声明，取前 10 个样本页
    for b in doc[pno].get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    font_stats[(span["font"], round(span["size"], 1))] += len(t)
print("\n字体字号 TOP12（按字符数）:")
for (font, size), n in font_stats.most_common(12):
    print(f"  {font:<28} {size:>5}pt  x{n}")

# ---------- 3. 页边距估算（正文 span 的 bbox 极值，样本页） ----------
xs0, xs1, ys0 = [], [], []
for pno in range(2, min(len(doc), 12)):
    W = doc[pno].rect.width
    for b in doc[pno].get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if len(t) < 20 or not span["font"].endswith("Regular") and "SimSun" not in span["font"] and "Song" not in span["font"]:
                    continue
                x0, y0, x1, _ = span["bbox"]
                xs0.append(x0); xs1.append(W - x1); ys0.append(y0)
if xs0:
    import statistics as st
    print(f"\n边距估算(宋体长行 span): 左≈{st.median(xs0)/72*25.4:.1f}mm "
          f"右≈{st.median(xs1)/72*25.4:.1f}mm 首行顶距≈{st.median(ys0):.0f}pt")

# ---------- 4. 章节标题与图表编号风格（全文搜索） ----------
import re
pat_ch = re.compile(r"^第[一二三四五六七八九十\d]+章")
pat_sec = re.compile(r"^\d\.\d(\.\d)?\s")
pat_fig = re.compile(r"^图\s?\d+[-‐–]\d+")
pat_tab = re.compile(r"^表\s?\d+[-‐–]\d+")
found = {"ch": [], "sec": [], "fig": [], "tab": []}
for pno in range(len(doc)):
    for line in doc[pno].get_text().splitlines():
        s = line.strip()
        if pat_ch.match(s) and len(s) < 40:
            found["ch"].append((pno + 1, s))
        elif pat_sec.match(s) and len(s) < 40:
            found["sec"].append((pno + 1, s))
        elif pat_fig.match(s) and len(s) < 50:
            found["fig"].append((pno + 1, s))
        elif pat_tab.match(s) and len(s) < 50:
            found["tab"].append((pno + 1, s))
print("\n章标题样例:", [s for _, s in found["ch"]][:8])
print("节标题样例:", [s for _, s in found["sec"]][:8])
print("图编号样例:", [s for _, s in found["fig"]][:5])
print("表编号样例:", [s for _, s in found["tab"]][:5])
print(f"计数: 章{len(found['ch'])} 节{len(found['sec'])} 图{len(found['fig'])} 表{len(found['tab'])}")

# ---------- 5. 标题字体（章/节标题 span） ----------
title_spans = []
for pno in range(4, min(len(doc), 40)):
    for b in doc[pno].get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if pat_ch.match(text) and len(text) < 40:
                sp = line["spans"][0]
                title_spans.append(("章", sp["font"], round(sp["size"], 1)))
            elif pat_sec.match(text) and len(text) < 40:
                sp = line["spans"][0]
                title_spans.append(("节", sp["font"], round(sp["size"], 1)))
print("\n标题字体样例(章):", title_spans[:4])
print("标题字体样例(节):", title_spans[4:8])

# ---------- 6. 参考文献样式 ----------
for pno in range(len(doc) - 15, len(doc)):
    txt = doc[pno].get_text()
    if "参考文献" in txt:
        lines = [l.strip() for l in txt.splitlines()]
        idx = next((i for i, l in enumerate(lines) if l == "参考文献"), None)
        if idx is not None:
            print(f"\n参考文献（第{pno+1}页）样式样例:")
            for l in lines[idx:idx + 8]:
                if l:
                    print("  ", l[:80])
        break

# ---------- 7. 摘要/目录结构 ----------
print("\n前 6 页首行（判断封面/摘要/目录布局）:")
for pno in range(min(6, len(doc))):
    first = [l.strip() for l in doc[pno].get_text().splitlines() if l.strip()][:3]
    print(f"  p{pno+1}: {first}")
