"""实验 6 出图:短片段(exp2)vs 长音频(exp6)的 content CER 对比。

读 results/exp2_summary.csv 与 results/exp6_summary.csv, 在 4 个 (cond,level)
格子上并排 L1/L3/L4/L5 的 short vs long, 直观回答"处理顺序/分离结论是否短片段
特有"。英文标签(本机 matplotlib 未配中文字体)。

输出: results/exp6_length.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS = ["L1", "L3", "L4", "L5"]
CELLS = [("clean", "light"), ("clean", "heavy"),
         ("babble_0dB", "light"), ("babble_0dB", "heavy"),
         ("white_0dB", "light"), ("white_0dB", "heavy")]
OUT = os.path.join(BASE, "results", "exp6_length.png")


def load(path):
    """(link,cond,level) -> content_cer(%)，仅 whisper。"""
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["engine"] != "whisper":
            continue
        if r["content_cer"]:
            d[(r["link"], r["cond"], r["level"])] = float(r["content_cer"]) * 100
    return d


short = load(os.path.join(BASE, "results", "exp2_summary.csv"))
long = load(os.path.join(BASE, "results", "exp6_summary.csv"))

fig, axes = plt.subplots(3, 2, figsize=(11, 10))
x = np.arange(len(LINKS))
w = 0.38
for ax, (cond, level) in zip(axes.ravel(), CELLS):
    s = [short.get((L, cond, level), np.nan) for L in LINKS]
    g = [long.get((L, cond, level), np.nan) for L in LINKS]
    ax.bar(x - w / 2, s, w, label="short ~2.4s (exp2)", color="#7aa6c2")
    ax.bar(x + w / 2, g, w, label="long ~12s (exp6)", color="#c2785a")
    for i, (sv, gv) in enumerate(zip(s, g)):
        if not np.isnan(sv):
            ax.text(i - w / 2, sv + 1, f"{sv:.0f}", ha="center", va="bottom", fontsize=8)
        if not np.isnan(gv):
            ax.text(i + w / 2, gv + 1, f"{gv:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{cond} / {level}")
    ax.set_xticks(x)
    ax.set_xticklabels(LINKS)
    ax.set_ylabel("content CER (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
axes.ravel()[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Experiment 6: does the 'separation/order' conclusion hold on long audio?\n"
             "(Whisper, content CER; lower is better)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=130)
print(f"wrote {os.path.relpath(OUT, BASE)}")
