"""实验 4 出图:VAD on/off 各输入类型的幻觉率对比(英文标签,避开中文字体问题)。

读 results/exp4_hallucination.csv,按输入类型(white / silence / babble)统计
非空转写(=幻觉)比例,VAD off vs on 分组柱状图 -> results/exp4_vad_compare.png
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(BASE, "results", "exp4_hallucination.csv")
OUT = os.path.join(BASE, "results", "exp4_vad_compare.png")

GROUPS = ["white", "silence", "babble"]


def kind(rel):
    name = rel.split("/")[-1]
    if name.startswith("white"):
        return "white"
    if name.startswith("babble"):
        return "babble"
    return "silence"  # zero_* / floor_*


def main():
    # tag -> group -> [n_total, n_hallucinated]
    stat = {t: {g: [0, 0] for g in GROUPS} for t in ("exp4_vadoff", "exp4_vadon")}
    with open(CSV, encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["tag"] not in stat:
                continue
            g = kind(row["file"])
            stat[row["tag"]][g][0] += 1
            if row["text"].strip():
                stat[row["tag"]][g][1] += 1

    def rate(tag):
        return [100 * stat[tag][g][1] / stat[tag][g][0] if stat[tag][g][0] else 0
                for g in GROUPS]

    off, on = rate("exp4_vadoff"), rate("exp4_vadon")
    x = np.arange(len(GROUPS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, off, w, label="VAD off", color="#d1495b")
    b2 = ax.bar(x + w / 2, on, w, label="VAD on (Silero)", color="#30638e")
    ax.bar_label(b1, fmt="%.0f%%", fontsize=9)
    ax.bar_label(b2, fmt="%.0f%%", fontsize=9)
    ax.set_xticks(x, [g + f"\n(n={stat['exp4_vadoff'][g][0]})" for g in GROUPS])
    ax.set_ylabel("Hallucination rate (% of clips with non-empty output)")
    ax.set_ylim(0, 115)
    ax.set_title("Whisper hallucination on speechless audio: VAD off vs on")
    ax.legend()
    ax.text(0.5, -0.22,
            "VAD silences white-noise & silence completely, but barely dents babble (speech-like noise).",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("VAD off:", dict(zip(GROUPS, [f'{v:.0f}%' for v in off])))
    print("VAD on :", dict(zip(GROUPS, [f'{v:.0f}%' for v in on])))


if __name__ == "__main__":
    main()
