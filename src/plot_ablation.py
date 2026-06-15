"""消融出图:实验 2 五链路 content CER 均值对比(英文标签,避开中文字体)。

读 results/exp2_summary.csv,画 15 格(cond×level)content CER 均值条形图,
标注每条链路在 15 格中的最优次数 -> results/ablation_pipelines.png
直观展示:baseline 直接 ASR(L1)碾压所有预处理链路。
"""
import csv
import os
import statistics as st

import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(BASE, "results", "exp2_summary.csv")
OUT = os.path.join(BASE, "results", "ablation_pipelines.png")

LINKS = ["L1", "L2", "L3", "L4", "L5"]
LABEL = {
    "L1": "L1\ndirect ASR\n(baseline)",
    "L2": "L2\n+denoise",
    "L3": "L3\n+separate",
    "L4": "L4\ndenoise\n>separate",
    "L5": "L5\nseparate\n>denoise",
}


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    cer = {L: [] for L in LINKS}
    cells = {}
    for r in rows:
        cer[r["link"]].append(float(r["content_cer"]))
        cells.setdefault((r["cond"], r["level"]), {})[r["link"]] = float(r["content_cer"])
    means = [100 * st.mean(cer[L]) for L in LINKS]
    wins = {L: 0 for L in LINKS}
    for d in cells.values():
        wins[min(d, key=d.get)] += 1

    colors = ["#2a9d8f"] + ["#bbb"] * 4  # baseline 绿,预处理灰
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(range(len(LINKS)), means, color=colors, edgecolor="#444")
    for i, L in enumerate(LINKS):
        ax.text(i, means[i] + 1.5, f"best in\n{wins[L]}/15", ha="center",
                fontsize=8, color="#2a9d8f" if wins[L] else "#999",
                fontweight="bold" if wins[L] else "normal")
    ax.bar_label(bars, fmt="%.1f%%", padding=-18, fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(len(LINKS)), [LABEL[L] for L in LINKS], fontsize=8)
    ax.set_ylabel("Mean content CER over 15 noise×overlap cells (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Ablation: every preprocessing chain hurts vs direct ASR\n(overlapped 2-4s debate clips)")
    ax.text(0.5, -0.30, "Simplest chain (L1) wins 13 of 15 conditions; separation only edges ahead under heavy overlap.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print("means:", {L: f"{m:.1f}%" for L, m in zip(LINKS, means)}, "| wins:", wins)


if __name__ == "__main__":
    main()
