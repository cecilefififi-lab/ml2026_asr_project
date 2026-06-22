"""Exp2 pipeline comparison: content CER by overlap level (Whisper).

Reads results/exp2_summary.csv and draws a grouped bar chart of the five
pipelines (L1-L5) averaged over the noise conditions, grouped by overlap level.
Visualises the over-separation story: L1 (direct ASR) is lowest at no/light
overlap, separation pipelines only catch up under heavy overlap.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "exp2_summary.csv")

links = ["L1", "L2", "L3", "L4", "L5"]
labels = {
    "L1": "L1 direct ASR",
    "L2": "L2 denoise->ASR",
    "L3": "L3 separate->ASR",
    "L4": "L4 denoise->separate",
    "L5": "L5 separate->denoise",
}
levels = ["no", "light", "heavy"]
level_x = ["no overlap", "light (0.3)", "heavy (0.8)"]

piv = (
    df.groupby(["link", "level"])["content_cer"].mean().unstack("level") * 100
).loc[links, levels]

x = np.arange(len(levels))
w = 0.16
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(links)))
for i, l in enumerate(links):
    bars = ax.bar(x + (i - 2) * w, piv.loc[l].values, w,
                  label=labels[l], color=colors[i])
    ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=1)

ax.set_xticks(x)
ax.set_xticklabels(level_x)
ax.set_ylabel("content CER (%)   (lower is better)")
ax.set_title("Exp2: pipeline comparison by overlap level\n(mean over 5 noise conditions, Whisper)")
ax.legend(fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 105)
fig.tight_layout()

out = ROOT / "results" / "exp2_pipelines.png"
fig.savefig(out, dpi=150)
print("saved", out)
print(piv.round(1))
