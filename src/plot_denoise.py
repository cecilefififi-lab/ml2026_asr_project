"""实验 1 出图: 降噪方法对比。

读 results/summary.csv (tag 格式: {条件} 或 {条件}__{方法}), 输出:
  results/exp1_denoise_curves.png  退化曲线 2x2 (噪声 x 引擎), 每图三线: 无处理/frcrn/specsub
  results/exp1_denoise_heatmap.png 热力图: 行=条件, 列=引擎x方法, 值=CER%
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
SUM_CSV = os.path.join(BASE, "results", "summary.csv")
OUT_CURVES = os.path.join(BASE, "results", "exp1_denoise_curves.png")
OUT_HEAT = os.path.join(BASE, "results", "exp1_denoise_heatmap.png")

SNR_ORDER = ["clean", "15dB", "5dB", "0dB"]
METHODS = ["none", "frcrn", "specsub"]
METHOD_LABEL = {"none": "no processing", "frcrn": "FRCRN", "specsub": "spectral sub."}

if __name__ == "__main__":
    # (noise, engine, method) -> {snr_label: cer}
    curves = defaultdict(dict)
    for r in csv.DictReader(open(SUM_CSV, encoding="utf-8-sig")):
        tag, engine, cer = r["tag"], r["engine"], float(r["mean_cer"])
        cond, _, method = tag.partition("__")
        method = method or "none"
        if cond == "clean":
            for noise in ("white", "babble"):
                curves[(noise, engine, method)]["clean"] = cer
        elif "_" in cond and cond.endswith("dB"):
            noise, snr = cond.rsplit("_", 1)
            curves[(noise, engine, method)][snr] = cer

    # 图 1: 2x2 退化曲线
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    method_style = {"none": ("black", "o", "-"),
                    "frcrn": ("tab:green", "s", "--"),
                    "specsub": ("tab:orange", "^", ":")}
    for i, noise in enumerate(["white", "babble"]):
        for j, engine in enumerate(["whisper", "funasr"]):
            ax = axes[i][j]
            for method in METHODS:
                pts = curves.get((noise, engine, method), {})
                xs = [s for s in SNR_ORDER if s in pts]
                ys = [pts[s] * 100 for s in xs]
                color, marker, ls = method_style[method]
                ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                        label=METHOD_LABEL[method])
            ax.set_title(f"{noise} / {engine}")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("SNR")
            if j == 0:
                ax.set_ylabel("CER (%)")
    axes[0][0].legend()
    fig.suptitle("Does denoising help ASR? (26 clips, zh)")
    fig.tight_layout()
    fig.savefig(OUT_CURVES, dpi=150)
    print(f"wrote {OUT_CURVES}")

    # 图 2: 热力图 行=条件, 列=引擎x方法
    conds = ["clean"] + [f"{n}_{s}" for n in ("white", "babble") for s in ("15dB", "5dB", "0dB")]
    cols = [(e, m) for e in ("whisper", "funasr") for m in METHODS]
    mat = np.full((len(conds), len(cols)), np.nan)
    for ci, cond in enumerate(conds):
        noise = cond.rsplit("_", 1)[0] if cond != "clean" else "white"
        snr = cond.rsplit("_", 1)[1] if cond != "clean" else "clean"
        for cj, (engine, method) in enumerate(cols):
            pts = curves.get((noise, engine, method), {})
            if snr in pts:
                mat[ci, cj] = pts[snr] * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)),
                  [f"{e}\n{METHOD_LABEL[m]}" for e, m in cols], fontsize=8)
    ax.set_yticks(range(len(conds)), conds)
    for ci in range(len(conds)):
        for cj in range(len(cols)):
            if not np.isnan(mat[ci, cj]):
                ax.text(cj, ci, f"{mat[ci, cj]:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if mat[ci, cj] > 60 else "black")
    fig.colorbar(im, label="CER (%)")
    ax.set_title("CER (%) by condition x engine x denoising")
    fig.tight_layout()
    fig.savefig(OUT_HEAT, dpi=150)
    print(f"wrote {OUT_HEAT}")
