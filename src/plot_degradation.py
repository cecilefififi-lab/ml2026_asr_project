"""画噪声退化曲线: CER vs SNR, 按 噪声类型 x ASR引擎 分线。

读 results/summary.csv (tag 格式: clean 或 {noise}_{snr}dB), 输出 results/degradation_curve.png
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
SUM_CSV = os.path.join(BASE, "results", "summary.csv")
OUT_PNG = os.path.join(BASE, "results", "degradation_curve.png")

SNR_ORDER = ["clean", "15dB", "5dB", "0dB"]

if __name__ == "__main__":
    # (noise, engine) -> {snr_label: cer}
    curves = defaultdict(dict)
    for r in csv.DictReader(open(SUM_CSV, encoding="utf-8-sig")):
        tag, engine, cer = r["tag"], r["engine"], float(r["mean_cer"])
        if tag == "clean":
            for noise in ("white", "babble"):
                curves[(noise, engine)]["clean"] = cer
        elif "_" in tag and tag.endswith("dB"):
            noise, snr = tag.rsplit("_", 1)
            curves[(noise, engine)][snr] = cer

    plt.figure(figsize=(7, 4.5))
    styles = {("white", "whisper"): ("tab:blue", "o", "-"),
              ("white", "funasr"): ("tab:blue", "s", "--"),
              ("babble", "whisper"): ("tab:red", "o", "-"),
              ("babble", "funasr"): ("tab:red", "s", "--")}
    for (noise, engine), pts in sorted(curves.items()):
        xs = [s for s in SNR_ORDER if s in pts]
        ys = [pts[s] * 100 for s in xs]
        color, marker, ls = styles.get((noise, engine), ("gray", "x", ":"))
        plt.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                 label=f"{noise} / {engine}")
    plt.xlabel("SNR")
    plt.ylabel("CER (%)")
    plt.title("ASR degradation under noise (26 clips, zh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")
