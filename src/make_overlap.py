"""从已校对的干净片段合成可控重叠语音(实验 2 用)。

把反方(con)与正方(pro)同序号片段配对,模拟辩论抢话,按三档重叠
比例错位叠加,等响度(0 dB SIR)混合。两个说话人都有 ground truth
(refs/),可严格算 CER 与说话人归属。

重叠比例定义 = 重叠时长 / min(两条时长);A 先开始,B 在 A 末尾错位插入。
    no    ratio=0.0  顺序拼接,无重叠(对照组)
    light ratio=0.3  B 在 A 结束前 30% 处切入
    heavy ratio=0.8  大段重叠,分离最难

用法:
    python src/make_overlap.py                 # 全部配对 x 三档
    python src/make_overlap.py --level light

输出:
    data/overlap_synth/{level}/{pair_id}.wav
    data/overlap_synth/manifest.csv
"""
import argparse
import csv
import glob
import os

import numpy as np
import soundfile as sf

SR = 16000
# 重叠比例 = 重叠时长 / min(两条时长)
LEVELS = {"no": 0.0, "light": 0.3, "heavy": 0.8}
BASE = os.path.join(os.path.dirname(__file__), "..")
CLEAN_DIR = os.path.join(BASE, "data", "clean")
OUT_DIR = os.path.join(BASE, "data", "overlap_synth")


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_pair(a, b, ratio):
    """A 先开始, B 按 ratio 在 A 末尾错位插入; B 缩放到与 A 等 RMS。
    返回 (mix, offset_samp, overlap_samp, total_samp)。"""
    b = b * (rms(a) / rms(b))                  # 等响度 0 dB SIR
    overlap = int(round(ratio * min(len(a), len(b))))
    offset = len(a) - overlap                  # B 起点
    total = max(len(a), offset + len(b))
    mix = np.zeros(total, dtype=np.float64)
    mix[:len(a)] += a
    mix[offset:offset + len(b)] += b
    peak = np.max(np.abs(mix))
    if peak > 0.99:                            # 防削波
        mix *= 0.99 / peak
    return mix, offset, overlap, total


def pairs():
    """con_i <-> pro_i 同序号配对(辩论正反方)。"""
    out = []
    for cf in sorted(glob.glob(os.path.join(CLEAN_DIR, "con_*.wav"))):
        idx = os.path.basename(cf).split("_")[1]      # "001.wav"
        pf = os.path.join(CLEAN_DIR, f"pro_{idx}")
        if os.path.exists(pf):
            out.append((cf, pf))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", nargs="*", choices=LEVELS.keys(),
                    default=list(LEVELS.keys()))
    args = ap.parse_args()

    rows = []
    for level in args.level:
        ratio = LEVELS[level]
        out_dir = os.path.join(OUT_DIR, level)
        os.makedirs(out_dir, exist_ok=True)
        for cf, pf in pairs():
            a, sr = sf.read(cf)
            assert sr == SR
            b, sr = sf.read(pf)
            assert sr == SR
            mix, offset, overlap, total = mix_pair(a, b, ratio)
            ci = os.path.splitext(os.path.basename(cf))[0]
            pi = os.path.splitext(os.path.basename(pf))[0]
            pair_id = f"{ci}_{pi}"
            mix_path = os.path.join(out_dir, f"{pair_id}.wav")
            sf.write(mix_path, mix.astype(np.float32), SR)
            rows.append({
                "pair_id": pair_id, "level": level,
                "ratio_target": ratio,
                "ratio_actual": round(overlap / min(len(a), len(b)), 3),
                "spkA": ci, "spkB": pi,
                "dA_sec": round(len(a) / SR, 2),
                "dB_sec": round(len(b) / SR, 2),
                "offset_sec": round(offset / SR, 2),
                "overlap_sec": round(overlap / SR, 2),
                "total_sec": round(total / SR, 2),
                "mix_path": os.path.relpath(mix_path, BASE).replace("\\", "/"),
            })
        print(f"{level} (ratio {ratio}): {len(pairs())} mixes")

    man = os.path.join(OUT_DIR, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"manifest -> {os.path.relpath(man, BASE)} ({len(rows)} rows)")
