"""实验 6 长音频:把多条单说话人短片段拼成长说话人, 再按重叠档混合。

动机:实验 2 在 2-4s 短片段上得到"分离/处理顺序几乎无差异"的结论, 但该结论
可能是短片段本身难分(spk CER 84-88%)的退化情形。本实验把同一说话人的多条
片段拼成 ~12s 长音频再混合, 验证"处理顺序无差异"是否短片段特有 → 把退化观察
升级为边界条件定位。

数据来源同实验 2(学长 62.4s 双人辩论质询片段切出的单人片段, 论文表 4.12):
con_* 全为反方一人, pro_* 全为正方一人 → 拼接得到的是连贯的真实单说话人,
不构成"假说话人"。两个说话人都有 ground truth(refs 按拼接顺序拼接)。

用法:
    python src/make_exp6.py                 # 5 条样本 x {light,heavy}
    python src/make_exp6.py --n 6 --target 14

输出:
    data/exp6/clean/{level}/s{i}.wav
    data/exp6/manifest.csv  (含每条样本拼接的 con/pro 片段及拼接后 refA/refB)
"""
import argparse
import csv
import glob
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_overlap import mix_pair, SR  # 复用实验 2 等响度错位混合逻辑

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE, "data", "clean")
REF_DIR = os.path.join(BASE, "refs")
OUT_DIR = os.path.join(BASE, "data", "exp6")
LEVELS = {"light": 0.3, "heavy": 0.8}  # 无重叠(no)对分离无意义, 砍掉
SEED = 6


def ref_text(stem):
    p = os.path.join(REF_DIR, stem + ".txt")
    return open(p, encoding="utf-8").read().strip()


def build_long(clips):
    """按给定顺序拼接片段为长音频; 返回 (signal, ref 拼接)。"""
    sig = []
    for stem in clips:
        y, sr = sf.read(os.path.join(CLEAN_DIR, stem + ".wav"))
        assert sr == SR
        sig.append(y)
    return np.concatenate(sig), "".join(ref_text(s) for s in clips)


def pick_until(order, target_s):
    """按 order 取片段直到累计时长 >= target_s; 返回 stem 列表。"""
    chosen, dur = [], 0.0
    for stem in order:
        info = sf.info(os.path.join(CLEAN_DIR, stem + ".wav"))
        chosen.append(stem)
        dur += info.frames / info.samplerate
        if dur >= target_s:
            break
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="长样本条数")
    ap.add_argument("--target", type=float, default=11.0, help="每说话人目标时长 s")
    args = ap.parse_args()

    con = sorted(os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(os.path.join(CLEAN_DIR, "con_*.wav")))
    pro = sorted(os.path.splitext(os.path.basename(f))[0]
                 for f in glob.glob(os.path.join(CLEAN_DIR, "pro_*.wav")))

    rows = []
    for level in LEVELS:
        os.makedirs(os.path.join(OUT_DIR, "clean", level), exist_ok=True)

    for i in range(args.n):
        rng = random.Random(SEED + i)              # 每条样本固定可复现的抽样顺序
        a_clips = pick_until(rng.sample(con, len(con)), args.target)
        b_clips = pick_until(rng.sample(pro, len(pro)), args.target)
        longA, refA = build_long(a_clips)
        longB, refB = build_long(b_clips)
        sid = f"s{i}"
        for level, ratio in LEVELS.items():
            mix, offset, overlap, total = mix_pair(longA, longB, ratio)
            out = os.path.join(OUT_DIR, "clean", level, f"{sid}.wav")
            sf.write(out, mix.astype(np.float32), SR)
            rows.append({
                "sample_id": sid, "level": level,
                "conA_clips": "+".join(a_clips), "proB_clips": "+".join(b_clips),
                "dA_sec": round(len(longA) / SR, 2),
                "dB_sec": round(len(longB) / SR, 2),
                "offset_sec": round(offset / SR, 2),
                "overlap_sec": round(overlap / SR, 2),
                "total_sec": round(total / SR, 2),
                "refA": refA, "refB": refB,
            })
            print(f"  {sid}/{level}: A={len(a_clips)}条 {len(longA)/SR:.1f}s "
                  f"B={len(b_clips)}条 {len(longB)/SR:.1f}s "
                  f"overlap={overlap/SR:.1f}s total={total/SR:.1f}s")

    man = os.path.join(OUT_DIR, "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{args.n} 条样本 x {len(LEVELS)} 档 -> {os.path.relpath(man, BASE)} "
          f"({len(rows)} 行)")
