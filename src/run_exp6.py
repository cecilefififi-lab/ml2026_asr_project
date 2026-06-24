"""实验 6 链路编排:长音频上的处理顺序对比(L1/L3/L4/L5)。

复用实验 2 的组件脚本(denoise / separate_mossformer_ms / run_asr), 仅换 exp6
数据目录与条件。砍掉 L2(纯降噪, 不涉及顺序), 保留:
    L1 直接          mix
    L3 分离->ASR     separate(mix)
    L4 降噪->分离    separate(denoise(mix))   <- 顺序对比主角
    L5 分离->降噪    denoise(separate(mix))   <- 顺序对比主角

按环节分进程批处理(每模型只加载一次, 避开 6GB 显存同时驻留)。ASR 结果写
results/exp6_asr.csv, file 列 = {cond}/{level}/{sample}[_spkN], 供 eval_exp6 解析。

用法:
    python src/run_exp6.py                       # 全流程
    python src/run_exp6.py --skip-prep           # 仅重跑 ASR
"""
import argparse
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
EXP6 = os.path.join(BASE, "data", "exp6")
CLEAN = os.path.join(EXP6, "clean")
NOISY = os.path.join(EXP6, "noisy")
OUT_CSV = os.path.join(BASE, "results", "exp6_asr.csv")
PY = sys.executable
ALL_COND = ["clean", "babble_0dB"]


def sh(script, *a):
    cmd = [PY, os.path.join(SRC, script), *[str(x) for x in a]]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_mix(conds):
    """汇集各 cond 的混合到 data/exp6/mix/{cond}/{level}/*.wav。"""
    mix = os.path.join(EXP6, "mix")
    for cond in conds:
        src = CLEAN if cond == "clean" else os.path.join(NOISY, cond)
        assert os.path.isdir(src), f"缺少输入: {src}(先跑 make_exp6 / add_noise)"
        dst = os.path.join(mix, cond)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return mix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoise", choices=["frcrn", "specsub"], default="specsub")
    ap.add_argument("--engine", nargs="+", default=["whisper"])
    ap.add_argument("--cond", nargs="+", default=ALL_COND)
    ap.add_argument("--skip-prep", action="store_true",
                    help="跳过降噪/分离, 仅重跑 ASR(中间产物须已存在)")
    args = ap.parse_args()

    m = args.denoise
    mix = build_mix(args.cond)
    D = os.path.join(EXP6, "L2_denoise", m, "exp6_L2")   # denoise(mix), 供 L4
    S = os.path.join(EXP6, "L3_sep")                     # separate(mix)
    SD = os.path.join(EXP6, "L4_sep")                    # separate(D)
    DS = os.path.join(EXP6, "L5_denoise", m, "exp6_L5")  # denoise(S)

    if not args.skip_prep:
        sh("denoise.py", "--method", m, "--input", mix,
           "--tag", "exp6_L2", "--out-dir", os.path.join(EXP6, "L2_denoise"))
        sh("separate_mossformer_ms.py", mix, "--out-dir", S,
           "--out-sr", 16000, "--tag", "exp6_L3")
        sh("separate_mossformer_ms.py", D, "--out-dir", SD,
           "--out-sr", 16000, "--tag", "exp6_L4")
        sh("denoise.py", "--method", m, "--input", S,
           "--tag", "exp6_L5", "--out-dir", os.path.join(EXP6, "L5_denoise"))

    for eng in args.engine:
        for tag, tree in [("L1", mix), ("L3", S), ("L4", SD), ("L5", DS)]:
            sh("run_asr.py", "--engine", eng, "--input", tree,
               "--tag", tag, "--out-csv", OUT_CSV)

    print(f"\n完成。链路 ASR 结果 -> {os.path.relpath(OUT_CSV, BASE)}")
