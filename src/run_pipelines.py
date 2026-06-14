"""实验 2 五链路编排:噪声 + 重叠语音的处理顺序对比。

链路(处理后送 ASR):
    L1 直接          mix
    L2 降噪->ASR     denoise(mix)
    L3 分离->ASR     separate(mix)
    L4 降噪->分离    separate(denoise(mix))
    L5 分离->降噪    denoise(separate(mix))

按环节分进程批处理(每个模型只加载一次,避免 6GB 显存同时驻留三模型),
中间产物落盘到 data/exp2/。ASR 结果写 results/exp2_asr.csv,
file 列为 {cond}/{level}/{pair}[_spkN],供评测解析说话人与重叠档。

用法:
    python src/run_pipelines.py --denoise specsub --engine whisper
    python src/run_pipelines.py --denoise frcrn --engine whisper funasr \
        --cond clean babble_5dB babble_0dB
    python src/run_pipelines.py --engine funasr --skip-prep   # 仅补跑 ASR
"""
import argparse
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
EXP2 = os.path.join(BASE, "data", "exp2")
SYNTH = os.path.join(BASE, "data", "overlap_synth")
NOISY = os.path.join(BASE, "data", "overlap_noisy")
OUT_CSV = os.path.join(BASE, "results", "exp2_asr.csv")
PY = sys.executable
ALL_COND = ["clean", "babble_5dB", "babble_0dB", "white_5dB", "white_0dB"]


def sh(script, *a):
    cmd = [PY, os.path.join(SRC, script), *[str(x) for x in a]]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_mix(conds):
    """汇集各 cond 的混合到 data/exp2/mix/{cond}/{level}/*.wav。"""
    mix = os.path.join(EXP2, "mix")
    for cond in conds:
        src = SYNTH if cond == "clean" else os.path.join(NOISY, cond)
        assert os.path.isdir(src), f"缺少输入: {src}(先跑 make_overlap / add_noise)"
        dst = os.path.join(mix, cond)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.csv"))
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
    D = os.path.join(EXP2, "L2_denoise", m, "exp2_L2")   # denoise(mix)
    S = os.path.join(EXP2, "L3_sep")                     # separate(mix)
    SD = os.path.join(EXP2, "L4_sep")                    # separate(D)
    DS = os.path.join(EXP2, "L5_denoise", m, "exp2_L5")  # denoise(S)

    if not args.skip_prep:
        # 环节顺序固定:先 mix->D,再分离 mix/D,最后对分离结果降噪
        sh("denoise.py", "--method", m, "--input", mix,
           "--tag", "exp2_L2", "--out-dir", os.path.join(EXP2, "L2_denoise"))
        sh("separate_mossformer_ms.py", mix, "--out-dir", S,
           "--out-sr", 16000, "--tag", "exp2_L3")
        sh("separate_mossformer_ms.py", D, "--out-dir", SD,
           "--out-sr", 16000, "--tag", "exp2_L4")
        sh("denoise.py", "--method", m, "--input", S,
           "--tag", "exp2_L5", "--out-dir", os.path.join(EXP2, "L5_denoise"))

    for eng in args.engine:
        for tag, tree in [("L1", mix), ("L2", D), ("L3", S), ("L4", SD), ("L5", DS)]:
            sh("run_asr.py", "--engine", eng, "--input", tree,
               "--tag", tag, "--out-csv", OUT_CSV)

    print(f"\n完成。链路 ASR 结果 -> {os.path.relpath(OUT_CSV, BASE)}")
