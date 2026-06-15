"""阶段三:真实录音泛化抽查编排。

对 6 条真实录音(已由 convert_audio.py 转 16k mono wav)跑:
  baseline = 直接 ASR (whisper VAD off / VAD on / funasr)
  最佳链路按录音类型自动选:
    单人噪声条(dorm/canteen/classroom): 降噪->ASR (specsub + frcrn)
    重叠条(discussion_*):                分离->ASR (L3) / 降噪->分离->ASR (L4)

中间产物落盘 data/real/ 供视频前后试听; ASR 结果写 results/real_asr.csv。
不算 CER(无逐字 ground truth, 文稿可意译), 文本对照交人工 spot-check。

用法:
    python src/run_real.py                 # 全跑
    python src/run_real.py --skip-prep     # 仅重跑 ASR(中间产物须已存在)
"""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
REAL = os.path.join(BASE, "data", "real")
RAW = os.path.join(REAL, "raw")
OUT_CSV = os.path.join(BASE, "results", "real_asr.csv")
PY = sys.executable

SINGLE = ["dorm_clean_01", "canteen_01", "classroom_01"]
OVERLAP = ["discussion_quiet_01", "discussion_canteen_01", "discussion_canteen_heavy_01"]


def sh(script, *a):
    cmd = [PY, os.path.join(SRC, script), *[str(x) for x in a]]
    print(">>", " ".join(os.path.relpath(c, BASE) if os.path.isabs(c) else c for c in cmd))
    subprocess.run(cmd, check=True)


def stage(stems, src_dir, dst_dir):
    """把指定 stem 的 wav 收集到 dst_dir(分离脚本按目录批处理)。"""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    for s in stems:
        shutil.copy(os.path.join(src_dir, s + ".wav"), os.path.join(dst_dir, s + ".wav"))
    return dst_dir


def main():
    skip_prep = "--skip-prep" in sys.argv
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)  # 本实验单独成表, 重跑覆盖

    dns = os.path.join(REAL, "denoised")           # denoised/{method}/real/*.wav
    ov_raw = os.path.join(REAL, "_ov_raw")         # 重叠原始(分离 L3 输入)
    ov_dns = os.path.join(REAL, "_ov_dns")         # 重叠 frcrn 降噪(分离 L4 输入)
    sep_l3 = os.path.join(REAL, "sep_L3")
    sep_l4 = os.path.join(REAL, "sep_L4")

    if not skip_prep:
        # 降噪(全 6 条, 两法): 单人条用于降噪链路, frcrn 重叠用于 L4
        sh("denoise.py", "--method", "specsub", "--input", RAW, "--tag", "real",
           "--out-dir", dns)
        sh("denoise.py", "--method", "frcrn", "--input", RAW, "--tag", "real",
           "--out-dir", dns)
        # 分离 L3: 重叠原始
        stage(OVERLAP, RAW, ov_raw)
        sh("separate_mossformer_ms.py", ov_raw, "--out-dir", sep_l3,
           "--out-sr", 16000, "--tag", "real_L3")
        # 分离 L4: 重叠 frcrn 降噪后
        stage(OVERLAP, os.path.join(dns, "frcrn", "real"), ov_dns)
        sh("separate_mossformer_ms.py", ov_dns, "--out-dir", sep_l4,
           "--out-sr", 16000, "--tag", "real_L4")

    # ---- baseline ----
    sh("run_asr.py", "--engine", "whisper", "--input", RAW, "--tag", "base",
       "--out-csv", OUT_CSV)
    sh("run_asr.py", "--engine", "whisper", "--input", RAW, "--tag", "base_vad",
       "--vad-filter", "--out-csv", OUT_CSV)
    sh("run_asr.py", "--engine", "funasr", "--input", RAW, "--tag", "base",
       "--out-csv", OUT_CSV)

    # ---- 降噪链路(单人条主用, 全 6 条都跑便于对照) ----
    sh("run_asr.py", "--engine", "whisper", "--input",
       os.path.join(dns, "specsub", "real"), "--tag", "specsub", "--out-csv", OUT_CSV)
    sh("run_asr.py", "--engine", "whisper", "--input",
       os.path.join(dns, "frcrn", "real"), "--tag", "frcrn", "--out-csv", OUT_CSV)

    # ---- 分离链路(重叠条) ----
    sh("run_asr.py", "--engine", "whisper", "--input", sep_l3, "--tag", "L3_sep",
       "--out-csv", OUT_CSV)
    sh("run_asr.py", "--engine", "whisper", "--input", sep_l4, "--tag", "L4_dns_sep",
       "--out-csv", OUT_CSV)

    print(f"\n完成。真实录音抽查 ASR -> {os.path.relpath(OUT_CSV, BASE)}")


if __name__ == "__main__":
    main()
