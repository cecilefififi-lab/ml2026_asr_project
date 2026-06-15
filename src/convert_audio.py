"""把 m4a/任意音频解码并重采样为 16kHz mono wav (用 PyAV, 不依赖系统 ffmpeg)。

本机无 ffmpeg 二进制, 但 venv 自带 av(PyAV), 其内置 ffmpeg 库可解码 m4a。
下游 add_noise / denoise / separate / run_asr 统一吃 16kHz mono wav。

用法:
    python src/convert_audio.py luyin --out-dir data/real/raw
    python src/convert_audio.py luyin/canteen_01.m4a --out-dir data/real/raw
"""
import argparse
import glob
import os

import av
import numpy as np
import soundfile as sf

SR = 16000
BASE = os.path.join(os.path.dirname(__file__), "..")


def decode_mono16k(path):
    """解码 -> float32 mono 16kHz 一维数组。"""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
    container = av.open(path)
    chunks = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    for rf in resampler.resample(None):  # flush
        chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="音频文件或目录")
    ap.add_argument("--out-dir", default=os.path.join("data", "real", "raw"))
    ap.add_argument("--ext", default=".m4a", help="目录模式下匹配的扩展名")
    args = ap.parse_args()

    in_path = args.input if os.path.isabs(args.input) else os.path.join(BASE, args.input)
    if os.path.isdir(in_path):
        files = sorted(glob.glob(os.path.join(in_path, f"*{args.ext}")))
    else:
        files = [in_path]
    assert files, f"no input found in {in_path}"

    out_root = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(BASE, args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    for f in files:
        y = decode_mono16k(f)
        stem = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(out_root, stem + ".wav")
        sf.write(dst, y, SR)
        print(f"  {os.path.basename(f)} -> {os.path.relpath(dst, BASE)} ({len(y)/SR:.1f}s)")
    print(f"converted {len(files)} files -> {out_root}")
