"""按目标 SNR 给干净音频加噪。

用法:
    python src/add_noise.py                      # 全部 clean x 全部噪声 x 全部 SNR
    python src/add_noise.py --snr 5 --noise white

输出: data/noisy/{noise}_{snr}dB/{原文件名}.wav
同时打印实测 SNR 用于校验(应与目标一致, 容差 ±0.1 dB)。
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf

SR = 16000
SNRS = [15, 5, 0]
SEED = 42
BASE = os.path.join(os.path.dirname(__file__), "..")
CLEAN_DIR = os.path.join(BASE, "data", "clean")
NOISE_DIR = os.path.join(BASE, "data", "noise")
NOISY_DIR = os.path.join(BASE, "data", "noisy")


def mix_at_snr(clean, noise, snr_db, rng):
    """返回 (混合信号, 实测SNR)。噪声随机取段, 必要时循环补齐。"""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    start = rng.integers(0, len(noise) - len(clean) + 1)
    noise = noise[start:start + len(clean)]

    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    scaled_noise = scale * noise
    mixed = clean + scaled_noise

    peak = np.max(np.abs(mixed))
    if peak > 0.99:  # 防削波, 等比缩放不改变 SNR
        mixed *= 0.99 / peak
        scaled_noise *= 0.99 / peak
        clean = clean * 0.99 / peak
    actual_snr = 10 * np.log10(np.mean(clean ** 2) / np.mean(scaled_noise ** 2))
    return mixed, actual_snr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", nargs="*", help="噪声名(默认全部)")
    ap.add_argument("--snr", nargs="*", type=int, default=SNRS)
    ap.add_argument("--in-dir", default=CLEAN_DIR, help="输入音频目录(递归)")
    ap.add_argument("--out-dir", default=NOISY_DIR, help="输出根目录")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
    noise_files = [f for f in noise_files if not os.path.basename(f).startswith("_")]
    if args.noise:
        noise_files = [f for f in noise_files
                       if os.path.splitext(os.path.basename(f))[0] in args.noise]
    clean_files = sorted(glob.glob(os.path.join(args.in_dir, "**", "*.wav"),
                                   recursive=True))

    for nf in noise_files:
        noise_name = os.path.splitext(os.path.basename(nf))[0]
        noise, sr = sf.read(nf)
        assert sr == SR
        for snr in args.snr:
            out_dir = os.path.join(args.out_dir, f"{noise_name}_{snr}dB")
            errs = []
            for cf in clean_files:
                clean, sr = sf.read(cf)
                assert sr == SR
                mixed, actual = mix_at_snr(clean, noise, snr, rng)
                dst = os.path.join(out_dir, os.path.relpath(cf, args.in_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                sf.write(dst, mixed.astype(np.float32), SR)
                errs.append(abs(actual - snr))
            print(f"{noise_name} @ {snr}dB: {len(clean_files)} files, "
                  f"max SNR error {max(errs):.3f} dB")
