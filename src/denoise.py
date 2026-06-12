"""对一个目录批量降噪, 输出到 data/denoised/{method}/{tag}/, 记录每条耗时。

方法:
    frcrn   : ClearVoice FRCRN_SE_16K (神经, 16kHz 原生)
    specsub : noisereduce 谱减法/spectral gating (stationary=True)

用法:
    python src/denoise.py --method frcrn   --input data/noisy/white_5dB --tag white_5dB
    python src/denoise.py --method specsub --input data/clean --tag clean

输出: data/denoised/{method}/{tag}/*.wav
      results/denoise_raw.csv (列: method, tag, file, audio_s, proc_s, rtf)
"""
import argparse
import csv
import glob
import os
import time

import soundfile as sf

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_CSV = os.path.join(BASE, "results", "denoise_raw.csv")


def run_frcrn(files, out_dir):
    from clearvoice import ClearVoice
    cv = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    for f in files:
        t0 = time.perf_counter()
        out = cv(input_path=f, online_write=False)
        proc_s = time.perf_counter() - t0
        sf.write(os.path.join(out_dir, os.path.basename(f)), out.squeeze(), 16000)
        yield f, proc_s


def run_specsub(files, out_dir):
    import noisereduce as nr
    for f in files:
        y, sr = sf.read(f)
        t0 = time.perf_counter()
        out = nr.reduce_noise(y=y, sr=sr, stationary=True)
        proc_s = time.perf_counter() - t0
        sf.write(os.path.join(out_dir, os.path.basename(f)), out, sr)
        yield f, proc_s


METHODS = {"frcrn": run_frcrn, "specsub": run_specsub}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--input", required=True, help="wav 目录")
    ap.add_argument("--tag", required=True, help="噪声条件标签, 如 clean / white_5dB")
    args = ap.parse_args()

    in_dir = os.path.join(BASE, args.input) if not os.path.isabs(args.input) else args.input
    files = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    assert files, f"no wav found in {in_dir}"
    out_dir = os.path.join(BASE, "data", "denoised", args.method, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if write_header:
            w.writerow(["method", "tag", "file", "audio_s", "proc_s", "rtf"])
        for f, proc_s in METHODS[args.method](files, out_dir):
            info = sf.info(f)
            dur = info.frames / info.samplerate
            w.writerow([args.method, args.tag, os.path.basename(f),
                        f"{dur:.2f}", f"{proc_s:.2f}", f"{proc_s / dur:.3f}"])
            fp.flush()
            print(f"  {os.path.basename(f)} ({dur:.1f}s, rtf={proc_s / dur:.2f})")
    print(f"denoised {len(files)} files -> {out_dir}")
