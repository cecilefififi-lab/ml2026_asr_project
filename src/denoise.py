"""对一个目录批量降噪, 递归保留子目录结构, 记录每条耗时。

方法:
    frcrn   : ClearVoice FRCRN_SE_16K (神经, 16kHz 原生)
    specsub : noisereduce 谱减法/spectral gating (stationary=True)

用法:
    python src/denoise.py --method frcrn   --input data/noisy/white_5dB --tag white_5dB
    python src/denoise.py --method specsub --input data/clean --tag clean
    # 链路用: 自定义输出根, 递归保留 level 子目录
    python src/denoise.py --method frcrn --input data/overlap_noisy/babble_5dB \
        --tag ov_babble_5dB --out-dir data/exp2/denoise

输出: {out_dir}/{method}/{tag}/{相对路径}.wav
      results/denoise_raw.csv (列: method, tag, file, audio_s, proc_s, rtf)
"""
import argparse
import csv
import glob
import os
import time

import soundfile as sf

SR = 16000
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_CSV = os.path.join(BASE, "results", "denoise_raw.csv")
DENOISE_ROOT = os.path.join(BASE, "data", "denoised")


def run_frcrn(files):
    from clearvoice import ClearVoice
    cv = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    for f in files:
        t0 = time.perf_counter()
        out = cv(input_path=f, online_write=False)
        yield f, out.squeeze(), time.perf_counter() - t0


def run_specsub(files):
    import noisereduce as nr
    for f in files:
        y, sr = sf.read(f)
        t0 = time.perf_counter()
        out = nr.reduce_noise(y=y, sr=sr, stationary=True)
        yield f, out, time.perf_counter() - t0


METHODS = {"frcrn": run_frcrn, "specsub": run_specsub}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--input", required=True, help="wav 目录(递归)")
    ap.add_argument("--tag", required=True, help="条件标签, 如 clean / white_5dB")
    ap.add_argument("--out-dir", default=DENOISE_ROOT, help="输出根(下接 method/tag)")
    args = ap.parse_args()

    in_dir = os.path.join(BASE, args.input) if not os.path.isabs(args.input) else args.input
    files = sorted(glob.glob(os.path.join(in_dir, "**", "*.wav"), recursive=True))
    assert files, f"no wav found in {in_dir}"
    rels = [os.path.relpath(f, in_dir) for f in files]
    out_root = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(BASE, args.out_dir)
    out_dir = os.path.join(out_root, args.method, args.tag)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if write_header:
            w.writerow(["method", "tag", "file", "audio_s", "proc_s", "rtf"])
        for (f, audio, proc_s), rel in zip(METHODS[args.method](files), rels):
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, audio, SR)
            info = sf.info(f)
            dur = info.frames / info.samplerate
            w.writerow([args.method, args.tag, rel.replace("\\", "/"),
                        f"{dur:.2f}", f"{proc_s:.2f}", f"{proc_s / dur:.3f}"])
            fp.flush()
            print(f"  {rel} ({dur:.1f}s, rtf={proc_s / dur:.2f})")
    print(f"denoised {len(files)} files -> {out_dir}")
