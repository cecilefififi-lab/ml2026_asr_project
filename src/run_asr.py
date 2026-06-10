"""对一个目录(或单文件)跑 ASR, 记录文本 + 耗时 + RTF, 追加写入 CSV。

用法:
    python src/run_asr.py --engine whisper --input data/clean --tag clean
    python src/run_asr.py --engine funasr  --input data/noisy/white_5dB --tag white_5dB

输出: results/asr_raw.csv (列: tag, engine, file, text, audio_s, proc_s, rtf)
"""
import argparse
import csv
import glob
import os
import time

import soundfile as sf

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_CSV = os.path.join(BASE, "results", "asr_raw.csv")


def audio_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_whisper(files, model_size):
    from faster_whisper import WhisperModel
    device = get_device()
    compute = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    print(f"faster-whisper {model_size} on {device} ({compute})")
    for f in files:
        t0 = time.perf_counter()
        segments, _ = model.transcribe(f, language="zh", beam_size=5)
        text = "".join(s.text for s in segments)  # 迭代器在此真正执行
        yield f, text, time.perf_counter() - t0


def run_funasr(files, _model_size):
    from funasr import AutoModel
    device = get_device()
    model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                      punc_model="ct-punc", device=device, disable_update=True)
    print(f"funasr paraformer-zh on {device}")
    for f in files:
        t0 = time.perf_counter()
        res = model.generate(input=f, batch_size=1)
        text = res[0]["text"] if res else ""
        yield f, text, time.perf_counter() - t0


ENGINES = {"whisper": run_whisper, "funasr": run_funasr}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=ENGINES)
    ap.add_argument("--input", required=True, help="wav 目录或单个 wav")
    ap.add_argument("--tag", required=True, help="实验条件标签, 如 clean / white_5dB")
    ap.add_argument("--model-size", default="large-v3", help="whisper 模型规格")
    args = ap.parse_args()

    path = os.path.join(BASE, args.input) if not os.path.isabs(args.input) else args.input
    files = sorted(glob.glob(os.path.join(path, "*.wav"))) if os.path.isdir(path) else [path]
    assert files, f"no wav found in {path}"

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if write_header:
            w.writerow(["tag", "engine", "file", "text", "audio_s", "proc_s", "rtf"])
        for f, text, proc_s in ENGINES[args.engine](files, args.model_size):
            dur = audio_duration(f)
            w.writerow([args.tag, args.engine, os.path.basename(f), text,
                        f"{dur:.2f}", f"{proc_s:.2f}", f"{proc_s / dur:.3f}"])
            fp.flush()
            print(f"  {os.path.basename(f)} ({dur:.1f}s, rtf={proc_s / dur:.2f}): {text[:50]}")
    print(f"appended {len(files)} rows -> {OUT_CSV}")
