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


def run_whisper(files, model_size, vad_filter=False):
    from faster_whisper import WhisperModel
    device = get_device()
    compute = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    print(f"faster-whisper {model_size} on {device} ({compute}), vad_filter={vad_filter}")
    for f in files:
        t0 = time.perf_counter()
        segments, _ = model.transcribe(f, language="zh", beam_size=5, vad_filter=vad_filter)
        text = "".join(s.text for s in segments)  # 迭代器在此真正执行
        yield f, text, time.perf_counter() - t0


def run_funasr(files, _model_size, _vad_filter=False):
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
    ap.add_argument("--input", required=True, help="wav 目录(递归)或单个 wav")
    ap.add_argument("--tag", required=True, help="实验条件标签, 如 clean / L1")
    ap.add_argument("--model-size", default="large-v3", help="whisper 模型规格")
    ap.add_argument("--out-csv", default=OUT_CSV, help="结果 csv 路径")
    ap.add_argument("--vad-filter", action="store_true",
                    help="whisper 启用 Silero VAD 过滤(实验4 VAD on/off 对照)")
    args = ap.parse_args()

    path = os.path.join(BASE, args.input) if not os.path.isabs(args.input) else args.input
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.wav"), recursive=True))
        base = path
    else:
        files = [path]
        base = os.path.dirname(path)
    assert files, f"no wav found in {path}"

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(BASE, args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if write_header:
            w.writerow(["tag", "engine", "file", "text", "audio_s", "proc_s", "rtf"])
        for f, text, proc_s in ENGINES[args.engine](files, args.model_size, args.vad_filter):
            dur = audio_duration(f)
            rel = os.path.relpath(f, base).replace("\\", "/")
            w.writerow([args.tag, args.engine, rel, text,
                        f"{dur:.2f}", f"{proc_s:.2f}", f"{proc_s / dur:.3f}"])
            fp.flush()
            print(f"  {rel} ({dur:.1f}s, rtf={proc_s / dur:.2f}): {text[:50]}")
    print(f"appended {len(files)} rows -> {out_csv}")
