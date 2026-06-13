"""MossFormer2 双说话人分离 (ModelScope damo/speech_mossformer2_separation_temporal_8k, 8kHz)。
学长论文使用的模型。模型在 8kHz 分离, 输出可上采样到 --out-sr 供下游 ASR / 降噪 使用。

用法:
    # 单文件 (输出 8kHz, 兼容旧用法)
    python src/separate_mossformer_ms.py data/overlap/MidOverlap.wav
    # 目录批处理 (递归, 保留子目录结构, 上采样到 16k, 记录耗时)
    python src/separate_mossformer_ms.py data/overlap_synth --out-dir data/exp2/sep/clean \
        --out-sr 16000 --tag ov_clean

输出: {out_dir}/{相对路径去扩展名}_spk{1,2}.wav
      results/separate_raw.csv (列: tag, file, audio_s, proc_s, rtf)  [仅 --tag 时]
"""
import argparse
import csv
import glob
import os
import time

import pyarrow.dataset  # noqa: F401  必须先于 librosa 导入, 否则 Windows 下 DLL 冲突段错误

import librosa
import numpy as np
import soundfile as sf

SEP_SR = 8000
BASE = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_OUT = os.path.join(BASE, "data", "separated_mossformer_ms")
SEP_CSV = os.path.join(BASE, "results", "separate_raw.csv")


def separate(p, in_path, out_stem, out_sr):
    """分离单文件, 写两路到 {out_stem}_spk{1,2}.wav; 返回 (audio_s, proc_s)。"""
    y, _ = librosa.load(in_path, sr=SEP_SR)
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    tmp = out_stem + "_8k_tmp.wav"
    sf.write(tmp, y, SEP_SR)
    t0 = time.perf_counter()
    result = p(tmp)
    proc_s = time.perf_counter() - t0
    os.remove(tmp)
    for i, pcm in enumerate(result["output_pcm_list"]):
        sig = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if out_sr != SEP_SR:
            sig = librosa.resample(sig, orig_sr=SEP_SR, target_sr=out_sr)
        sf.write(f"{out_stem}_spk{i + 1}.wav", sig, out_sr)
    return len(y) / SEP_SR, proc_s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="wav 文件或目录(目录递归)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="输出根目录")
    ap.add_argument("--out-sr", type=int, default=SEP_SR, help="输出采样率(默认 8k)")
    ap.add_argument("--tag", help="给定则把耗时写入 results/separate_raw.csv")
    args = ap.parse_args()

    in_path = args.input if os.path.isabs(args.input) else os.path.join(BASE, args.input)
    out_root = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(BASE, args.out_dir)
    if os.path.isdir(in_path):
        files = sorted(glob.glob(os.path.join(in_path, "**", "*.wav"), recursive=True))
        rels = [os.path.relpath(f, in_path) for f in files]
    else:
        files = [in_path]
        rels = [os.path.basename(in_path)]
    assert files, f"no wav found in {in_path}"

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    p = pipeline(Tasks.speech_separation,
                 model="damo/speech_mossformer2_separation_temporal_8k")

    rows = []
    for f, rel in zip(files, rels):
        out_stem = os.path.join(out_root, os.path.splitext(rel)[0])
        dur, proc_s = separate(p, f, out_stem, args.out_sr)
        rows.append((rel.replace("\\", "/"), dur, proc_s))
        print(f"  {rel} ({dur:.1f}s, rtf={proc_s / dur:.2f}) -> _spk1/_spk2")

    if args.tag:
        os.makedirs(os.path.dirname(SEP_CSV), exist_ok=True)
        write_header = not os.path.exists(SEP_CSV)
        with open(SEP_CSV, "a", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp)
            if write_header:
                w.writerow(["tag", "file", "audio_s", "proc_s", "rtf"])
            for rel, dur, proc_s in rows:
                w.writerow([args.tag, rel, f"{dur:.2f}", f"{proc_s:.2f}",
                            f"{proc_s / dur:.3f}"])
    print(f"separated {len(files)} files -> {out_root}")
