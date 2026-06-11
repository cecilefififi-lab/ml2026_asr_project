"""MossFormer2 双说话人分离 (ModelScope damo/speech_mossformer2_separation_temporal_8k, 8kHz)。
学长论文使用的模型。

用法:
    python src/separate_mossformer_ms.py data/overlap/MidOverlap.wav

输出: data/separated_mossformer_ms/{原名}_spk{1,2}.wav (8kHz)
"""
import os
import sys
import time

import pyarrow.dataset  # noqa: F401  必须先于 librosa 导入, 否则 Windows 下 DLL 冲突段错误

import librosa
import numpy as np
import soundfile as sf

SEP_SR = 8000
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "separated_mossformer_ms")

if __name__ == "__main__":
    in_path = sys.argv[1]
    if not os.path.isabs(in_path):
        in_path = os.path.join(BASE, in_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    p = pipeline(Tasks.speech_separation,
                 model="damo/speech_mossformer2_separation_temporal_8k")

    y, _ = librosa.load(in_path, sr=SEP_SR)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    tmp_8k = os.path.join(OUT_DIR, f"_{stem}_8k_tmp.wav")
    sf.write(tmp_8k, y, SEP_SR)

    t0 = time.perf_counter()
    result = p(tmp_8k)
    proc_s = time.perf_counter() - t0
    os.remove(tmp_8k)

    for i, pcm in enumerate(result["output_pcm_list"]):
        sig = np.frombuffer(pcm, dtype=np.int16)
        out = os.path.join(OUT_DIR, f"{stem}_spk{i + 1}.wav")
        sf.write(out, sig, SEP_SR)
        print(f"wrote {out}")
    dur = len(y) / SEP_SR
    print(f"separation: {proc_s:.1f}s for {dur:.1f}s audio (rtf={proc_s / dur:.2f})")
