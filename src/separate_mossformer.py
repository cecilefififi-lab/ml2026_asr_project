"""MossFormer2 双说话人分离 (ClearerVoice-Studio, MossFormer2_SS_16K, 16kHz)。

用法:
    python src/separate_mossformer.py data/overlap/MidOverlap.wav

输出: data/separated_mossformer/{原名}/ 下的 spk1/spk2 wav (16kHz)
"""
import os
import sys
import time

import soundfile as sf

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "separated_mossformer")

if __name__ == "__main__":
    in_path = sys.argv[1]
    if not os.path.isabs(in_path):
        in_path = os.path.join(BASE, in_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    from clearvoice import ClearVoice
    cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])

    dur = sf.info(in_path).frames / sf.info(in_path).samplerate
    t0 = time.perf_counter()
    cv(input_path=in_path, online_write=True, output_path=OUT_DIR)
    proc_s = time.perf_counter() - t0
    print(f"separation done: {proc_s:.1f}s for {dur:.1f}s audio (rtf={proc_s / dur:.2f})")
    print(f"outputs under {OUT_DIR}")
