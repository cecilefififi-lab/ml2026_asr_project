"""生成实验用噪声文件: white(合成) + babble(英文语音多副本错位叠加)。

输出到 data/noise/{white,babble}.wav, 16kHz mono, 60s。
babble 源为英文语音, 与中文测试语料无文本泄漏。
"""
import os

import numpy as np
import soundfile as sf

SR = 16000
DUR_S = 60
NOISE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "noise")
BABBLE_SRC = os.path.join(NOISE_DIR, "_babble_source.wav")
N_BABBLE_LAYERS = 6
SEED = 42


def make_white(n_samples, rng):
    return rng.standard_normal(n_samples) * 0.1


def make_babble(n_samples, rng):
    src, sr = sf.read(BABBLE_SRC)
    assert sr == SR, f"babble source sr={sr}, expected {SR}"
    if len(src) < n_samples:
        src = np.tile(src, int(np.ceil(n_samples / len(src))))
    out = np.zeros(n_samples)
    for _ in range(N_BABBLE_LAYERS):
        start = rng.integers(0, len(src) - n_samples + 1)
        out += src[start:start + n_samples]
    return out / N_BABBLE_LAYERS


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    n = SR * DUR_S
    for name, fn in [("white", make_white), ("babble", make_babble)]:
        y = fn(n, rng)
        path = os.path.join(NOISE_DIR, f"{name}.wav")
        sf.write(path, y.astype(np.float32), SR)
        print(f"wrote {path} ({DUR_S}s)")
