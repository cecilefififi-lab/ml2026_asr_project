"""实验 4(Whisper 幻觉)素材生成:纯噪声短片段 + 静音段(纯零 / 近静音底噪)。

纯噪声:从 data/noise/{white,babble}.wav 60s 源按固定起点截取,混合长度
(看长度对幻觉量的影响;babble 不同段落叠加的英文人声不同,幻觉文本各异)。
静音:pure_zero(全 0)+ low_floor(极低电平白噪,模拟安静房间底噪)。
三档电平梯度(满电平噪声 → 近静音底噪 → 纯零)用于展示幻觉触发难度,
配合 VAD on/off 对照说明"为什么需要 VAD"。

输出 data/exp4/{noise,silence}/*.wav,16kHz mono。固定 seed/起点,可复现。
"""
import os

import numpy as np
import soundfile as sf

SR = 16000
SEED = 4
BASE = os.path.join(os.path.dirname(__file__), "..")
NOISE_DIR = os.path.join(BASE, "data", "noise")
OUT_NOISE = os.path.join(BASE, "data", "exp4", "noise")
OUT_SILENCE = os.path.join(BASE, "data", "exp4", "silence")

# (源文件, 起点秒, 时长秒):起点错开,babble 各段英文人声不同
NOISE_CLIPS = [
    ("white.wav", 0, 3), ("white.wav", 20, 10), ("white.wav", 30, 30),
    ("babble.wav", 0, 3), ("babble.wav", 15, 10), ("babble.wav", 28, 30),
]
ZERO_CLIPS = [10, 30]                       # 纯零静音时长(秒)
FLOOR_CLIPS = [(-60, 10), (-50, 30), (-45, 10)]  # (dBFS 电平, 时长秒),电平递增更易触发


def cut(name, start_s, dur_s):
    y, sr = sf.read(os.path.join(NOISE_DIR, name))
    assert sr == SR, f"{name} sr={sr}"
    return y[start_s * SR:(start_s + dur_s) * SR]


def main():
    os.makedirs(OUT_NOISE, exist_ok=True)
    os.makedirs(OUT_SILENCE, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for name, st, dur in NOISE_CLIPS:
        y = cut(name, st, dur)
        out = os.path.join(OUT_NOISE, f"{name[:-4]}_{st}s_{dur}s.wav")
        sf.write(out, y.astype(np.float32), SR)
        print(f"noise   {os.path.basename(out)} ({dur}s)")

    for dur in ZERO_CLIPS:
        out = os.path.join(OUT_SILENCE, f"zero_{dur}s.wav")
        sf.write(out, np.zeros(dur * SR, dtype=np.float32), SR)
        print(f"silence {os.path.basename(out)} (pure zero, {dur}s)")

    for db, dur in FLOOR_CLIPS:
        y = rng.standard_normal(dur * SR) * 10 ** (db / 20)
        out = os.path.join(OUT_SILENCE, f"floor_{abs(db)}dBFS_{dur}s.wav")
        sf.write(out, y.astype(np.float32), SR)
        print(f"silence {os.path.basename(out)} ({db} dBFS floor, {dur}s)")


if __name__ == "__main__":
    main()
