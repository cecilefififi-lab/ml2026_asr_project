# 实验 6 长音频处理顺序对比 (content CER %)

链路: L1=直接 / L3=分离→ASR / L4=降噪→分离 / L5=分离→降噪

## funasr · content CER

| cond | level | L1 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| babble_0dB | light | 48.2 | 98.1 | 51.4 | 99.8 |
| babble_0dB | heavy | 71.0 | 96.6 | 73.9 | 82.3 |
| clean | light | 21.7 | 94.8 | 87.6 | 91.4 |
| clean | heavy | 49.0 | 62.5 | 56.1 | 54.6 |
| white_0dB | light | 76.9 | 44.7 | 44.3 | 60.0 |
| white_0dB | heavy | 88.7 | 56.9 | 67.6 | 55.5 |

## funasr · per-speaker CER (分离质量, 仅 L3/L4/L5)

| cond | level | L3 | L4 | L5 |
|---|---|---|---|---|
| babble_0dB | light | 97.7 | 91.0 | 99.8 |
| babble_0dB | heavy | 96.6 | 81.5 | 81.9 |
| clean | light | 98.2 | 89.4 | 94.0 |
| clean | heavy | 65.2 | 58.6 | 56.6 |
| white_0dB | light | 92.3 | 85.9 | 78.4 |
| white_0dB | heavy | 62.7 | 75.3 | 63.8 |

## whisper · content CER

| cond | level | L1 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| babble_0dB | light | 53.6 | 93.7 | 64.0 | 90.6 |
| babble_0dB | heavy | 86.6 | 89.1 | 92.9 | 89.2 |
| clean | light | 17.9 | 82.9 | 73.7 | 86.9 |
| clean | heavy | 52.1 | 49.1 | 51.3 | 47.7 |
| white_0dB | light | 29.9 | 48.1 | 47.2 | 51.8 |
| white_0dB | heavy | 59.5 | 65.7 | 75.4 | 63.0 |

## whisper · per-speaker CER (分离质量, 仅 L3/L4/L5)

| cond | level | L3 | L4 | L5 |
|---|---|---|---|---|
| babble_0dB | light | 94.0 | 84.4 | 90.8 |
| babble_0dB | heavy | 89.1 | 92.4 | 89.5 |
| clean | light | 86.4 | 84.7 | 89.9 |
| clean | heavy | 50.7 | 52.9 | 51.8 |
| white_0dB | light | 83.8 | 75.7 | 83.5 |
| white_0dB | heavy | 70.6 | 78.7 | 76.6 |

