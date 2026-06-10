# When Does Audio Preprocessing Help ASR?

噪声与重叠语音场景下,本地音频预处理对 ASR 鲁棒性的影响研究。

## 环境

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -r requirements.txt
```

## 最小流程

```bash
python src/make_noises.py                                   # 生成 white/babble 噪声
python src/add_noise.py                                     # clean x 噪声 x SNR{15,5,0}
python src/run_asr.py --engine whisper --input data/clean --tag clean
python src/run_asr.py --engine funasr  --input data/clean --tag clean
python src/run_asr.py --engine whisper --input data/noisy/white_5dB --tag white_5dB
python src/evaluate.py                                      # CER/RTF 汇总表
python src/separate.py data/overlap/MidOverlap.wav          # SepFormer 分离
```

实验记录见 `LOG.md`。
