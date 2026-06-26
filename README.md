# When Does Audio Preprocessing Help ASR?

本项目研究噪声与重叠语音场景下，本地音频预处理对 ASR 鲁棒性的影响。我们以辩论赛“AI 书记员”场景为例，比较 faster-whisper 与 FunASR 等 ASR 后端在降噪、语音分离、VAD、LLM 后处理等不同路径下的表现，并用 CER、RTF 等指标观察准确率和效率之间的权衡。

## 项目结构

- `src/`：核心代码和实验脚本，包括噪声生成、加噪、ASR 调用、降噪、语音分离、评估、绘图和 demo 生成等流程。
- `data/`：实验音频数据，主要包含 `clean/` 干净语音、`noise/` 噪声素材、`overlap/` 重叠语音样本，以及部分实验输入文件。
- `refs/`：参考转写文本、热词表和草稿转写，用于 CER 评估和 ASR 纠错对照。
- `results/`：实验结果与图表输出，包含 `.csv` 指标表、`.md` 汇总说明，以及 `.png` 图表，例如降噪曲线、pipeline 对比、VAD 幻觉对比、长度消融和 trade-off 图。
- `demo/`：Streamlit 展示页面，入口为 `demo/Home.py`，各实验展示页位于 `demo/pages/`，示例说明在 `demo/cases.md`。
- `luyin/`：真实录音样本和录音要求，用于真实场景 spot check。
- `test/`：少量 sanity-check 音频样本。
- `REPORT.md`、`REPORT_en.md`、`REPORT_en.pdf`：中文/英文报告与英文 PDF 版本。
- `LOG.md`：实验过程记录。
- `requirements.txt`：Python 依赖列表。

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

实验记录见 `LOG.md`，主要结果汇总见 `results/summary.md` 和 `results/summary.csv`。
