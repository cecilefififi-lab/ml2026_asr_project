# 实验日志

## 项目假设与边界(2026-06-10)

**总问题**:噪声与多人重叠条件下,哪些本地预处理(降噪/VAD/分离)真的能提升 ASR,哪些反而有害?

**假设**:
1. 噪声增强(SNR 15→5→0 dB)时,Whisper 和 FunASR 的 CER 单调上升,babble(人声背景)比 white 伤害更大
2. 降噪在低 SNR 下有收益,但在高 SNR 下可能因伪影反而提高 CER
3. 噪声 + 重叠并存时,处理顺序(先降噪 vs 先分离)对结果有显著影响

**边界(scope-out)**:波束成形、回声消除、移动端部署、模型训练均不做。

**数据说明**:
- 干净基底:学长辩论片段 26 条(`data/clean/`),16kHz mono,**每条仅 2-4 秒**(短于计划的 10-30 秒,如 Whisper 短音频不稳定再拼接加长)
- 重叠样本:学长 5 条(No/Light/Mid/Heavy/Opposite),42-52 秒
- 噪声:white(合成)+ babble(学长英文样本 6 副本错位叠加,与中文目标无文本泄漏);第三类计划用真实录音
- ground truth:人工转写学长片段(先用 whisper large-v3 干净音频输出做草稿,人工校对后入 `refs/`)
- **注意**:ground truth 草稿来自被测模型之一,人工校对必须逐条听音频修正,否则偏向 Whisper

**评测口径**:
- CER:去标点/空白后字符级编辑距离(jiwer)
- RTF:处理耗时 / 音频时长,单条记录,按条件取均值
- 环境:RTX 4050 Laptop 6GB,faster-whisper large-v3 int8_float16;FunASR paraformer-zh

---

## 2026-06-10(Day 1)

- 建仓库骨架、venv、脚本(add_noise / run_asr / evaluate / separate / make_noises)
- 数据就位:26 条干净片段 + 5 条重叠样本拷入
- 待办:装依赖 → 跑通 baseline → 生成 ref 草稿 → SepFormer 分离一条
