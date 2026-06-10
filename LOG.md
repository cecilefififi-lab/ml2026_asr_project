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

- 建仓库骨架、venv(CUDA torch 2.12 + cu126)、脚本(add_noise / run_asr / evaluate / separate / make_noises / make_ref_drafts)
- 数据就位:26 条干净片段 + 5 条重叠样本;生成 white/babble 噪声,26×2×3=156 个加噪文件,**实测 SNR 误差 0.000 dB**
- 跑通 baseline + 一个噪声条件 + 分离,端到端全链路 OK

### 第一轮数字(⚠️ 参考文本为未校对的 whisper 草稿,clean/whisper CER=0 是构造性的,数字仅验证流程,待人工校对后重算)

| tag | engine | n | mean_CER | mean_RTF |
|---|---|---|---|---|
| clean | whisper(large-v3 int8) | 26 | 0.0000* | 0.275 |
| clean | funasr(paraformer-zh) | 26 | 0.1227* | 0.136 |
| babble_0dB | whisper | 26 | 0.7161 | 0.260 |
| babble_0dB | funasr | 26 | 0.4657 | 0.100 |

初步观察(待校对参考后确认):
- babble 0dB 下两模型都严重退化;whisper 退化幅度更大,且出现整句幻觉(如原文"我都熬了两小时"→"謝謝"/"我没有看到")
- FunASR 的 RTF 约为 whisper 的一半

### 分离的坏消息(重要,记入失败案例)

SepFormer(sepformer-wsj02mix, 英文 8k 训练)分离 MidOverlap.wav:RTF 0.06 很快,
但 spk1/spk2 的 whisper 转写内容几乎相同 → 两路均残留混合人声,疑似分离失败。
学长目录里有当年的 separated_audio 可对比。阶段二需要:
1. 对比学长当年分离输出的质量
2. 尝试 MossFormer2(ModelScope/ClearerVoice)替代

### 踩坑记录

- Windows 无特权创建 symlink → speechbrain `from_hparams` 需传 `local_strategy=LocalStrategy.COPY`
- Python 3.12 + speechbrain 1.1.0:`inspect.getmodule` 触发 k2/flair 懒加载崩溃 → 导入后从 `sys.modules` 摘除 LazyModule
- speechbrain `separate_file` 会在路径前拼 cwd → 必须传相对路径
- HF 下载偶发瞬时失败,重试即可(本机有代理 127.0.0.1:7890)

### 待办(Day 2)

- [ ] **人工校对 refs/draft/*.txt**(逐条听 data/clean 音频改文本,覆盖到 refs/)→ 重算 baseline
- [ ] 补跑其余 5 个噪声条件(white_{15,5,0}, babble_{15,5})→ 第一条退化曲线
- [ ] 对比学长 separated_audio 与本次 SepFormer 输出,决定分离方案是否换 MossFormer2
- [ ] 确定最终数据规模与模型范围(阶段一验收项)
