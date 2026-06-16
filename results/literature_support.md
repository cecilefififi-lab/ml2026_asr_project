# 研究结论 × 文献佐证对照

> 目的:回应老师"实验结论需有现实研究/博客论证"的要求。
> 把本项目的核心结论逐条对照公开论文与社区博客,标注**证实 / 收窄 / 新发现**三种状态。
> 数据来源以 `results/` 与 `LOG.md` 为准;本表只做"结论 ↔ 文献"映射,不改实验数字。

## 总览表

| # | 项目结论 | 状态 | 关键佐证 |
|---|---|---|---|
| 1 | Whisper 在静音/非语音上幻觉,吐字幕残留(Amara.org、点赞订阅) | ✅ 强证实 | arXiv 2501.11378;whisper Discussion #928 |
| 2 | VAD 压得住白噪/静音,压不住 babble | ✅ 证实(babble 端为本项目实证) | faster-whisper / WhisperX;VAD 机制文献 |
| 3 | babble 比 white 更毒,0dB 处两模型曲线交叉 | ✅ 证实(需注意 SNR 依赖) | 信息掩蔽理论;Whisper-AT;Conformer NoiseX |
| 4 | 降噪/语音增强经常损害 ASR(伪影 + 失配) | ✅ 强证实 | arXiv 2201.06685;arXiv 2512.17562 |
| 5 | 神经降噪在 babble 上消掉目标人声 | ✅ 间接证实 | 同 #4 + over-separation 文献 |
| 6 | 语音分离在无/轻重叠、短片段上负收益(over-separation) | ✅ 强证实 | arXiv 2106.00949;arXiv 2503.17886 |
| 7 | LLM 是"过滤器"非"纠错器":拒幻觉,修不了声学已丢的词 | ✅ 强证实 | arXiv 2409.09785;arXiv 2505.24347 |
| 8 | Whisper 抗 white、FunASR 相反(模型抗噪画像不同) | ✅ 大方向证实(FunASR 端为新发现) | Whisper 原论文;Whisper-AT;crossover 文献 |
| A | 处理顺序几乎不重要(L4 vs L5) | ⚠️ 需收窄 | 无直接文献;本项目属退化情形 |
| B | MossFormer2 对 2–4s 短片段根本没分开 | ⚠️ 本项目实证 | over-separation 有据,短片段失效无直接背书 |

---

## 逐条说明

### ✅ 1. Whisper 静音/非语音幻觉(对应 exp4)

文献几乎一字不差。arXiv 2501.11378《Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio》系统证实:**约 35% 的幻觉是两个固定短语,前 10 个输出占一半以上**——印证我们 exp4 的"固定模式、非随机"判断。whisper Discussion #928 专门讨论 "Amara.org Community" 字幕污染;社区幻觉黑名单含 `amara.org`、`like and subscribe`。训练数据污染机制与 exp4 写法一致。

### ✅ 2. VAD 边界(对应 exp4 表 2)

"VAD 减少 Whisper 幻觉"被广泛证实(faster-whisper 默认 Silero VAD;WhisperX 把 "VAD preprocessing reduces hallucination" 列为核心特性)。"压不住 babble"的机制(VAD 把"像人话"的连续噪声判为语音放行)有文献支持,而 67%→67% 的具体数字是**本项目直接证据**,可作为贡献点。

### ✅ 3. babble vs white、0dB 交叉(对应 exp1)

babble = energetic + **informational masking**(词汇/音素竞争),比纯能量掩蔽的 white 更难分离目标;Whisper 对 babble/spoken noise 显著弱于其他噪声,尤其 SNR < 0dB。
**注意(避免外推)**:white vs babble 的相对难度**依赖 SNR 区间与频谱**,中等 SNR 下 white 反而可能更差。故保持"低 SNR / 交叉点附近两模型弱点相反"的表述,**不要写成单调关系**。

### ✅ 4. 降噪伤 ASR(对应 exp1 / ablation 主论点)

支撑"高级前端不划算"的关键证据,文献很硬:
- arXiv 2201.06685《How Bad Are Artifacts?》:正交投影分解证明 **artifact 是 ASR 退化主因**。
- arXiv 2512.17562《When De-noising Hurts》(医疗 ASR):**40/40 配置降噪后全部更差**,退化 1.1%–46.6%;结论"现代 ASR 直接吃噪声音频比吃降噪后更好"。
- 现代大模型自带抗噪能力,降噪引入分布失配抵消甚至超过收益。**直接支持 L1 直接 ASR 最划算。**

### ✅ 5. 神经降噪消掉目标人声(对应 exp1 babble + FRCRN)

属 #4 特例:增强/分离对"非重叠、单通道、人声型噪声"常无益甚至有害,因模型分不清"目标人声 vs 背景人声"。与 over-separation 同源。

### ✅ 6. over-separation(对应 exp2)

- arXiv 2106.00949《Should We Always Separate?》:标题即我们的问题,提出"按是否真有重叠切换是否分离",正为避免非重叠段过分离伪影。
- arXiv 2503.17886:分离前端引入 processing artifacts,劣化 clean-backend ASR;多说话人 ASR 在单说话人输入上退化。
- 撑稳"L1 在 15 格里 13 格最优、分离只在 heavy 重叠回本"。

### ✅ 7. LLM 是过滤器(对应 exp3)

- text-only GER **无法恢复解码时已丢失/被剪枝的声学信息**——即"声学已丢的通顺错词修不了"。
- GER 完全改写会**虚构未说内容**(经典例:"I like algorithms" → "I like Al Gore",文本音近);热词表帮倒忙也有据(exp3 "玩具"案例 0.40→1.0 吻合)。
- 业界用三阶段验证、logit-space anchoring 专门防 GER 幻觉——与我们"LLM 当过滤器"同向。

### ✅ 8. 模型抗噪画像相反(对应 exp1 跨模型)

- Whisper 因 680K 小时多样训练,对 white/ambient 强;babble 是其**特定弱点**。
- 存在 **crossover point**:clean 下最强的模型在噪声下可能反转。直接支持"两条退化曲线 0dB 交叉"。
- **诚实标注**:文献无 Whisper vs FunASR(Paraformer)直接对比。"FunASR 抗 babble 强于 Whisper"是**本项目新发现**,但符合"抗噪画像随训练数据而异"的总规律,作为贡献而非已知结论陈述。

---

## ⚠️ 需修正/收窄的两点

### A.「处理顺序几乎不重要」(L4 vs L5)——不要外推

文献对级联顺序无"顺序不重要"的普适结论。更关键:本项目 **spk CER 始终 84–88%,即 MossFormer2 在 2–4s 片段上根本没分开**。失效组件的前后顺序当然看不出差异——这是**退化情形**,非普适规律。
**建议表述**:"在本数据(短片段导致分离基本失效)条件下,L4/L5 无稳定差异,因此无法验证'先降噪更稳'假设。"

### B.「MossFormer2 对 2–4s 短片段根本没分开」——本项目实证

over-separation / 非重叠段伪影有文献,但"短片段(2–4s)导致分离失效"的具体因果无直接背书。建议作为**本项目观察现象**陈述(附 spk CER 数字作证),不写成公认结论。

---

## Sources

- Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio — arXiv:2501.11378 — https://arxiv.org/abs/2501.11378
- openai/whisper Discussion #928 (Dataset bias "Amara.org") — https://github.com/openai/whisper/discussions/928
- Calm-Whisper: Reduce Whisper Hallucination on Non-Speech — arXiv:2505.12969 — https://arxiv.org/pdf/2505.12969
- WhisperX (VAD preprocessing reduces hallucination) — https://github.com/m-bain/whisperX
- How Bad Are Artifacts? Analyzing the Impact of Speech Enhancement Errors on ASR — arXiv:2201.06685 — https://arxiv.org/abs/2201.06685
- When De-noising Hurts: Speech Enhancement Effects on Medical ASR — arXiv:2512.17562 — https://arxiv.org/abs/2512.17562
- Should We Always Separate? Switching Between Enhanced and Observed Signals — arXiv:2106.00949 — https://arxiv.org/abs/2106.00949
- Decoupling Speaker Separation and Speech Recognition — arXiv:2503.17886 — https://arxiv.org/abs/2503.17886
- LLM-Based Generative Error Correction: A Challenge and Baselines — arXiv:2409.09785 — https://arxiv.org/html/2409.09785v2
- Fewer Hallucinations, More Verification: Three-Stage LLM ASR Correction — arXiv:2505.24347 — https://arxiv.org/pdf/2505.24347
- Whisper-AT: Noise-Robust ASR are Strong Audio Event Taggers — arXiv:2307.03183 — https://arxiv.org/pdf/2307.03183
- Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) — arXiv:2212.04356 — https://arxiv.org/pdf/2212.04356
- Audio-Visual Efficient Conformer (white vs babble WER) — arXiv:2301.01456 — https://arxiv.org/pdf/2301.01456
