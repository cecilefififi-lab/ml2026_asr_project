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

- [x] 人工校对 refs(26 条全部校对,8 条相对草稿有修正)→ 重算 baseline
- [x] 补跑其余 5 个噪声条件 → 第一条退化曲线
- [x] 对比学长 separated_audio 与本次 SepFormer 输出
- [ ] 确定最终数据规模与模型范围(待含男声片段清单确认后定)

## 2026-06-11(Day 2)

### 校对后完整噪声矩阵(refs 已人工校对,数字可信)

| SNR | white/whisper | white/funasr | babble/whisper | babble/funasr |
|---|---|---|---|---|
| clean | 5.3% | 10.4% | 5.3% | 10.4% |
| 15dB | 7.8% | 10.0% | 5.7% | 8.8% |
| 5dB | 12.5% | 29.8% | 20.9% | 22.5% |
| 0dB | 42.1% | 68.4% | **71.9%** | 43.6% |

图:`results/degradation_curve.png`

### 发现(可进视频)

1. **两模型弱点相反**:Whisper 抗 white 强(5dB 仅 12.5%)但 babble 0dB 崩溃(71.9%,大量整句幻觉);FunASR 对 white 退化快(0dB 68.4%)但 babble 0dB 反而比 Whisper 稳(43.6%)。0dB 处两条曲线交叉。
2. 假设修正:原假设"babble 比 white 伤害更大"只对 Whisper 成立,FunASR 相反。可能解释:Whisper 是多任务/多语言生成式模型,babble(可懂人声)诱发语言模型"接管"产生幻觉;FunASR 是判别式 CTC/paraformer,对宽带能量掩蔽(white)更敏感。
3. RTF 全程稳定:whisper ~0.26-0.31,funasr ~0.08-0.15,与噪声条件无关。

### 分离方案对比(Day 1 疑点确认)

学长 separated_audio/MidOverlap 两路转写内容明显不同(真分开了);
我们 SepFormer(wsj02mix 8k)两路几乎相同(分离失败)。
学长论文用 MossFormer2,`chongdie.py` 的 SepFormer 只是样例。
→ **决策:阶段二分离方案换 MossFormer2(ModelScope/ClearerVoice),SepFormer 留作失败案例。**

### 男声片段决策(Day 3,2026-06-11)

校对时未单独记录含男声的片段。改用数据验证:逐条比对 clean 条件下
hyp 与 ref 的长度差,**无一条出现明显插入(最大 +2 字)**,即两引擎都没有
把背景男声转写出来 → 男声对评测无实质影响,**26 条全部保留,不剔除**。

抽查高 CER 条目发现 2 条参考文本可疑(两引擎一致对抗参考),待重听:
- con_003:"我得在这里" vs 两引擎一致的"我方这里"
- con_005:"初中觉得" vs whisper 无/funasr"光"

已确认的真实 ASR 错误案例(留作素材):pro_005 whisper 把"晚睡晚起"听成"玩具";con_009 funasr 输出"两点,睡"。

### 最终数据规模与模型范围(阶段一验收项,定稿)

- 数据:26 条干净片段(2-4s,辩论语料,人工校对参考)+ 5 条重叠样本(42-52s)
- 噪声:white + babble,SNR {15, 5, 0} dB;阶段三补 1 类真实录音噪声
- ASR:faster-whisper large-v3(int8_float16, GPU)+ FunASR paraformer-zh,两模型贯穿全部实验
- 分离:MossFormer2(待阶段二验证环境);SepFormer 已确认失败,留作对比案例
- 降噪(实验 1 用):一个神经方法(DeepFilterNet 或 RNNoise)+ 谱减法,阶段二定

### 待办(Day 3)

- [x] 重听 con_003 / con_005,确认参考文本(con_005 改为"哦觉得";con_003 维持人工听写)
- [x] MossFormer2 环境验证
- [x] 阶段一验收检查(见下)

### MossFormer2 环境验证(Day 3 下午,一波三折)

尝试路线与结果:
1. **ClearerVoice MossFormer2_SS_16K**:装通跑通(RTF 0.33),但 MidOverlap 分离质量差(s1 仍混两人,s2 碎片化)
2. **ModelScope MossFormer2 8k(学长论文用的)**:
   - 踩坑 1:缺依赖 addict/datasets,逐个补齐
   - 踩坑 2:**Windows 段错误**——librosa 先于 pyarrow 加载时 DLL 冲突,
     `import pyarrow.dataset` 必须放在 librosa 之前(已写入脚本注释)
   - 修复后跑通,RTF 0.13

质量验证(波形相关系数):
- 我们分 MidOverlap:spk1↔spk2 = 0.90,spk1↔mix = 0.97 → **输出≈混合,没分开**
- 学长的 separated_audio:spk1↔spk2 = 0.001 → 真分开了(同模型同输入,他怎么做到的未知,可能有未公开的预处理步骤)
- **受控实验**:用 pro_001+con_001 合成完全重叠混合 → spk1↔srcA=0.82, spk2↔srcB=0.79,交叉 0.42 → **模型和环境没问题**
- 截 MidOverlap 中段 10s 再分:0.94 → 排除长度因素,该录音本身难分(疑似两说话人声纹接近)

### 决策:实验 2 数据方案变更

实验 2 改用**自制重叠样本**:从 26 条已校对片段中选对,按重叠比例(无/轻/重)
错位混合。优点:两个说话人都有 ground truth,CER 可严格计算;重叠程度可控。
学长的 5 条重叠样本降级为定性演示素材。分离方案:ModelScope MossFormer2 8k。

### 阶段一验收清单(2026-06-11 晚,全部通过)

- [x] ≥3 条音频跑通 baseline → 26 条 × 2 引擎 × 7 条件
- [x] ≥1 条重叠样本完成分离且可听 → 合成混合分离成功(相关系数验证);MidOverlap 失败已记录为案例
- [x] 能输出 CER/WER 和处理耗时 → summary.csv/md + 退化曲线图
- [x] LOG.md 有第一轮 baseline 数字 → Day 1/2 表格
- [x] 明确数据规模与模型范围 → Day 3 定稿(实验 2 数据方案今日更新)
