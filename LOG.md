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

## 2026-06-12

### 叙事与计划调整(课堂获悉评分标准后)

老师课上明确评分标准:insight / nuances / depth / difficulties / **fun**,
看迭代过程,不看代码行数。对照检查:前四项已有覆盖(LOG 假设-反直觉记录、
SepFormer 失败案例、"预处理何时有害"本身即 nuance 问题),短板是 fun 和叙事。

**决策(只改包装,不改实验设计,详见 PLAN.md):**

1. **视频叙事背景定为辩论赛**:数据本来就是辩论片段,包装为
   "Who Said That?! Teaching ASR to Survive a Debate"——
   噪声 = 观众起哄,重叠 = 抢话,LLM 纠错 = 书记员校对
2. **Whisper 幻觉实验从加餐 backlog 提升为实验 4**(约半天):
   喂纯噪声/静音段记录幻觉输出,与实验 1 中 babble 0dB 整句幻觉案例
   (如"我都熬了两小时"→"謝謝")呼应,引出 VAD 的必要性。fun + insight 双收
3. 视频加"人类 vs Whisper"互动环节;demo 加处理前后音频试听按钮

### 真实录音处理规则(定稿)

已有真实录音(食堂/教室等,待入库 `data/raw_recordings/`)双重用途:

- **作第三类噪声源**:仅转 16kHz mono + 截掉含清晰可懂语句的段落(防文本
  泄漏)+ `add_noise` 按 SNR 受控混合;叙事上即"辩论现场观众噪声"
- **作泛化抽查**:原始录音零处理直接跑
- **禁止人工调音**(EQ/混响/手动音量):退化曲线依赖 SNR 精确受控
  (阶段一实测误差 0.000 dB),手工调制破坏可复现性

## 2026-06-12(实验 1:噪声 × 降噪 × ASR)

### 假设(实验前)

1. 降噪在低 SNR 下有收益,高 SNR 下可能因伪影反而提高 CER(沿用项目假设 2)
2. 神经降噪(FRCRN)优于简单 DSP(谱减法)

### 方法变更:DeepFilterNet → FRCRN_SE_16K

Python 3.12 无 deepfilterlib 预编译 wheel,源码安装需 Rust 工具链 → 放弃。
改用 clearvoice(阶段一已装)自带的 **FRCRN_SE_16K**(阿里, ICASSP 2022),
16kHz 原生免重采样,零新增依赖。DSP 方法用 noisereduce(spectral gating,
stationary=True),即谱减法变体。

### 设置

- 26 条 × 7 条件(clean + {white,babble}×{15,5,0}dB)× 3 处理(无/frcrn/specsub)× 2 引擎
- 新脚本:`denoise.py`(批量降噪+逐条耗时 → denoise_raw.csv)、
  `plot_denoise.py`(曲线+热力图)、`find_cases.py`(逐条 CER 差挖案例)
- 降噪 RTF:specsub **0.018**(CPU)/ frcrn **0.105**(GPU),约差 6 倍;
  链路总 RTF = 降噪 RTF + ASR RTF
- clean 也过降噪:直接验证"高 SNR 伪影有害"假设

### 结果(CER %,无处理 → FRCRN → 谱减)

| 条件 | whisper | funasr |
|---|---|---|
| clean | 4.9 → 9.3 → 14.7 | 10.1 → 8.5 → 13.0 |
| white_15dB | 7.4 → 16.6 → 14.4 | 9.6 → 12.0 → 14.6 |
| white_5dB | 12.2 → 33.2 → 32.3 | 29.4 → 17.0 → 16.8 |
| white_0dB | 41.9 → 49.9 → 61.7 | **68.4 → 27.5** → 40.1 |
| babble_15dB | 5.4 → 17.0 → 19.0 | 8.5 → 12.7 → 16.1 |
| babble_5dB | 20.6 → 78.6 → 47.1 | 22.3 → 52.3 → 30.1 |
| babble_0dB | 71.7 → 92.5 → 83.7 | 43.3 → **95.1** → 46.2 |

图:`results/exp1_denoise_curves.png` / `exp1_denoise_heatmap.png`
案例全文:`results/exp1_cases.txt`

### 发现

1. **降噪只在一种组合下大幅有效**:FunASR × white × 低 SNR(0dB: 68.4→27.5)。
   与"FunASR 怕宽带能量掩蔽、FRCRN 擅长去稳态噪声"的解释一致。
2. **Whisper 任何场景都不受益于降噪**:自身鲁棒性已超过"降噪收益-伪影代价"净值。
3. **babble + 神经降噪 = 灾难**:FunASR babble_0dB 从 43.3% 飙到 95.1%。
   机理见案例 1:FRCRN 分不清目标人声和背景人声,把目标语音消了。
4. **降噪改变 Whisper 的错误模式**:幻觉 → 沉默。降噪后空转写 78 条,
   集中在 whisper × 低 SNR(babble_0dB+specsub 15/26 条为空)。
5. 假设 2 证实:clean/15dB 下降噪全面有害(whisper clean 4.9→9.3/14.7)。
6. 假设"神经 > DSP"**只对 white 成立**;babble 下谱减反而比 FRCRN 伤害小
   (谱减只是减能量,FRCRN 会"重写"频谱)。
7. 待查:whisper 在谱减输出上 RTF 升高(0.26 → 0.49-0.55),疑似伪影拖慢解码。

### 典型案例(`find_cases.py` 产出,已选定)

- **帮倒忙①(删错了人)**:con_001 babble_0dB funasr,无处理 CER 28%(中文基本正确)
  → FRCRN 后输出英文 "Because here's that with kind of talthink against this."
  —— 目标中文被当噪声消掉,留下背景英文 babble。95.1% 格子的机理解释。
- **帮倒忙②(诱发新幻觉)**:pro_002 white_0dB whisper,specsub 后输出
  "請不吝点赞 订阅 转发 打赏支持明镜与点点栏目" —— 训练数据字幕残留,实验 4 素材。
- **帮倒忙③(消灭正确答案)**:con_007/pro_006 babble 下无处理 CER 0%,降噪后全空。
- **帮忙①(救活 FunASR)**:pro_015 white_5dB funasr,无处理输出空 → specsub 后
  完美转写"我多了两个小时"(CER 100→0)。
- **帮忙②(压制幻觉)**:con_009 white_0dB whisper,幻觉"小明星大跟班下次再見"
  → 降噪后幻觉消失。

### 踩坑记录

- pip 装 deepfilternet 失败:Python 3.12 无 wheel,要求 Rust 编译 → 换 FRCRN
- funasr 批跑第 25/28 个目录时在模型初始化阶段卡死(jieba 缓存加载后挂起,
  疑似网络检查),73 分钟无进展 → 杀掉补跑缺失 4 条件,数据无损。
  后续长批跑考虑按引擎分段、或 funasr 离线模式
- Windows 控制台 GBK:含中文的脚本输出须 `PYTHONIOENCODING=utf-8` 重定向到文件

## 2026-06-13(实验 2:噪声 + 重叠语音处理顺序)

### 假设(实验前)

1. **噪声破坏分离**:SNR 越低,MossFormer2 两路残留越多(串话/复制),分离后 CER 不降反升。
2. **处理顺序有差异**:先降噪再分离(L4)比先分离再降噪(L5)更稳——先清噪让分离器更易工作;反向则分离器先被噪声干扰、降噪再引入伪影。
3. **最复杂链路不一定最优**:低重叠或高 SNR 下简单链路(L1/L2)可能已够,分离反而引入误差。

### 数据与设置

- **自制重叠样本**(`make_overlap.py`):con_i ↔ pro_i 同序号配对(辩论正反方)11 对,等响度(0 dB SIR)错位混合,三档重叠 no/light/heavy = 0/0.3/0.8(重叠比例 = 重叠时长 / min 两条时长);manifest 记录 offset/重叠时长/各自时长。**双说话人均有 ground truth,CER 可严格计算**。
- **噪声**:clean + {babble,white}×{5,0}dB = 5 个条件(`add_noise.py` 扩展支持递归输入 + 自定义输出根,SNR 误差 0.000)。
- **五链路**(`run_pipelines.py` 按环节分进程批处理,避开 6GB 显存同时驻留分离+降噪+ASR 三模型):
  - L1 直接 / L2 降噪→ASR / L3 分离→ASR / L4 降噪→分离 / L5 分离→降噪
- **模型**:降噪 specsub(实验 1 证明 babble 下比 FRCRN 温和、不灾难性消人声);分离 MossFormer2(ModelScope 8k,输出上采样 16k);ASR Whisper large-v3 单模型(按砍单优先级,FunASR 留待 `--skip-prep` 复用中间产物补跑)。
- **评测口径**(`eval_pipelines.py`,复用实验 1 char_cer):
  - content CER(五链路可比):单路 hyp vs 拼接 ref 取较优顺序;双路取 permutation 较优配对后 hypA+hypB vs refA+refB
  - per-speaker CER(仅分离链路):permutation 较优配对后两路平均,衡量分离质量

### 结果

规模:11 对 × 3 重叠档 × 5 噪声条件 × 5 链路 = 825 个 pair-链路,Whisper 共 1320 条转写。

**整体内容 CER %(content,五链路同口径,`results/exp2_pivot.md`):**

| cond | level | L1 直接 | L2 降噪 | L3 分离 | L4 降噪→分离 | L5 分离→降噪 |
|---|---|---|---|---|---|---|
| clean | no | **4.4** | 17.9 | 70.7 | 51.8 | 51.0 |
| clean | light | **11.3** | 23.2 | 57.6 | 61.3 | 48.8 |
| clean | heavy | 50.0 | 57.0 | **46.9** | 58.9 | 57.4 |
| babble_5dB | no | **19.3** | 35.3 | 71.5 | 66.2 | 68.1 |
| babble_5dB | light | **28.1** | 45.0 | 77.7 | 65.9 | 73.7 |
| babble_5dB | heavy | **57.8** | 76.3 | 73.8 | 85.7 | 72.4 |
| babble_0dB | no | **50.6** | 71.2 | 102.5 | 88.0 | 84.4 |
| babble_0dB | light | **58.3** | 74.6 | 100.6 | 76.5 | 80.5 |
| babble_0dB | heavy | 90.7 | 89.6 | **70.3** | 86.3 | 77.3 |
| white_5dB | no | **17.3** | 28.9 | 58.8 | 65.6 | 64.6 |
| white_5dB | light | **31.2** | 36.7 | 53.5 | 52.8 | 56.8 |
| white_5dB | heavy | 58.0 | 66.0 | 65.9 | 65.1 | **60.1** |
| white_0dB | no | **27.4** | 40.4 | 45.4 | 46.4 | 53.5 |
| white_0dB | light | **32.5** | 51.5 | 64.2 | 68.7 | 64.4 |
| white_0dB | heavy | 60.1 | 73.6 | **63.0** | 71.8 | 66.4 |

**per-speaker CER %(分离质量,仅 L3/L4/L5):普遍 63–112%,分离基本未成功**

| cond/level | L3 | L4 | L5 | | cond/level | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|---|
| clean/no | 102.6 | 85.8 | 74.1 | | babble_0dB/no | 109.7 | 92.2 | 83.9 |
| clean/light | 79.5 | 86.8 | 65.4 | | babble_0dB/heavy | 81.4 | 88.5 | 83.6 |
| clean/heavy | 63.7 | 74.0 | 75.3 | | white_0dB/no | 82.6 | 81.2 | 83.7 |

工程代价:链路 RTF ≈ 各环节之和(specsub ~0.018、separate ~0.13、whisper ~0.2);分离链路(L3-5)耗时约为 L1 的 2-3 倍(分离 + 双路 ASR),换来的却是更差的 CER —— trade-off 全程负向。

### 发现

1. **直接 ASR(L1)在 no/light 重叠下全面、大幅最优**:clean/no L1=4.4% vs 任何分离链路 47–70%。2-4s 短片段即便轻度重叠,Whisper 直接识别远胜任何预处理。L1 在 15 个格子里 11 个最优。
2. **分离(L3-5)几乎总是有害**:仅在 heavy 重叠时劣势缩小,个别条件微弱反超(clean/heavy 46.9<50.0;babble_0dB/heavy 70.3<90.7)。即"两人长时间同时说话"时分离才勉强值得。
3. **per-speaker CER 始终 63–112%,MossFormer2 对短片段分离根本没成功**。典型失败模式 = 把混合整体**复制到两路**(案例 A:spk1≈spk2≈完整混合句)。content CER 远低于 spk CER,正因两路都召回了全部内容、却都没分开。
4. **降噪(L2)在重叠场景同样有害**:全部 15 个条件 L2>L1,与实验 1 specsub 伪影结论一致。
5. **处理顺序 L4 vs L5 差异小、无稳定赢家**:clean 下 L5(先分离再降噪)略优,但整体互有胜负 —— **假设 2 不成立**。
6. **0dB babble 触发 Whisper 幻觉**:案例 B 中 L1 输出空、分离两路全是训练字幕残留("感谢观看""中文字幕志愿者"),呼应实验 4。

**假设回顾:**

- 假设 1(噪声破坏分离):**被更强效应掩盖**。clean/no 分离就已失败(spk_cer 102.6),主因是短片段 + 轻重叠本身难分,噪声只是雪上加霜,非主因。
- 假设 2(L4 先降噪更稳):**否**,L4/L5 差异小且 L5 略优。
- 假设 3(最复杂链路不一定最优):**强烈成立** —— 这是实验 2 主结论,直接呼应项目总问题"哪些预处理看起来高级但不划算"。

### 典型案例

- **案例 A(分离复制失败)**:clean/light `con_004_pro_004`,两人台词接续成一句。L1 直接近乎完美转写整句;L3 分离 spk1/spk2 **几乎一字不差地都等于整句混合** —— 分离器没分开,反把内容翻倍。轻度重叠 ≈ 顺序说话,分离纯属画蛇添足。
- **案例 B(0dB 幻觉)**:babble_0dB/heavy `con_001_pro_001`,L1 输出空;L3 spk1="中文字幕志愿者 杨茜茜"、spk2="感谢观看" —— 两路皆 Whisper 字幕幻觉,实验 4 高光素材。

### 视频叙事点

辩论书记员费力上声源分离 + 降噪,结果还不如直接听写 —— 因为辩论片段太短、真重叠太少,分离器把两人的话复制成两份。"高级预处理"在真实约束下反成负担,只有两人长时间抢话(heavy)时才勉强回本。

### 踩坑记录

- 6GB 显存装不下分离(MossFormer2)+ 降噪(FRCRN)+ ASR 三模型同时驻留 → `run_pipelines.py` 按环节分进程批处理,每模型只加载一次,中间产物落盘串联。
- 分离模型输出 8kHz,ASR/FRCRN 需 16kHz → `separate` 统一上采样到 16k 再下游。
- content CER 可 >100%(babble_0dB 分离两路双双幻觉,拼接超长,插入数 > 参考字数)。

## 2026-06-14(实验 3:小型语义纠错)

### 假设(实验前)

1. LLM 能修同音/近音字错(尤其有上下文时)。
2. 热词表/术语库辅助能进一步提升术语级修正。
3. 声学信息已完全丢失的错误,LLM 修不了。

### 数据与设置

- **5 条代表性 ASR 错误**(`data/exp3_inputs.json`),选自实验 1/2 最差/最典型输出,覆盖四类:术语错(上下文全对 / 声学差远)、近音错、综艺幻觉、字幕幻觉。
- **三档对比**(`src/correct.py`):无纠错 / 纯 LLM / 热词辅助 LLM。
- **盲测**:纠错时只给「ASR 转写(+ 热词表)」,**不给 ref**;system prompt 要求「整段是幻觉就输出 `[无法识别]`、不准脑补」。
- 模型:`claude-opus-4-8`(adaptive thinking);热词表 `refs/hotwords.txt`(13 个作息辩论术语);CER 复用 char_cer。

### 结果(content CER)

| 案例 | 类型 | raw | 纯LLM | 热词 | 实质 |
|---|---|---|---|---|---|
| case1 | 术语错·错词本身通顺(好意) | 0.158 | 0.158 | 0.158 | 好意→好事/好的,**未修成"熬夜"** |
| case2 | 术语错·声学差远(玩具) | 0.40 | 0.40 | **1.0** | 纯LLM保留原样;热词版反判[无法识别](更差) |
| case3 | 近音错(说/多) | 0.077 | 0.077 | 0.077 | 保留"说了",**未修成"多了"** |
| case4 | 综艺名幻觉 | 2.286 | **0.571** | 0.571 | 正确输出 [无法识别],**不脑补** |
| case5 | 字幕幻觉 | 1.579 | **0.842** | 0.842 | 正确输出 [无法识别] |

全文:`results/exp3_correction.md`

### 发现

1. **LLM 在本语料唯一可靠的价值 = 识别并拒绝幻觉**(case4/5):面对综艺名、"点赞订阅"字幕残留,准确判为"与辩论无关"、输出 [无法识别],不编造。注:case4/5 的 CER 下降是副产品(拒绝文本比原幻觉短),真正意义是**拒绝脑补**——信息全丢,本来就不可能恢复成"早睡晚睡"。
2. **声学信息已丢失的字词错,LLM 修不了**(case1/2/3):错词在语法语义上**本身通顺**("好意/说了/玩具"都读得通),LLM 既无声学线索、也没理由把通顺的词改掉 → 要么保留错误,要么换成另一个通顺的错词("好意"→"好事")。
3. **热词表无增益,甚至帮倒忙**(case2 纯LLM 0.40 → 热词 1.0,反直觉):术语库只在"ASR 输出已接近某术语、需轻微纠偏"时有用;当声学错误使输出与术语相距甚远("玩具" vs "晚睡晚起"),热词表无法桥接,反而诱导模型对无法对应的输入更倾向判 [无法识别],拉高 CER。

**假设回顾:**

- 假设 1(LLM 修近音):**否**。声学区分信息丢失后,通顺的错词 LLM 修不了(case3 全程没把"说"改回"多")。
- 假设 2(热词提升):**否,甚至有害**(case2)。
- 假设 3(声学丢失修不了):**成立**,是本实验主结论。

**目标结论回答:**

- LLM 后处理在此场景**最适合"过滤幻觉/无意义内容"**,而非修复字词级 ASR 错误;一旦错误词本身通顺(= 声学信息已丢失),就不能指望 LLM 兜底——它没有声学线索,只能靠语言模型猜,而猜不过"本身就通顺"这一关。

> **caveat**:样本仅 5 条且偏向"严重退化"案例(选自最差输出),结论为定性。若选轻度错误(ASR 输出已接近正确),LLM / 热词可能转为正增益——本实验刻意展示的是 LLM 兜底的**边界**。

### 踩坑记录

- 用第三方中转(xuedingtoken,key 非官方 `sk-ant-` 格式):官方端点 401,需显式配 `base_url`;且本机已有的 `ANTHROPIC_BASE_URL`(官方)会干扰,脚本改为从 `.env` 显式读 key+base_url 传参。
- 该中转把 adaptive thinking 的推理用 `<thinking>...</thinking>` 标签塞进**正文 block**(非标准实现)→ 正则 strip 标签才得到干净答案。
- Opus 4.8 thinking **关闭**时会把推理泄漏进正文(首轮 case1 CER 飙到 67);开 adaptive thinking + strip 标签后稳定。

## 2026-06-15(实验 4:Whisper 幻觉小实验)

### 假设(实验前)

1. 喂无语音内容的音频(纯噪声/静音),Whisper 不会沉默,而是吐训练数据残留型幻觉(字幕/片尾文本)——沿用实验 1/2 已观察到的字幕残留现象。
2. 开 VAD(`vad_filter`)能压制这类幻觉。
3. 输入越接近绝对静音越安全:纯零静音应不触发幻觉。

### 数据与设置

- 模型:faster-whisper large-v3(int8_float16, GPU),`language=zh, beam_size=5` —— 与实验 1 完全同设置(复现同款幻觉的关键)。
- 素材(`make_exp4_clips.py` → `data/exp4/`,11 条,固定 seed=4 可复现):
  - 纯噪声 6 条:white / babble 各 3 条(3s / 10s / 30s),从 60s 噪声源固定起点截取。
  - 静音 5 条:纯零 `zero` ×2(10s/30s)+ 近静音底噪 `floor`(-60 / -50 / -45 dBFS)×3。三档电平梯度(满电平噪声 → 微弱底噪 → 纯零)用于探幻觉触发阈值。
- 对照:同批素材跑 **VAD off** vs **VAD on**(Silero,`run_asr.py` 新增 `--vad-filter` 开关,默认 off 不影响实验 1-3 复现)各一遍。
- 数据:`results/exp4_hallucination.csv`(tag = `exp4_vadoff` / `exp4_vadon`,共 22 条)。

### 结果

**表 1:无语音输入 → 幻觉(VAD off,11 条命中 10 条 = 91%)**

| 输入 | 类型 | 时长 | Whisper 输出 | 归类 |
|---|---|---|---|---|
| babble_0s_3s | babble | 3s | *(空)* | — |
| babble_15s_10s | babble | 10s | 请不吝点赞 订阅 转发 打赏支持明镜与点点栏目 | 直播打赏字幕 |
| babble_28s_30s | babble | 30s | 字幕志愿者 杨茜茜 | 字幕组署名 |
| white_0s_3s | white | 3s | 響鐘 | 短脑补 |
| white_20s_10s | white | 10s | 由 Amara.org 社群提供的字幕 | 字幕平台残留 |
| white_30s_30s | white | 30s | 響鐘 | 短脑补 |
| floor_60dBFS_10s | 近静音底噪 | 10s | 字幕志愿者 杨茜茜 | 字幕组署名 |
| floor_50dBFS_30s | 近静音底噪 | 30s | 響鐘 | 短脑补 |
| floor_45dBFS_10s | 近静音底噪 | 10s | 由 Amara.org 社群提供的字幕 | 字幕平台残留 |
| zero_10s | 纯零静音 | 10s | 由 Amara.org 社群提供的字幕 | 字幕平台残留 |
| zero_30s | 纯零静音 | 30s | 由 Amara.org 社群提供的字幕 | 字幕平台残留 |

**表 2:VAD on/off 各类型幻觉率(`results/exp4_vad_compare.png`)**

| 输入类型(n) | VAD off | VAD on | 效果 |
|---|---|---|---|
| white(3) | 100% | **0%** | ✅ 完全压制 |
| silence: zero+floor(5) | 100% | **0%** | ✅ 完全压制 |
| babble(3) | 67% | **67%** | ❌ 几乎无效(漏 "感谢观看" / "字幕志愿者 杨茜茜") |

产出:案例集 `results/exp4_hallucination.md`(1 页,视频高光素材)+ 对比图 `results/exp4_vad_compare.png`。

### 发现

1. **训练数据污染型幻觉**:纯噪声、甚至纯零静音上稳定吐 YouTube/字幕平台残留(`由 Amara.org 社群提供的字幕`、`请不吝点赞订阅打赏`、`字幕志愿者 杨茜茜`)。这是训练集"片尾/无语音段"字幕的条件反射,与有没有声学能量无关。
2. **VAD 对白噪/静音 100% 有效,对 babble 几乎无效**:white + 全部 silence(8 条)VAD on 后幻觉清零;babble 3 条仍漏 2 条 —— Silero VAD 把可懂人声型噪声判成语音放行。**这解释了实验 1 为何 babble 0dB 最毒:它骗得过 VAD。** 辩论现场观众嘈杂声本质即 babble,VAD 非万能。
3. **同款幻觉跨实验复现,证明是 Whisper 固定模式而非随机**:
   - babble 10s 的"请不吝点赞订阅..." = 实验 1 `pro_002` white_0dB+specsub 同款。
   - babble 30s 的"字幕志愿者 杨茜茜" / VAD on 漏出的"感谢观看" = 实验 2 案例 B 同款。
   - 呼应 LOG Day1 `pro_005`:真句被替换成"謝謝"。
4. **机理串联**:噪声/降噪伪影冲掉目标语音的声学信息后,Whisper 的语言模型"接管",从训练记忆调出最熟悉的"无语音段字幕"填空 → 必须在 ASR 前置 VAD/能量闸门拦下"根本没人说话"的段落。

**假设回顾:**

- 假设 1(无语音输入触发幻觉):**强成立**,91% 命中。
- 假设 2(VAD 压制幻觉):**部分成立** —— 对白噪/静音完全有效,对 babble 失效。这是本实验最大 insight。
- 假设 3(纯零最安全):**否**,纯零静音照样稳定吐字幕残留。Whisper 不靠能量门控决定是否解码。

### 典型案例

- **案例 A(纯零幻觉)**:`zero_30s`(全 0 采样)→ "由 Amara.org 社群提供的字幕"。绝对静音也无中生有,VAD 必要性的最强论据。
- **案例 B(VAD 的盲区)**:`babble_15s_10s` VAD off → "请不吝点赞订阅...",VAD on → "感谢观看"。VAD 没拦住,只是换了个幻觉文本 —— 人声型噪声始终骗得过 VAD。

### 踩坑记录

- 本机无 ffmpeg,`luyin/*.m4a` 真实录音第三类噪声 soundfile 读不了 → 实验 4 暂跳过真实录音素材(加餐性质,不阻塞),纯噪声/静音已足够说明问题。
- 含中文幻觉文本的脚本输出在 Windows GBK 控制台会崩 → 沿用实验 1 经验,`PYTHONIOENCODING=utf-8` 跑。
- 出图用英文标签:本机 matplotlib 未配中文字体(`plot_denoise` 同),`plot_exp4.py` 全英文标签避开豆腐块。

## 2026-06-15(消融总表整理 · 阶段二收尾)

把实验 1–4 汇成一张消融主表,正面回答项目总问题"哪些预处理真有用、哪些看起来高级却不划算"。

### 产出

- `results/ablation_summary.md`:消融主表(6 行链路 baseline → +降噪 → +分离 → +降噪后分离 → +分离后降噪 → +LLM 纠错)+ 三域口径声明 + 反直觉发现汇总 + 规则版推荐链路。
- `results/ablation_pipelines.png`:实验 2 五链路 content CER 均值对比图(英文标签,视频用)。
- `src/plot_ablation.py`:出图脚本。

### 关键聚合(原始 csv 精确计算,非手抄)

- **域 A 降噪**(实验 1,6 噪声条件 CER 均值):whisper none **26.5%** → frcrn 47.9% / specsub 43.0%(全害);funasr none **30.3%** → frcrn 36.1% / specsub 27.3%(specsub 均值上略益)。唯一大幅救场:funasr white_0dB 68.4→**27.5**。
- **域 B 分离/顺序**(实验 2,15 格 content CER 均值):L1 **39.8%** / L2 52.5% / L3 68.2% / L4 67.4% / L5 65.3%;spk CER L3-5 均 84–88%(短片段没分开)。L1 在 15 格中 **13 格最优**,L3 仅 2 格(均 heavy 重叠)。
- **域 C 纠错**(实验 3,5 案例 content CER 均值):raw **90.0%** → 纯 LLM 41.0% → 热词 53.0%(降幅来自拒绝 2 条幻觉,非修字词;热词反升)。

### 口径提醒

三域绝对 CER 不可跨行直接比(单人 vs 重叠双人 vs 挑选案例),消融主表跨行只比净效果方向。

### 主结论

2–4s 短辩论片段 + 噪声/重叠约束下,**直接 ASR(L1)几乎总是性价比最高**;复杂前端只在极窄条件回本(FunASR×white×低 SNR 的降噪、heavy 重叠的分离)。复杂度每加一层,多数条件 CER 不降反升 → 正面回答项目总问题。

### 阶段二收尾状态

实验 1–4 + 消融总表完成。每个实验的假设/设置/结果/失败案例/反直觉发现均已在上方各实验段落记录(PLAN 消融要求项)。下一步进入阶段三(真实录音泛化抽查 + Streamlit demo)。

## 2026-06-16(阶段三 · 真实录音泛化抽查)

把阶段一/二的结论拿到**手机实录的真实音频**上做泛化抽查:换一套独立采集的数据,直接 ASR 是否仍是强 baseline?降噪/分离/VAD 在真实噪声上是否还是负收益为主?

### 假设

阶段二的主结论(直接 ASR 强、降噪/分离多负收益、只有 heavy 重叠分离回本、VAD 对人声型噪声盲区、幻觉跨数据稳定)是合成/受控数据上得到的;真实录音应当能独立复现,否则结论不可泛化。

### 设置

- 6 条真实录音(5 必录 + 1 可选 heavy,`luyin/*.m4a`)。**本机无 ffmpeg**,改用 venv 自带 PyAV(av 17.1,内置 ffmpeg 库)写 `src/convert_audio.py` 解码重采样为 16k mono wav(`data/real/raw/`),解开实验四曾因此跳过真实噪声的阻塞。
- 分两类:单人条(dorm_clean 干净 / canteen 食堂强噪 / classroom 教室轻噪)、双人重叠条(discussion_quiet 安静 / discussion_canteen 噪声+自然插话 / discussion_canteen_heavy 噪声+刻意同步重叠)。
- baseline = 直接 ASR(whisper VAD off / VAD on / funasr);链路按类型自动选:单人降噪→ASR(specsub + frcrn),重叠分离→ASR(L3)/ 降噪→分离→ASR(L4)。编排 `src/run_real.py`,结果 `results/real_asr.csv`。
- **不算 CER**:文稿允许意译(录音要求即"意思和热词说到即可")、重叠条无逐字 GT。改人工 spot-check,并排表 `results/real_spotcheck.md`(`src/report_real.py` 生成)。dorm_clean 与 canteen **同稿**,留作真实"clean vs 强噪"直接对照。
- 中间产物(降噪/分离 wav)落盘 `data/real/{denoised,sep_L3,sep_L4}/` 作视频前后试听素材。

### 结果与发现(真实数据逐条复现阶段二)

1. **直接 ASR 是真实数据上的强 baseline**:clean、轻噪、自然插话条几乎全文转出;heavy 强噪仍保留主干句。复现"L1 几乎总最优"。
2. **降噪对 Whisper 真实噪声基本无益,且引入幻觉**:canteen+谱减末尾冒"请不吝点赞订阅订阅";frcrn 把"噪声"转成"造孙"、尾巴更乱。复现域 A"降噪对 Whisper 全害"。
3. **分离在自然/轻重叠上失败(复制混合)**:discussion_quiet 的 L3 spk2 ≈ 整句混合复制、spk1 是碎片;L4 spk1 陷入循环复读。**只有 heavy 真重叠回本**:discussion_canteen_heavy 直接 ASR 丢掉 A 的同步句("谱减法对平稳噪声…对食堂"),分离后 spk2 把它找回。复现"只有 heavy 重叠分离才勉强回本"。
4. **L4(降噪→分离)最脆**:discussion_canteen 的 L4 spk2、heavy 的 L4 spk1 都吐"中文字幕志愿者 李宗盛"。复现实验四字幕幻觉,且是处理链放大出来的。
5. **VAD 双刃**:classroom(轻噪+尾静音)VAD on 消掉尾部"感谢观看"幻觉(有益);canteen(强连续噪声)VAD on 把真实语音中段整块切掉、内容大面积缺失(有害)。复现实验四"VAD 对连续/人声型噪声盲区"。
6. **热词普遍转错,支撑实验三动机**:英文专名在真实录音上稳定出错(FunASR→方ASR、MossFormer2→MouseFormer / MOSFORMER、DeepFilterNet→DeepFlatNet、谱减法→铺剪法 / 蒲间法、信噪比→心噪笔)。亮点:discussion_canteen 里清晰念出时 Whisper 把 DeepFilterNet 转对了("deep filter net"),说明是声学清晰度问题而非词表缺失。

### 反直觉 / nuance

- **真实"噪声+自然插话"(discussion_canteen)的直接 ASR 反而比"安静重叠"(discussion_quiet)的同步段更完整** —— 前者是自然轮流+少量句尾插话(时间上几乎不重叠),后者含刻意的 5–6s 完全同步段。证明伤害 ASR 的是**同步重叠时长**,不是有没有噪声。
- 同一句脚本 clean(dorm)vs 强噪(canteen):专名从"Whisper/FunASR 基本对"退化到"FoundDSR / Most Form Pro",噪声主要冲掉专名/低频词的声学细节,而非整体语义。

### 典型案例(视频高光)

- **案例 A(降噪反害)**:canteen + 谱减 → 末尾"请不吝点赞订阅订阅" —— 降噪伪影直接触发字幕幻觉。
- **案例 B(分离唯一回本)**:discussion_canteen_heavy 直接 ASR 丢 A 的同步句,分离 spk2 找回 → 分离价值的最强论据。
- **案例 C(L4 幻觉放大)**:discussion_canteen / heavy 降噪→分离 → "中文字幕志愿者 李宗盛"。
- **案例 D(VAD 双刃)**:classroom VAD 消尾部幻觉(益)vs canteen VAD 切掉真实语音(害),同一开关相反效果。

### 踩坑记录

- 本机无 ffmpeg → m4a 读不了(实验四曾因此跳过真实噪声)。发现 venv 自带 PyAV 内置 ffmpeg 库,`convert_audio.py` 用 `av.AudioResampler` 解码重采样,无需装系统 ffmpeg。
- 文稿非逐字(允许意译)→ 不算 CER 只做 spot-check;靠 dorm/canteen 同稿保留一处可比的 clean-vs-noise 对照。
- 沿用 `PYTHONIOENCODING=utf-8` 跑(中文+幻觉文本在 Windows GBK 控制台会崩)。

### 小结

真实录音独立复现阶段二全部主结论,泛化成立。结论一句话:**换一套真实数据,"直接 ASR 最稳、复杂前端多半帮倒忙、只有 heavy 重叠分离回本、VAD 对人声噪声有盲区"依旧成立**。

### Demo 固定素材打包(同日)

无需补录,直接用已有抢话录音打包固定 demo 素材(`src/make_demo.py` → `demo/audio/` 10 条前后音频 + `demo/cases.md` 5 案例,文本从 `results/real_asr.csv` 精确取出):案例 A=heavy 抢话分离唯一回本(直接 ASR 丢 A 同步句、spk2 找回)/ B=自然插话直接 ASR 够用 / C=安静重叠分离复制混合 / D=canteen 降噪反害幻觉 / E=VAD 双刃。目的是录视频时有固定样例,不现场随机翻车。0dB babble 合成重叠样本(human-vs-Whisper 互动段)按需从 `data/overlap_noisy/babble_0dB/heavy/` 取。

阶段三剩余:Streamlit demo 界面 + 视频脚本大纲。

## 2026-06-16(阶段三收尾 + 阶段四图表:demo 界面 / 视频脚本大纲 / trade-off 简表)

承上,补齐阶段三最后两项,并推进阶段四第 12 天的图表整理。

### Streamlit demo 最小版(`demo/app.py`,混合形态)

- 一页两 tab:**Tab1 固定案例**(读 `make_demo.CASES` + `demo/audio/` + `real_asr.csv`,选案例 A–E → 前后音频试听 + 前后文本对照 + 耗时/RTF + 讲点,**零现场跑、不翻车**);**Tab2 上传现场跑**(上传音频 → 直接 ASR,faster-whisper large-v3,可选 VAD → 文本 + 时长/耗时/RTF + 试听,模型 `@st.cache_resource` 缓存)。
- 降噪/分离等重链路只在 Tab1 以预跑结果展示,现场不跑(沿用 `make_demo` "避免录视频翻车"的初衷)。
- 依赖:`streamlit==1.58.0` 入 `requirements.txt`。自测:headless 启动 `/_stcore/health=ok` 无报错;Tab1 数据完备校验(5 案例的音频/csv 行/文本/RTF 齐)通过。
- 注:`demo/audio/` 在 `.gitignore` 内(音频不入库),clone 后需本地重跑 `make_demo.py` 复原 Tab1 音频;`app.py` 与 `cases.md` 入库。

### 视频脚本大纲(`video_script_outline.md`,中文规划稿)

- 12 段时间轴(约 12 分钟,可裁到 10),辩论赛叙事(噪声=起哄 / 重叠=抢话 / LLM=校对),每段标【画面 + 素材路径】【中文旁白要点】【数据出处】【评分维度 insight/nuances/depth/difficulties/fun】。
- 实验数字全部回填 LOG 出处;片尾含团队分工 + AI 协作署名。成片英文录制,故本稿为中文规划稿。

### 精度-耗时 trade-off 简表(`src/plot_tradeoff.py` → `results/tradeoff_summary.md` + `tradeoff.png`)

- 链路总 RTF 由各环节**实测** RTF 组装(非手填):ASR 取 `exp2_summary.asr_rtf`、降噪取 `denoise_raw` 的 `exp2_L2/exp2_L5`、分离取 `separate_raw` 的 `exp2_L3/exp2_L4`;双路环节(分离链路 ASR、L5 降噪)按 2 路计;L2/L4 共用"对原混合降噪"、L5 复用 L3 分离。

| 链路 | content CER | 链路 RTF | ×L1 |
|---|---|---|---|
| L1 直接 | 39.8% | 0.191 | 1.0 |
| L2 降噪 | 52.5% | 0.235 | 1.2 |
| L3 分离 | 68.2% | 0.520 | 2.7 |
| L4 降噪→分离 | 67.4% | 0.533 | 2.8 |
| L5 分离→降噪 | 65.3% | 0.645 | 3.4 |

- CER 与 `ablation_summary.md` 完全一致(校验聚合口径正确);RTF 的 L3–L5 = 2.7–3.4× L1,印证本文 2026-06-13"分离链路约 L1 的 2–3 倍耗时"。
- 结论:**精度-耗时平面上没有任何链路落在 L1 的右下方(更快更准)** —— 量化坐实"复杂前端多花算力却更不划算"的项目总结论。

### 阶段进度

- 阶段三全部完成(真实录音抽查 + Streamlit demo + 视频脚本大纲)。
- 阶段四第 12 天图表整理完成(退化曲线 / 降噪对比 / 处理顺序 / 失败案例 / trade-off 五项齐 + 结论 `ablation_summary.md`)。下一步(本次未做,按用户要求停在写脚本前):写英文视频脚本逐字稿 → 录制剪辑 → 提交材料打包。

## 2026-06-16(结论 × 文献佐证对照)

回应老师"实验结论需有现实研究/博客论证"的要求,把 8 条核心结论逐条对照公开论文/社区博客,产出 `results/literature_support.md`(团队共享用)。

### 核验结论

- **6 条强证实**:Whisper 静音/非语音幻觉(arXiv 2501.11378;whisper Discussion #928)、降噪伤 ASR(arXiv 2201.06685《How Bad Are Artifacts?》/ 2512.17562《When De-noising Hurts》40/40 配置全劣化)、over-separation(arXiv 2106.00949《Should We Always Separate?》/ 2503.17886)、LLM 是过滤器非纠错器(arXiv 2409.09785 / 2505.24347)、VAD 边界、模型抗噪画像差异(Whisper 原论文 2212.04356 / Whisper-AT 2307.03183)。
- **1 条证实但需注意 SNR 依赖**:babble vs white 的相对难度依赖 SNR 区间,保持"0dB 交叉/低 SNR 两模型弱点相反"表述,勿写成单调关系。
- **2 条需收窄/标为新发现**:(A)"处理顺序不重要"实为分离失效(spk CER 84–88%)的退化情形,应收窄为"分离失效条件下无稳定差异";(B)"FunASR 抗 babble 强于 Whisper"文献无直接对比,作为本项目新发现陈述(符合"抗噪画像随训练数据而异"总规律)。

### 备注

仅做"结论 ↔ 文献"映射,未改任何实验数字。

## 2026-06-19（加餐实验 5b：预处理 × 语音情感）

回应老师建议探索 “Emotion in Audio/Voice/ASR”。作为加餐长在“高级预处理不一定划算”主线上，问：降噪/分离在改善文字的同时，是否擦掉副语言情绪。零新增依赖，沿用 FunASR + ModelScope `iic/` 栈。

### 模型与先验探针

- SER：`iic/emotion2vec_plus_large`（9 类，16kHz，FunASR `AutoModel`）。
- 探针：对 26 条 clean 片段跑 SER → **18/26 非中立（平均置信 0.891），其中 angry=17**（辩论本身对抗性语气）。源情绪分明，E5b 用原数据即可，无需 ESD。
- 指标：**P(angry) 均值**（免 spk 配对），比较处理前后愤怒强度是否被削平。

### 实验假设

降噪/分离会系统性降低 P(angry)；分离（短片段失真严重）比降噪伤得更重。

### 设置（`src/run_emotion.py`）

- A 降噪伪影：`clean` vs `denoised/{frcrn,specsub}/clean`（对干净片段也降噪，无噪声混淆）。
- A' 真实管线：`clean` → `denoised/{frcrn,specsub}/{noise_snr}`，按 SNR 聚合。
- B 分离：源 con/pro vs `exp2/{L3_sep,L4_sep}/clean/{no,light,heavy}` 分离两路。

### 实验结果

- **A（17 条 angry 片段 P(angry) 均值）：clean 0.89 → FRCRN 0.85（翻转 1/17）→ SpecSub 0.65（翻转 4/17）。** 便宜谱减伤情绪明显重于神经降噪。
- **B（同批源说话人）：source 0.63 → L3 no/light/heavy 0.37 / 0.47 / 0.35。** 分离把愤怒腰斩，且与重叠程度无单调关系。
- 口径：A 基线为 17 条 angry-top 片段；B source 为 22 个源说话人（含中立）。有效对比是各支路内部前后差，A、B 绝对值不直接可比。

### 失败 / 反直觉

- 同一句两种降噪相反结局：`pro_014` angry(0.88)→谱减 neutral(0.02) 但 FRCRN 仍 angry(0.95)；`con_005` angry(0.92)→谱减 **happy(0.001)**（愤怒被误读成开心）。
- 分离对情绪的破坏 > 降噪，且 no(0.37)≈heavy(0.35) 无单调关系，与既有“分离在 2-4s 短片段普遍失效”自洽。

### 结论（接主线）

“高级预处理不一定划算”新增**副语言层**：即便对文字有时有用，降噪/分离会系统性擦掉“话说得多激动”；**分离 > 降噪、谱减 > 神经降噪**。回应老师 emotion 方向。

### 产物

`results/exp5_emotion.md` / `exp5_emotion_clips.csv` / `exp5_emotion_sep.csv` / `exp5_emotion_drift.png`；脚本 `src/run_emotion.py`。

## 2026-06-23（按老师反馈撰写正式研究报告）

老师群内反馈：提交在 depth / justification / supporting evidence 上不足——选型缺论证、缺文献、缺"前人局限"与"测试什么假设"的显式表述。实验已基本完成，本次补"研究报告"这一层（非新实验），把已有 PLAN/LOG/project_notes/results 的真实迭代与决策提炼成研究推理体（去日期、去流水账）。

### 产物：REPORT.md（中文主稿，术语 English）

九节正文 + 复现 + References：Abstract / Introduction / Related Work / Hypotheses(H1–H5) / **Design Rationale（选型论证，写最详）** / Experimental Setup / Experiments 1–6 / Discussion / Limitations。
- 正面回应老师点名问题：为何 faster-whisper 而非 WhisperX（短片段不需 alignment/diarization、且要单控 VAD 变量）；为何加 FunASR 做异构对照；DeepFilterNet→FRCRN、SepFormer→MossFormer2 的真实试错。
- 显式列 H1–H5 假设及"证实/推翻/收窄"；§2.6 单列前人工作（学长 `xutong_paper`）三点局限。

### 新增图表

- `results/exp2_pipelines.png`（脚本 `src/plot_exp2.py`）：五链路按 overlap 程度 content CER 分组柱状，直观呈现 over-separation（no-overlap L1=23.8% vs 分离 ≈64–70%；heavy 时 L1=63.3%≈L3=64.0%）。三档均值复算 = 39.8，与 ablation 表 L1 自洽。
- REPORT 内嵌 3 张 mermaid：系统架构 / L1–L5 链路 / 文献三态（证实/收窄/新发现）。
- §6.6 补 audio examples 案例表（A–E，指向 `demo/audio/` + `demo/cases.md`）。

### 文献编号核实

联网核对所用模型原始论文 arXiv 编号并补入 References：Paraformer 2206.08317 / FunASR 2305.11013 / MossFormer2 2312.11825 / FRCRN 2206.07293 / ClearerVoice-Studio 2506.19398 / emotion2vec 2312.15185 / SepFormer 2010.13154 / Conv-TasNet 1809.07454 / DeepFilterNet 2110.05588。

### 待补（不阻塞）

- demo 截图 → `results/demo_screenshot.png`（§10 已留图位 + 启动命令）。
- 实验 6 长音频 length ablation（占位，验证处理顺序结论的边界条件）。
- mermaid → PNG（仅导出 PDF / 视频 slides 时需要）。
