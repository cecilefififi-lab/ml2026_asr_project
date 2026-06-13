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
