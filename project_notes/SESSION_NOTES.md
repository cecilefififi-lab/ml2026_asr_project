# Session Notes

## 2026-06-12

- 确定新增 `project_notes/` 文件夹，用来存放“给人看的”项目总结材料。
- 明确 `LOG.md` 的定位：详细记录实验过程、踩坑和 agent 可复盘的信息；`project_notes/` 的定位：直接服务课程报告、视频脚本和答辩说明。
- 约定每日记录文件命名规则：`YYYY-MM-DD_序号.md`，如果同一天有多个重要产出，就依次写成 `_01`、`_02`。
- 新增 `2026-06-12_01.md`，把阶段一产出整理为人类可读版本，重点记录产出位置、实验流程、实验结果、课程知识点、失败案例和报告表述。
- 新增 `2026-06-12_02.md`，把实验一“噪声 × 降噪 × ASR”的产出整理为人类可读版本，覆盖产出目录、关键 CER/RTF 数字、图表位置、典型成功/失败案例、课程知识点、实验结论和中英文报告表述。
- 本次仅更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件。
- hzy提交噪声分析相关py和csv
- tjx提交实验一总结数据处理文档
- jxy提交清洗好的音频数据
- hsf、gxy录制实验二所需音频

## 2026-06-13

- 新增 `2026-06-13_01.md`，把实验二“噪声 + 重叠语音处理顺序”的产出整理成人类可读版本。
- 本次总结覆盖数据、表格、文档和实验脚本位置，并说明实验二没有新增独立 PNG 图，主可视化材料是 `results/exp2_pivot.md`。
- 重新核对 `results/exp2_summary.csv`：直接 ASR 链路 L1 在 15 个条件中有 13 个条件 content CER 最低，L3 只在 2 个 heavy 重叠条件中最低。
- 总结中补充了 content CER、per-speaker CER、RTF、典型失败案例、机器学习课程知识点、实验结论、后续建议，以及可直接放入课程报告的中英文表述。
- 本次只更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件。

## 2026-06-14

- 新增 `2026-06-14_01.md`，把实验三“小型 LLM 语义纠错实验”的产出整理成人类可读版本。
- 本次总结自动参考了 `PLAN.md`、`LOG.md`、`project_notes/SESSION_NOTES.md`，并核对了今天主要产出 `results/exp3_correction.md`、`results/exp3_correction.csv`、`data/exp3_inputs.json`、`refs/hotwords.txt` 和 `src/correct.py`。
- 总结中说明了实验三的数据、表格、文档和脚本位置；实验三没有新增独立 PNG 图，核心可视化材料是 `results/exp3_correction.md` / `.csv` 的 5 案例对比表。
- 总结补充了 raw / 纯 LLM / 热词辅助三档 CER 数字、LLM 能拒绝幻觉但不能恢复声学丢失信息的结论、机器学习课程知识点、失败与踩坑，以及可直接放入课程报告的中英文表述。
- 本次只更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件。
- jxy提交每日md和session_note
- hzy提供correct.py和ep3-input-json
- tjx提供ep3—correction.md/csv
- 今后提交加入ai协作者如codex和claudecode，直观表现

## 2026-06-15

- 新增 `2026-06-15_01.md`，把阶段二收尾部分整理成人类可读版本，重点覆盖实验四“Whisper 幻觉小实验”和消融总表。
- 本次总结参考了 `PLAN.md`、`LOG.md`、`project_notes/SESSION_NOTES.md`，并核对了今天主要产出：`data/exp4/`、`results/exp4_hallucination.csv`、`results/exp4_hallucination.md`、`results/exp4_vad_compare.png`、`results/ablation_summary.md`、`results/ablation_pipelines.png`。
- 总结中说明了实验四的数据、图、表格和文档位置，补充了 VAD off 91% 幻觉率、white/silence 100%→0%、babble 67%→67% 等关键数字。
- 总结中整理了消融表主结论：直接 ASR 在实验二 15 个条件中 13 个最优；降噪、分离、处理顺序和 LLM 后处理都有明显适用边界。
- 总结补充了机器学习课程知识点、失败或不顺之处、后续建议，以及可直接放入课程报告和视频答辩的中英文表述。
- 本次只更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件。

## 2026-06-16

- 新增 `2026-06-16_01.md`，把阶段三“真实录音泛化抽查”和 demo 固定素材整理成人类可读版本。
- 本次总结参考了 `PLAN.md`、`LOG.md`、`project_notes/SESSION_NOTES.md`，并核对了今天主要产出：`data/real/raw/`、`data/real/denoised/`、`data/real/sep_L3/`、`data/real/sep_L4/`、`results/real_asr.csv`、`results/real_spotcheck.md`、`demo/audio/` 和 `demo/cases.md`。
- 总结中说明了真实录音的数据、图/可视化材料、表格、文档和 demo 素材位置，并明确本次没有新增独立 PNG 图；同时强调本次因缺少逐字 ground truth 而采用人工 spot-check，不强行计算 CER。
- 总结整理了真实录音复现出的关键现象：直接 ASR 仍最稳；食堂降噪可能触发“请不吝点赞订阅”等字幕式幻觉；分离只在 heavy 同步重叠中有回本价值；VAD 在轻噪下有益、强人声噪声下可能误切真实语音。
- 总结补充了机器学习课程知识点、失败或不顺之处、后续建议，以及可直接放入课程报告、视频和答辩的中英文表述。
- 本次只更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件。

- 新增 `2026-06-16_02.md`，把今天后续新增的 Streamlit demo、视频脚本大纲、精度-耗时 trade-off 图表和文献佐证对照整理成人类可读版本。
- 本次总结参考了 `PLAN.md`、`LOG.md`、`project_notes/SESSION_NOTES.md`，并核对了今天的主要产出：`demo/app.py`、`demo/cases.md`、`demo/audio/`、`results/tradeoff_summary.md`、`results/tradeoff.png`、`results/literature_support.md`、`video_script_outline.md`、`requirements.txt`、`results/ablation_summary.md` 和 `results/real_spotcheck.md`。
- 总结中说明了本次数据、图、表格和文档分别所在目录，整理了 demo 五个固定案例、L1-L5 的 CER/RTF trade-off、视频 10-12 分钟叙事结构，以及哪些结论有文献支撑、哪些需要收窄表述。
- 总结补充了机器学习课程知识点、实验失败或不顺之处、后续建议，并给出可直接写入课程报告和视频答辩的中英文表述。
- 本次只更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件。

## 2026-06-19

- 新增 `2026-06-19_01.md`，把实验 5b“预处理是否抹掉语音情绪”整理成人类可读版本。
- 本次总结参考了 `PLAN.md`、`Project.md`、`asr-robustness/LOG.md`、`project_notes/SESSION_NOTES.md`，并核对了今天更新的主要产出：`src/run_emotion.py`、`results/exp5_emotion.md`、`results/exp5_emotion_clips.csv`、`results/exp5_emotion_sep.csv`、`results/exp5_emotion_drift.png`。
- 总结中说明了数据、图、表格、文档和脚本的位置，整理了 emotion2vec P(angry) 指标、clean 到降噪、噪声加降噪、分离链路三组结果，以及可直接写入课程报告和英文视频的中英文表述。
- 本次只更新 `project_notes/` 下的 Markdown 总结文件；未修改代码，未删除文件，也未涉及无关文件。
