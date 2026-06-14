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
