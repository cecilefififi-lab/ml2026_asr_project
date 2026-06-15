"""把真实录音抽查里最有代表性的案例打包成固定 demo 素材, 供录视频/demo 直接调用,
避免现场随机翻车。音频从 data/real/ 已有产物复制并清晰重命名, 前后文本从
results/real_asr.csv 精确取出(不手抄)。

产出:
    demo/audio/*.wav   固定演示音频(前/后)
    demo/cases.md      每个案例: 类型 + 前后文本 + 一句话讲点

用法: python src/make_demo.py
"""
import csv
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(BASE, "data", "real")
CSV_PATH = os.path.join(BASE, "results", "real_asr.csv")
DEMO = os.path.join(BASE, "demo")
AUDIO = os.path.join(DEMO, "audio")

# 源音频(相对 data/real/)
RAW = lambda s: os.path.join(REAL, "raw", s + ".wav")
SPECSUB = lambda s: os.path.join(REAL, "denoised", "specsub", "real", s + ".wav")
SEP = lambda s, n: os.path.join(REAL, "sep_L3", f"{s}_spk{n}.wav")

# 每个案例: 复制哪些音频(demo文件名 -> 源路径) + 展示哪些文本行(标签 -> (tag,engine,file))
CASES = [
    {
        "id": "A", "title": "两人抢话 · 分离唯一回本",
        "type": "双人·食堂噪声+重度同步重叠 (discussion_canteen_heavy)",
        "audio": {
            "caseA_heavy_raw.wav": RAW("discussion_canteen_heavy_01"),
            "caseA_heavy_sep_spk1.wav": SEP("discussion_canteen_heavy_01", 1),
            "caseA_heavy_sep_spk2.wav": SEP("discussion_canteen_heavy_01", 2),
        },
        "rows": [
            ("直接 ASR(混合)", "base", "whisper", "discussion_canteen_heavy_01.wav"),
            ("分离 spk1", "L3_sep", "whisper", "discussion_canteen_heavy_01_spk1.wav"),
            ("分离 spk2", "L3_sep", "whisper", "discussion_canteen_heavy_01_spk2.wav"),
        ],
        "point": "两人同时念 5–6 秒时,直接 ASR 只锁住一个人、丢掉 A 的同步句"
                 "(谱减法对平稳噪声…对食堂);分离后某一路把它找回 —— 这是整套实验里"
                 "分离唯一明显回本的场景。",
    },
    {
        "id": "B", "title": "自然插话 · 直接 ASR 够用",
        "type": "双人·食堂噪声+自然插话 (discussion_canteen)",
        "audio": {"caseB_canteen_raw.wav": RAW("discussion_canteen_01")},
        "rows": [("直接 ASR", "base", "whisper", "discussion_canteen_01.wav")],
        "point": "自然轮流+少量句尾插话(时间上几乎不重叠)时,直接 ASR 几乎全文转出。"
                 "对照案例 A 说明:伤害 ASR 的是『同步重叠时长』,不是『有没有噪声』。",
    },
    {
        "id": "C", "title": "安静重叠 · 分离复制混合(帮倒忙)",
        "type": "双人·安静重叠 (discussion_quiet)",
        "audio": {
            "caseC_quiet_raw.wav": RAW("discussion_quiet_01"),
            "caseC_quiet_sep_spk1.wav": SEP("discussion_quiet_01", 1),
            "caseC_quiet_sep_spk2.wav": SEP("discussion_quiet_01", 2),
        },
        "rows": [
            ("直接 ASR(混合)", "base", "whisper", "discussion_quiet_01.wav"),
            ("分离 spk1", "L3_sep", "whisper", "discussion_quiet_01_spk1.wav"),
            ("分离 spk2", "L3_sep", "whisper", "discussion_quiet_01_spk2.wav"),
        ],
        "point": "自然轮流式重叠下分离失败:spk2≈把整句混合复制了一遍,spk1 只剩碎片。"
                 "复杂前端在这里纯属帮倒忙。",
    },
    {
        "id": "D", "title": "降噪反害 · 伪影触发字幕幻觉",
        "type": "单人·食堂强噪 (canteen)",
        "audio": {
            "caseD_canteen_raw.wav": RAW("canteen_01"),
            "caseD_canteen_specsub.wav": SPECSUB("canteen_01"),
        },
        "rows": [
            ("直接 ASR", "base", "whisper", "canteen_01.wav"),
            ("谱减降噪→ASR", "specsub", "whisper", "canteen_01.wav"),
        ],
        "point": "食堂强噪下谱减降噪不仅没救回内容,末尾还凭空冒出『请不吝点赞订阅订阅』"
                 "—— 降噪伪影直接触发训练数据里的字幕式幻觉。",
    },
    {
        "id": "E", "title": "VAD 双刃 · 同一开关相反效果",
        "type": "单人·教室轻噪 (classroom) vs 食堂强噪 (canteen)",
        "audio": {"caseE_classroom_raw.wav": RAW("classroom_01")},
        "rows": [
            ("classroom 直接 ASR", "base", "whisper", "classroom_01.wav"),
            ("classroom 直接 ASR+VAD", "base_vad", "whisper", "classroom_01.wav"),
            ("canteen 直接 ASR", "base", "whisper", "canteen_01.wav"),
            ("canteen 直接 ASR+VAD", "base_vad", "whisper", "canteen_01.wav"),
        ],
        "point": "classroom(轻噪+尾静音)开 VAD 消掉尾部『感谢观看』幻觉(有益);"
                 "canteen(强连续噪声)开 VAD 反而把真实语音中段整块切掉(有害)。",
    },
]


def main():
    lut = {(r["tag"], r["engine"], r["file"]): r["text"].strip()
           for r in csv.DictReader(open(CSV_PATH, encoding="utf-8-sig"))}
    if os.path.exists(AUDIO):
        shutil.rmtree(AUDIO)
    os.makedirs(AUDIO)

    out = ["# Demo 固定素材 · 真实录音案例集\n",
           "录视频/demo 直接用的固定样例。音频在 `demo/audio/`,文本取自 "
           "`results/real_asr.csv`。每个案例配一句讲点。\n",
           "> 视频互动『人类 vs Whisper』段另有 0dB babble 合成重叠样本可用,见 "
           "`data/overlap_noisy/babble_0dB/heavy/`(本包未复制,按需取)。\n"]

    n_audio = 0
    for c in CASES:
        for dst_name, src in c["audio"].items():
            assert os.path.exists(src), f"缺音频: {src}"
            shutil.copy(src, os.path.join(AUDIO, dst_name))
            n_audio += 1
        out.append(f"## 案例 {c['id']} — {c['title']}\n")
        out.append(f"**录音**:{c['type']}\n")
        if c["audio"]:
            out.append("**音频**:" + " / ".join(f"`audio/{k}`" for k in c["audio"]) + "\n")
        out.append("| 阶段 | ASR 输出 |")
        out.append("|---|---|")
        for label, tag, eng, fname in c["rows"]:
            out.append(f"| {label} | {lut.get((tag, eng, fname), '(缺)')} |")
        out.append(f"\n**讲点**:{c['point']}\n")

    with open(os.path.join(DEMO, "cases.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(out))
    print(f"copied {n_audio} audio -> demo/audio/, wrote demo/cases.md")


if __name__ == "__main__":
    main()
