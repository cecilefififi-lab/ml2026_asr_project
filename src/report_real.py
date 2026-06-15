"""把 results/real_asr.csv 整理成人工 spot-check 文档 results/real_spotcheck.md。

每条录音并排展示: 参考文稿 + 各阶段(baseline / 降噪 / 分离)ASR 输出 + 热词命中。
不算 CER(文稿可意译、重叠条无逐字 ground truth), 供人工对照是否变好。
参考文稿取自 luyin/录音要求.txt; 重叠条为"参考对话", 非逐字。
"""
import csv
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "..")
CSV_PATH = os.path.join(BASE, "results", "real_asr.csv")
OUT_MD = os.path.join(BASE, "results", "real_spotcheck.md")

HOTWORDS = ["Whisper", "FunASR", "MossFormer2", "DeepFilterNet", "谱减法", "信噪比",
            "语音分离", "说话人归属", "字错误率", "热词表", "声纹", "重叠语音"]

TYPE = {
    "dorm_clean_01": "单人·干净对照", "canteen_01": "单人·食堂强噪",
    "classroom_01": "单人·教室轻噪", "discussion_quiet_01": "双人·安静重叠",
    "discussion_canteen_01": "双人·食堂噪声+重叠",
    "discussion_canteen_heavy_01": "双人·食堂噪声+重度重叠",
}

# 参考文稿(luyin/录音要求.txt); 对话条为参考对话, 非逐字
REF = {
    "dorm_clean_01": "这学期我们做的项目是语音识别的鲁棒性研究,主要对比 Whisper 和 FunASR "
    "两个模型在噪声和重叠语音下的表现。预处理这边,降噪打算用谱减法和 DeepFilterNet 各做一组,"
    "语音分离用 MossFormer2。评价指标主要看字错误率,再加上说话人归属的准确率,"
    "最后比较不同信噪比下,哪一条处理链路效果最好、代价最小。",
    # canteen_01 与 dorm_clean_01 同稿(条2=条1), 末尾 main() 里补
    "classroom_01": "实验里我们发现,噪声一大,模型就开始把背景人声也转写出来,甚至在静音段冒出"
    "幻觉文本。所以我们先做了说话人归属测试,看声纹特征在轻噪声下还稳不稳定,再用热词表辅助大模型"
    "做语义纠错。像 Whisper、MossFormer2 这种英文名词,不加热词表十有八九会被转错,"
    "这正好是我们要对比的地方。",
    "discussion_quiet_01": "[参考对话] A:实验先跑 Whisper 比较稳,FunASR 放第二阶段。"
    "B:语音分离这块,MossFormer2 环境装好了吗? A:装好了,短片段分离不太稳定,要加上下文补齐。"
    "B:先用长片段把流程跑通。// (轻度重叠)A:信噪比太低分离效果会—— B:会崩,残留很明显。"
    "A:要不要先降噪再分离? B:先降噪可能把人声细节削掉,声纹都对不上。"
    "// (重度重叠)A:谱减法对平稳噪声有效,对食堂人声噪声基本没用。"
    "B:说话人归属在重叠语音下错误率最高,声纹特征会互相污染。"
    "// A:按降噪后分离先做一组,对照组直接转写。B:字错误率我统计,热词表你整理。",
    "discussion_canteen_01": "[参考对话] A:背景全是人声,正好测谱减法效果。B:不行,非平稳噪声得上 "
    "DeepFilterNet。A:处理顺序先降噪还是先分离—— B:先分离,不然降噪把 B 的声音当噪声削了。"
    "A:先分离每一路还带食堂背景音,信噪比照样上不去。B:两种顺序各跑一遍,用字错误率说话。"
    "A:回去拿这条录音试,看说话人归属对不对。",
    "discussion_canteen_heavy_01": "[参考对话] 前段正常对话(提到 Whisper、语音分离)// "
    "(重度重叠)A:谱减法对平稳噪声有效,对食堂人声噪声基本没用。"
    "B:说话人归属在重叠语音下错误率最高,声纹特征会互相污染。// 正常收尾一句。",
}

STAGE_ORDER = ["base", "base_vad", "specsub", "frcrn", "L3_sep", "L4_dns_sep"]
STAGE_LABEL = {
    "base": "直接ASR", "base_vad": "直接ASR+VAD", "specsub": "谱减降噪→ASR",
    "frcrn": "FRCRN降噪→ASR", "L3_sep": "分离→ASR", "L4_dns_sep": "降噪→分离→ASR",
}


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def hot_hits(text):
    t = norm(text)
    return [h for h in HOTWORDS if norm(h) in t]


def parse_stem(fname):
    """real_asr.csv 的 file 列 -> (录音 stem, spk标签或'')。"""
    name = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"(.+)_(spk\d)$", name)
    return (m.group(1), m.group(2)) if m else (name, "")


REF["canteen_01"] = REF["dorm_clean_01"]  # 条2 与条1 同稿


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    # stem -> list of (stage, engine, spk, text, rtf)
    by_stem = {}
    for r in rows:
        stem, spk = parse_stem(r["file"])
        by_stem.setdefault(stem, []).append(
            (r["tag"], r["engine"], spk, r["text"].strip(), r["rtf"]))

    order = ["dorm_clean_01", "canteen_01", "classroom_01",
             "discussion_quiet_01", "discussion_canteen_01", "discussion_canteen_heavy_01"]

    out = ["# 阶段三 · 真实录音泛化抽查 (spot-check)\n",
           "对手机实录 6 条(转 16kHz mono)跑 baseline 与按类型自动选的最佳链路。",
           "不算 CER(文稿可意译、重叠条无逐字 ground truth);下表供人工对照\"是否变好\"。",
           "热词命中只做朴素子串匹配,英文名近音错(如 MouseFormer / DeepFlat)算未命中,正文另述。\n"]

    for stem in order:
        if stem not in by_stem:
            continue
        out.append(f"## {stem} — {TYPE.get(stem, '')}\n")
        ref = REF.get(stem, "")
        out.append(f"**参考文稿**:{ref}\n")
        ref_hot = hot_hits(ref)
        out.append(f"_文稿含热词_:{', '.join(ref_hot) if ref_hot else '—'}\n")
        out.append("| 阶段 | 引擎 | 路 | 命中热词 | ASR 输出 |")
        out.append("|---|---|---|---|---|")
        items = by_stem[stem]

        def sort_key(x):
            tag, eng, spk = x[0], x[1], x[2]
            return (STAGE_ORDER.index(tag) if tag in STAGE_ORDER else 9,
                    0 if eng == "whisper" else 1, spk)
        for tag, eng, spk, text, rtf in sorted(items, key=sort_key):
            hits = hot_hits(text)
            disp = text if text else "(空)"
            out.append(f"| {STAGE_LABEL.get(tag, tag)} | {eng} | {spk or '-'} | "
                       f"{len(hits)}/{len(ref_hot)} | {disp} |")
        out.append("")

    with open(OUT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out))
    print(f"wrote {os.path.relpath(OUT_MD, BASE)}")


if __name__ == "__main__":
    main()
