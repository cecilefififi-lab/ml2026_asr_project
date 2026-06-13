"""实验 2 五链路评测:整体内容 CER + per-speaker CER + 处理顺序透视表。

CER 口径复用实验 1 evaluate.py 的 char_cer(NFKC + 去标点空白 + 字符级 jiwer)。
每个 pair 有两个 ground truth: refA=con_XXX, refB=pro_XXX。

- 单路链路 L1/L2(混合一条转写 hyp):
    content_cer = min over 拼接顺序 char_cer(refA+refB, hyp)   # 整体内容召回
- 双路链路 L3/L4/L5(分离两路 spk1/spk2, 顺序不定):
    取 permutation 较优配对(spk1->A,spk2->B vs 反之),
    content_cer = char_cer(refA+refB, hypA+hypB)               # 与单路同口径, 可比
    spk_cer     = (char_cer(refA,hypA)+char_cer(refB,hypB))/2  # 分离后每路准确度

输入: results/exp2_asr.csv, refs/
输出: results/exp2_summary.csv, results/exp2_pivot.md
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import char_cer  # 复用实验 1 CER 口径

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR_CSV = os.path.join(BASE, "results", "exp2_asr.csv")
REF_DIR = os.path.join(BASE, "refs")
SUM_CSV = os.path.join(BASE, "results", "exp2_summary.csv")
PIVOT_MD = os.path.join(BASE, "results", "exp2_pivot.md")
LINKS = ["L1", "L2", "L3", "L4", "L5"]
LINK_NAME = {"L1": "直接", "L2": "降噪→ASR", "L3": "分离→ASR",
             "L4": "降噪→分离", "L5": "分离→降噪"}
LEVELS = ["no", "light", "heavy"]


def read_ref(name):
    p = os.path.join(REF_DIR, name + ".txt")
    return open(p, encoding="utf-8").read().strip() if os.path.exists(p) else None


def parse(file):
    """'cond/level/con_001_pro_001[_spkN].wav' -> (cond, level, pair, spk)"""
    cond, level, name = file.split("/")
    name = os.path.splitext(name)[0]
    spk = None
    if "_spk" in name:
        name, spk = name.rsplit("_spk", 1)
    return cond, level, name, spk


def pair_refs(pair):
    """con_001_pro_001 -> (refA=con_001, refB=pro_001)"""
    parts = pair.split("_")
    return read_ref("_".join(parts[0:2])), read_ref("_".join(parts[2:4]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-csv", default=ASR_CSV)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.asr_csv, encoding="utf-8-sig")))
    # (engine,tag,cond,level,pair) -> {spk_or_"_": text}; 同时累计 RTF
    grp = defaultdict(dict)
    rtf = defaultdict(list)
    for r in rows:
        cond, level, pair, spk = parse(r["file"])
        grp[(r["engine"], r["tag"], cond, level, pair)][spk or "_"] = r["text"]
        rtf[(r["engine"], r["tag"], cond, level)].append(float(r["rtf"]))

    # (engine,tag,cond,level) -> content/spk CER 列表
    agg = defaultdict(lambda: {"content": [], "spk": [], "n": 0})
    for (engine, tag, cond, level, pair), texts in grp.items():
        refA, refB = pair_refs(pair)
        if refA is None or refB is None:
            continue
        a = agg[(engine, tag, cond, level)]
        a["n"] += 1
        if "_" in texts:  # 单路
            hyp = texts["_"]
            a["content"].append(min(char_cer(refA + refB, hyp),
                                    char_cer(refB + refA, hyp)))
        else:             # 双路 spk1/spk2, permutation-invariant
            h1, h2 = texts.get("1", ""), texts.get("2", "")
            if char_cer(refA, h1) + char_cer(refB, h2) <= \
               char_cer(refA, h2) + char_cer(refB, h1):
                hypA, hypB = h1, h2
            else:
                hypA, hypB = h2, h1
            a["spk"].append((char_cer(refA, hypA) + char_cer(refB, hypB)) / 2)
            a["content"].append(char_cer(refA + refB, hypA + hypB))

    mean = lambda xs: sum(xs) / len(xs) if xs else None

    # summary.csv
    header = ["engine", "link", "cond", "level", "n_pairs",
              "content_cer", "spk_cer", "asr_rtf"]
    out = []
    for (engine, tag, cond, level), a in sorted(agg.items()):
        mc, ms = mean(a["content"]), mean(a["spk"])
        mr = mean(rtf[(engine, tag, cond, level)])
        out.append([engine, tag, cond, level, a["n"],
                    f"{mc:.4f}" if mc is not None else "",
                    f"{ms:.4f}" if ms is not None else "",
                    f"{mr:.3f}" if mr is not None else ""])
    with open(SUM_CSV, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(out)

    # pivot.md: (cond,level) x L1..L5 的 content CER (%)
    def cc(e, t, c, l):
        return mean(agg.get((e, t, c, l), {"content": []})["content"])

    engines = sorted({k[0] for k in agg})
    with open(PIVOT_MD, "w", encoding="utf-8") as fp:
        fp.write("# 实验 2 处理顺序对比 (content CER %)\n\n")
        fp.write("链路: " + " / ".join(f"{k}={v}" for k, v in LINK_NAME.items()) + "\n\n")
        for e in engines:
            fp.write(f"## {e}\n\n| cond | level | " + " | ".join(LINKS) + " |\n")
            fp.write("|" + "---|" * (len(LINKS) + 2) + "\n")
            for c in sorted({k[2] for k in agg if k[0] == e}):
                for l in LEVELS:
                    vals = [cc(e, t, c, l) for t in LINKS]
                    cells = [f"{v * 100:.1f}" if v is not None else "-" for v in vals]
                    fp.write(f"| {c} | {l} | " + " | ".join(cells) + " |\n")
            fp.write("\n")
    print(f"wrote {SUM_CSV} and {PIVOT_MD}")
    for o in out:
        print("  ", o)
