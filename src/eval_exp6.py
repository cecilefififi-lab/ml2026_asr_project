"""实验 6 评测:长音频上的 content CER + per-speaker CER + 处理顺序透视表。

与 eval_pipelines 同口径(char_cer / permutation-invariant 配对), 区别仅在
ground truth 来自 data/exp6/manifest.csv 的拼接 refA/refB(每条长样本由多条
con/pro 片段拼成), 而非单片段 refs。

输入: results/exp6_asr.csv, data/exp6/manifest.csv
输出: results/exp6_summary.csv, results/exp6_pivot.md
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import char_cer  # 复用实验 1 CER 口径

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR_CSV = os.path.join(BASE, "results", "exp6_asr.csv")
MANIFEST = os.path.join(BASE, "data", "exp6", "manifest.csv")
SUM_CSV = os.path.join(BASE, "results", "exp6_summary.csv")
PIVOT_MD = os.path.join(BASE, "results", "exp6_pivot.md")
LINKS = ["L1", "L3", "L4", "L5"]
LINK_NAME = {"L1": "直接", "L3": "分离→ASR", "L4": "降噪→分离", "L5": "分离→降噪"}
LEVELS = ["light", "heavy"]


def parse(file):
    """'cond/level/s0[_spkN].wav' -> (cond, level, sample, spk)"""
    cond, level, name = file.split("/")
    name = os.path.splitext(name)[0]
    spk = None
    if "_spk" in name:
        name, spk = name.rsplit("_spk", 1)
    return cond, level, name, spk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-csv", default=ASR_CSV)
    args = ap.parse_args()

    # sample_id -> (refA, refB)(各 level 拼接片段相同, 取任一行即可)
    refs = {}
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig")):
        refs[r["sample_id"]] = (r["refA"], r["refB"])

    rows = list(csv.DictReader(open(args.asr_csv, encoding="utf-8-sig")))
    grp = defaultdict(dict)
    rtf = defaultdict(list)
    for r in rows:
        cond, level, sample, spk = parse(r["file"])
        grp[(r["engine"], r["tag"], cond, level, sample)][spk or "_"] = r["text"]
        rtf[(r["engine"], r["tag"], cond, level)].append(float(r["rtf"]))

    agg = defaultdict(lambda: {"content": [], "spk": [], "n": 0})
    for (engine, tag, cond, level, sample), texts in grp.items():
        refA, refB = refs[sample]
        a = agg[(engine, tag, cond, level)]
        a["n"] += 1
        if "_" in texts:  # 单路 L1
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

    header = ["engine", "link", "cond", "level", "n_samples",
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

    def cc(e, t, c, l):
        return mean(agg.get((e, t, c, l), {"content": []})["content"])

    def sc(e, t, c, l):
        return mean(agg.get((e, t, c, l), {"spk": []})["spk"])

    engines = sorted({k[0] for k in agg})
    with open(PIVOT_MD, "w", encoding="utf-8") as fp:
        fp.write("# 实验 6 长音频处理顺序对比 (content CER %)\n\n")
        fp.write("链路: " + " / ".join(f"{k}={v}" for k, v in LINK_NAME.items()) + "\n\n")
        for e in engines:
            fp.write(f"## {e} · content CER\n\n| cond | level | " + " | ".join(LINKS) + " |\n")
            fp.write("|" + "---|" * (len(LINKS) + 2) + "\n")
            for c in sorted({k[2] for k in agg if k[0] == e}):
                for l in LEVELS:
                    vals = [cc(e, t, c, l) for t in LINKS]
                    cells = [f"{v * 100:.1f}" if v is not None else "-" for v in vals]
                    fp.write(f"| {c} | {l} | " + " | ".join(cells) + " |\n")
            fp.write("\n")
            fp.write(f"## {e} · per-speaker CER (分离质量, 仅 L3/L4/L5)\n\n")
            fp.write("| cond | level | L3 | L4 | L5 |\n|---|---|---|---|---|\n")
            for c in sorted({k[2] for k in agg if k[0] == e}):
                for l in LEVELS:
                    vals = [sc(e, t, c, l) for t in ["L3", "L4", "L5"]]
                    cells = [f"{v * 100:.1f}" if v is not None else "-" for v in vals]
                    fp.write(f"| {c} | {l} | " + " | ".join(cells) + " |\n")
            fp.write("\n")
    print(f"wrote {SUM_CSV} and {PIVOT_MD}")
    for o in out:
        print("  ", o)
