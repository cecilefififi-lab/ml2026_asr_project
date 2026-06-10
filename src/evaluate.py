"""根据 results/asr_raw.csv 和 refs/ 下的参考文本计算 CER, 汇总成表。

参考文本: refs/{音频文件名去扩展名}.txt (UTF-8, 一行原文)
中文 CER: 去标点和空白后按字符算编辑距离 (jiwer.cer)。

用法:
    python src/evaluate.py            # 输出汇总表到 results/summary.csv + summary.md
"""
import csv
import os
import re
import unicodedata
from collections import defaultdict

import jiwer

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV = os.path.join(BASE, "results", "asr_raw.csv")
REF_DIR = os.path.join(BASE, "refs")
SUM_CSV = os.path.join(BASE, "results", "summary.csv")
SUM_MD = os.path.join(BASE, "results", "summary.md")


def normalize(text):
    """去标点/空白, 全角转半角, 小写。中文按字符级评测。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w]", "", text)  # 去掉所有标点和空白
    return text.lower()


def char_cer(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return None
    return jiwer.cer(" ".join(ref), " ".join(hyp))  # 按字符切分


if __name__ == "__main__":
    rows = list(csv.DictReader(open(RAW_CSV, encoding="utf-8-sig")))
    # (tag, engine) -> 累计
    agg = defaultdict(lambda: {"cer": [], "rtf": [], "n": 0, "no_ref": 0})
    for r in rows:
        stem = os.path.splitext(r["file"])[0]
        ref_path = os.path.join(REF_DIR, stem + ".txt")
        key = (r["tag"], r["engine"])
        agg[key]["n"] += 1
        agg[key]["rtf"].append(float(r["rtf"]))
        if os.path.exists(ref_path):
            ref = open(ref_path, encoding="utf-8").read().strip()
            cer = char_cer(ref, r["text"])
            if cer is not None:
                agg[key]["cer"].append(cer)
        else:
            agg[key]["no_ref"] += 1

    out_rows = []
    for (tag, engine), a in sorted(agg.items()):
        mean_cer = sum(a["cer"]) / len(a["cer"]) if a["cer"] else float("nan")
        mean_rtf = sum(a["rtf"]) / len(a["rtf"])
        out_rows.append([tag, engine, a["n"], len(a["cer"]),
                         f"{mean_cer:.4f}", f"{mean_rtf:.3f}"])
        if a["no_ref"]:
            print(f"warning: {tag}/{engine} 有 {a['no_ref']} 条缺参考文本")

    header = ["tag", "engine", "n_files", "n_scored", "mean_cer", "mean_rtf"]
    with open(SUM_CSV, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(out_rows)
    with open(SUM_MD, "w", encoding="utf-8") as fp:
        fp.write("| " + " | ".join(header) + " |\n")
        fp.write("|" + "---|" * len(header) + "\n")
        for r in out_rows:
            fp.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"wrote {SUM_CSV} and {SUM_MD}")
    for r in out_rows:
        print("  ", r)
