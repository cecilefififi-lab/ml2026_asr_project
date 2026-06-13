"""实验 1 案例挖掘: 按条目对比降噪前后 CER, 找"帮忙最大"和"帮倒忙最大"的案例。

读 results/asr_raw.csv + refs/, 对每个 (条件, 引擎, 方法, 文件) 计算
delta = CER(降噪后) - CER(无处理), 输出 delta 最负(帮忙)/最正(帮倒忙)的条目及转写文本。

用法:
    python src/find_cases.py [--top 5]
"""
import argparse
import csv
import os

from evaluate import char_cer

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV = os.path.join(BASE, "results", "asr_raw.csv")
REF_DIR = os.path.join(BASE, "refs")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    # (cond, engine, method, file) -> text
    texts = {}
    for r in csv.DictReader(open(RAW_CSV, encoding="utf-8-sig")):
        cond, _, method = r["tag"].partition("__")
        texts[(cond, r["engine"], method or "none", r["file"])] = r["text"]

    rows = []
    for (cond, engine, method, fname), hyp in texts.items():
        if method == "none":
            continue
        base_hyp = texts.get((cond, engine, "none", fname))
        ref_path = os.path.join(REF_DIR, os.path.splitext(fname)[0] + ".txt")
        if base_hyp is None or not os.path.exists(ref_path):
            continue
        ref = open(ref_path, encoding="utf-8").read().strip()
        cer_base, cer_dn = char_cer(ref, base_hyp), char_cer(ref, hyp)
        if cer_base is None:
            continue
        rows.append({"cond": cond, "engine": engine, "method": method,
                     "file": fname, "ref": ref, "base": base_hyp, "dn": hyp,
                     "cer_base": cer_base, "cer_dn": cer_dn,
                     "delta": cer_dn - cer_base})

    rows.sort(key=lambda x: x["delta"])
    print(f"=== 降噪帮忙最大 top{args.top} (delta = 降噪后CER - 无处理CER) ===")
    for x in rows[:args.top]:
        print(f"\n[{x['cond']}/{x['engine']}/{x['method']}] {x['file']} "
              f"CER {x['cer_base']:.2f} -> {x['cer_dn']:.2f} (delta {x['delta']:+.2f})")
        print(f"  ref : {x['ref']}")
        print(f"  raw : {x['base']}")
        print(f"  dn  : {x['dn']}")
    print(f"\n=== 降噪帮倒忙最大 top{args.top} ===")
    for x in rows[:-args.top - 1:-1]:
        print(f"\n[{x['cond']}/{x['engine']}/{x['method']}] {x['file']} "
              f"CER {x['cer_base']:.2f} -> {x['cer_dn']:.2f} (delta {x['delta']:+.2f})")
        print(f"  ref : {x['ref']}")
        print(f"  raw : {x['base']}")
        print(f"  dn  : {x['dn']}")
