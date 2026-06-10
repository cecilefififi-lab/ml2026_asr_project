"""把 asr_raw.csv 中 tag=clean, engine=whisper 的输出导出到 refs/draft/。

人工逐条听音频校对后, 把改好的文件移到 refs/ 根目录作为 ground truth。
注意: 草稿来自被测模型, 不校对直接用会使评测偏向 Whisper。
"""
import csv
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV = os.path.join(BASE, "results", "asr_raw.csv")
DRAFT_DIR = os.path.join(BASE, "refs", "draft")

if __name__ == "__main__":
    os.makedirs(DRAFT_DIR, exist_ok=True)
    n = 0
    for r in csv.DictReader(open(RAW_CSV, encoding="utf-8-sig")):
        if r["tag"] == "clean" and r["engine"] == "whisper":
            stem = os.path.splitext(r["file"])[0]
            with open(os.path.join(DRAFT_DIR, stem + ".txt"), "w", encoding="utf-8") as fp:
                fp.write(r["text"].strip() + "\n")
            n += 1
    print(f"wrote {n} drafts -> {DRAFT_DIR} (人工校对后移到 refs/)")
