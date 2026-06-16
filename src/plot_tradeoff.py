"""精度-耗时 trade-off:实验 2 五链路 content CER 均值 vs 链路总 RTF。

链路总 RTF 由各环节实测 RTF 组装(非手填),口径见下:
  RTF = proc_s / audio_s(处理单位时长音频的耗时)。
  各环节均值取自原始 csv:
    ASR      : exp2_summary.csv 的 asr_rtf(每链路 15 格均值,单路口径)
    降噪     : denoise_raw.csv  specsub,tag=exp2_L2(对原始混合)/ exp2_L5(对分离双路)
    分离     : separate_raw.csv tag=exp2_L3(分离原混合)/ exp2_L4(分离降噪后)
  双路上执行的环节按 2 路计(分离链路 ASR 跑 spk1+spk2;L5 降噪跑双路)。
  L2/L4 第一步都是"对原始混合降噪",共用 exp2_L2;L5 第一步分离原混合,共用 exp2_L3。

链路组装:
  L1 = asr(L1)
  L2 = dns_L2 + asr(L2)
  L3 = sep_L3 + 2·asr(L3)
  L4 = dns_L2 + sep_L4 + 2·asr(L4)
  L5 = sep_L3 + 2·dns_L5 + 2·asr(L5)

产出:
  results/tradeoff_summary.md  精度-耗时简表(中文)
  results/tradeoff.png         散点图(英文标签;左下=又快又准=理想)
"""
import csv
import os
import statistics as st

import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(BASE, "results")
SUMMARY = os.path.join(RES, "exp2_summary.csv")
DENOISE = os.path.join(RES, "denoise_raw.csv")
SEPARATE = os.path.join(RES, "separate_raw.csv")
OUT_MD = os.path.join(RES, "tradeoff_summary.md")
OUT_PNG = os.path.join(RES, "tradeoff.png")

LINKS = ["L1", "L2", "L3", "L4", "L5"]
DESC = {
    "L1": "直接 ASR(baseline)",
    "L2": "降噪 → ASR",
    "L3": "分离 → ASR",
    "L4": "降噪 → 分离 → ASR",
    "L5": "分离 → 降噪 → ASR",
}
PLOT_LABEL = {"L1": "L1 direct", "L2": "L2 +denoise", "L3": "L3 +separate",
              "L4": "L4 denoise>sep", "L5": "L5 sep>denoise"}


def col_mean(path, value_col, where):
    """对 csv 取满足 where(row) 的行的 value_col 均值。"""
    rows = csv.DictReader(open(path, encoding="utf-8-sig"))
    vals = [float(r[value_col]) for r in rows if where(r)]
    return st.mean(vals)


def main():
    # ---- content CER 均值(15 格)+ 每链路 ASR RTF 均值 ----
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8-sig")))
    cer = {L: 100 * st.mean(float(r["content_cer"]) for r in rows if r["link"] == L)
           for L in LINKS}
    asr = {L: st.mean(float(r["asr_rtf"]) for r in rows if r["link"] == L)
           for L in LINKS}

    # ---- 环节 RTF 均值 ----
    dns_L2 = col_mean(DENOISE, "rtf", lambda r: r["method"] == "specsub" and r["tag"] == "exp2_L2")
    dns_L5 = col_mean(DENOISE, "rtf", lambda r: r["method"] == "specsub" and r["tag"] == "exp2_L5")
    sep_L3 = col_mean(SEPARATE, "rtf", lambda r: r["tag"] == "exp2_L3")
    sep_L4 = col_mean(SEPARATE, "rtf", lambda r: r["tag"] == "exp2_L4")

    rtf = {
        "L1": asr["L1"],
        "L2": dns_L2 + asr["L2"],
        "L3": sep_L3 + 2 * asr["L3"],
        "L4": dns_L2 + sep_L4 + 2 * asr["L4"],
        "L5": sep_L3 + 2 * dns_L5 + 2 * asr["L5"],
    }
    # 各链路相对 L1 的耗时倍数
    mult = {L: rtf[L] / rtf["L1"] for L in LINKS}

    # ---- 写 markdown 简表 ----
    lines = [
        "# 精度-耗时 trade-off 简表(实验 2 五链路)",
        "",
        "> 同一数据域(自制重叠双人 11 对 × 噪声 × 重叠档,15 个 cond×level 格)。",
        "> content CER = 15 格均值(越低越好);链路 RTF = 各环节实测 RTF 组装(越低越快)。",
        "> 数据取自 `exp2_summary.csv` / `denoise_raw.csv` / `separate_raw.csv` 原始记录,非手填。",
        "",
        "| 链路 | 组成 | content CER(%) | 链路 RTF | 相对 L1 耗时 | trade-off |",
        "|---|---|---|---|---|---|",
    ]
    verdict = {
        "L1": "**又快又准,基准**",
        "L2": "更慢且更差",
        "L3": "最慢之一且最差",
        "L4": "最慢且无改善",
        "L5": "最慢且无稳定改善",
    }
    for L in LINKS:
        lines.append(f"| {L} | {DESC[L]} | {cer[L]:.1f} | {rtf[L]:.3f} | "
                     f"{mult[L]:.1f}× | {verdict[L]} |")
    lines += [
        "",
        "## 环节 RTF(组装用)",
        "",
        f"- ASR(whisper,单路均值):L1 {asr['L1']:.3f} / L2 {asr['L2']:.3f} / "
        f"L3 {asr['L3']:.3f} / L4 {asr['L4']:.3f} / L5 {asr['L5']:.3f}",
        f"- 降噪(specsub):对原混合 {dns_L2:.3f} / 对分离双路单路 {dns_L5:.3f}",
        f"- 分离(MossFormer2):原混合 {sep_L3:.3f} / 降噪后 {sep_L4:.3f}",
        "- 双路环节按 2 路计(分离链路 ASR 跑 spk1+spk2;L5 降噪跑双路)。",
        "",
        "## 结论",
        "",
        f"最简单的 **L1 直接 ASR 同时是最快(RTF {rtf['L1']:.3f})和最准"
        f"(CER {cer['L1']:.1f}%)** 的链路;每叠一层预处理,耗时升到 "
        f"{min(mult[L] for L in LINKS[1:]):.1f}–{max(mult.values()):.1f}× 而 CER 不降反升。"
        "精度-耗时平面上没有任何链路落在 L1 的左下方(更快更准),"
        "即**没有一条复杂链路提供更优 trade-off** —— 正面印证项目总问题:"
        "短辩论片段 + 噪声/重叠下,高级预处理花了更多算力却更不划算。",
        "",
        "> 引擎层参考:实验 1 中 FunASR RTF 约为 Whisper 的一半"
        "(whisper ~0.26–0.31 / funasr ~0.08–0.15),但 CER 互有胜负(两模型弱点相反)。",
        "",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- 散点图 ----
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    off = {"L1": (10, -2), "L2": (8, 7), "L3": (6, 9), "L4": (6, -17), "L5": (8, 7)}
    for L in LINKS:
        is_base = L == "L1"
        ax.scatter(rtf[L], cer[L], s=160 if is_base else 110,
                   color="#2a9d8f" if is_base else "#e76f51",
                   edgecolor="#333", zorder=3)
        ax.annotate(PLOT_LABEL[L], (rtf[L], cer[L]),
                    textcoords="offset points", xytext=off[L], fontsize=9,
                    fontweight="bold" if is_base else "normal")
    ax.set_xlabel("Pipeline RTF  (lower = faster →)")
    ax.set_ylabel("Mean content CER over 15 cells (%)  ·  lower is better")
    ax.set_title("Accuracy–latency trade-off: no chain beats direct ASR\n"
                 "(overlapped 2-4s debate clips, Whisper)")
    ax.invert_xaxis()  # 让"快"在右,"准"在上,理想点落右上
    ax.set_xlabel("Pipeline RTF  (← slower | faster →)")
    ax.annotate("ideal\n(fast & accurate)", (rtf["L1"], cer["L1"]),
                textcoords="offset points", xytext=(14, -28), fontsize=8,
                color="#2a9d8f", ha="left")
    ax.grid(True, ls="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")

    print("CER%:", {L: round(cer[L], 1) for L in LINKS})
    print("RTF :", {L: round(rtf[L], 3) for L in LINKS})
    print("mult:", {L: round(mult[L], 1) for L in LINKS})
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
