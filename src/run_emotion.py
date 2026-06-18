"""E5b: 预处理（降噪 / 分离）是否抹掉语音情绪。

核心指标：emotion2vec 的 "angry" 概率均值（辩论片段以 angry 为主），
比较处理前后是否被削平；免 spk 配对。

支路：
  A 降噪伪影：clean vs denoised/{frcrn,specsub}/clean   （无噪声混淆）
  A' 真实管线：clean(ref) vs denoised/{frcrn,specsub}/{noise_snr}（按 SNR 聚合）
  B 分离：源 con/pro vs exp2/{L3_sep,L4_sep}/clean/{no,light,heavy} 分离两路

产出：
  results/exp5_emotion_clips.csv  逐 clean 片段（ref/frcrn/specsub）
  results/exp5_emotion_sep.csv    逐分离轨
  results/exp5_emotion.md         汇总
  results/exp5_emotion_drift.png  两面板图
"""
import os, sys, glob, csv

sys.stdout.reconfigure(encoding="utf-8")  # 避免 Windows GBK 乱码
os.environ.setdefault("MODELSCOPE_LOG_LEVEL", "40")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from funasr import AutoModel

EMO_MODEL = "iic/emotion2vec_plus_large"
CLEAN = "data/clean"
DEN = "data/denoised"
EXP2 = "data/exp2"
NOISE_CONDS = ["babble_15dB", "babble_5dB", "babble_0dB",
               "white_15dB", "white_5dB", "white_0dB"]
METHODS = ["frcrn", "specsub"]
SEP_LEVELS = ["no", "light", "heavy"]
SEP_LINKS = ["L3_sep", "L4_sep"]
os.makedirs("results", exist_ok=True)


def simplify(label):
    return label.split("/")[-1].strip().lower()  # "生气/angry" -> "angry"


class Emo:
    def __init__(self):
        self.m = AutoModel(model=EMO_MODEL, disable_update=True, disable_pbar=True)

    def predict(self, path):
        r = self.m.generate(path, granularity="utterance",
                            extract_embedding=False, disable_pbar=True)[0]
        labels = [simplify(x) for x in r["labels"]]
        scores = [float(s) for s in r["scores"]]
        i = max(range(len(scores)), key=lambda k: scores[k])
        d = dict(zip(labels, scores))
        return labels[i], float(scores[i]), d.get("angry", 0.0)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    emo = Emo()
    clips = sorted(os.path.basename(p) for p in glob.glob(f"{CLEAN}/*.wav"))

    # ---- ref：clean 情绪 ----
    ref = {}  # clip -> (top, conf, angry_prob)
    for c in clips:
        ref[c] = emo.predict(f"{CLEAN}/{c}")
    angry_clips = [c for c in clips if ref[c][0] == "angry"]
    print(f"clean: {len(clips)} 片段，其中 angry={len(angry_clips)}", flush=True)

    # ---- A 降噪伪影：clean vs frcrn(clean) vs specsub(clean) ----
    rows = []
    for c in clips:
        row = {"clip": c, "ref_emo": ref[c][0], "ref_conf": round(ref[c][1], 3),
               "ref_angry": round(ref[c][2], 3)}
        for mth in METHODS:
            p = f"{DEN}/{mth}/clean/{c}"
            top, conf, ang = emo.predict(p)
            row[f"{mth}_emo"] = top
            row[f"{mth}_conf"] = round(conf, 3)
            row[f"{mth}_angry"] = round(ang, 3)
            row[f"{mth}_flip"] = int(ref[c][0] == "angry" and top != "angry")
        rows.append(row)
    with open("results/exp5_emotion_clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    den_angry_mean = {"clean": mean([ref[c][2] for c in angry_clips])}
    den_flip = {}
    for mth in METHODS:
        den_angry_mean[mth] = mean([r[f"{mth}_angry"] for r in rows if r["clip"] in angry_clips])
        den_flip[mth] = sum(r[f"{mth}_flip"] for r in rows)

    # ---- A' 真实管线：按 SNR 聚合（仅 angry 片段的 angry 概率均值）----
    noisy_angry = {}  # (mth, cond) -> mean angry prob over angry_clips
    for mth in METHODS:
        for cond in NOISE_CONDS:
            vals = []
            for c in angry_clips:
                p = f"{DEN}/{mth}/{cond}/{c}"
                if os.path.exists(p):
                    vals.append(emo.predict(p)[2])
            noisy_angry[(mth, cond)] = mean(vals)

    # ---- B 分离：源 vs 分离两路（clean 噪声条件，免配对，取 angry 概率均值）----
    sep_rows = []
    pairs = sorted({os.path.basename(p)[:-4]
                    for p in glob.glob(f"{EXP2}/mix/clean/no/*.wav")})  # con_i_pro_i
    src_speakers = sorted({s for pr in pairs for s in
                           (pr.split("_pro_")[0], "pro_" + pr.split("_pro_")[1])})
    # 源说话人 angry 概率（这些 con/pro 都在 clean 目录里）
    src_angry_mean = mean([ref[f"{s}.wav"][2] for s in src_speakers
                           if f"{s}.wav" in ref])
    sep_angry_mean = {}  # (link, level) -> mean angry prob over both separated tracks
    for link in SEP_LINKS:
        for lvl in SEP_LEVELS:
            vals = []
            for pr in pairs:
                for spk in ("spk1", "spk2"):
                    p = f"{EXP2}/{link}/clean/{lvl}/{pr}_{spk}.wav"
                    if os.path.exists(p):
                        top, conf, ang = emo.predict(p)
                        vals.append(ang)
                        sep_rows.append({"pair": pr, "link": link, "level": lvl,
                                         "track": spk, "emo": top,
                                         "conf": round(conf, 3), "angry": round(ang, 3)})
            sep_angry_mean[(link, lvl)] = mean(vals)
    with open("results/exp5_emotion_sep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "link", "level", "track",
                                          "emo", "conf", "angry"])
        w.writeheader()
        w.writerows(sep_rows)

    # ---- 图：两面板 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel A 降噪
    a_labels = ["clean", "FRCRN", "SpecSub"]
    a_vals = [den_angry_mean["clean"], den_angry_mean["frcrn"], den_angry_mean["specsub"]]
    ax1.bar(a_labels, a_vals, color=["#4C72B0", "#DD8452", "#C44E52"])
    ax1.set_title(f"Denoise on clean: mean P(angry)\nover {len(angry_clips)} angry clips")
    ax1.set_ylabel("mean P(angry)")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(a_vals):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center")
    # Panel B 分离 (L3)
    b_labels = ["source"] + [f"sep-{lvl}" for lvl in SEP_LEVELS]
    b_vals = [src_angry_mean] + [sep_angry_mean[("L3_sep", lvl)] for lvl in SEP_LEVELS]
    ax2.bar(b_labels, b_vals, color=["#4C72B0", "#55A868", "#8172B3", "#C44E52"])
    ax2.set_title("Separation (L3, clean): mean P(angry)\nsource vs separated tracks")
    ax2.set_ylabel("mean P(angry)")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(b_vals):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig("results/exp5_emotion_drift.png", dpi=130)

    # ---- 汇总 md ----
    with open("results/exp5_emotion.md", "w", encoding="utf-8") as f:
        f.write("# 实验 5b：预处理是否抹掉语音情绪\n\n")
        f.write(f"模型 `{EMO_MODEL}`；指标 = emotion2vec P(angry) 均值"
                f"（辩论片段以 angry 为主）。\n\n")
        f.write(f"clean 片段 {len(clips)} 条，其中 angry={len(angry_clips)}。\n\n")

        f.write("## A 降噪伪影（clean → 降噪，无噪声混淆）\n\n")
        f.write("| 处理 | angry 片段 P(angry) 均值 | angry→非angry 翻转数 |\n")
        f.write("|---|---|---|\n")
        f.write(f"| clean（基线） | {den_angry_mean['clean']:.3f} | - |\n")
        for mth in METHODS:
            f.write(f"| {mth} | {den_angry_mean[mth]:.3f} | "
                    f"{den_flip[mth]}/{len(angry_clips)} |\n")
        f.write("\n")

        f.write("## A' 真实管线：噪声 + 降噪后 P(angry)（angry 片段均值）\n\n")
        f.write("| 方法 | " + " | ".join(NOISE_CONDS) + " |\n")
        f.write("|---|" + "---|" * len(NOISE_CONDS) + "\n")
        for mth in METHODS:
            f.write(f"| {mth} | " +
                    " | ".join(f"{noisy_angry[(mth, c)]:.3f}" for c in NOISE_CONDS) + " |\n")
        f.write("\n")

        f.write("## B 分离（clean 噪声条件）：源 vs 分离两路 P(angry) 均值\n\n")
        f.write(f"源说话人均值：{src_angry_mean:.3f}\n\n")
        f.write("| 链路 | " + " | ".join(SEP_LEVELS) + " |\n")
        f.write("|---|" + "---|" * len(SEP_LEVELS) + "\n")
        for link in SEP_LINKS:
            f.write(f"| {link} | " +
                    " | ".join(f"{sep_angry_mean[(link, l)]:.3f}" for l in SEP_LEVELS) + " |\n")
        f.write("\n图：`results/exp5_emotion_drift.png`\n")

    print("done. ->", "results/exp5_emotion.md / *.csv / *.png", flush=True)
    print(f"  A 降噪: clean {den_angry_mean['clean']:.3f} | "
          f"frcrn {den_angry_mean['frcrn']:.3f}(flip {den_flip['frcrn']}) | "
          f"specsub {den_angry_mean['specsub']:.3f}(flip {den_flip['specsub']})", flush=True)
    print(f"  B 分离 source {src_angry_mean:.3f} | "
          f"L3 no/light/heavy "
          f"{sep_angry_mean[('L3_sep','no')]:.3f}/"
          f"{sep_angry_mean[('L3_sep','light')]:.3f}/"
          f"{sep_angry_mean[('L3_sep','heavy')]:.3f}", flush=True)


if __name__ == "__main__":
    main()
