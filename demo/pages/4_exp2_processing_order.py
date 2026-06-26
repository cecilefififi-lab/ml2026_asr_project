"""Page 4 -- Experiment 2: noise + overlap, processing order. Standalone:
streamlit run demo/pages/4_exp2_processing_order.py
Speaking script: demo/pages/4_exp2_processing_order.md.
Numbers from REPORT.md sec. 6.2, results/exp2_summary.csv; audio from demo/audio/.
"""
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exp 2 - processing order", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")
AUDIO = os.path.join(BASE, "demo", "audio")


def img(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(p, width='stretch')
    else:
        st.warning(f"missing figure: {name}")


def play(fname, label, text):
    st.caption(label)
    p = os.path.join(AUDIO, fname)
    if os.path.exists(p):
        st.audio(p)
    else:
        st.warning(f"missing audio: {fname}")
    st.markdown(f"<span style='font-size:13px'>{text}</span>", unsafe_allow_html=True)


st.title("Experiment 2 - Noise, overlap, and the order of processing")
st.markdown("##### Five pipelines: when does separation pay off, and does the order matter?")

st.subheader("Data and method")
st.markdown(
    "- **Overlap:** built in a controlled way, so every speaker has ground truth. We mix con_i with "
    "pro_i at equal loudness (0 dB SIR) with a time offset, at overlap ratios 0, 0.3, and 0.8. The "
    "senior thesis' real overlap samples are kept only as qualitative failure cases, because they "
    "have no per-speaker reference.\n"
    "- **Noise on top:** babble and white at 5 and 0 dB, plus a clean baseline.\n"
    "- **Five pipelines** compared on the same audio:"
)
st.graphviz_chart(
    """
    digraph {
        rankdir=LR; bgcolor="transparent";
        node [shape=box, style=rounded, fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
        edge [arrowsize=0.6];
        I [label="noisy +\\noverlapped"];
        I -> "ASR" -> "L1 direct";
        I -> "denoise " -> "ASR " -> "L2";
        I -> "separate" -> "ASR  " -> "L3";
        I -> "denoise  " -> "separate " -> "ASR   " -> "L4";
        I -> "separate  " -> "denoise   " -> "ASR    " -> "L5";
    }
    """,
    width='stretch',
)
st.markdown(
    "- **Metrics:** content CER (same across pipelines) and per-speaker CER (separation quality)."
)

st.divider()

st.subheader("Results")
img("exp2_pipelines.png")
st.markdown(
    "| Pipeline | content CER | per-speaker CER | best in (of 15 cells) |\n|---|---|---|---|\n"
    "| **L1 direct ASR** | **39.8** | - | **13** |\n"
    "| L2 denoise -> ASR | 52.5 | - | 0 |\n"
    "| L3 separate -> ASR | 68.2 | 88.1 | 2 (both heavy) |\n"
    "| L4 denoise -> separate | 67.4 | 84.9 | 0 |\n"
    "| L5 separate -> denoise | 65.3 | 85.2 | 0 |\n"
)
st.caption("CER in %. Lower is better. Separation pipelines cost about 2-3x the runtime of L1.")

c1, c2, c3 = st.columns(3)
c1.metric("L1 wins", "13 / 15 cells")
c2.metric("Separation per-speaker CER", "84-88%", "never actually separated", delta_color="off")
c3.metric("L4 vs L5", "no stable diff.", "on these short clips", delta_color="off")

st.caption("Per-condition data actually produced (results/exp2_summary.csv, Whisper):")
df = pd.read_csv(os.path.join(RESULTS, "exp2_summary.csv"), encoding="utf-8-sig")
df = df[df["engine"] == "whisper"][["link", "cond", "level", "content_cer", "spk_cer"]]
st.dataframe(df, width='stretch', hide_index=True, height=240)

st.divider()

st.subheader("Audio evidence")
st.markdown("**Case A -- heavy crosstalk: the one time separation pays off.** "
            "In the raw mix, one speaker's overlapping sentence is lost; separation recovers it on one channel.")
a1, a2, a3 = st.columns(3)
with a1:
    play("caseA_heavy_raw.wav", "Raw mix (direct ASR)",
         "...说话人归属在重点语一下错误力最高声纹特征会相互污染 (one speaker only)")
with a2:
    play("caseA_heavy_sep_spk1.wav", "Separated spk1", "咱们用威士忽尔传写前可以先做语言分析吗...")
with a3:
    play("caseA_heavy_sep_spk2.wav", "Separated spk2",
         "...但是对食堂这种人设置会相互污染 (the lost sentence, recovered)")

st.markdown("**Case C -- quiet overlap: separation just copies the mix (backfires).** "
            "spk2 is roughly the whole mixture again; spk1 is only fragments.")
b1, b2, b3 = st.columns(3)
with b1:
    play("caseC_quiet_raw.wav", "Raw mix (direct ASR)", "我觉得实验先捧Whisper比较稳...（near-complete）")
with b2:
    play("caseC_quiet_sep_spk1.wav", "Separated spk1", "我觉得这些东西有我突然间想要...（fragments）")
with b3:
    play("caseC_quiet_sep_spk2.wav", "Separated spk2", "我觉得实验先捧Whisper比较稳...（copy of the mix）")

st.divider()

st.subheader("Findings")
st.markdown(
    "1. **Over-separation.** L1, direct ASR with no front-end, is best in 13 of 15 noise/overlap "
    "cells. Separation pays off only under heavy overlap (clean/heavy 50.0 -> 46.9).\n"
    "2. **The separator never actually separated.** Per-speaker CER stays at 84-88%, which means "
    "MossFormer2 copied the mixture onto both channels on these 2-4 s clips.\n"
    "3. **No stable order effect here** (L4 vs L5). Because the separator was already failing, this "
    "is a degenerate case, not a general rule -- Experiment 6 retests it on longer audio."
)
st.info(
    "Prior work confirms over-separation: separation should be applied conditionally "
    "(arXiv:2106.00949), and a separation front-end injects artifacts that degrade a clean-backend "
    "recognizer (arXiv:2503.17886)."
)
