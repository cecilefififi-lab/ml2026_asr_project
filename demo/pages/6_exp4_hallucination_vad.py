"""Page 6 -- Experiment 4: Whisper hallucination + VAD boundary. Standalone:
streamlit run demo/pages/6_exp4_hallucination_vad.py
Speaking script: demo/pages/6_exp4_hallucination_vad.md.
Numbers/text from REPORT.md sec. 6.4 and results/exp4_hallucination.csv (.md).
"""
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exp 4 - hallucination & VAD", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data")


def img(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(p, width='stretch')
    else:
        st.warning(f"missing figure: {name}")


st.title("Experiment 4 - Whisper hallucination and the limits of VAD")
st.markdown("##### What an ASR model says when nobody is speaking")

st.subheader("Data and method")
st.markdown(
    "- **Inputs with no speech:** 11 clips in `data/exp4/` (fixed seed, reproducible) -- white and "
    "babble noise (3, 10, 30 s) plus silence (pure-zero samples and near-silent room tone at -60 / "
    "-50 / -45 dBFS).\n"
    "- **Model:** faster-whisper large-v3, exactly the same settings as Experiment 1.\n"
    "- **Comparison:** the same clips are transcribed with Silero VAD off and with it on."
)

st.divider()

st.subheader("Results -- non-speech input produces text (VAD off)")
df = pd.read_csv(os.path.join(RESULTS, "exp4_hallucination.csv"), encoding="utf-8-sig")
off = df[df["tag"] == "exp4_vadoff"][["file", "text", "audio_s"]].copy()
off["text"] = off["text"].fillna("(empty)")
st.dataframe(off, width='stretch', hide_index=True)
st.caption("10 of 11 non-speech clips produced hallucinated text (91%). Only babble 3 s stayed empty.")

st.markdown("**Hallucinations are fixed phrases, not random** -- all trace to subtitle/credit text "
            "in the training data:")
st.markdown(
    "- 由 Amara.org 社群提供的字幕  (subtitle-platform credit)\n"
    "- 请不吝点赞 订阅 转发 打赏支持明镜与点点栏目  (livestream donation caption)\n"
    "- 字幕志愿者 杨茜茜  (subtitle-team credit)"
)

st.divider()

st.subheader("Does VAD stop it?")
c1, c2 = st.columns([3, 2])
with c1:
    img("exp4_vad_compare.png")
with c2:
    st.markdown(
        "| Input type | VAD off | VAD on |\n|---|---|---|\n"
        "| white (3) | 100% | **0%** |\n"
        "| silence (5) | 100% | **0%** |\n"
        "| babble (3) | 67% | **67%** |\n"
    )
    st.markdown(
        "VAD removes hallucination completely for white noise and silence, but does nothing for "
        "babble: babble sounds enough like speech that the VAD lets it through."
    )

st.markdown("**Listen** -- non-speech input, and what Whisper produced from it:")
demos = [
    ("noise/white_20s_10s.wav", "10 s of white noise", "由 Amara.org 社群提供的字幕"),
    ("noise/babble_15s_10s.wav", "10 s of babble noise", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"),
]
cols = st.columns(len(demos))
for col, (rel, label, out) in zip(cols, demos):
    with col:
        st.caption(label)
        p = os.path.join(DATA, "exp4", rel)
        if os.path.exists(p):
            st.audio(p)
        else:
            st.caption("(audio not found in data/exp4/)")
        st.markdown(f"<span style='font-size:13px'>Whisper output: <b>{out}</b></span>",
                    unsafe_allow_html=True)

st.divider()

st.subheader("Findings")
st.markdown(
    "1. **Hallucination is a fixed pattern, not noise.** Whisper falls back on subtitle/credit text "
    "memorised from its training data, even on pure-zero silence -- closely matching arXiv:2501.11378.\n"
    "2. **VAD has a hard boundary.** It silences white noise and silence completely (100% -> 0%), but "
    "fails on babble (67% -> 67%). The 67%-to-67% number is our own direct evidence.\n"
    "3. **This is why babble at 0 dB was the worst case in Experiment 1**: crowd babble is exactly the "
    "speech-like noise that slips past the VAD, and a debate audience is made of babble."
)
st.info(
    "Prior work: Whisper's non-speech hallucinations are documented and traceable to training-data "
    "subtitle artifacts (arXiv:2501.11378); VAD preprocessing is reported to reduce hallucination "
    "(WhisperX). Our contribution is quantifying where VAD stops working."
)
