"""Page 7 -- Experiment 5: does preprocessing erase emotion. Standalone:
streamlit run demo/pages/7_exp5_emotion_drift.py
Speaking script: demo/pages/7_exp5_emotion_drift.md.
Numbers from REPORT.md sec. 6.5, results/exp5_emotion.md, results/exp5_emotion_clips.csv.
Audio: data/clean vs data/denoised/specsub/clean.
"""
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exp 5 - emotion drift", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")
DATA = os.path.join(BASE, "data")


def img(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(p, width='stretch')
    else:
        st.warning(f"missing figure: {name}")


def play(path, label):
    st.caption(label)
    if os.path.exists(path):
        st.audio(path)
    else:
        st.caption("(audio not found)")


st.title("Experiment 5 - Does preprocessing erase emotion?")
st.markdown("##### A paralinguistic cost of cleaning audio that word accuracy never shows")

st.subheader("Data and method")
st.markdown(
    "- **Model:** emotion2vec_plus_large, run on the same clips before and after processing. It reuses "
    "the FunASR / ModelScope stack, so this adds no new dependencies and reuses audio already on disk.\n"
    "- **Data:** the 26 clean debate clips; 17 of them are clearly angry. The metric is the predicted "
    "probability of *angry*, averaged over those 17 clips, measured before vs after each front-end.\n"
    "- **Question:** does a front-end built for word accuracy quietly flatten how something was said?"
)

st.divider()

st.subheader("Results")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        "**A. Denoising (clean -> denoised, no noise)**\n\n"
        "| Front-end | mean P(angry) | angry -> non-angry flips |\n|---|---|---|\n"
        "| clean (baseline) | 0.893 | - |\n"
        "| FRCRN (neural) | 0.848 | 1 / 17 |\n"
        "| spectral subtraction | 0.652 | 4 / 17 |\n\n"
        "**B. Separation (clean overlap)**\n\n"
        "| Pipeline | no | light | heavy |\n|---|---|---|---|\n"
        "| source speakers | 0.628 | 0.628 | 0.628 |\n"
        "| L3 separate | 0.369 | 0.468 | 0.349 |\n"
        "| L4 denoise+separate | 0.148 | 0.096 | 0.101 |\n"
    )
with c2:
    img("exp5_emotion_drift.png")

m1, m2, m3 = st.columns(3)
m1.metric("Clean P(angry)", "0.89")
m2.metric("After spectral subtraction", "0.65", "-0.24", delta_color="inverse")
m3.metric("After separation", "~0.4", "emotion mostly gone", delta_color="inverse")

st.caption("Per-clip emotion predictions actually produced (results/exp5_emotion_clips.csv, angry clips):")
df = pd.read_csv(os.path.join(RESULTS, "exp5_emotion_clips.csv"), encoding="utf-8-sig")
df = df[df["ref_emo"] == "angry"][
    ["clip", "ref_angry", "frcrn_emo", "frcrn_angry", "specsub_emo", "specsub_angry", "specsub_flip"]
]
st.dataframe(df, width='stretch', hide_index=True, height=240)

st.divider()

st.subheader("Audio evidence -- the same clip, before and after denoising")
st.markdown("Both clips are clearly **angry** before processing. Spectral subtraction keeps the words "
            "but the emotion classifier flips the label.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**pro_014: angry -> neutral** (P(angry) 0.88 -> 0.02)")
    play(os.path.join(DATA, "clean", "pro_014.wav"), "Before (clean): angry")
    play(os.path.join(DATA, "denoised", "specsub", "clean", "pro_014.wav"),
         "After spectral subtraction: neutral")
with c2:
    st.markdown("**con_005: angry -> happy** (P(angry) 0.92 -> 0.00)")
    play(os.path.join(DATA, "clean", "con_005.wav"), "Before (clean): angry")
    play(os.path.join(DATA, "denoised", "specsub", "clean", "con_005.wav"),
         "After spectral subtraction: happy")

st.divider()

st.subheader("Findings")
st.markdown(
    "1. **Preprocessing systematically flattens emotion.** Clips that were clearly angry drift toward "
    "neutral after processing.\n"
    "2. **Separation erases more than denoising, and spectral subtraction more than the neural model.** "
    "Separation drops mean P(angry) to about 0.4 or below.\n"
    "3. **New observation for this project.** We found no prior work measuring this specific effect; we "
    "frame it as extending the artifact finding from word accuracy to the paralinguistic channel."
)
st.info(
    "The transcript can keep every word and the same content meaning, yet the system loses *how* it was "
    "said. For a debate record, that is the difference between logging an argument and logging a "
    "shouting match."
)
