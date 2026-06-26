"""Page 9 -- Real-recording generalization spot-check. Standalone:
streamlit run demo/pages/9_real_recordings.py
Speaking script: demo/pages/9_real_recordings.md.
Reuses make_demo.CASES (single source) + results/real_asr.csv; audio in demo/audio/.
"""
import os
import sys
import csv
import streamlit as st

st.set_page_config(page_title="Real recordings", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE, "src")
AUDIO = os.path.join(BASE, "demo", "audio")
CSV_PATH = os.path.join(BASE, "results", "real_asr.csv")
sys.path.insert(0, SRC)
import make_demo  # noqa: E402  CASES is a top-level data structure, no side effects


@st.cache_data
def load_rows():
    rows = {}
    with open(CSV_PATH, encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            rows[(r["tag"], r["engine"], r["file"])] = r
    return rows


def audio_label(key):
    if "raw" in key:
        return "Raw (mixed / noisy)"
    if "spk1" in key:
        return "Separated spk1"
    if "spk2" in key:
        return "Separated spk2"
    if "specsub" in key:
        return "After spectral-subtraction denoising"
    return key


st.title("Real-recording generalization spot-check")
st.markdown("##### Do the synthetic-data conclusions hold on real phone recordings?")

st.subheader("Data and method")
st.markdown(
    "- **Recordings:** 6 real phone recordings -- dorm (quiet), canteen (strong noise), classroom "
    "(light noise), and three two-person discussions (quiet / natural / heavy crosstalk).\n"
    "- **Runs:** direct ASR vs a rule-based 'best pipeline per condition' (denoise for single-speaker "
    "noisy clips, separation for overlapping clips). Audio was converted to 16 kHz mono with PyAV "
    "(no ffmpeg on the machine).\n"
    "- **Evaluation:** manual spot-check, not verbatim CER -- these recordings have no ground-truth "
    "transcript, so we compare before/after text qualitatively.\n"
    "- Below are 5 fixed cases. Each plays the actual audio and shows the real ASR output for each stage."
)

st.divider()

lut = load_rows()
for case in make_demo.CASES:
    st.subheader(f"Case {case['id']} - {case['title']}")
    st.markdown(f"**Recording:** {case['type']}")

    if case["audio"]:
        cols = st.columns(len(case["audio"]))
        for col, key in zip(cols, case["audio"]):
            with col:
                st.caption(audio_label(key))
                p = os.path.join(AUDIO, key)
                if os.path.exists(p):
                    st.audio(p)
                else:
                    st.warning(f"missing audio: {key}")

    table = []
    for label, tag, eng, fname in case["rows"]:
        r = lut.get((tag, eng, fname))
        table.append({
            "Stage": label,
            "ASR output": r["text"] if r else "(missing)",
            "Audio (s)": r["audio_s"] if r else "",
            "RTF": r["rtf"] if r else "",
        })
    st.table(table)
    st.info(case["point"])
    st.divider()

st.subheader("Finding")
st.markdown(
    "The real recordings independently reproduce every main conclusion from the synthetic data: "
    "direct ASR is the most stable option; under strong canteen noise, spectral subtraction triggers "
    "the same subtitle hallucination (請不吝点赞 订阅); the denoise-then-separate pipeline triggers "
    "中文字幕志愿者; separation only recovers a lost sentence under heavy synchronous crosstalk; VAD is "
    "double-edged; and English proper nouns are consistently mis-transcribed, which motivates the "
    "hot-word idea tested in Experiment 3. **Generalization holds.**"
)
