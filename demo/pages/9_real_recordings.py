"""Page 9 - real-recording generalization check. Standalone:
streamlit run demo/pages/9_real_recordings.py
Speaking script: demo/pages/9_real_recordings.md.
Uses results/real_asr.csv; fixed audio clips live in demo/audio/.
"""
import os
import csv
import streamlit as st

st.set_page_config(page_title="Real recordings", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO = os.path.join(BASE, "demo", "audio")
CSV_PATH = os.path.join(BASE, "results", "real_asr.csv")


@st.cache_data
def load_rows():
    rows = {}
    with open(CSV_PATH, encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            rows[(r["tag"], r["engine"], r["file"])] = r
    return rows


def row(tag, fname, engine="whisper"):
    return load_rows().get((tag, engine, fname))


def audio_player(title, filename):
    st.markdown(f"**{title}**")
    path = os.path.join(AUDIO, filename)
    if os.path.exists(path):
        st.audio(path)
    else:
        st.warning(f"missing audio: {filename}")


def transcript(label, tag, fname, engine="whisper", note=None):
    r = row(tag, fname, engine)
    st.markdown(f"**{label}**")
    if not r:
        st.warning(f"missing transcript: {tag} / {engine} / {fname}")
        return
    st.write(r["text"])
    st.caption(f"{r['audio_s']} s audio, RTF {r['rtf']}")
    if note:
        st.info(note)


def compact_table(items):
    table = []
    for label, tag, fname in items:
        r = row(tag, fname)
        table.append({
            "Stage": label,
            "ASR output": r["text"] if r else "(missing)",
            "RTF": r["rtf"] if r else "",
        })
    st.table(table)


st.title("Page 9 - Real recordings")
st.markdown("##### External validity check: do controlled-experiment patterns survive real phone recordings?")

st.info(
    "This page is a qualitative external-validity check. The audio is intentionally real and messy; "
    "the goal is to inspect whether the same ASR failure modes appear outside controlled data."
)

st.subheader("Slide focus")
st.markdown(
    "- Controlled experiments support reliable CER measurement; real recordings do not have verbatim "
    "ground truth.\n"
    "- The page therefore compares before/after ASR outputs qualitatively instead of reporting a CER "
    "score.\n"
    "- The selected clips test whether two controlled-experiment patterns survive real recordings: "
    "front-end processing can backfire, and real overlap is hard to handle cleanly."
)

st.divider()

st.subheader("Data and method")
st.markdown(
    "- **Recordings:** 6 unscripted phone recordings: dorm, canteen, classroom, and three two-person "
    "discussion/crosstalk clips.\n"
    "- **Engineering step:** m4a -> 16 kHz mono wav using PyAV, because this machine did not have "
    "system ffmpeg.\n"
    "- **Evaluation:** no verbatim ground truth, so this page is not a CER table. It is an external "
    "validity check using before/after ASR outputs."
)

st.divider()

st.subheader("Demo 1 - Real heavy overlap: separation is not clean enough")
c1, c2 = st.columns(2)
with c1:
    audio_player("Mixed heavy-overlap recording", "caseA_heavy_raw.wav")
    transcript(
        "Direct ASR on the mixed recording",
        "base",
        "discussion_canteen_heavy_01.wav",
        note="Observation: direct ASR struggles under synchronous overlap and loses information."
    )
with c2:
    audio_player("Separated output, still mixed", "caseA_heavy_sep_spk2.wav")
    transcript(
        "ASR after separation, channel 2",
        "L3_sep",
        "discussion_canteen_heavy_01_spk2.wav",
        note="Observation: this output is not a clean speaker-isolated channel, so it should not be presented as a separation win."
    )
st.success(
    "Conclusion: this real-recording check does not provide a clean positive example for separation; it is better treated as a limitation."
)

st.divider()

st.subheader("Demo 2 - Denoising can backfire in real canteen noise")
c1, c2 = st.columns(2)
with c1:
    audio_player("Raw canteen recording", "caseD_canteen_raw.wav")
    transcript("Direct ASR", "base", "canteen_01.wav")
with c2:
    audio_player("After spectral subtraction", "caseD_canteen_specsub.wav")
    transcript(
        "Spectral subtraction -> ASR",
        "specsub",
        "canteen_01.wav",
        note="Observation: the transcript ends with a subtitle-style like-and-subscribe hallucination."
    )
st.success(
    "Conclusion: denoising does not reliably recover content and can introduce artifacts that trigger hallucination."
)

st.divider()

st.subheader("Optional backup - VAD is double-edged")
compact_table([
    ("classroom direct ASR", "base", "classroom_01.wav"),
    ("classroom direct ASR + VAD", "base_vad", "classroom_01.wav"),
    ("canteen direct ASR", "base", "canteen_01.wav"),
    ("canteen direct ASR + VAD", "base_vad", "canteen_01.wav"),
])
st.info(
    "Observation: VAD removes a trailing hallucination in the classroom clip, but cuts real speech "
    "from the canteen clip."
)

with st.expander("Backup cases if someone asks why not play all recordings", expanded=False):
    st.markdown("**Natural turn-taking, canteen:** direct ASR is already enough because overlap is brief.")
    compact_table([("Direct ASR", "base", "discussion_canteen_01.wav")])
    st.markdown("**Quiet overlap:** separation mostly copies or fragments the mix, so the heavier front-end backfires.")
    compact_table([
        ("Direct ASR, mixed", "base", "discussion_quiet_01.wav"),
        ("Separated spk1", "L3_sep", "discussion_quiet_01_spk1.wav"),
        ("Separated spk2", "L3_sep", "discussion_quiet_01_spk2.wav"),
    ])

st.divider()

st.subheader("Takeaway")
st.markdown(
    "Real recordings do not replace the controlled experiments. They support the main risk pattern: "
    "**direct ASR remains the most stable baseline, denoising and VAD can backfire, and this real "
    "heavy-overlap sample is not clean enough to claim a successful separation case.**"
)
