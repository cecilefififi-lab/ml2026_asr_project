"""Page 8 -- Experiment 6: long-audio length ablation. Standalone:
streamlit run demo/pages/8_exp6_length_ablation.py
Speaking script: demo/pages/8_exp6_length_ablation.md.
Numbers from REPORT.md sec. 6.7, results/exp6_summary.csv, results/exp6_pivot.md.
"""
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exp 6 - length ablation", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")


def img(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(p, width='stretch')
    else:
        st.warning(f"missing figure: {name}")


st.title("Experiment 6 - Does clip length change the answer?")
st.markdown("##### Re-testing the Experiment 2 null result on longer audio, with two engines")

st.subheader("Data and method")
st.markdown(
    "- **Why:** in Experiment 2, denoise-first and separate-first looked identical. Before generalising "
    "that, we test whether it was caused by clips too short (2-4 s) for the separator to work at all.\n"
    "- **Data:** we concatenate annotated single-speaker clips into ~12 s passages, then mix two "
    "speakers with a controlled offset (light = 0.3, heavy = 0.8). The source clips come from one 62.4 s "
    "two-person debate segment, verified against the senior thesis, so each concatenated role is a "
    "coherent real single speaker and the per-speaker ground truth still holds.\n"
    "- **Runs:** clean + babble_0dB + white_0dB; pipelines L1 / L3 / L4 / L5; **both Whisper and FunASR**, "
    "to check the order effect is not specific to one engine. 5 samples x 2 levels x 3 noise x 4 "
    "pipelines x 2 engines = 420 transcripts."
)

st.divider()

st.subheader("Results")
img("exp6_length.png")
st.markdown("**Content CER, short (Experiment 2) -> long (this experiment), Whisper:**")
st.markdown(
    "| Condition | level | L1 | L3 | L4 | L5 |\n|---|---|---|---|---|---|\n"
    "| clean | light | 11.3 -> 17.9 | 57.6 -> 82.9 | 61.3 -> 73.7 | 48.8 -> 86.9 |\n"
    "| clean | heavy | 50.0 -> 52.1 | 46.9 -> 49.1 | 58.9 -> 51.3 | 57.4 -> **47.7** |\n"
    "| babble_0dB | light | 58.3 -> 53.6 | 100.6 -> 93.7 | 76.5 -> **64.0** | 80.5 -> 90.6 |\n"
    "| babble_0dB | heavy | 90.7 -> 86.6 | 70.3 -> 89.1 | 86.3 -> 92.9 | 77.3 -> 89.2 |\n"
    "| white_0dB | light | 32.5 -> 29.9 | 64.2 -> 48.1 | 68.7 -> **47.2** | 64.4 -> 51.8 |\n"
    "| white_0dB | heavy | 60.1 -> 59.5 | 63.0 -> 65.7 | 71.8 -> 75.4 | 66.4 -> 63.0 |\n"
)

st.markdown("**The order effect appears on long audio -- L4 (denoise first) vs L5 (separate first), both engines:**")
st.markdown(
    "| Condition / level | Whisper L4 / L5 | FunASR L4 / L5 |\n|---|---|---|\n"
    "| clean / light | **73.7** / 86.9 | **87.6** / 91.5 |\n"
    "| babble_0dB / light | **64.0** / 90.6 | **51.4** / 99.8 |\n"
    "| white_0dB / light | **47.2** / 51.8 | **44.3** / 60.0 |\n"
    "| clean / heavy | 51.3 / **47.7** | 56.1 / **54.6** |\n"
    "| babble_0dB / heavy | 92.9 / **89.2** | **73.9** / 82.3 |\n"
    "| white_0dB / heavy | 75.4 / **63.0** | 67.6 / **55.5** |\n"
)
st.caption("Bold = winner (lower CER). Under light overlap, denoise-first (L4) wins all 6 engine x noise cells.")

c1, c2, c3 = st.columns(3)
c1.metric("Light overlap, denoise-first wins", "6 / 6", "Whisper + FunASR x 3 noise")
c2.metric("Biggest L4 gain (babble)", "Whisper -27 / FunASR -48", "CER points", delta_color="inverse")
c3.metric("Separation still fails", "75-94%", "per-speaker CER (long)", delta_color="off")

st.caption("Per-condition long-audio data actually produced (results/exp6_summary.csv):")
df = pd.read_csv(os.path.join(RESULTS, "exp6_summary.csv"), encoding="utf-8-sig")
st.dataframe(df, width='stretch', hide_index=True, height=240)

st.divider()

st.subheader("Failure case -- the mechanism behind the order effect")
st.markdown(
    "`babble_0dB / light / speaker 2`: with **L5 (separate first)**, the noisy mixture goes straight into "
    "the separator and both channels collapse into the same subtitle hallucination, *中文字幕志愿者 李宗盛*. "
    "With **L4 (denoise first)**, the separator gets a clean input and recovers one real channel. "
    "Denoising first stops the separator from being driven into a double hallucination by the noise."
)

st.divider()

st.subheader("Findings -- and the reversal")
st.markdown(
    "1. **Separation still fails on long audio** (per-speaker CER mostly 75-94%), so 'clips were too "
    "short' is not the only cause -- the model genuinely struggles with overlapped Chinese speech.\n"
    "2. **But processing order does matter, and it holds across both engines.** Under light overlap, "
    "denoise-first wins all 6 engine x noise cells, strongest under babble. FunASR confirms it "
    "independently, so it is not a Whisper quirk.\n"
    "3. **Whether separation pays off depends on the engine's noise weakness.** FunASR, which fears "
    "white noise, recovers strongly under white_0dB where Whisper does not -- echoing Experiment 1's "
    "opposite-weakness result."
)
st.info(
    "This upgrades the Experiment 2 null result from a degenerate observation to a located boundary "
    "condition, confirmed across two engines: order is invisible only when the separator fails on short "
    "clips; with enough length, denoise-first wins under light overlap. Caveat: N = 5 per cell, so this "
    "is qualitative, and some heavy-overlap pipelines sit near the error floor."
)
