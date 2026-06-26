"""Page 3 -- Experiment 1: noise x denoising x ASR. Standalone:
streamlit run demo/pages/3_exp1_noise_denoising.py
Speaking script: demo/pages/3_exp1_noise_denoising.md.
Numbers from REPORT.md sec. 6.1, results/summary.csv, results/exp1_cases.txt.
"""
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exp 1 - noise x denoising", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")


def img(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(p, width='stretch')
    else:
        st.warning(f"missing figure: {name}")


st.title("Experiment 1 - Noise, denoising, and ASR")
st.markdown("##### What noise does to recognition, and whether denoising actually helps")

st.subheader("Data and method")
st.markdown(
    "- **Speech:** 26 clean Chinese debate clips, 2-4 s, 16 kHz mono. Transcripts were drafted by "
    "Whisper, then corrected by listening to each clip, so the ground truth is not biased toward "
    "Whisper's own output.\n"
    "- **Noise:** white (synthetic) and babble (offset overlay of English speech, no text leakage "
    "into the Chinese target), mixed at SNR 15 / 5 / 0 dB. The measured SNR error was 0.000 dB.\n"
    "- **Front-end:** each condition is transcribed with no denoising, with FRCRN (neural), and with "
    "spectral subtraction (classical DSP). Clean audio is also passed through the denoisers, as a "
    "falsification check.\n"
    "- **Back-end:** faster-whisper large-v3 and FunASR Paraformer-zh.\n"
    "- **Grid:** 26 clips x 2 noise x 3 SNR x 3 front-ends x 2 engines. Metric: character error rate "
    "(CER); also denoising real-time factor (RTF)."
)
st.caption("Aggregated output actually produced by the runs (results/summary.csv):")
df = pd.read_csv(os.path.join(RESULTS, "summary.csv"), encoding="utf-8-sig")
st.dataframe(df, width='stretch', hide_index=True)

st.divider()

st.subheader("Results")
st.markdown("**Degradation curves and denoising heatmap**")
c1, c2 = st.columns(2)
with c1:
    img("exp1_denoise_curves.png")
with c2:
    img("exp1_denoise_heatmap.png")

st.markdown("**CER by condition (none / FRCRN / spectral subtraction)** -- pick a noise type:")
noise = st.radio("Noise type", ["white", "babble"], horizontal=True, label_visibility="collapsed")
data = {
    "white": [("15 dB", "7.4 / 16.6 / 14.4", "9.6 / 12.0 / 14.6"),
              ("5 dB", "12.2 / 33.2 / 32.3", "29.4 / 17.0 / 16.8"),
              ("0 dB", "41.9 / 49.9 / 61.7", "68.4 / 27.5 / 40.1")],
    "babble": [("15 dB", "5.4 / 17.0 / 19.0", "8.5 / 12.7 / 16.1"),
               ("5 dB", "20.6 / 78.6 / 47.1", "22.3 / 52.3 / 30.1"),
               ("0 dB", "71.7 / 92.5 / 83.7", "43.3 / 95.1 / 46.2")],
}
tbl = "| Condition | Whisper (none / FRCRN / specsub) | FunASR (none / FRCRN / specsub) |\n|---|---|---|\n"
tbl += "| clean | 4.9 / 9.3 / 14.7 | 10.1 / 8.5 / 13.0 |\n"
for snr, w, f in data[noise]:
    tbl += f"| {noise} {snr} | {w} | {f} |\n"
st.markdown(tbl)
st.caption("CER in %. Lower is better. Denoising RTF: spectral subtraction ~0.018, FRCRN ~0.105 (about 6x slower).")

c1, c2, c3 = st.columns(3)
c1.metric("Best case (clean, Whisper)", "4.9%")
c2.metric("Only large denoising win", "68.4 -> 27.5%", "FunASR x white x 0 dB", delta_color="inverse")
c3.metric("Neural denoise on babble", "43.3 -> 95.1%", "FunASR, a disaster")

st.divider()

st.subheader("Failure case")
st.markdown(
    "`babble_0dB / FunASR / FRCRN / con_001.wav` -- the neural denoiser mistook the target Chinese "
    "voice for background speech and removed it, leaving English background that took over the output."
)
st.markdown(
    "- **Reference:** 他让你在那个时间段里拥有了自己\n"
    "- **No denoising (CER 0.28):** 哦，他让你在那个时间段去拥有的是不会。\n"
    "- **After FRCRN (CER 2.59):** *Because here's that with kind of talthink against this.*"
)

st.divider()

st.subheader("Findings")
st.markdown(
    "1. **Denoising is not a stable gain.** It helped substantially in only one corner of the grid: "
    "FunASR x white x low SNR (68.4 -> 27.5).\n"
    "2. **Babble plus neural denoising is a disaster** (FunASR 43.3 -> 95.1): FRCRN cannot tell the "
    "target voice from background voices and deletes the target.\n"
    "3. **Denoising clean audio is generally harmful** (Whisper 4.9 -> 9.3 / 14.7), which falsifies the "
    "idea that denoising is safe to apply at high SNR -- the artifacts, not residual noise, drive it.\n"
    "4. **The two engines have opposite weaknesses** (new observation): Whisper is hurt most by babble, "
    "FunASR by white. Their degradation curves cross near 0 dB."
)
st.info(
    "Prior work confirms the artifact mechanism (arXiv:2201.06685, arXiv:2512.17562) and that babble is "
    "harder than white for ASR. The opposite Whisper-vs-FunASR noise profile is our own observation: the "
    "literature has no direct Whisper-vs-Paraformer comparison."
)
