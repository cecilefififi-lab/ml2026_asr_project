"""Page 0 -- Title & framing. Standalone: streamlit run demo/pages/0_title_framing.py
Speaking script: demo/pages/0_title_framing.md. Framing only, no result numbers.
"""
import streamlit as st

st.set_page_config(page_title="Title & framing", layout="wide")

st.title("When does audio preprocessing help ASR?")
st.markdown(
    "##### Front-end robustness in noisy and overlapped speech &mdash; a debate-room case study"
)
st.caption("Machine Learning course project, 2026")

st.divider()

st.subheader("The problem")
st.markdown(
    "In real debates and meetings, two problems hit automatic speech recognition "
    "(ASR) at the same time: **background noise** and **overlapping speakers**. "
    "The common assumption is that adding an audio front-end &mdash; denoising and "
    "speaker separation &mdash; before the recognizer will improve accuracy. "
    "This project measures when that assumption holds and when the front-end "
    "actually makes recognition worse."
)

st.divider()

st.subheader("What we test")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Debate clips", "26")
c2.metric("Noise types x SNR", "2 x 3")
c3.metric("ASR engines", "2")
c4.metric("Pipelines compared", "6")
st.caption(
    "Two noise types (white, babble) at 15 / 5 / 0 dB SNR; light and heavy overlap; "
    "Whisper large-v3 and FunASR Paraformer-zh; six pipelines from direct ASR to a "
    "full front-end with denoising, separation, and LLM correction."
)

st.divider()

st.subheader("Pipeline under study")
st.graphviz_chart(
    """
    digraph {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style=rounded, fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
        edge [arrowsize=0.7];
        "Audio in" -> "Noise\\nprobe" -> "Denoise" -> "Separation" -> "ASR" -> "LLM\\ncorrection" -> "Evaluation";
    }
    """,
    width='stretch',
)
st.caption("Each stage is justified on the next page (System design).")

st.divider()

st.subheader("Research questions")
st.markdown(
    "1. **Noise degradation.** How do Whisper and FunASR accuracy change as noise increases?\n"
    "2. **Front-end benefit.** When do denoising and VAD help, and when do their artifacts hurt?\n"
    "3. **Processing order.** With noise and overlap together, denoise first or separate first?\n"
    "4. **Correction limits.** What can an LLM fix after the fact, and what is already lost acoustically?\n"
    "5. **Engineering trade-off.** Is the more complex pipeline worth its added cost?"
)
