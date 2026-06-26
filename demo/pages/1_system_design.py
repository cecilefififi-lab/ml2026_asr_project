"""Page 1 -- System design & component choices. Standalone:
streamlit run demo/pages/1_system_design.py
Speaking script: demo/pages/1_system_design.md.
Content from REPORT.md sec. 4.1-4.6 (candidate space -> problem hit -> decision per module).
"""
import streamlit as st

st.set_page_config(page_title="System design", layout="wide")

st.title("System design")
st.markdown("##### Why each component, and why not the alternatives")

st.markdown(
    "None of these components were chosen by default. For each one we list the candidates we "
    "considered, the problem we actually hit while testing, and the decision we made. The "
    "selection converged through repeated trial and failure, not a single up-front guess."
)

st.divider()

st.subheader("Pipeline")
st.graphviz_chart(
    """
    digraph {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style=rounded, fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
        edge [arrowsize=0.7];
        "Audio in" -> "Noise\\nprobe" -> "Front-end\\n(denoise / VAD)" -> "Separation" -> "ASR" -> "LLM\\ncorrection" -> "Evaluation";
    }
    """,
    width='stretch',
)

st.divider()

st.subheader("Component choices")
rows = [
    ("ASR backend",
     "openai-whisper, faster-whisper, WhisperX, wav2vec2, Conformer, FunASR (Paraformer)",
     "**faster-whisper large-v3 (int8) + FunASR Paraformer-zh.** WhisperX bundles forced VAD, "
     "alignment and diarization, which mainly help long audio; our clips are 2-4 s and we need "
     "VAD as a switch we can toggle for Exp 4, so a welded-in VAD would remove that control. We "
     "add FunASR as a non-Whisper engine to test whether noise robustness is a property of one "
     "model or of ASR in general. The two engines turned out to have opposite weaknesses (Exp 1)."),
    ("Denoising",
     "DeepFilterNet (neural), FRCRN (neural), spectral subtraction (DSP)",
     "**FRCRN + spectral subtraction.** DeepFilterNet had no Python 3.12 wheel and needed a Rust "
     "toolchain, which would hurt reproducibility, so we used ClearVoice's FRCRN as the neural "
     "representative and spectral subtraction as the classical one -- one method from each end of "
     "the complexity range."),
    ("Separation",
     "Conv-TasNet, SepFormer, MossFormer2",
     "**MossFormer2, on self-made controlled overlap.** SepFormer (trained on English 8 kHz) "
     "returned two near-identical channels on Chinese audio, a domain-mismatch failure. "
     "MossFormer2 (Chinese) still struggled on real recordings but separated self-made fully "
     "overlapped clips at 0.82 / 0.79 source correlation, so we moved the overlap experiments to "
     "controlled synthetic mixtures where every speaker has ground truth."),
    ("VAD / SER / LLM",
     "supporting components",
     "**Silero VAD (kept as a toggle), emotion2vec, Claude Opus.** VAD is studied as an on/off "
     "variable rather than always on. emotion2vec reuses the FunASR / ModelScope stack with no new "
     "dependencies. LLM correction runs blind, with no reference text, across three conditions."),
]
table = "| Module | Candidates considered | Decision and rationale |\n|---|---|---|\n"
for mod, cand, why in rows:
    table += f"| **{mod}** | {cand} | {why} |\n"
st.markdown(table)

st.divider()

st.subheader("A design choice for rigour")
st.info(
    "We also run **clean audio** through the denoisers. If denoising still helped at high SNR, "
    "our artifact explanation would be wrong. It did not help -- clean-audio denoising raised "
    "Whisper CER from 4.9% to 9.3-14.7%. That supports the view that artifacts, not residual "
    "noise, drive the degradation, and it shapes the experiments that follow."
)
