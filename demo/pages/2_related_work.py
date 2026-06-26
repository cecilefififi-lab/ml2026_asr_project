"""Page 2 -- Related work. Standalone: streamlit run demo/pages/2_related_work.py
Speaking script: demo/pages/2_related_work.md.
Literature review across four areas (denoising / separation / GER / SER), grounded in
specific findings from each paper. Refs verified in REPORT.md sec. 11 /
results/literature_support.md.
"""
import streamlit as st

st.set_page_config(page_title="Related work", layout="wide")


def arxiv(num):
    return f"[arXiv:{num}](https://arxiv.org/abs/{num})"


st.title("Related work")
st.markdown("##### What existing research already shows, across the four stages of our pipeline")

st.markdown(
    "We surveyed the literature for each stage of the pipeline. In every case an existing "
    "result already warns against applying that stage blindly. We summarise what each line of "
    "work actually found, and where we put it to the test."
)

st.divider()

st.subheader("1. Denoising and its effect on ASR")
st.markdown(
    "The assumption that cleaner audio means better recognition does not hold. *How Bad Are "
    "Artifacts?* uses an orthogonal-projection decomposition of the enhanced signal to show that "
    "**enhancement artifacts, not residual noise, are the dominant cause** of recognition errors. "
    "A medical-ASR study, *When De-noising Hurts*, evaluated 40 enhancement configurations and "
    "found **all 40 degraded accuracy** (by 1.1% to 46.6%), concluding that modern ASR does better "
    "on raw noisy audio than on denoised audio."
)
st.caption(f"Key work: {arxiv('2201.06685')}, {arxiv('2512.17562')}  ·  We test this in Exp 1.")

st.subheader("2. Speaker separation and over-separation")
st.markdown(
    "Separation is not safe to apply everywhere. *Should We Always Separate?* proposes switching "
    "between the separated signal and the observed signal **depending on whether overlap is "
    "actually present**, precisely to avoid artifacts on non-overlapping regions. *Decoupling "
    "Speaker Separation and Speech Recognition* shows that a separation front-end injects "
    "processing artifacts that **degrade a clean-backend recognizer**, and that multi-speaker ASR "
    "loses accuracy on single-speaker input."
)
st.caption(f"Key work: {arxiv('2106.00949')}, {arxiv('2503.17886')}  ·  We test this in Exp 2.")

st.subheader("3. LLM-based generative error correction (GER)")
st.markdown(
    "Correcting ASR output with a language model has a hard limit. The GER *challenge and "
    "baselines* paper shows a text-only model **cannot restore acoustic information already pruned "
    "during decoding**, and that an unconstrained rewrite fabricates content the speaker never "
    "said. Follow-up work adds a **three-stage, logit-anchored verification** step specifically to "
    "stop the corrector from hallucinating."
)
st.caption(f"Key work: {arxiv('2409.09785')}, {arxiv('2505.24347')}  ·  We test this in Exp 3.")

st.subheader("4. Speech emotion recognition (SER)")
st.markdown(
    "To examine the paralinguistic side, we build on self-supervised speech-emotion "
    "representation learning. *emotion2vec* pre-trains a **universal emotion representation** that "
    "transfers across languages and downstream tasks, which lets us score the same clip before and "
    "after processing and detect whether a front-end flattens the emotion it carries."
)
st.caption(f"Key work: {arxiv('2312.15185')}  ·  We test this in Exp 5.")

st.divider()

st.info(
    "A common gap across all four areas: the evidence above is drawn almost entirely from "
    "**English or single-condition data**. None of it has been established for noisy, overlapped "
    "Chinese speech, which is exactly the regime this project measures."
)

st.caption(
    "Whisper's noise robustness and its non-speech hallucinations (arXiv:2212.04356, 2307.03183, "
    "2501.11378) are reviewed on the Exp 4 page, where they are revisited directly."
)
