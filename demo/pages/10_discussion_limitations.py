"""Page 10 -- Discussion, limitations, conclusion. Standalone:
streamlit run demo/pages/10_discussion_limitations.py
Speaking script: demo/pages/10_discussion_limitations.md.
Numbers from REPORT.md sec. 7-9, results/tradeoff_summary.md, results/literature_support.md.
"""
import os
import streamlit as st

st.set_page_config(page_title="Discussion & limitations", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")


def img(name):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        st.image(p, width='stretch')
    else:
        st.warning(f"missing figure: {name}")


st.title("Discussion, limitations, and conclusion")
st.markdown("##### What we claim, what we deliberately do not claim, and the cost-benefit picture")

st.subheader("Counter-intuitive findings")
st.markdown(
    "| # | Finding |\n|---|---|\n"
    "| 1 | The two ASR engines have **opposite** noise weaknesses; their curves cross near 0 dB. |\n"
    "| 2 | Advanced preprocessing is **net-negative on short clips**; only heavy overlap makes separation pay off. |\n"
    "| 3 | Processing order is invisible on short clips but **appears on long audio** (denoise-first wins under light overlap). |\n"
    "| 4 | Denoising changes Whisper's error mode from hallucination to **silence** (empty transcripts). |\n"
    "| 5 | VAD has a boundary: it stops white/silence (100->0%) but not babble (67->67%). |\n"
    "| 6 | The LLM is a **filter, not a corrector**: it rejects hallucination but cannot fix acoustically-lost words. |\n"
    "| 7 | Preprocessing **flattens emotion**, erasing paralinguistic information. |\n"
)

st.divider()

st.subheader("Accuracy vs efficiency trade-off")
c1, c2 = st.columns([3, 2])
with c1:
    img("tradeoff.png")
with c2:
    st.markdown(
        "| Pipeline | CER % | RTF | vs L1 |\n|---|---|---|---|\n"
        "| **L1 direct** | **39.8** | **0.191** | 1.0x |\n"
        "| L2 denoise | 52.5 | 0.235 | 1.2x |\n"
        "| L3 separate | 68.2 | 0.520 | 2.7x |\n"
        "| L4 denoise+sep | 67.4 | 0.533 | 2.8x |\n"
        "| L5 sep+denoise | 65.3 | 0.645 | 3.4x |\n"
    )
    st.markdown("No pipeline sits below-left of L1 (faster **and** more accurate). The simplest "
                "pipeline is both the fastest and the most accurate.")

st.markdown("**Rule-based pipeline selector** (what we would actually deploy):")
st.code(
    "babble + low SNR     -> direct ASR (denoising deletes the voice; add VAD as a guard)\n"
    "white + low SNR      -> FunASR + neural denoise (the only stable positive combination)\n"
    "overlap none / light -> direct ASR (separation is pure overhead)\n"
    "overlap heavy        -> consider separation, but expect limited gain\n"
    "output looks hallucinated -> LLM filters it to [unrecognized]; do not expect word fixes",
    language="text",
)

st.divider()

st.subheader("How our results sit against the literature")
a, b, c = st.columns(3)
with a:
    st.markdown("**Confirmed by prior work**")
    st.success(
        "- Whisper non-speech hallucination (2501.11378)\n"
        "- denoising hurts ASR via artifacts (2201.06685 / 2512.17562)\n"
        "- over-separation (2106.00949 / 2503.17886)\n"
        "- LLM is a filter (2409.09785 / 2505.24347)"
    )
with b:
    st.markdown("**Claims we narrow**")
    st.warning(
        "- 'order does not matter' -> only on short clips where separation fails; Exp 6 shows order "
        "matters on long audio\n"
        "- 'white vs babble, which is worse' -> depends on the SNR range, not a monotonic rule"
    )
with c:
    st.markdown("**New observations (ours)**")
    st.info(
        "- FunASR resists babble better than Whisper (no prior direct comparison)\n"
        "- VAD on babble: 67% -> 67%\n"
        "- MossFormer2 fails on 2-4 s Chinese overlap"
    )

st.divider()

st.subheader("Limitations (stated plainly)")
st.markdown(
    "- **We do not claim processing order is irrelevant.** The Experiment 2 null result was a "
    "short-clip artifact; Experiment 6 shows the opposite on longer audio.\n"
    "- **'MossFormer2 fails on 2-4 s Chinese speech' is our measured observation, not a documented "
    "property of the model.** We report it with the per-speaker CER of 84-88% as evidence, not as an "
    "established fact.\n"
    "- **Small, short corpus.** Mostly 2-4 s Chinese clips, a small sample; results should not be "
    "extrapolated to much longer audio or other languages.\n"
    "- **Real recordings have no verbatim ground truth**, so that check is qualitative."
)

st.divider()

st.subheader("Conclusion")
st.markdown(
    "Under noise and overlap on short debate clips, **the simplest direct ASR is almost always the "
    "best value**: it wins 13 of 15 overlap conditions, and complex front-ends pay off only in narrow "
    "corners (FunASR + neural denoise on white at low SNR; separation under heavy overlap). Most "
    "'advanced preprocessing' is not worth its cost under realistic constraints, and that cost is not "
    "only CER and runtime but also the loss of paralinguistic emotion. We treat the negative result as "
    "a first-class result: knowing **when not to process** is as valuable as knowing what helps."
)
