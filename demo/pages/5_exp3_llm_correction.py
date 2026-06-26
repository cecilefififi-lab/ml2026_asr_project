"""Page 5 -- Experiment 3: LLM-based generative correction. Standalone:
streamlit run demo/pages/5_exp3_llm_correction.py
Speaking script: demo/pages/5_exp3_llm_correction.md.
Numbers/text from REPORT.md sec. 6.3 and results/exp3_correction.csv (.md).
"""
import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Exp 3 - LLM correction", layout="wide")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(BASE, "results")

st.title("Experiment 3 - LLM-based generative correction")
st.markdown("##### Can a language model fix ASR errors, or only filter them?")

st.subheader("Data and method")
st.markdown(
    "- **Inputs:** 5 representative ASR outputs taken from Experiments 1 and 2 -- two terminology "
    "errors, one near-homophone error, one variety-show-name hallucination, and one subtitle-text "
    "hallucination.\n"
    "- **Conditions:** no correction; plain LLM correction; and hot-word-assisted LLM correction "
    "(the model is given a glossary of debate terms).\n"
    "- **Protocol:** blind -- the model is never shown the reference transcript. Model: claude-opus-4-8. "
    "Metric: content CER before and after."
)

st.divider()

st.subheader("Results -- the five cases (actual model outputs)")
df = pd.read_csv(os.path.join(RESULTS, "exp3_correction.csv"), encoding="utf-8-sig")
show = df[["id", "ref", "raw", "llm", "llm_hot", "cer_raw", "cer_llm", "cer_hot"]]
st.dataframe(show, width='stretch', hide_index=True)
st.caption(
    "ref = ground truth; raw = no correction; llm = plain LLM; llm_hot = hot-word-assisted. "
    "Empty llm_hot in case 2 is the model returning nothing. CER columns: lower is better."
)

c1, c2, c3 = st.columns(3)
c1.metric("Variety-show hallucination", "2.29 -> 0.57", "LLM rejected it", delta_color="inverse")
c2.metric("Subtitle hallucination", "1.58 -> 0.84", "LLM rejected it", delta_color="inverse")
c3.metric("Fluent wrong word", "0.16 -> 0.16", "unchanged", delta_color="off")

st.divider()

st.subheader("What the LLM could and could not do")
st.markdown(
    "- **Reliably good at one thing: rejecting hallucinations.** For the variety-show name "
    "(小明星大跟班...) and the subtitle text (請不吝点赞 订阅...), the model returned `[无法识别]` "
    "(unrecognized) instead of inventing content, which lowered CER.\n"
    "- **Could not fix fluent, acoustically-lost errors.** When the wrong word is itself grammatical "
    "(好意 for 熬夜, 说 for 多), the acoustic evidence needed to catch it is already gone, and the "
    "correction left CER unchanged.\n"
    "- **The hot-word list gave no gain, and sometimes hurt.** In case 2 (玩具 for 晚睡晚起, acoustically "
    "far apart) the glossary pushed CER from 0.40 to 1.0 -- it could not bridge a gap that large."
)
st.info(
    "This matches the generative-error-correction literature: text-only correction cannot recover "
    "acoustic information lost in decoding and can fabricate content (arXiv:2409.09785), which is why "
    "newer work adds verification stages (arXiv:2505.24347). **In our setting the LLM is a filter, "
    "not a corrector.**"
)
