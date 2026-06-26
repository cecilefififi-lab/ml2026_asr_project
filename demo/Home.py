"""Presentation entry point. Enables Streamlit's sidebar navigation over demo/pages/.
Run:  streamlit run demo/Home.py
No model runs here -- this is a results presentation. Each page in demo/pages/ is also
runnable on its own (streamlit run demo/pages/<N>_<slug>.py).
"""
import streamlit as st

st.set_page_config(page_title="ASR robustness - presentation", layout="wide")

st.title("ASR robustness under noise and overlapping speech")
st.markdown("##### When does audio preprocessing help ASR, and when does it hurt? "
            "A debate-room case study")
st.caption("Machine Learning course project, 2026. Use the left sidebar to move through the sections.")

st.divider()

st.subheader("Contents")
st.markdown(
    "0. **Title & framing** - the problem, the data, the research questions\n"
    "1. **System design** - why each component, and why not the alternatives\n"
    "2. **Related work** - what existing research already shows\n"
    "3. **Experiment 1** - noise x denoising x ASR\n"
    "4. **Experiment 2** - noise + overlap, and the order of processing\n"
    "5. **Experiment 3** - LLM-based generative correction\n"
    "6. **Experiment 4** - Whisper hallucination and the limits of VAD\n"
    "7. **Experiment 5** - does preprocessing erase emotion?\n"
    "8. **Experiment 6** - does clip length change the answer?\n"
    "9. **Real recordings** - generalization spot-check\n"
    "10. **Discussion, limitations, conclusion**"
)

st.divider()

st.markdown(
    "**One-line takeaway:** under noise and overlap on short debate clips, the simplest direct ASR "
    "is almost always the best value; most advanced preprocessing costs accuracy, runtime, and even "
    "the emotion in the speech."
)
