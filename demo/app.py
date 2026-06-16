"""Streamlit demo 最小版(混合形态)。

两部分:
  Tab1 固定案例:复用 make_demo.CASES + demo/audio/ + results/real_asr.csv,
                选案例 -> 前后音频试听 + 前后文本对照 + 耗时/RTF + 讲点。零现场跑,稳。
  Tab2 上传现场跑:上传音频 -> 直接 ASR(faster-whisper large-v3, 可选 VAD)
                  -> 文本 + 耗时 + RTF + 试听。模型用 @st.cache_resource 缓存。

降噪/分离等重链路只在 Tab1 以预跑结果展示, 现场不跑(避免录视频时翻车)。

运行: streamlit run demo/app.py   (cwd = asr-robustness)
"""
import os
import sys
import tempfile
import time

import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "src")
AUDIO = os.path.join(BASE, "demo", "audio")
CSV_PATH = os.path.join(BASE, "results", "real_asr.csv")
sys.path.insert(0, SRC)
import make_demo  # noqa: E402  复用案例定义, 顶层无副作用(main 在 __main__ 下)

st.set_page_config(page_title="ASR 鲁棒性 Demo", layout="wide")


@st.cache_data
def load_rows():
    import csv
    rows = {}
    with open(CSV_PATH, encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            rows[(r["tag"], r["engine"], r["file"])] = r
    return rows


@st.cache_resource
def load_whisper():
    import torch
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel("large-v3", device=device, compute_type=compute), device


def audio_label(key):
    if "raw" in key:
        return "① 原始(混合/带噪)"
    if "spk1" in key:
        return "分离 spk1"
    if "spk2" in key:
        return "分离 spk2"
    if "specsub" in key:
        return "谱减降噪后"
    return key


st.title("噪声与重叠语音下 ASR 鲁棒性 · Demo")
st.caption("辩论赛书记员:观众起哄(噪声)+ 选手抢话(重叠),AI 还能听清谁说了什么?")

tab1, tab2 = st.tabs(["📦 固定案例(预跑·稳)", "🎤 上传现场跑(直接 ASR)"])

with tab1:
    lut = load_rows()
    case = st.selectbox(
        "选择案例", make_demo.CASES,
        format_func=lambda c: f"案例 {c['id']} — {c['title']}",
    )
    st.markdown(f"**录音类型**:{case['type']}")

    if case["audio"]:
        st.markdown("**音频试听(处理前/后)**")
        cols = st.columns(len(case["audio"]))
        for col, key in zip(cols, case["audio"]):
            path = os.path.join(AUDIO, key)
            with col:
                st.caption(audio_label(key))
                if os.path.exists(path):
                    st.audio(path)
                else:
                    st.warning(f"缺音频 {key}")

    st.markdown("**ASR 输出对照**")
    table = []
    for label, tag, eng, fname in case["rows"]:
        r = lut.get((tag, eng, fname))
        table.append({
            "阶段": label,
            "ASR 输出": r["text"] if r else "(缺)",
            "音频时长(s)": r["audio_s"] if r else "",
            "处理耗时(s)": r["proc_s"] if r else "",
            "RTF": r["rtf"] if r else "",
        })
    st.table(table)
    st.info("💡 " + case["point"])

with tab2:
    st.markdown("上传一段音频,现场跑 **直接 ASR**(faster-whisper large-v3)。"
                "降噪/分离等重链路见左侧固定案例,现场不跑(避免翻车)。")
    up = st.file_uploader("上传音频", type=["wav", "mp3", "m4a", "flac", "ogg"])
    vad = st.checkbox("启用 VAD 过滤(Silero)", value=False)
    if up is not None:
        st.audio(up)
        if st.button("▶ 跑直接 ASR"):
            suffix = os.path.splitext(up.name)[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.getvalue())
            tmp.close()
            try:
                with st.spinner("加载模型 + 转写中…"):
                    model, device = load_whisper()
                    t0 = time.perf_counter()
                    segs, info = model.transcribe(
                        tmp.name, language="zh", beam_size=5, vad_filter=vad)
                    text = "".join(s.text for s in segs)
                    proc = time.perf_counter() - t0
                dur = info.duration
                st.success(text or "(空)")
                c1, c2, c3 = st.columns(3)
                c1.metric("音频时长", f"{dur:.1f}s")
                c2.metric("处理耗时", f"{proc:.1f}s")
                c3.metric("RTF", f"{proc / dur:.2f}" if dur else "—")
                st.caption(f"device={device}, vad_filter={vad}")
            finally:
                os.remove(tmp.name)
