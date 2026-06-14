"""实验 3: LLM 语义纠错 三档对比(无纠错 / 纯 LLM / 热词辅助 LLM)。

盲测: 纠错时只给 ASR 转写(+ 热词表), **不提供 ground truth**。
模型: claude-opus-4-8(anthropic SDK, 从 .env 读 ANTHROPIC_API_KEY)。
CER 口径复用实验 1 evaluate.char_cer。

用法: python src/correct.py
输出: results/exp3_correction.md, results/exp3_correction.csv
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import char_cer  # 复用实验 1 CER 口径

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(BASE, "data", "exp3_inputs.json")
HOTWORDS = os.path.join(BASE, "refs", "hotwords.txt")
OUT_MD = os.path.join(BASE, "results", "exp3_correction.md")
OUT_CSV = os.path.join(BASE, "results", "exp3_correction.csv")
MODEL = "claude-opus-4-8"

SYS_BASE = (
    "你是中文语音识别(ASR)转写的校对员。输入是一段可能含识别错误的中文转写,"
    "来自一场关于“熬夜与作息”的辩论赛录音。请输出纠正后的文本。\n"
    "规则:\n"
    "1. 只修正明显的识别错误(同音/近音字、读不通顺的词);\n"
    "2. 若整段是与口语辩论无关的幻觉(如视频字幕、综艺节目名、“点赞订阅”之类),"
    "输出 [无法识别];\n"
    "3. 不要脑补或新增信息,不要解释;\n"
    "4. 只输出纠正后的文本本身,不要任何前后缀、引号或推理过程。"
)


def load_env():
    """读取 .env 返回 dict(KEY->VALUE)。显式返回而非写环境变量,
    避免本机已有的 ANTHROPIC_BASE_URL(官方端点)干扰中转 base_url。"""
    env = {}
    p = os.path.join(BASE, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def load_hotwords():
    return [s.strip() for s in open(HOTWORDS, encoding="utf-8")
            if s.strip() and not s.startswith("#")]


def correct(client, text, hotwords=None):
    sys_prompt = SYS_BASE
    if hotwords:
        sys_prompt += "\n已知该辩论的常用术语(优先参考纠正):" + "、".join(hotwords)
    # adaptive thinking: 让模型先推理再答。注: 本中转(xuedingtoken)把推理
    # 用 <thinking>...</thinking> 标签塞进正文 block(非标准实现), 故下面 strip 掉。
    msg = client.messages.create(
        model=MODEL, max_tokens=2000, system=sys_prompt,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": text}])
    out = "".join(b.text for b in msg.content if b.type == "text")
    out = re.sub(r"<thinking>.*?</thinking>", "", out, flags=re.DOTALL)
    return out.strip()


if __name__ == "__main__":
    env = load_env()
    api_key = env.get("ANTHROPIC_API_KEY")
    assert api_key, "缺 ANTHROPIC_API_KEY —— 在 asr-robustness/.env 写 ANTHROPIC_API_KEY=..."
    import anthropic
    client = anthropic.Anthropic(api_key=api_key,
                                 base_url=env.get("ANTHROPIC_BASE_URL") or None)

    cases = json.load(open(INPUTS, encoding="utf-8"))
    hot = load_hotwords()

    rows = []
    for c in cases:
        raw, ref = c["asr_raw"], c["ref"]
        out_llm = correct(client, raw)              # 纯 LLM(无热词)
        out_hot = correct(client, raw, hot)         # 热词辅助
        r = {
            "id": c["id"], "ref": ref, "raw": raw,
            "llm": out_llm, "llm_hot": out_hot,
            "cer_raw": round(char_cer(ref, raw), 3),
            "cer_llm": round(char_cer(ref, out_llm), 3),
            "cer_hot": round(char_cer(ref, out_hot), 3),
            "note": c.get("note", ""),
        }
        rows.append(r)
        print(f'{r["id"]}: CER raw {r["cer_raw"]} -> llm {r["cer_llm"]} -> hot {r["cer_hot"]}')

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# 实验 3: LLM 语义纠错(无纠错 / 纯 LLM / 热词辅助)\n\n")
        f.write(f"模型: {MODEL} | 盲测(纠错时不提供参考答案)\n\n")
        for r in rows:
            f.write(f"## {r['id']}\n")
            f.write(f"- 参考 ref: {r['ref']}\n")
            f.write(f"- 无纠错 raw: {r['raw']}  (CER {r['cer_raw']})\n")
            f.write(f"- 纯 LLM: {r['llm']}  (CER {r['cer_llm']})\n")
            f.write(f"- 热词辅助: {r['llm_hot']}  (CER {r['cer_hot']})\n")
            f.write(f"- 说明: {r['note']}\n\n")
    print(f"wrote {OUT_MD} and {OUT_CSV}")
