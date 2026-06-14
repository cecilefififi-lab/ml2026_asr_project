"""SepFormer 双说话人分离 (沿用学长方案, speechbrain sepformer-wsj02mix, 8kHz)。

用法:
    python src/separate.py data/overlap/MidOverlap.wav

输出: data/separated/{原名}_spk1.wav / _spk2.wav (8kHz)
"""
import os
import sys
import time

import librosa
import soundfile as sf
import torch

SEP_SR = 8000
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE, "data", "separated")


if __name__ == "__main__":
    in_path = sys.argv[1]
    if not os.path.isabs(in_path):
        in_path = os.path.join(BASE, in_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    import speechbrain  # noqa: F401
    from speechbrain.utils.importutils import LazyModule
    # Python 3.12 inspect.getmodule 会触发未安装依赖(k2/flair/spacy)的懒加载并崩溃
    for _name, _mod in list(sys.modules.items()):
        if isinstance(_mod, LazyModule):
            sys.modules.pop(_name, None)

    from speechbrain.inference.separation import SepformerSeparation as separator
    from speechbrain.utils.fetching import LocalStrategy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = separator.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=os.path.join(BASE, "tools", "sepformer-wsj02mix"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,  # Windows 无特权创建 symlink
    )

    y, _ = librosa.load(in_path, sr=SEP_SR)
    stem = os.path.splitext(os.path.basename(in_path))[0]
    tmp_8k = os.path.join(OUT_DIR, f"_{stem}_8k_tmp.wav")
    sf.write(tmp_8k, y, SEP_SR)

    t0 = time.perf_counter()
    # speechbrain 会在路径前拼 cwd, 绝对路径会出错, 必须传相对路径
    est = model.separate_file(path=os.path.relpath(tmp_8k))
    proc_s = time.perf_counter() - t0
    os.remove(tmp_8k)

    for i in range(2):
        spk = est[:, :, i].detach().cpu().numpy().squeeze()
        out = os.path.join(OUT_DIR, f"{stem}_spk{i + 1}.wav")
        sf.write(out, spk, SEP_SR)
        print(f"wrote {out}")
    dur = len(y) / SEP_SR
    print(f"separation on {device}: {proc_s:.1f}s for {dur:.1f}s audio (rtf={proc_s / dur:.2f})")
         