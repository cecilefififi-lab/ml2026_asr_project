# When Does Audio Preprocessing Help ASR?

This project studies when local audio preprocessing improves ASR robustness under noisy and overlapped-speech conditions. Using a debate "AI secretary" scenario, it compares ASR backends such as faster-whisper and FunASR across denoising, speech separation, VAD, and LLM-based correction pipelines. The experiments measure both character error rate (CER) and real-time factor (RTF), so the accuracy-efficiency trade-off of each processing path can be observed.

## Project Structure

- `src/`: Core code and experiment scripts, including noise generation, noise mixing, ASR inference, denoising, speech separation, evaluation, plotting, and demo generation.
- `data/`: Experimental audio data, including clean speech in `clean/`, noise materials in `noise/`, overlapped-speech samples in `overlap/`, and selected experiment input files.
- `refs/`: Reference transcripts, hotwords, and draft transcripts used for CER evaluation and ASR correction comparisons.
- `results/`: Experiment outputs and figures, including `.csv` metric tables, `.md` summaries, and `.png` charts such as denoising curves, pipeline comparisons, VAD hallucination comparisons, length ablations, and trade-off plots.
- `demo/`: Streamlit demo pages. The entry point is `demo/Home.py`, experiment pages are in `demo/pages/`, and example case notes are in `demo/cases.md`.
- `luyin/`: Real recording samples and recording instructions for real-world spot checks.
- `test/`: A small set of sanity-check audio samples.
- `REPORT.md`, `REPORT_en.md`, `REPORT_en.pdf`: Chinese and English reports, plus the English PDF version.
- `LOG.md`: Experiment log and process notes.
- `requirements.txt`: Python dependency list.

## Environment

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/pip install -r requirements.txt
```

## Minimal Workflow

```bash
python src/make_noises.py                                   # Generate white/babble noise
python src/add_noise.py                                     # clean x noise x SNR{15,5,0}
python src/run_asr.py --engine whisper --input data/clean --tag clean
python src/run_asr.py --engine funasr  --input data/clean --tag clean
python src/run_asr.py --engine whisper --input data/noisy/white_5dB --tag white_5dB
python src/evaluate.py                                      # CER/RTF summary tables
python src/separate.py data/overlap/MidOverlap.wav          # SepFormer separation
```

See `LOG.md` for experiment notes. Main result summaries are available in `results/summary.md` and `results/summary.csv`.
