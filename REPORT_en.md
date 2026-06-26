# When Does Audio Preprocessing Help ASR? A Study of Front-End Robustness in Noisy and Overlapped Speech

---

## Abstract

Modern large-scale weakly supervised ASR systems have been trained on massive amounts of noisy speech and therefore already possess substantial built-in noise robustness. This makes the engineering intuition of "denoise or separate first, then recognize" worth re-examining. When the backend model is already strong, does additional front-end processing compensate for degradation, or does it introduce new distribution shift and processing artifacts? Existing work often studies a single processing component in a single data domain or under clean conditions. It rarely provides a systematic comparison of multiple processing chains under a unified protocol when noise and overlap coexist.

This report uses a "debate AI secretary" scenario as the carrier task: speaker interruptions correspond to overlapped speech, while audience reactions correspond to background noise. On two heterogeneous ASR backends, faster-whisper large-v3 and FunASR paraformer-zh, we conduct controlled ablation studies over four classes of components: denoising, speech separation, VAD, and LLM-based correction. Evaluation uses both character error rate (CER) and real-time factor (RTF), so that the net effect and boundary conditions of each processing layer can be measured along both accuracy and efficiency axes.

The main findings are threefold. **Front-end benefit boundaries.** Under 2-4 s short clips, low SNR, and real overlap constraints, most complex front-end processing produces negative gains. The simplest direct ASR path achieves the lowest CER in 13 of 15 overlap conditions. Denoising gives stable positive gains only in the narrow condition `FunASR x white noise x low SNR`. Speech separation only partly offsets its additional cost under heavy synchronous overlap. The reliable value of LLM post-processing is to detect and reject hallucinations, not to restore words that have already been lost acoustically. **Backend differences.** The two backends have almost opposite noise robustness profiles: Whisper is more sensitive to babble, while FunASR is more sensitive to white noise, and their degradation curves cross near 0 dB. This comparison shows that using only one model can easily turn a model-specific weakness into a mistaken universal claim about ASR. **Cost spillover.** The cost of preprocessing is not limited to text accuracy: denoising and separation also systematically flatten paralinguistic emotion cues such as anger. Every conclusion in this report is tied to corresponding failure cases and public literature, and the report explicitly distinguishes claims confirmed by prior work, claims that must be narrowed, and new observations from this project.

---

## 1. Introduction

### 1.1 Scenario and Narrative

This report places the research question in a concrete application scenario: an "AI secretary" for debate competitions. Such a system must accurately record "who said what and when" while two debaters interrupt each other with overlapped or cross-speech and while the audience produces background noise. This scenario is not merely rhetorical. It naturally combines two course topics: Topic 3, local audio preprocessing, and Topic 1, speech separation, speaker labeling, and LLM x ASR collaboration. In other words, noise and multi-speaker overlap are not isolated problems here. They are coexisting degradations that the system must handle together, making the scenario a suitable testbed for evaluating the value of front-end processing.

### 1.2 Why This Problem Matters

Engineering practice often assumes a pipeline intuition: once audio is contaminated by noise, it should first be denoised or separated before being sent to ASR. This intuition made sense in the era of traditional ASR, but it is no longer self-evident after the rise of large-scale weakly supervised models. Modern ASR systems represented by Whisper have been trained on hundreds of thousands of hours of noisy speech. Their acoustic front ends have implicitly absorbed many noise and accent variations. If another enhancement or separation layer is applied before ASR, the resulting input distribution may drift away from what the model saw during training, or it may be contaminated by artifacts introduced by the enhancement model itself. This leads to a question that has not been sufficiently tested: when the ASR backend is already strong, does additional front-end processing improve recognition quality, or does it introduce distribution shift and processing artifacts that cause additional degradation? This question is well suited to systematic ablation and failure analysis, rather than to an isolated claim that one pipeline "works well enough."

### 1.3 Contributions

This report makes three contributions.

1. **Adding the noise dimension to overlap research.** Prior senior work (`xutong_paper.pdf`) focused on speech separation, speaker labeling, and LLM correction under clean conditions. This report adds the more realistic setting where noise and overlap coexist, and treats "when not to process" as a question equally important to "when to process."
2. **A unified ablation protocol.** Under one evaluation protocol, this report systematically compares five processing chains: L1 direct ASR, L2 denoise -> ASR, L3 separate -> ASR, L4 denoise -> separate -> ASR, and L5 separate -> denoise -> ASR. The same protocol measures the net effect and narrow applicability of each layer, instead of presenting one pipeline in isolation.
3. **Actionable routing rules.** The report derives rule-like processing recommendations and reports several counter-intuitive findings. Each finding is compared with public literature and categorized as confirmed, narrowed, or newly observed, so that each conclusion can be traced back to data and references.

---

## 2. Research Process and Design Iteration

This section explains how the experimental design evolved. The project did not start with the final design already fixed. It went through a research-oriented iteration: first proposing intuitive hypotheses, then validating the pipeline through small-scale experiments, and then revising data, models, and evaluation protocols based on failure cases. Documenting this process matters because it explains why the experiments are designed this way, not only what numbers they eventually produce.

### 2.1 Early Stage: Problem Definition and Initial Hypotheses

The initial overall question was: under noise and multi-speaker overlap, which local preprocessing methods such as denoising, VAD, and separation can materially improve ASR, and which ones are harmful? The question came from a natural engineering intuition: once audio is contaminated, it seems reasonable to enhance or clean it before giving it to ASR. However, this intuition is not necessarily valid for modern large ASR models, because models such as Whisper have already been trained on large amounts of noisy speech. Additional front-end processing may introduce artifacts or distribution shift.

Based on this reasoning, we wrote three initial hypotheses in the experiment log on 2026-06-10. First, as SNR decreases from 15 dB to 0 dB, CER should increase for both Whisper and FunASR. Second, babble noise, which contains speech-like interference, may be more damaging than white noise. Third, when noise and overlap coexist, whether denoising comes before separation should affect the final recognition result. Later experiments did not simply "prove" these hypotheses. Instead, they made them more precise: whether babble is harder depends on the ASR backend; processing order is not visible on short clips, but becomes visible on longer audio and light-overlap conditions.

| Initial hypothesis | Rationale at the time | Later result |
|---|---|---|
| Lower SNR gives higher CER | Stronger energy masking makes target speech harder to hear | Mostly true; both backends degrade as SNR decreases |
| Babble is harder than white noise | Babble contains speech interference and informational masking | Must be narrowed: Whisper is more sensitive to babble, FunASR to white noise |
| Denoise before separation may be more stable | The separator may work better on cleaner input | Hidden by separation failure on short clips; after exp6 lengthening, light overlap supports L4 in 6/6 cases |

### 2.2 Data Preparation and Ground Truth Correction

The data initially came from Chinese debate short clips in the senior project. We first used Whisper large-v3 to produce draft transcriptions for clean clips, then manually listened to and corrected each item, and finally wrote the references into `refs/`. This step is essential. If Whisper drafts were directly used as references, the clean / Whisper CER would be structurally pushed toward 0, biasing the evaluation toward one of the tested models. Therefore, the first round of numbers in the log was explicitly marked as "pipeline validation, not final evidence," and the full noise matrix was recomputed only after manual correction.

This process also changed our judgment about data scale. We originally hoped for longer audio per clip, but the actual clean clips were mostly 2-4 s long. This limitation later became an important background for separation failure and for the exp6 length ablation. In other words, short duration was not a post-hoc explanation added late in the report. It was a constraint exposed during data preparation and directly shaped the subsequent experimental design.

### 2.3 Tool Selection and Adjustment

Component selection did not simply follow an existing pipeline. It resulted from repeated trade-offs among candidate methods, runtime constraints, reproducibility, and variable control. The table below summarizes the selection process.

| Component | Initial candidates | Issues or trade-offs | Final choice | Reason |
|---|---|---|---|---|
| ASR backend | Whisper family | Single-model conclusions may only reflect model-specific behavior | faster-whisper large-v3 + FunASR paraformer-zh | Use a Whisper and a non-Whisper backend throughout the experiments to avoid mistaking one model's weakness for a universal pattern |
| Whisper pipeline | WhisperX / faster-whisper | WhisperX alignment and diarization bring limited value for 2-4 s clips, and VAD is bound into the pipeline | faster-whisper | Faster on the local GPU and allows VAD to be studied as an independent variable |
| Denoising | DeepFilterNet / FRCRN / spectral subtraction | DeepFilterNet lacks a usable wheel for Python 3.12 and source installation requires Rust, increasing reproducibility cost | FRCRN + spectral subtraction | One neural method and one DSP method with stable dependencies, allowing comparison between spectral reconstruction and energy reduction |
| Speech separation | SepFormer / MossFormer2 | SepFormer produced nearly identical two-channel outputs on Chinese overlap clips, clearly failing | MossFormer2 | Closer to the senior work, and able to separate two correlated channels on a fully overlapped synthetic sample |
| LLM correction | Directly rewrite ASR text | Direct rewriting may generate unspoken content and cannot recover acoustically lost information | LLM as filter | Mainly detects and rejects hallucinations, rather than promising to correct all wrong words |

### 2.4 Mid Stage: How Failure Cases Changed the Design

One of the most important turns in the project came from separation failure. We first tried SepFormer on the senior project's real `MidOverlap.wav`, but the two output channels produced almost the same ASR text. This indicated that the model had not actually separated the two speakers. After switching to MossFormer2, real overlap clips remained unstable. However, on a synthetic fully overlapped `pro_001 + con_001` sample, the two outputs reached correlations of 0.82 / 0.79 with the two source audios. This showed that the model and runtime environment were not completely unusable.

This failure led us to change the main data for experiment 2 to synthetic controllable overlap samples. We mixed `con_i` and `pro_i` at equal loudness with 0 dB SIR and different offset ratios. Each speaker therefore had ground truth, allowing strict computation of content CER and per-speaker CER. The senior project's real overlap samples were moved to qualitative failure cases. This changed the experiment from "demonstrating a separation system" to "controlled testing of when separation should or should not be used."

Another key turn came from denoising. Installation problems with DeepFilterNet made us drop that candidate. This was not just a tool replacement; it strengthened the project's reproducibility requirement. The final choice of FRCRN and spectral subtraction did not cover every recent enhancement model, but it allowed the complete experimental matrix to run stably and produced a neural-vs-DSP comparison.

### 2.5 Late Stage: How Additional Experiments Answered Boundary Questions

The initial conclusion from experiment 2 was that L4, denoise -> separate, and L5, separate -> denoise, showed no stable difference on 2-4 s clips. Writing this directly as "processing order does not matter" would have overgeneralized the result. Given the per-speaker CER of 84-88%, a more reasonable explanation is that separation itself had already failed on short clips, making the order of a failed component naturally hard to observe.

Therefore, we added experiment 6 in the late stage. We concatenated multiple clips from the same role to around 12 s, reran L1/L3/L4/L5, and included both Whisper and FunASR backends. The result showed that under light overlap, L4 beat L5 in 6/6 cases across clean / babble / white and both backends. This means H3 was not completely wrong; it was masked by the degraded short-clip condition. Once audio duration gave separation more room, the boundary condition of processing order emerged. This is the research process the project aims to present: negative results are not the end point; they help locate boundaries, refine hypotheses, and design the next experiment.

---

## 3. Related Work

This section connects the five technical lines in the project to public literature and explains how each line relates to this work. Claim-level alignment is listed in `results/literature_support.md`, which includes 12 major sources.

### 3.1 Robust ASR and Whisper Hallucination

Whisper obtains strong robustness to ambient and white noise through roughly 680,000 hours of weakly supervised training (Robust Speech Recognition via Large-Scale Weak Supervision, arXiv:2212.04356). Its noise robustness is even strong enough for audio event tagging (Whisper-AT, arXiv:2307.03183). However, the same generative decoder can hallucinate on non-speech or low-confidence audio. A systematic analysis in arXiv:2501.11378 reports that about 35% of hallucinations concentrate in a few fixed phrases rather than being randomly distributed. The OpenAI community has also documented fixed outputs such as `Amara.org` and "like and subscribe" caused by subtitle-data bias (openai/whisper Discussion #928). This phenomenon directly matches our exp4 observations and explains why speech-like babble is especially dangerous for Whisper: it is easily misread by the decoder as decodable speech.

### 3.2 Effects of Speech Enhancement and Denoising on ASR

The claim that denoising always helps recognition does not hold. arXiv:2201.06685, "How Bad Are Artifacts?", decomposes enhancement errors into noise residuals and artificial artifacts using orthogonal projection, showing that enhancement artifacts rather than residual noise dominate downstream ASR degradation. arXiv:2512.17562, "When De-noising Hurts" for medical ASR, further reports that all 40 tested configurations became worse after denoising. These works form the core literature basis for this report's claim that advanced front ends are not automatically worthwhile, and they foreshadow the exp1 result that clean audio can have higher CER after denoising.

### 3.3 Speech Separation and Over-Separation

The title of arXiv:2106.00949, "Should We Always Separate?", is almost the same question asked here. It argues that whether separation should be applied should depend dynamically on whether overlap truly exists, in order to avoid artifacts caused by over-separation in non-overlapped segments. arXiv:2503.17886 further shows that placing a separation front end before clean-trained ASR can degrade recognition. Together, these works align with the over-separation phenomenon observed in exp2.

### 3.4 LLM-Based Generative Error Correction

arXiv:2409.09785 provides the challenge and baselines for generative error correction (GER). It clearly states that text-only rewriting cannot restore acoustic information already lost or pruned during decoding, and that full rewriting can hallucinate unspoken content. arXiv:2505.24347 uses three-stage verification and logit-space anchoring to suppress GER hallucinations. This matches the conclusion of exp3: in this setting, the LLM is a filter rather than a corrector.

### 3.5 Speech Emotion Recognition

We use the emotion2vec family of self-supervised speech emotion representation models to evaluate whether preprocessing flattens emotion, adding a paralinguistic dimension to the main argument (see exp5). Original citations for the models used in this project, including emotion2vec, Paraformer/FunASR, MossFormer2, FRCRN, SepFormer, Conv-TasNet, and DeepFilterNet, have been checked and are listed in References.

### 3.6 Prior Work and Its Limitations

The senior work (`xutong_paper.pdf`) built a clean-condition pipeline of speech separation -> speaker labeling -> LLM semantic correction, verifying that this approach is feasible under clean conditions. This report identifies three limitations that can be further studied. First, the noise dimension is missing: the prior work does not systematically examine how background noise, especially speech-like babble, simultaneously harms ASR, separation, and speaker attribution. Second, it lacks a criterion for whether processing should be applied at all: it assumes a separation -> recognition process and does not test whether separation is harmful on non-overlapped or short clips. Third, it lacks a front-end trade-off view: RTF, processing order, and artifact risk are not included in the accuracy-efficiency trade-off. This project is organized around these three gaps.

---

## 4. Problem Statement and Hypotheses

To make the overall problem falsifiable, we decomposed it at project start into five testable hypotheses and then marked each one after the experiments as confirmed, rejected, or narrowed. The overall question is: under noise and multi-speaker overlap, which local preprocessing methods truly improve ASR and which ones are harmful? The table below gives the five hypotheses and their test results.

| # | Research question | Hypothesis at project start | Test result |
|---|---|---|---|
| H1 | noise degradation | CER rises monotonically as SNR changes from 15 to 5 to 0 dB; babble background speech is more harmful than white noise | **Partly confirmed and narrowed:** monotonic degradation holds; the relative difficulty of white vs babble depends on the SNR range. Babble is more damaging to Whisper only near low SNR / crossover regions (exp1) |
| H2 | denoising benefit | Denoising helps under low SNR but hurts under high SNR because of artifacts | **Confirmed:** stable positive gain appears only for `FunASR x white x low SNR`; denoising generally hurts clean and high-SNR audio (exp1) |
| H3 | processing order | When noise and overlap coexist, denoising before separation vs separation before denoising has a significant effect | **Rejected first, then recovered through boundary localization and two-backend evidence:** L4/L5 show no stable difference on short clips, a degenerate case; after exp6 lengthens audio to around 12 s, order differences emerge. Under light overlap, denoise-first L4 is preferred in Whisper + FunASR 6/6 cases (see 7.7 / 9.3) |
| H4 | LLM correction boundary | LLMs can correct terminology or formatting but cannot repair acoustically lost errors | **Confirmed:** the only reliable value of the LLM is rejecting hallucinations; fluent wrong words cannot be repaired (exp3) |
| H5 | engineering trade-off | More complex chains are not necessarily worthwhile; low-cost near-optimal solutions may exist | **Confirmed:** direct ASR is best in 13 of 15 overlap conditions (exp2 + ablation) |

These hypotheses structure all later experiments. Exp1 tests H1/H2, exp2 tests H3/H5, exp3 tests H4, and exp4-exp6 strengthen or revise the conclusions from the perspectives of hallucination mechanism, paralinguistic cost, and audio length boundaries.

---

## 5. Methodology

This section gives the overall system design and the rationale for each component. Each component is described through the sequence: candidate space -> evaluation criteria -> problems observed in practice -> final choice -> evidence. This shows that the design converged through repeated testing and failure cases rather than a one-shot subjective choice.

### 5.1 Overall Pipeline

The system architecture is shown in Figure 5.1. Audio passes through audio probing, front-end processing, separation, ASR, speaker labeling, LLM post-processing, and evaluation. Each module can be independently switched on or off, allowing the net effect of each module to be measured separately.

**Figure 5.1 System architecture.**

![System architecture](results/report_pipeline.svg)

The core exp2 ablation consists of the five paths shown in Figure 5.2. L1 is direct ASR. L2 and L3 each add one layer, either denoising or separation. L4 and L5 include both layers but swap the processing order. This design isolates both the net effect of one processing layer and the effect of processing order.

**Figure 5.2 Five comparison paths L1-L5.**

![Five comparison paths](results/report_ablation_paths.svg)

### 5.2 ASR Backend Selection

The candidate space included openai-whisper, faster-whisper, WhisperX, wav2vec2, Conformer, and FunASR (Paraformer). The evaluation criteria were Chinese short-clip CER, ability to run on a 6 GB laptop GPU, acceptable RTF, and the ability to independently control variables, especially using VAD on/off as the independent variable in exp4.

We selected faster-whisper rather than WhisperX because WhisperX is essentially faster-whisper plus forced VAD, forced alignment, and diarization. Its gains mainly target long-audio time alignment and speaker separation. This report studies 2-4 s short clips, where alignment and diarization are unnecessary. Exp4 also requires VAD to be studied as a switchable independent variable; using WhisperX would lock VAD into the pipeline and reduce control. faster-whisper is based on CTranslate2, and int8_float16 quantization allows large-v3 to run stably on an RTX 4050 Laptop 6 GB GPU with RTF around 0.2-0.3. It is therefore the more suitable base engine.

In addition to Whisper, we added FunASR (paraformer-zh) as a non-Whisper heterogeneous control, to test whether noise robustness is a model-specific property or a universal pattern. This decision later supported one of the most important counter-intuitive findings: the two backends have opposite weaknesses (see 7.1). If only one model had been used, "Whisper's weakness" could have been mistaken for "ASR's weakness." The final choice was to use faster-whisper large-v3 (int8_float16) and FunASR paraformer-zh throughout the study.

### 5.3 Denoising Method Selection

The candidate space included DeepFilterNet (neural), FRCRN (neural), and spectral subtraction / spectral gating (DSP, `noisereduce`). The evaluation criteria were to choose one representative neural method and one representative DSP method while adding no heavy new dependency and remaining reproducible in the current environment.

In practice, the originally selected DeepFilterNet had no usable `deepfilterlib` wheel under Python 3.12, and source installation required the Rust toolchain. This would greatly increase dependency complexity and threaten reproducibility. We therefore used ClearVoice's FRCRN_SE_16K as the neural representative and spectral subtraction as the DSP representative. A key design decision was to run denoising even on clean audio. This was not to simulate noise, but to falsify H2: if denoising still reduced CER under high SNR, the artifact hypothesis would be weakened. The result was that clean denoising generally hurt performance, with Whisper rising from 4.9% to 9.3% / 14.7%, supporting the claim that artifacts are a major degradation source. The final choice was FRCRN (neural) and spectral subtraction (DSP).

### 5.4 Speech Separation Selection

The candidate space included Conv-TasNet, SepFormer, and MossFormer2. The evaluation criteria were the ability to handle Chinese overlap, direct inference with pretrained models, and outputs that could be verified using ASR or correlation.

A major failure case appeared in testing. We first used SepFormer (`sepformer-wsj02mix`, English 8 kHz training) to separate the senior project's real `MidOverlap.wav`. The two output channels had highly similar content, so we judged the separation to have failed and suspected domain mismatch between English training and Chinese testing. After switching to MossFormer2 (ModelScope Chinese), real `MidOverlap.wav` remained unstable, but on a synthetic fully overlapped sample it separated two channels with source correlations of 0.82 / 0.79. This led to a data decision: the quantitative overlap experiment would use synthetic controllable overlap samples, mixing `con_i` and `pro_i` at equal loudness with 0 dB SIR and overlap ratios of 0 / 0.3 / 0.8, ensuring that each speaker had ground truth. The senior project's real overlap samples were used as qualitative failure cases. This is a research decision made to keep ground truth controlled, not an arbitrary choice. The final choice was MossFormer2 with synthetic controllable overlap data.

### 5.5 VAD, SER, and LLM Selection

VAD uses the switchable Silero VAD built into faster-whisper, so that it can be the independent variable in exp4 rather than being always enabled. SER uses `iic/emotion2vec_plus_large`. It belongs to the same FunASR/ModelScope ecosystem as paraformer-zh, MossFormer2, and FRCRN, so already-generated before/after audio can be reused for emotion drift comparison with no heavy new dependency. LLM correction uses claude-opus-4-8 through a relay API, under a blind-test setting where the reference is not provided during correction. We compare three settings: no correction, pure LLM, and hotword-aided correction.

### 5.6 Evaluation Protocol

The evaluation uses CER rather than WER because the test corpus is Chinese. CER is computed as character-level edit distance with punctuation and whitespace removed using jiwer. RTF is defined as processing time divided by audio duration, recorded per stage (denoise RTF, separate RTF, ASR RTF), with chain RTF being the sum of stages. For speaker attribution, we use per-speaker CER with the best permutation matching rather than manually labeled attribution error rate, because separation often fails and both channels frequently copy the mixture, making manual attribution labels uninformative.

One cross-experiment caveat is important. Exp1 (domain A, single-speaker denoising), exp2 (domain B, overlap separation), and exp3 (domain C, LLM correction) use different data domains, so their absolute CER values should not be compared across rows. Only the direction of net effects should be compared across domains. Detailed protocol notes are in `results/ablation_summary.md`.

---

## 6. Experimental Setup

The table below summarizes the full setup for corpus, noise, backends, front ends, hardware, and metrics. The corpus is centered on manually corrected Chinese debate short clips. Noise and overlap are controllably synthesized, and six real phone recordings are used for generalization spot-checks.

| Item | Setting |
|---|---|
| Clean corpus | 26 Chinese debate short clips, 2-4 s, 16 kHz mono, manually corrected transcripts saved in `refs/`. Drafts came from Whisper and were manually corrected to avoid bias toward Whisper |
| Overlap corpus | Synthetic: `con_i` <-> `pro_i` equal-loudness offset mixing at 0 dB SIR, ratio {0, 0.3, 0.8}; plus five real overlap clips from the senior project for qualitative comparison |
| Noise | white synthetic noise + babble made from offset English speech, with no text leakage into Chinese targets; SNR {15, 5, 0} dB, measured SNR error 0.000 dB |
| Real recordings | Six real phone recordings: dorm, canteen, classroom, discussion x3 |
| Backend | faster-whisper large-v3 (int8_float16), FunASR paraformer-zh |
| Front-end | FRCRN_SE_16K, spectral subtraction, Silero VAD, MossFormer2 |
| SER | emotion2vec_plus_large |
| Hardware | RTX 4050 Laptop 6 GB |
| Metrics | CER, RTF, per-speaker CER, hallucination case |

Data generation and running scripts are in `src/`, including `make_noises.py`, `add_noise.py`, `make_overlap.py`, `run_asr.py`, `denoise.py`, `separate.py`, and `evaluate.py`. The minimal reproduction steps are in `README.md`.

---

## 7. Experiments and Results

The following six experiments test the hypotheses in sequence. Each experiment first states the tested hypothesis and design motivation, then provides data tables, and then explains the mechanism using failure cases. The definitive numbers are stored in `results/`.

### 7.1 Experiment 1: Noise x Denoising x ASR (Domain A)

Experiment 1 tests H1 and H2: how noise itself degrades CER, and whether denoising can compensate for this degradation. We add {white, babble} noise at {15, 5, 0} dB SNR to 26 clean clips. Clean clips also pass through denoising as a falsification test for H2. Each condition is recognized by Whisper and FunASR under three settings: no processing, FRCRN, and spectral subtraction. Table 7.1 reports CER in the format `none -> FRCRN -> specsub`. Full data are in `results/asr_raw.csv` and `results/summary.csv`.

| Condition | Whisper | FunASR |
|---|---|---|
| clean | 4.9 -> 9.3 -> 14.7 | 10.1 -> 8.5 -> 13.0 |
| white 15 dB | 7.4 -> 16.6 -> 14.4 | 9.6 -> 12.0 -> 14.6 |
| white 5 dB | 12.2 -> 33.2 -> 32.3 | 29.4 -> 17.0 -> 16.8 |
| white 0 dB | 41.9 -> 49.9 -> 61.7 | **68.4 -> 27.5** -> 40.1 |
| babble 15 dB | 5.4 -> 17.0 -> 19.0 | 8.5 -> 12.7 -> 16.1 |
| babble 5 dB | 20.6 -> 78.6 -> 47.1 | 22.3 -> 52.3 -> 30.1 |
| babble 0 dB | 71.7 -> 92.5 -> 83.7 | 43.3 -> **95.1** -> 46.2 |

In efficiency terms, spectral subtraction has RTF around 0.018, while FRCRN has RTF around 0.105, roughly six times slower. Degradation curves and heatmaps are shown in `results/exp1_denoise_curves.png` and `results/exp1_denoise_heatmap.png`; representative cases are in `results/exp1_cases.txt`.

![Denoising curves](results/exp1_denoise_curves.png)

![Denoising heatmap](results/exp1_denoise_heatmap.png)

Several consistent patterns emerge. **Boundary 1: denoising is not a stable gain.** Across seven noise conditions, denoising gives a large positive gain only in `FunASR x white x low SNR`, reducing CER from 68.4 to 27.5 at 0 dB; most other conditions are negative. **Boundary 2: babble and neural denoising can couple dangerously.** Under babble 0 dB, FunASR after FRCRN rises from 43.3 to 95.1 CER, nearly unusable. A failure case makes the mechanism clear: for `babble_0dB / FunASR / FRCRN / con_001.wav`, the original audio still recognized part of the Chinese target, while FRCRN removed the target Chinese as background speech and left English babble to dominate the output. **Boundary 3: high SNR can also be damaged by processing.** Clean clips generally become worse after denoising, with Whisper rising from 4.9% to 9.3% / 14.7%. This directly falsifies the idea that high-SNR audio should still be denoised and supports the artifact-dominance explanation. Finally, "neural is always better than DSP" is also false: whether FRCRN or spectral subtraction is better depends on noise type and backend, not on a general rule.

### 7.2 Experiment 2: Noise + Overlap x Processing Order (Domain B)

Experiment 2 tests H3 and H5: the relative performance of five processing paths L1-L5 when noise and overlap coexist, and whether processing order matters. The data use synthetic controllable overlap with ratios {0, 0.3, 0.8}, combined with {babble, white} x {5, 0} dB and clean. Table 7.2 reports the average content CER for each path across 15 overlap cells. Details are in `results/exp2_summary.csv` and `results/exp2_pivot.md`.

| Path | content CER | spk CER | Best count among 15 cells |
|---|---|---|---|
| L1 direct | **39.8** | -- | **13** |
| L2 denoise -> ASR | 52.5 | -- | 0 |
| L3 separate -> ASR | 68.2 | 88.1 | 2, both heavy |
| L4 denoise -> separate | 67.4 | 84.9 | 0 |
| L5 separate -> denoise | 65.3 | 85.2 | 0 |

For chain RTF, specsub is around 0.018, separation around 0.13, and Whisper around 0.2. Therefore, chains containing separation are about two to three times slower than L1. **Core phenomenon: over-separation.** L1 achieves the lowest CER in 13 of 15 cells. Separation paths only occasionally give positive gains under heavy overlap, for example clean / heavy drops from 50.0 to 46.9. Per-speaker CER remains at 84-88%, showing that MossFormer2 fails to effectively separate 2-4 s short clips. The two outputs often copy the entire mixture. Because separation has already failed in this condition, simply swapping the order between L4 and L5 does not show a stable difference. This means H3 is rejected under this condition, but it is a degenerate case. Section 7.7 locates the root cause by lengthening audio. Two classes of failure support this interpretation: under light overlap, the separator copies one-channel content into two channels; under 0 dB babble, both separated channels can trigger Whisper hallucination.

Figure 7.2 averages over five noise conditions for Whisper and visualizes how over-separation changes with overlap severity. The plotting script is `src/plot_exp2.py`.

![exp2 pipeline comparison](results/exp2_pipelines.png)

As shown, under no overlap, L1 at 23.8% is far below separation paths L3/L4/L5 at around 64-70%. As overlap becomes heavier, L1's advantage narrows. Under heavy overlap, L1 at 63.3% is almost tied with L3 at 64.0%, and separation only overtakes in a few heavy cells. In other words, separation can only approach parity under the most severe overlap; in most conditions it mainly adds artifacts and latency.

### 7.3 Experiment 3: LLM-Based Generative Correction (Domain C)

Experiment 3 tests H4, the boundary of LLM post-processing. We select five representative errors from exp1 / exp2: two terminology errors, one near-homophone error, one variety-show hallucination, and one subtitle hallucination. Under a blind-test setting where the reference is not provided during correction, we compare no correction, pure LLM, and hotword-aided correction. The correction model is claude-opus-4-8. Table 7.3 gives content CER for each case; details are in `results/exp3_correction.md` and the corresponding `.csv`.

| Case type | raw | pure LLM | hotword |
|---|---|---|---|
| Fluent terminology error | 0.158 | 0.158 | 0.158 |
| Acoustically distant terminology error | 0.40 | 0.40 | **1.0** |
| Near-homophone error | 0.077 | 0.077 | 0.077 |
| Variety-show hallucination | 2.286 | **0.571** | 0.571 |
| Subtitle hallucination | 1.579 | **0.842** | 0.842 |

The results point to a clear boundary. **The reliable role of the LLM is filtering, not repair.** The only reliable value of the LLM is to identify and reject hallucinations. It can replace fabricated content with `[unrecognizable]` in the variety-show hallucination (2.286 -> 0.571) and subtitle hallucination (1.579 -> 0.842), rather than continuing to generate content without acoustic evidence. But for words that are already lost acoustically while still producing fluent text, the LLM cannot help: the fluent terminology case remains 0.158 in all three settings, and the near-homophone case remains 0.077. Hotwords do not help either and can even hurt when the acoustic gap is too large. In the "toy" case, the hotword cannot establish a valid constraint between target and recognized text, so CER rises from 0.40 to 1.0. In short, in this project, the LLM is a filter rather than a corrector.

### 7.4 Experiment 4: Whisper Hallucination and VAD Boundaries

Experiment 4 tests a finer mechanism hypothesis: non-speech or low-confidence audio induces fixed-pattern hallucination, and VAD can suppress this phenomenon but has boundaries. We feed pure noise and silence into Whisper (`data/exp4/`) and compare VAD on/off. Details are in `results/exp4_hallucination.csv` and `.md`; the comparison plot is `results/exp4_vad_compare.png`.

![VAD hallucination comparison](results/exp4_vad_compare.png)

With VAD disabled, the hallucination rate is around 91%. With VAD enabled, hallucinations on white noise and silence are completely suppressed (100% -> 0%), but babble is almost unchanged (67% -> 67%). The hallucinated content follows fixed patterns rather than random text: `Amara.org`, "please like and subscribe," and subtitle-team signatures repeatedly appear, matching arXiv:2501.11378. VAD fails on babble because it judges continuous speech-like babble as speech and lets it pass. This is exactly the root cause of the severe damage caused by babble 0 dB in exp1 / exp2. The 67% -> 67% curve is direct evidence for this mechanism.

### 7.5 Experiment 5: Emotion x Preprocessing (Paralinguistic Dimension)

Experiment 5 extends the cost analysis from text to the paralinguistic dimension. It tests whether denoising and separation systematically flatten high-arousal emotion, especially anger, and whether processing meant to improve audibility also weakens how speech is expressed. We run emotion2vec on the same audio before and after processing and compare P(angry). The script is `src/run_emotion.py`, details are in `results/exp5_emotion*.csv/.md`, and the drift plot is `results/exp5_emotion_drift.png`.

![Emotion drift after preprocessing](results/exp5_emotion_drift.png)

Across 17 angry clips, the mean P(angry) drops from 0.89 for clean audio to 0.85 after FRCRN, with 1/17 emotion-label flips, and further to 0.65 after spectral subtraction, with 4/17 flips. Separation weakens the representation even more. Source audio at 0.63 drops after L3 to 0.37 / 0.47 / 0.35 under no / light / heavy overlap. Thus preprocessing systematically flattens emotion. In this sample, separation weakens paralinguistic information more than denoising, and spectral subtraction weakens it more than neural denoising. Representative cases include `pro_014`, which changes from angry to neutral after specsub, and `con_005`, which changes from angry to happy. This adds another layer to the main argument: front-end processing can harm not only ASR, but also emotion cues.

### 7.6 Real Recording Generalization Spot-Check

To test whether the above conclusions are merely artifacts of synthetic data, we run the baseline and best path once on six real phone recordings and conduct manual spot-checking. Since there is no word-level ground truth, we do not force CER computation. Details are in `results/real_asr.csv` and `results/real_spotcheck.md`, and intermediate outputs are stored under `data/real/{raw,denoised,sep_L3,sep_L4}/`. The real recordings independently reproduce all main conclusions from exp1 / exp2: direct ASR is the most stable; in canteen scenes, spectral subtraction triggers the subtitle hallucination "please like and subscribe," and L4 triggers "Chinese subtitle volunteers"; separation only gives positive gains under heavy synchronous overlap by recovering a synchronous sentence missed by direct ASR; VAD is double-edged; English proper nouns are consistently misrecognized, supporting the motivation for exp3 hotwords. This spot-check supports the trend of extrapolating the conclusions to real recordings.

Table 7.6 lists five representative cases as qualitative evidence for failure analysis. Each case corresponds to a concrete engineering boundary or error mechanism.

| Case | Scenario | Phenomenon in one sentence | Supported judgment |
|---|---|---|---|
| A | Two speakers, canteen noise, heavy synchronous overlap | **Only separation gives a positive gain:** direct ASR misses A's synchronous sentence, while one separated channel recovers it | Separation may be worthwhile only under heavy synchronous overlap |
| B | Two speakers, canteen noise, natural interruption | **Direct ASR is sufficient:** nearly the whole content is transcribed | The key damage to ASR is not whether noise exists, but the duration of synchronous overlap |
| C | Two speakers, quiet overlap | **Separation causes negative effects:** spk2 nearly copies the whole mixture, while spk1 retains only fragments | Unnecessary separation introduces over-separation artifacts |
| D | One speaker, strong canteen noise | **Denoising causes negative effects:** spectral-subtraction artifacts hallucinate "please like and subscribe" at the end | Denoising artifacts can induce Whisper subtitle-style hallucinations |
| E | One speaker, light classroom noise vs strong canteen noise | **VAD is double-edged:** it removes tail hallucination under light noise but cuts real mid-speech under strong noise | VAD is not a universal fix and depends on noise shape |

### 7.7 Experiment 6: Long-Audio Length Ablation (Domain B Boundary, H3 Root Cause)

Experiment 6 returns to the root cause of H3. Is the "no stable processing-order difference" observed in exp2 a general rule, or is it a degenerate artifact caused by complete separation failure on 2-4 s clips, where spk CER is 84-88%? We lengthen audio to around 12 s and retest two questions: (a) whether separation produces positive gains; and (b) whether L4 and L5 become different.

The data source was verified as reliable. According to Table 4.12 in the senior `xutong_paper.pdf`, `con` and `pro` originate from a 62.4 s two-speaker debate questioning segment and were manually split into single-speaker clips. Therefore, concatenating clips from the same role yields coherent real single-speaker audio rather than an artificial speaker, and two-speaker ground truth remains strict. In setup, `src/make_exp6.py` reuses `make_overlap.mix_pair`; each role is concatenated with a fixed seed to around 12 s, mixed at equal loudness under {light=0.3, heavy=0.8}, and covers clean, babble_0dB, and white_0dB. For each item, we run L1/L3/L4/L5, dropping L2 because it does not involve order. Both Whisper and FunASR are used to verify that the order effect is not model-specific. Denoising uses specsub and separation uses MossFormer2. The total is 5 samples x 2 levels x 3 noise conditions x 4 paths x 2 engines = 420 transcriptions. Scripts are `src/{make_exp6,run_exp6,eval_exp6,plot_exp6}.py`; details are in `results/exp6_{summary.csv,pivot.md}` and `data/exp6/manifest.csv`. Table 7.7a gives Whisper content CER and compares "short clip -> long audio."

| cond | level | L1 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| clean | light | 11.3 -> 17.9 | 57.6 -> 82.9 | 61.3 -> 73.7 | 48.8 -> 86.9 |
| clean | heavy | 50.0 -> 52.1 | 46.9 -> 49.1 | 58.9 -> 51.3 | 57.4 -> **47.7** |
| babble_0dB | light | 58.3 -> 53.6 | 100.6 -> 93.7 | 76.5 -> **64.0** | 80.5 -> 90.6 |
| babble_0dB | heavy | 90.7 -> 86.6 | 70.3 -> 89.1 | 86.3 -> 92.9 | 77.3 -> 89.2 |
| white_0dB | light | 32.5 -> 29.9 | 64.2 -> 48.1 | 68.7 -> **47.2** | 64.4 -> 51.8 |
| white_0dB | heavy | 60.1 -> 59.5 | 63.0 -> 65.7 | 71.8 -> 75.4 | 66.4 -> 63.0 |

![Length ablation](results/exp6_length.png)

Processing order differences appear on long audio. Table 7.7b gives two-backend comparisons between L4 and L5 (content CER, lower is better). Under light overlap across three noise conditions and two engines, all six conditions favor L4, denoising first.

| cond/level | whisper L4 / L5 | funasr L4 / L5 |
|---|---|---|
| clean/light | **73.7** / 86.9 | **87.6** / 91.5 |
| babble_0dB/light | **64.0** / 90.6 | **51.4** / 99.8 |
| white_0dB/light | **47.2** / 51.8 | **44.3** / 60.0 |
| clean/heavy | 51.3 / **47.7** | 56.1 / **54.6** |
| babble_0dB/heavy | 92.9 / **89.2** | **73.9** / 82.3 |
| white_0dB/heavy | 75.4 / **63.0** | 67.6 / **55.5** |

Per-speaker CER for long Whisper only drops to around 50% in clean / heavy; most other cells remain 75-94%. The short-vs-long comparison is shown in `results/exp6_length.png`.

The mechanism of the order effect appears in a failure case. Under `babble_0dB / light / s2`, L5, separate-first, sends a noisy mixture directly into the separator. Both spk1 and spk2 collapse into "Chinese subtitle volunteers Li Zongsheng," meaning both channels become subtitle hallucinations and completely fail. L4, denoise-first, gives the separator a cleaner input and preserves real content in one channel. In other words, denoising first prevents the separator from being noise-induced into two-channel hallucination, which explains why L4 leads strongly under light-overlap + noise.

Experiment 6 gives three conclusions. **Long audio still does not fully solve separation.** Separation still fails widely on long audio, with spk CER mostly 75-94%, so short duration is not the only cause. The deeper issue is insufficient separation ability for Chinese overlapped speech. **Processing order has an observable effect.** Under light overlap, two backends x three noise conditions all favor denoise-first L4; the gain is largest under babble, with Whisper -27 and FunASR -48 points. Under heavy overlap, most conditions slightly favor L5. FunASR's independent evidence shows that the order effect is not a Whisper-specific property. **Separation gains depend on backend weaknesses.** Whether separation helps depends on the backend's noise weakness. Whisper gains only slightly under clean / heavy, while FunASR does not gain under clean / heavy but gains strongly under white_0dB, reducing L1's 76.9 / 88.7 to 44.3 / 55.5 after separation. The reason is that FunASR itself is more sensitive to white noise (exp1), which echoes the finding that the two engines have opposite weaknesses.

Therefore, H3 should be upgraded from a degenerate-case negative observation to a boundary condition supported by heterogeneous backends: on short clips where separation fails entirely, the order is invisible; after providing enough duration, the order does matter. Under light overlap, denoise-first L4 is preferred, with both backends agreeing in 6/6 cases. This partially restores the initial hypothesis that denoising first can make separation easier. The caveat is that each cell has N=5 and is qualitative in scale; under heavy conditions some chains are close to a performance floor, making differences unreliable. The strongest evidence is L4 over L5 under light overlap across both backends, plus the positive gain of FunASR x white.

---

## 8. Cross-Experiment Analysis

The six experiments above answer separate questions about noise, denoising, overlap, LLM correction, VAD, emotion, and length boundaries. To avoid leaving the report as a list of results, this section combines the experiments: which processing steps are actually useful, which ones mainly add complexity, and why direct ASR becomes such a strong baseline.

### 8.1 Which Processing Steps Are Actually Useful?

The results support a conservative engineering conclusion: more front-end processing is not necessarily better, and a component should be added only under explicit trigger conditions. In most cases, direct ASR is not only the simplest path, but also the most stable, most efficient, and least likely to introduce artifacts. The table below condenses the experiments into scenario-based routing recommendations.

| Scenario | Recommended path | Evidence |
|---|---|---|
| clean / no overlap / light overlap | L1 direct ASR | In exp2, L1 has the lowest CER in 13 of 15 overlap conditions; unnecessary separation clearly introduces over-separation artifacts |
| white noise + low SNR + FunASR | denoise -> ASR | In exp1, `FunASR x white x low SNR` is the only stable positive-gain denoising condition |
| heavy synchronous overlap | try separation | Direct ASR tends to miss synchronous speakers; separation can occasionally recover lost content under heavy overlap |
| babble + Whisper | direct ASR, optionally with VAD as boundary protection | Babble easily induces Whisper hallucination; denoising / separation may introduce additional artifacts |
| obviously irrelevant or subtitle-style output | LLM filter | Exp3/exp4 show that the LLM is better at rejecting hallucination than restoring acoustically lost words |
| emotion or speaking style must be preserved | avoid excessive preprocessing | Exp5 shows that denoising / separation flatten angry and other paralinguistic information |

### 8.2 Accuracy-Efficiency Trade-Off

Looking only at CER can overestimate complex chains, because denoising and separation add RTF cost, and each additional module creates another opportunity for artifacts and distribution shift. Combining `results/tradeoff_summary.md` and `results/tradeoff.png`, no complex path stably falls into L1's "faster and more accurate" region.

![Accuracy-efficiency trade-off](results/tradeoff.png)

The rule version of path selection can be summarized as:

```text
babble and low SNR      -> direct ASR; denoising may weaken target speech; Whisper may need VAD as boundary protection
white and low SNR       -> FunASR + neural denoise; the only stable positive-gain combination
overlap no / light      -> direct ASR; unnecessary separation mainly adds artifacts
overlap heavy           -> consider separation only for severe interruptions, and expect limited gains
suspected hallucination -> LLM filters output to [unrecognizable], not expected to correct words
```

### 8.3 Why Is Direct ASR Often Best?

The advantage of direct ASR is not that it "does no explicit processing." Rather, modern ASR backends have already absorbed large amounts of noise and accent variation. As long as the input has not moved far outside the training distribution, the benefit of additional front-end processing is easily outweighed by its side effects. Denoising may weaken target speech or introduce spectral artifacts. Separation can damage single-channel content under short clips and light overlap. VAD is unstable around speech-like babble. LLM correction cannot recover information already lost during acoustic decoding. The experiments above provide direct evidence for each of these four side effects, and together they explain why direct ASR wins so often.

Therefore, the final message is not "never preprocess." It is a narrower and more useful rule: treat direct ASR as a strong baseline, and add a front-end module only when noise type, overlap degree, and backend weakness jointly point to that module. This is the value of studying negative results.

---

## 9. Discussion

### 9.1 Expected Findings

Several results are consistent with the initial project expectations and prior literature. Lower SNR generally raises CER. Denoising under clean or high-SNR conditions can hurt ASR because of artifacts. LLM-based correction cannot recover words already lost in the acoustic stage and can only act as a filter when hallucination is obvious. Forced separation under no overlap or light overlap can create over-separation. These findings are not the most surprising part of the project, but they form the evidential base for the main conclusions.

### 9.2 Unexpected Findings

**Unexpected finding 1: opposite backend robustness profiles.** The most surprising result is that the two backends have opposite weaknesses. Whisper is robust to white noise but more sensitive to babble, where hallucination is easier to trigger. FunASR shows the reverse pattern. The two degradation curves cross near 0 dB. This means that using only one model can easily mistake a model-specific weakness for a universal ASR weakness. Related to this, complex preprocessing generally produces negative gains on short clips: separation and denoising mostly hurt 2-4 s debate clips, and only heavy overlap gives limited positive gains.

**Unexpected finding 2: length dependence of processing order.** Processing-order differences emerge with longer audio and are robust across engines. On short clips, L4 and L5 differ little because separation has already failed. After exp6 extends audio to around 12 s, the difference becomes visible: light overlap should be denoised first (L4). Whisper and FunASR agree in 6/6 conditions, and the gain is largest under babble (-27 / -48 points). At the same time, whether separation itself gives positive gains changes with the backend's noise weakness. FunASR x white gains from separation, echoing the opposite-weakness finding (see 7.7 / 8.1).

**Unexpected finding 3: processing costs spill beyond text.** Other unexpected findings also point to the theme that processing has costs. Denoising changes Whisper's error pattern, sometimes turning hallucination into silence through many empty transcripts. VAD has clear boundaries: it suppresses white noise and silence (100% -> 0%) but not babble (67% -> 67%). The LLM is a filter, not a corrector: it rejects hallucination but cannot repair fluent wrong words that were acoustically lost. Preprocessing also systematically flattens emotion such as anger (exp5).

### 9.3 Alignment with Literature: Confirmed, Narrowed, and New

The full itemized alignment is in `results/literature_support.md`. Here, the findings are grouped into three categories. **Strongly confirmed:** Whisper non-speech hallucination (2501.11378), denoising hurting ASR (2201.06685 / 2512.17562), over-separation (2106.00949 / 2503.17886), and LLM as filter (2409.09785 / 2505.24347). **Claims to narrow:** first, "processing order does not matter" has no universal support in the literature. In this project, short-clip spk CER of 84-88% shows that separation simply fails to separate two channels, so the lack of L4/L5 difference is a degenerate case and cannot be generalized. After exp6 lengthens audio to around 12 s, the boundary becomes clear: under light overlap, denoise-first L4 is preferred, and Whisper and FunASR agree across six conditions (see 7.7). Second, "whether white or babble is more damaging" depends on the SNR range and cannot be written as a monotonic relation. **New observations:** FunASR is more robust to babble than Whisper (there is no direct Whisper vs Paraformer comparison in the literature), VAD's 67% -> 67% behavior on babble, and separation failure caused by 2-4 s short clips. Figure 9.3 visualizes these three categories.

![Literature alignment map](results/report_literature_map.svg)

---

## 10. Limitations

The conclusions of this project should be understood within explicit boundaries. The corpus mainly consists of 2-4 s Chinese short clips, and the sample size is limited: 26 clean clips plus synthetic overlap data and five long-audio samples in exp6. The conclusions should not be directly extrapolated to longer audio, multilingual settings, or large-scale meeting scenarios. MossFormer2 generally fails on this data, so the separation-related conclusions in exp2 are degraded observations under the premise that separation is mostly unusable. Exp6 confirms that even after lengthening to around 12 s, separation still fails in most conditions, but this does not mean that all separation models will fail on all Chinese overlapped speech. The real recording spot-check lacks word-level ground truth and can only provide qualitative evidence, unlike the controllable synthetic data where CER can be computed strictly. The emotion experiment is exploratory: drift in emotion2vec P(angry) shows that preprocessing affects paralinguistic representations, but it is not equivalent to a full SER benchmark. In addition, this project does not cover beamforming, echo cancellation, mobile deployment, or model training; those belong to larger engineering systems and are outside the scope of this study.

---

## 11. Future Work

Future work can proceed in several directions. First, the data boundary should be expanded by adding longer audio, more speakers, and more real-world recordings, and by adding word-level annotations to real recordings so that CER can be evaluated there as well. Second, front-end models should be replaced and expanded: more speech separation and speech enhancement models should be compared to determine whether MossFormer2's failure comes from the model, the data, or the short-clip setting. Third, the rule-based strategy can be turned into automatic path selection. Signals such as ASR confidence, VAD confidence, noise estimation, and overlap detection can be combined so that the system first decides whether processing is needed and then selects among L1-L5. Fourth, the emotion dimension should be strengthened by using labeled emotion corpora such as ESD, CASIA, and RAVDESS to plot emotion x CER degradation curves, testing whether "denoising flattens emotion" still holds on standard SER data. Finally, the structured output layer can be improved by sending `[emotion]`, speaker, ASR text, and hallucination flags together to the LLM, producing structured records better suited to the debate secretary scenario.

---

## 12. Reproducibility

The core output of this project is not an interactive demo system, but a controlled experimental and analytical evidence chain around the question "when is front-end preprocessing useful or harmful under noisy and overlapped speech?" Environment installation, data directories, and minimal running steps are in `README.md`; daily experimental process, numerical changes, pitfalls, and design decisions are in `LOG.md`; final report statistics, figures, failure cases, and literature alignment are concentrated in `results/`. From the perspective of reproducibility, the project contains four classes of materials.

| Material | Path | Role |
|---|---|---|
| Experiment log | `LOG.md` | Records the complete process from hypothesis proposal, tool trial, data-plan adjustment, to exp6 boundary validation |
| Core scripts | `src/` | Generate noise, create overlapped speech, run ASR / denoising / separation / LLM correction, and compute CER and RTF |
| Manual reference text | `refs/` | Stores manually corrected ground truth, avoiding bias from directly using Whisper drafts |
| Results and figures | `results/` | Stores experiment CSVs, summary tables, degradation curves, processing-chain comparisons, trade-off plots, and emotion-drift figures |

Therefore, the report focuses on explaining what the project actually did: first define falsifiable hypotheses, then gradually test them through controlled noise, controlled overlap, processing-chain ablations, real recording spot-checks, and long-audio boundary experiments. Reproducibility materials support the traceability of conclusions rather than serving as extra feature demos.

---

## 13. Conclusion

This project ultimately answers a seemingly simple but boundary-sensitive question: under noise and overlapped speech, when is local audio preprocessing worth doing? The experiments show that on 2-4 s Chinese debate clips, the simplest direct ASR path is usually best under the combined trade-off of accuracy and efficiency. It achieves the lowest CER in 13 of 15 overlap conditions. Complex front ends only offset their additional cost under narrow conditions, such as denoising for `FunASR x white x low SNR` and separation under heavy synchronous overlap.

More importantly, the project shows that adding processing modules is not the same as making the system more robust. Denoising can create artifacts or weaken target speech. Separation can introduce over-separation on short clips. VAD has clear limits on babble. LLM correction cannot restore information already lost acoustically. Meanwhile, we find several results not preset at project start: Whisper and FunASR have opposite noise robustness profiles; FunASR is stronger than Whisper under babble; denoising and separation affect not only text accuracy but also flatten emotion cues such as anger.

Therefore, the conclusion is not "do not preprocess." It is "prove that preprocessing is worth doing first." **Final conclusion:** negative results are core research results here. Knowing when not to denoise, when not to separate, and when not to let an LLM rewrite ASR text is as important as knowing when a module is useful.

---

## 14. References

1. Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) - arXiv:2212.04356
2. Whisper-AT: Noise-Robust ASR are Strong Audio Event Taggers - arXiv:2307.03183
3. Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio - arXiv:2501.11378
4. Calm-Whisper: Reduce Whisper Hallucination on Non-Speech - arXiv:2505.12969
5. openai/whisper Discussion #928 (dataset bias "Amara.org")
6. WhisperX (VAD preprocessing reduces hallucination) - github.com/m-bain/whisperX
7. How Bad Are Artifacts? Analyzing the Impact of Speech Enhancement Errors on ASR - arXiv:2201.06685
8. When De-noising Hurts: Speech Enhancement Effects on Medical ASR - arXiv:2512.17562
9. Should We Always Separate? Switching Between Enhanced and Observed Signals - arXiv:2106.00949
10. Decoupling Speaker Separation and Speech Recognition - arXiv:2503.17886
11. LLM-Based Generative Error Correction: A Challenge and Baselines - arXiv:2409.09785
12. Fewer Hallucinations, More Verification: Three-Stage LLM ASR Correction - arXiv:2505.24347
13. Audio-Visual Efficient Conformer (white vs babble WER) - arXiv:2301.01456
14. Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition - arXiv:2206.08317
15. FunASR: A Fundamental End-to-End Speech Recognition Toolkit - arXiv:2305.11013
16. MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation - arXiv:2312.11825
17. FRCRN: Boosting Feature Representation using Frequency Recurrence for Monaural Speech Enhancement - arXiv:2206.07293
18. ClearerVoice-Studio: Bridging Advanced Speech Processing (FRCRN / MossFormer2 toolkit) - arXiv:2506.19398
19. emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation - arXiv:2312.15185
20. Attention is All You Need in Speech Separation (SepFormer) - arXiv:2010.13154
21. Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation - arXiv:1809.07454
22. DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering - arXiv:2110.05588
