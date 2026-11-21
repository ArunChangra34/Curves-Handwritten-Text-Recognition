# Curves — Technical Report

**Handwritten word recognition for Devanagari and English with a single CRNN**
Date: 20 Nov 2026 · Code: `ml/` · Raw numbers: `results/results.json`

## Abstract
Curves recognises handwritten words and short phrases in **Devanagari and English** with one network: a convolutional feature extractor, a bidirectional LSTM and a CTC output layer over a unified 154-symbol alphabet. On a held-out-font test set the joint model reaches **7.7 % character error rate (CER)**, against 50.6 % for a plain CNN baseline. It also beats two single-script CRNNs of identical size on their own script, which shows that sharing one model between scripts helps rather than hurts. The model (4.7 MB) is exported to ONNX and runs in the browser.

**Scope caveat.** All training and test images are *synthetic handwriting-style* renderings (see §2). Real-handwriting accuracy has not been measured here; §6 lists how to measure it with the supplied loader.

## 1. Problem
Given an image of one handwritten word or short phrase, output its text. Input scripts: Devanagari (Hindi), Latin (English), or both in one line. Lines are not segmented into characters; the model must learn the alignment itself — the reason CTC is used.

## 2. Data
No real handwriting corpus (IAM, Kaggle Devanagari sets) could be reached from the build environment, so training data were generated.

| Item | Detail |
|---|---|
| Text | 40k most frequent English and Hindi words (word-frequency lists), filtered to the alphabet; mixed-script phrases made by joining one word of each |
| Rendering | 9 open-licensed handwriting-style fonts (Raqm shaping, so matras and conjuncts are correct) |
| Distortion | wavy baseline, slant, elastic jitter, rotation, pen-thickness change, blur, paper noise |
| Size | train 60,000 · val 6,000 · test_seen 6,000 · test_held 6,000; models trained on 40,000 (single-script models on the matching ~20,000 subset) |
| Held-out fonts | 3 fonts never used for train or validation → `test_held`, the closest available proxy for "unseen handwriting" |
| Image | greyscale, 32 × 160 after ink-cropping |

## 3. Models
| Model | Description | Parameters |
|---|---|---|
| **CRNN (joint)** | 6 conv blocks → height-collapsed feature sequence → 2-layer BiLSTM → linear → CTC; trained on both scripts | 1.18 M |
| CRNN English-only / Devanagari-only | same architecture and recipe, one script | 1.18 M each |
| CNN baseline | same style of conv stack, but a dense layer predicts a fixed number of character slots (no recurrence, no CTC alignment) | 3.54 M |

Training: AdamW, one-cycle learning rate, 8 epochs, CPU only (≈ 2–3 min per epoch), best-validation checkpoint kept. Metrics: CER and WER from edit distance; exact-match is the share of images read perfectly.

## 4. Results

### 4.1 Overall (all test images)
| Model | test_seen CER | test_seen WER | test_held CER | test_held WER | test_held exact |
|---|---|---|---|---|---|
| **CRNN joint** | **2.48** | **10.67** | **7.73** | **30.49** | **63.4 %** |
| CNN baseline | 47.32 | 90.47 | 50.59 | 93.57 | 7.6 % |
| CRNN English-only | 50.58 | 57.98 | 55.68 | 76.65 | 19.7 % |
| CRNN Devanagari-only | 56.40 | 60.59 | 58.37 | 68.39 | 26.4 % |

(The single-script models score badly "overall" only because they are tested on a script they never saw; the fair comparison is next.)

### 4.2 Per script, held-out fonts (CER %, lower is better)
| Model | English | Devanagari | Mixed |
|---|---|---|---|
| **CRNN joint** | **8.12** | **7.43** | **7.49** |
| CRNN English-only | 12.56 | 104.59 | 55.50 |
| CRNN Devanagari-only | 101.66 | 9.40 | 58.37 |
| CNN baseline | 43.17 | 52.90 | 60.31 |

On seen fonts the joint model scores CER 1.02 (English), 3.48 (Devanagari) and 3.54 (mixed); the specialists score 1.18 and 4.17 on their own script. Mixed-script lines are only possible with the joint model (specialists: exact-match 0 %).

### 4.3 Script selector
Restricting decoding to the chosen script (the UI's "Devanagari" / "English" buttons) changes held-out CER from 7.73 to 7.71 — practically nothing. The network almost never emits the wrong script, so the selector is a convenience and a safety net, and **Auto** is just as accurate.

### 4.4 Speed and size
≈ 1.8 ms per image on one CPU in PyTorch; 10–40 ms end-to-end in the browser including preprocessing. ONNX output matches PyTorch to 4.8 × 10⁻⁶.

## 5. Analysis
1. **CTC + recurrence is what matters.** Same-ish convolutional features, but the slot-based CNN baseline never learns where letters are (CER ≈ 50 %), while the CRNN solves the alignment problem.
2. **One model for two scripts is better, not worse.** On unseen fonts the joint model beats the specialists by 4.4 points (English) and 2.0 points (Devanagari) of CER. Seeing twice the variety of strokes and fonts acts as regularisation; the shared recurrent layers also learn script-independent sequence structure.
3. **Generalisation gap.** CER rises from 2.5 % (seen fonts) to 7.7 % (unseen fonts): the network partly memorises font styles. More fonts, more distortion types and real data are the main levers.
4. **Error patterns** (from `results.json`): dropped or merged matras (सूर्यकांत → सर्यकांत; ट्रेडर्स → ट्रडर्स), lost nukta dots (ज़ाहिर → जाहिर), and look-alike Latin letters (l/L, i/d). Devanagari errors are mostly diacritics, which are small relative to the 32-pixel height.

## 6. Limitations and next steps
- **Synthetic data only.** Treat the numbers as a controlled comparison between architectures, not a claim about real handwriting. Expect higher error on real pages.
- Words and short phrases only; full-page documents need line segmentation first.
- CTC predicts Unicode code points, so a recogniser output can in principle be an invalid combination (e.g. a matra with no consonant). A dictionary or language-model decoder would fix many of the errors in §5.4.
- **Measuring on real handwriting:** `ml/real_data.py` converts any dataset listed as `path<TAB>text<TAB>script` into the same tensors; evaluate with `evaluate.py`, then fine-tune the joint model on it (guide, Phase 10). Suggested sources: IAM (English), Kaggle/academic Devanagari word datasets.
- Ablations not run: depth of BiLSTM, data-size scaling, augmentation strength.

## 7. Reproducibility
```
cd ml && python make_data.py --n 60000 && ./run_all.sh && python evaluate.py && python export_onnx.py
```
Fonts are SIL OFL (Google Fonts). Random seeds are fixed in `config.py`; training is on CPU, so results reproduce to within small numerical differences.
