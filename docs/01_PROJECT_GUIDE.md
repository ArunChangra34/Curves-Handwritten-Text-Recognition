# Curves — Project Guide

Handwritten text recognition for **Devanagari and English** with one network: a CRNN (CNN + BiLSTM + CTC).
This guide explains the whole project phase by phase — what each phase is for, what was decided, why, and where the code lives.

---

## 0. The idea in one page

| Question | Answer |
|---|---|
| **What does Curves do?** | Takes an image of handwritten text (a word or a short phrase) and returns the text. |
| **Which scripts?** | Devanagari (Hindi) and English (Latin). Both are read by the *same* model; a selector can restrict the output to one script. |
| **Model** | CNN feature extractor → 2-layer bidirectional LSTM → linear classifier → CTC loss. |
| **Baseline** | A plain CNN with no sequence model and no alignment (one softmax per character slot). |
| **Why CTC?** | Handwriting has no per-character labels. CTC learns the alignment between image columns and characters on its own. |
| **Where does it run?** | In the browser (ONNX Runtime Web), deployed as a static site on Vercel. Nothing is uploaded to a server. |

### Repository map

```
curves/
├── ml/                      # everything that trains and evaluates the model
│   ├── config.py            # image size, max length, font splits
│   ├── charset.py           # unified Devanagari + English alphabet, script masks
│   ├── imgproc.py           # image → 32×160 model input (mirrored in web/app.js)
│   ├── synth.py             # handwriting-style image generator
│   ├── make_data.py         # builds train / val / test sets
│   ├── real_data.py         # plug in real datasets (IAM, Devanagari word sets)
│   ├── models.py            # CRNN and plain-CNN baseline
│   ├── train.py             # training loop (CTC)
│   ├── evaluate.py          # CER / WER per model, per script
│   ├── export_onnx.py       # PyTorch → ONNX, copies web assets
│   ├── infer.py             # command-line inference
│   ├── make_samples.py      # demo images for the web page
│   ├── run_all.sh           # trains every model in the report
│   └── fonts/               # handwriting-style fonts (SIL Open Font License)
├── data/                    # generated datasets (.npz)
├── checkpoints/             # trained weights + training logs
├── results/results.json     # evaluation output
├── web/                     # the Vercel site (static)
└── docs/                    # this guide, the step log, the report, extras
```

---

## Phase 1 — Define the task and scope

**Goal.** Decide exactly what the system reads. Curves reads **single words and short phrases (up to 16 characters)** in Devanagari, English, or both on one line.

**Decisions**
- *Word/phrase level, not full pages.* Page-level handwriting needs line segmentation, a separate and much larger problem. Word-level keeps the model small enough to train on a CPU and to run in a browser.
- *One model for two scripts* ("Option A"). Separate models are simpler, but a single shared network is the more interesting design, and it supports mixed-script lines for free.
- *Metrics.* **CER** (character error rate), **WER** (word error rate) and exact-match accuracy. CER = edit distance ÷ reference length.

---

## Phase 2 — The unified character set

**File:** `ml/charset.py`

The network's output layer has one unit per symbol. A *unified* set simply lists both alphabets together:

- index **0** = the CTC *blank* (a "no character here" symbol, explained in Phase 6),
- space,
- Latin letters (a–z, A–Z), digits and `. , - '`,
- Devanagari: chandrabindu/anusvara/visarga, independent vowels, all consonants, nukta, vowel signs (matras), the *halant* (virama), digits ०–९ and the danda.

Total: **154 classes** (153 symbols + blank), of which 86 are Devanagari.

**Why code points, not whole syllables?** Devanagari writes conjuncts such as क्ष by combining `क` + `्` + `ष`. Predicting Unicode code points keeps the alphabet small (86 symbols) and lets the model compose syllables itself. The cost is that the model must learn that a matra can appear before or after the consonant it modifies, which is exactly what the BiLSTM's two-way context helps with.

**Script masks.** For a chosen script (`en`, `hi`, `auto`) we build a boolean mask over the 154 classes. At decoding time, classes outside the mask are set to −∞, so choosing "English" can never emit a Devanagari symbol. `auto` leaves everything open. This is how one model offers an Auto / Devanagari / English selector with no extra weights.

---

## Phase 3 — Data

**Files:** `ml/synth.py`, `ml/make_data.py`, `ml/real_data.py`, `ml/fonts/`

### 3.1 What data is used
Public handwriting datasets were not reachable from the build environment (it could only reach package registries and GitHub), so the training data is **synthetic handwriting-style text** rendered from open-licensed handwriting fonts and then heavily distorted. The pipeline is built so that real data can replace or extend it without changing any other code (Phase 10).

### 3.2 How a sample is made
1. **Pick text.** Real word lists (top 40,000 English and Hindi words by frequency; only words whose every character is in our alphabet). 10 % of samples are *random character strings* so the network learns letter shapes instead of memorising a dictionary. 25 % are two-word phrases; 15 % of "both" samples are mixed-script phrases (one English word + one Hindi word).
2. **Render** with a handwriting-style font (shaping done with HarfBuzz/Raqm, which is essential: Devanagari matras and conjuncts only look right when the text is properly shaped).
3. **Distort** to imitate a human hand: wavy baseline, slant/shear, elastic jitter, rotation (±4°), pen thickness variation, blur, paper tone, ink darkness, sensor noise.
4. **Preprocess** with the *same function used at inference* (Phase 4).

### 3.3 Splits — and the part that keeps the test honest
| Set | Size | Text | Fonts |
|---|---|---|---|
| train | 40,000 used (60,000 generated) | seen | training fonts |
| val | 6,000 | new | training fonts |
| test_seen | 6,000 | new | training fonts |
| **test_held** | 6,000 | new | **fonts never used in training** |

Training fonts — English: Caveat, Indie Flower, Kalam (×2). Devanagari: Kalam (×3 weights), Amita, Sahitya, Karma.
Held-out fonts — English: Patrick Hand. Devanagari: Sumana, Laila.

`test_held` measures *generalisation to a new handwriting style*, which is much closer to real use than testing on the training fonts.

---

## Phase 4 — Preprocessing

**File:** `ml/imgproc.py` (Python) and `web/app.js` (browser) — deliberately the same algorithm.

1. Convert to greyscale.
2. If the image is mostly dark (median < 128), invert it, so ink is always darker than paper.
3. Estimate the paper level (median) and the ink level (0.5th percentile); scale to ink = 1, paper = 0; zero anything below 0.15 (paper noise).
4. Smooth lightly and find the bounding box of the writing; crop to it with a 2-pixel margin. This makes the model independent of where the writing sits on the page.
5. Resize to **28 pixels high**, keeping the aspect ratio (capped at 152 wide), and place it on a 32 × 160 canvas.

Output is a 32 × 160 float image where **ink is bright**. A model that always sees the same layout is far easier to train.

---

## Phase 5 — The models

**File:** `ml/models.py`

### 5.1 CRNN (the main model) — 1.18 M parameters
```
image 1×32×160
 → 6 conv blocks (3×3 conv + BatchNorm + ReLU, 4 max-pools)   → 128 channels × 2 rows × 40 columns
 → treat each of the 40 columns as one time step (stack channels × rows = 256 features)
 → linear 256 → 256
 → 2-layer bidirectional LSTM, 128 hidden units per direction
 → linear → 154 class scores for each of the 40 steps
```
- **CNN** learns local stroke features (curves, loops, the Devanagari headline, matras).
- Pooling shrinks height 16× but width only 4×, so the image becomes a **sequence of 40 steps** — enough room for 16 characters.
- **BiLSTM** reads that sequence left-to-right and right-to-left, so each step knows its context. This matters for Devanagari, where a vowel sign can sit before the consonant it belongs to.

### 5.2 Plain CNN baseline — 3.54 M parameters
Same convolutional backbone, then flatten → dense layer → **16 independent softmax heads** (one per character position). It has no sequence model and no alignment: character *k* must appear at slot *k*. It is the "what if we just used a CNN" comparison.

### 5.3 Single-script CRNNs
The same CRNN, trained only on English or only on Devanagari. They answer: *does sharing one network across scripts help or hurt?*

---

## Phase 6 — Training with CTC

**File:** `ml/train.py`

### 6.1 CTC in plain words
The network outputs 40 predictions (one per column), but the label has only a few characters. **CTC (Connectionist Temporal Classification)** sums over *every* way the label could be stretched across the 40 steps, using two rules: a **blank** may fill any step, and **repeated symbols are merged**. Example: `h h - e e l - l l o` → `h e l l o` (the blank between the two `l` steps keeps both letters). Training maximises the total probability of the correct label, so no one ever has to mark where each letter is.

**Decoding** is greedy: take the best class at each step, merge repeats, drop blanks.

### 6.2 Settings
| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | one-cycle schedule, peak 2e-3 |
| Batch size | 64 |
| Epochs | 8 |
| Gradient clipping | 5.0 |
| Extra augmentation | light Gaussian noise on half the batches |
| Model selection | best validation CER is saved |

`run_all.sh` trains all four models in sequence (joint CRNN, CNN baseline, English-only CRNN, Devanagari-only CRNN).

---

## Phase 7 — Evaluation

**File:** `ml/evaluate.py` → `results/results.json`

For each model, on `test_seen` and `test_held`, and per script group (English, Hindi, Mixed, All):
CER, WER, exact-match, and CPU milliseconds per image. For the joint model we also report decoding **with the script selector** (the true script is used as the mask) versus fully automatic decoding. A list of error examples is saved to understand failure modes.

The full numbers and analysis are in `03_REPORT.md`.

---

## Phase 8 — Export to ONNX

**File:** `ml/export_onnx.py`

PyTorch weights are converted to ONNX (opset 17). The script runs the same input through both PyTorch and ONNX Runtime and prints the maximum difference — it must be ~1e-5 or smaller. It also writes `charset.json` (alphabet and script masks) and `results.json` to `web/model/`.

---

## Phase 9 — The web application

**Folder:** `web/` — plain HTML/CSS/JS, no build step.

| File | Role |
|---|---|
| `index.html` | Three views: Recognize, Model, Pipeline |
| `styles.css` | Design system: Inter + JetBrains Mono, cyan accent, dark/light themes |
| `app.js` | Theme toggle, preprocessing, ONNX inference, CTC decoding, drawing pad, results |
| `model/` | `curves.onnx`, `charset.json`, `results.json` |
| `vendor/` | ONNX Runtime Web (WebAssembly, single-threaded) |
| `fonts/` | Inter and JetBrains Mono, self-hosted |
| `samples/` | Demo images (rendered from held-out fonts) |
| `vercel.json` | Clean URLs + cache headers |

**Features:** drag-and-drop / click / clipboard-paste upload; freehand drawing pad with auto-recognition; Auto / Devanagari / English selector; confidence, detected script, latency; a view of the exact 32 × 160 image the model sees; light and dark themes (remembered between visits).

Because inference is local, the site needs no backend and no API keys, and the user's images never leave their device.

---

## Phase 10 — Going further with real handwriting

Synthetic fonts are a good start but real pens vary more. To improve real-world accuracy:

1. Get a word-level dataset (e.g. **IAM** words for English; a Devanagari word/line dataset). Isolated-character datasets are *not* suitable — the model needs words.
2. Write a tab-separated labels file: `relative/path.png<TAB>text<TAB>en|hi|mix`.
3. Run `python real_data.py ROOT labels.tsv real_train` (and `real_val`, `real_test`) — it applies the same preprocessing and writes the same `.npz` format.
4. Fine-tune: load `crnn_joint.pt`, train a few epochs on real data (mix 50 % synthetic to avoid forgetting).
5. Re-run `evaluate.py` and `export_onnx.py`, replace `web/model/`.

See `04_DEPLOYMENT.md` for deployment and `README.md` for commands.

---

## Glossary

- **CER / WER** — character / word error rate: edit distance divided by reference length.
- **CNN** — convolutional network; learns visual features.
- **BiLSTM** — recurrent layer reading a sequence both ways.
- **CTC** — loss that learns alignment between a long output sequence and a short label.
- **Blank** — CTC's "nothing here" symbol, index 0.
- **Matra** — Devanagari vowel sign attached to a consonant.
- **Halant (virama)** — Devanagari sign that joins consonants into conjuncts.
- **ONNX** — portable model format; **ONNX Runtime Web** runs it in a browser.

## Questions you should be ready to answer

1. *Why CTC instead of fixed character slots?* Handwriting length and spacing vary; CTC removes the need for per-character labels or fixed positions (see the baseline's results).
2. *Why a BiLSTM after the CNN?* Characters depend on neighbours — especially Devanagari matras — so each time step needs left and right context.
3. *How can one model read two scripts?* Their symbols are disjoint, so a single softmax over the union is unambiguous; masks restrict the choice when the script is known.
4. *Why test on held-out fonts?* Otherwise the model could memorise the fonts' shapes; held-out fonts measure transfer to new handwriting.
5. *What are the limits?* Words and short phrases only; trained on synthetic handwriting-style data, so real-pen accuracy needs fine-tuning on real datasets (Phase 10).
