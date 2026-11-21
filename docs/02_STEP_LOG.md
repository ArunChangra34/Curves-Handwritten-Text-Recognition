# Curves — Step Log

A chronological record of every step taken, the command or file involved, and what happened — including the problems met and how they were fixed. Date: 20 Nov 2026.

| # | Step | Detail / command | Outcome |
|---|---|---|---|
| 1 | **Chose the project** | CRNN handwriting recogniser; decided on one model for Devanagari **and** English (single shared alphabet) | Scope fixed: word / short-phrase level |
| 2 | **Checked the workspace** | Python 3.13, 2 CPU cores, 7 GB RAM, no GPU, no PyTorch preinstalled | Plan sized for CPU training |
| 3 | **Installed dependencies** | `pip install torch wordfreq` (ONNX, onnxruntime, Pillow already present) | PyTorch 2.15 CPU working |
| 4 | **Looked for real datasets** | Tested reachability of dataset hosts | Only package registries and GitHub reachable → decided on synthetic handwriting-style data, with a plug-in for real data (`real_data.py`) |
| 5 | **Collected fonts** | Downloaded open-licensed (OFL) handwriting-style fonts: Caveat, Indie Flower, Patrick Hand, Kalam (3 weights), Amita, Sahitya, Karma, Sumana, Laila | Fonts in `ml/fonts/`; three kept aside as **held-out test fonts** |
| 6 | **Verified text shaping** | Rendered `हिन्दी किताब Hello` with the Raqm layout engine | Matras and conjuncts render correctly (required for Devanagari) |
| 7 | **Built the alphabet** | `ml/charset.py` | 154 classes = blank + space + 66 Latin/digit/punctuation + 86 Devanagari |
| 8 | **Built the word lists** | `wordfreq` top 40k English / Hindi words, filtered to the alphabet | 40,849 English and 23,401 Hindi word forms |
| 9 | **Wrote the preprocessing** | `ml/imgproc.py` — greyscale, invert-if-dark, ink cut-out, crop, resize to 32×160 | First version amplified paper noise and failed to crop (text came out tiny) |
| 10 | **Fixed preprocessing** | Paper level = median, ink level = 0.5th percentile; drop values < 0.15; bounding box from a smoothed mask | Clean, tightly-cropped inputs |
| 11 | **Wrote the generator** | `ml/synth.py` — font rendering + wavy baseline, slant, elastic jitter, rotation, pen thickness, blur, noise | First version took 100 ms per image |
| 12 | **Sped up the generator** | Crop to the text box *before* warping, smaller canvas | **7.6 ms per image** (13× faster); visual check of 12 samples |
| 13 | **Generated datasets** | `python make_data.py --n 60000` (2 worker processes) | train 60,000 · val 6,000 · test_seen 6,000 · test_held 6,000 (≈ 4 min) |
| 14 | **Wrote the models** | `ml/models.py` — CRNN and plain-CNN baseline | First CRNN: 2.09 M params, 0.80 s per batch of 64 on CPU |
| 15 | **Slimmed the CNN** | Channels 32-64-128-128-256-256 → 16-32-64-64-128-128; baseline dense layer 512 → 256 | CRNN **1.18 M params, 0.31 s/batch**; baseline 3.54 M params, 0.20 s/batch |
| 16 | **Wrote training + metrics** | `ml/train.py`, `ml/metrics.py` (edit distance → CER / WER / exact match) | CTC loss, AdamW, one-cycle LR, best-val checkpointing |
| 17 | **Trained the joint CRNN** | `python train.py --model crnn --scripts both --epochs 8 --n 40000 --out crnn_joint` | Validation CER 93 % → 15.9 % → 8.9 % → 5.9 % → 4.0 % → 3.0 % → … → **2.63 %** after 8 epochs (~3 min/epoch) |
| 18 | **Wrote the web app** | `web/index.html`, `styles.css`, `app.js` — layout, fonts, colours and theme switch taken from the supplied screenshots | Dark and light themes match the reference |
| 19 | **Vendored the runtime** | Copied ONNX Runtime Web (`ort.wasm.min.js` + `.wasm`) into `web/vendor/`; self-hosted Inter and JetBrains Mono | Site has no external dependencies |
| 20 | **Exported to ONNX** | `python export_onnx.py` | PyTorch vs ONNX Runtime max difference **4.8 × 10⁻⁶** |
| 21 | **First browser test** | Headless Chrome via Playwright | "Model failed to load": ONNX Runtime needs an absolute URL for its WebAssembly path |
| 22 | **Fixed the runtime path** | `wasmPaths = new URL("vendor/", location.href).href` | Model loads, 10–40 ms per recognition |
| 23 | **Cross-checked browser vs Python** | Same sample images through `infer.py` and the page | Same output; one demo image (Devanagari "किताब") failed in both |
| 24 | **Found the cause** | The demo image itself was cropped at the bottom (font with tall line metrics) | Regenerated demo images with bounding-box placement; all 5 read correctly in browser and Python |
| 25 | **Trained comparison models** | `./run_all.sh` — plain CNN baseline, English-only CRNN, Devanagari-only CRNN | See report |
| 26 | **Re-ran interrupted trainings** | The English-only and Devanagari-only runs had been cut short; re-ran both (8 epochs each, ≈ 100 s/epoch) | Final validation CER: English-only ≈ 2 %, Devanagari-only 4.3 % |
| 27 | **Evaluated all four models** | `python evaluate.py` on seen-font and held-out-font test sets, per script | `results/results.json` written; joint CRNN CER 2.48 % / 7.73 %, baseline 47 % / 51 % |
| 28 | **Checked the interface against the screenshots** | Rendered the site in dark and light themes at 1440 × 900 and compared colours, fonts and layout | Both themes match the reference |
| 29 | **Wrote the report** | `docs/03_REPORT.md` — data, models, tables, analysis, limitations | Honest scope note: data are synthetic |
| 30 | **Final clean-up** | Removed stray download folders; scanned the whole project for tool/vendor wording | None found; project ready to deploy from `web/` |
