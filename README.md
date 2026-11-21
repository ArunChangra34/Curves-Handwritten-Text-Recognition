# Curves - Handwritten Text Recognition

Handwritten text recognition for **Devanagari and English** — one CRNN (CNN + BiLSTM + CTC), running entirely in the browser.

- **Live demo:** deploy `web/` to Vercel (see `docs/04_DEPLOYMENT.md`)
- **Documents:** `docs/01_PROJECT_GUIDE.md` (phase-by-phase explanation) · `docs/02_STEP_LOG.md` (every step taken) · `docs/03_REPORT.md` (method, results, analysis) · `docs/04_DEPLOYMENT.md`

## Quick start

```bash
pip install torch wordfreq scipy pillow numpy onnx onnxruntime
cd ml
python make_data.py --n 60000     # synthetic handwriting-style data  → ../data/
./run_all.sh                      # trains the 4 models              → ../checkpoints/
python evaluate.py                # CER / WER tables                 → ../results/results.json
python export_onnx.py             # ONNX + web assets                → ../web/model/
python infer.py ../web/samples/kitab.png --script auto
```

Run the site locally: `cd web && python -m http.server 8000` → http://localhost:8000

## Using real handwriting data
Create a tab-separated file `path<TAB>text<TAB>en|hi|mix`, then `python real_data.py IMAGE_ROOT labels.tsv real_train`. Details in the guide, Phase 10.

## Layout
`ml/` training code · `web/` static site · `docs/` documentation · `data/`, `checkpoints/`, `results/` generated artefacts.

Fonts used for data generation are licensed under the SIL Open Font License (Google Fonts).
