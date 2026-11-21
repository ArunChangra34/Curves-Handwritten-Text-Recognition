# Curves — Deployment on Vercel

`web/` is a static site: no build step, no server, no environment variables.

## Option A — Vercel CLI
```bash
npm i -g vercel
cd curves/web
vercel            # first run: log in, accept the defaults (no framework, output dir ".")
vercel --prod     # publish to production
```

## Option B — GitHub + dashboard
1. Push the `curves/` folder to a GitHub repository.
2. In Vercel choose **Add New → Project** and import the repository.
3. Set **Root Directory** to `web`, **Framework Preset** to *Other*, leave build and output settings empty.
4. Deploy. Every push redeploys automatically.

## What gets served
| Path | Size | Notes |
|---|---|---|
| `model/curves.onnx` | ≈ 4.7 MB | the network |
| `vendor/ort-wasm-simd-threaded.wasm` | ≈ 15 MB | ONNX Runtime engine (cached for a week via `vercel.json`) |
| `fonts/` | ≈ 1 MB | Inter + JetBrains Mono |

First load downloads ≈ 21 MB; later visits come from the browser cache. Everything runs on the visitor's device, so hosting cost is effectively zero.

## Updating the model
```bash
cd ml
python train.py ...          # or fine-tune on real data
python evaluate.py
python export_onnx.py        # overwrites web/model/*
cd .. && vercel --prod
```

## Troubleshooting
- **"Model failed to load"** — open the browser console; the usual cause is `model/curves.onnx` or `vendor/*.wasm` not deployed (check *Root Directory = web*).
- **Blank Devanagari text** — the visitor's device lacks a Devanagari font; all current Windows, macOS, Android and iOS versions include one.
- **Slow first recognition** — the WebAssembly engine is compiled on first use; later recognitions take 10–40 ms.
