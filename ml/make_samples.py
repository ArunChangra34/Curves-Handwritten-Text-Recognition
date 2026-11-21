"""Render demo images for the web app (held-out fonts, so they were never seen in training)."""
import os, random, numpy as np
from PIL import Image, ImageDraw
import synth
rng = random.Random(3); np.random.seed(3); os.makedirs("../web/samples", exist_ok=True)
S = [("notebook", "PatrickHand-Regular.ttf", "notebook"), ("Forest 24", "PatrickHand-Regular.ttf", "forest24"),
     ("किताब", "Sumana-Regular.ttf", "kitab"), ("भारत", "Laila-Regular.ttf", "bharat"), ("hello नमस्ते", "Laila-Regular.ttf", "mixed")]
for text, f, name in S:
    im = Image.new("L", (900, 400), 0); ImageDraw.Draw(im).text((40, 40), text, font=synth.font(f, 96), fill=255)
    a = np.asarray(im, np.float32) / 255; ys, xs = np.where(a > .2); a = np.pad(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 50)
    a = synth.warp(a, rng)
    im = Image.fromarray((a * 255).astype(np.uint8)).rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, expand=True)
    arr = 245 - np.asarray(im, np.float32) / 255 * 215; arr = arr + np.random.randn(*arr.shape) * 4
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(f"../web/samples/{name}.png")
print("ok")
