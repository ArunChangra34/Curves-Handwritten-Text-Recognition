"""Image -> model input. Mirrored in web/app.js so training and the browser agree."""
import numpy as np
from PIL import Image
from config import IMG_H, IMG_W

def to_model_input(img):
    """PIL image -> float32 array (IMG_H, IMG_W) in [0,1]; ink = high values."""
    from scipy.ndimage import gaussian_filter
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if np.median(g) < 128: g = 255 - g                  # light-on-dark -> dark-on-light
    bg, dark = np.median(g), np.percentile(g, 0.5)
    ink = np.clip((bg - g) / max(bg - dark, 25.0), 0, 1)
    ink = np.where(ink < 0.15, 0, ink)                  # drop paper noise
    m = gaussian_filter(ink, 1.0) > 0.3
    ys, xs = np.where(m)
    if len(xs) > 0:
        pad = 2
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, g.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, g.shape[1])
        ink = ink[y0:y1, x0:x1]
    h, w = ink.shape
    nh = IMG_H - 4
    nw = max(1, min(IMG_W - 8, int(round(w * nh / h))))
    r = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255
    out = np.zeros((IMG_H, IMG_W), np.float32)
    out[2:2 + nh, 4:4 + nw] = r
    return out
