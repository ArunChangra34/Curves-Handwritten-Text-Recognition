"""Synthetic handwriting-style word/phrase generator (fonts + handwriting-like distortions)."""
import os, re, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter
from wordfreq import top_n_list
from config import *
from charset import C2I
from imgproc import to_model_input

FD = os.path.join(os.path.dirname(__file__), "fonts")
_cache = {}
def font(name, size):
    k = (name, size)
    if k not in _cache: _cache[k] = ImageFont.truetype(os.path.join(FD, name), size, layout_engine=ImageFont.Layout.RAQM)
    return _cache[k]

def words():
    en = [w for w in top_n_list("en", 40000) if re.fullmatch(r"[A-Za-z]{1,10}", w)]
    en += [w.capitalize() for w in en[:6000]]
    hi = [w for w in top_n_list("hi", 40000) if re.fullmatch(r"[ऀ-ॣ०-ॿ]{1,10}", w)]
    hi = [w for w in hi if all(c in C2I for c in w)]
    return {"en": en, "hi": hi}

def rand_word(rng, W, script):
    r = rng.random()
    if r < 0.1:   # random character strings force the model to read shapes, not memorise a lexicon
        pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if script == "en" else \
               "कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंँ्०१२३४५६७८९"
        return "".join(rng.choice(pool) for _ in range(rng.randint(2, 7)))
    if r < 0.15 and script == "en": return str(rng.randint(0, 99999))
    return rng.choice(W[script])

def rand_text(rng, W, script):
    """script in en/hi/mix -> (text, label script)"""
    if script == "mix": a, b = (("en", "hi") if rng.random() < .5 else ("hi", "en")); t = rand_word(rng, W, a) + " " + rand_word(rng, W, b)
    else: t = rand_word(rng, W, script) if rng.random() < .75 else rand_word(rng, W, script) + " " + rand_word(rng, W, script)
    return t[:MAX_LEN].rstrip()

def warp(a, rng):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp, per = rng.uniform(0, 2.5), rng.uniform(25, 80)
    yy = yy + amp * np.sin(xx / per * 2 * np.pi + rng.uniform(0, 6.3))          # wavy baseline
    xx = xx + yy * rng.uniform(-0.25, 0.25)                                    # slant / shear
    dx = gaussian_filter(np.random.randn(h, w), 6) * rng.uniform(0, 14)        # elastic jitter
    dy = gaussian_filter(np.random.randn(h, w), 6) * rng.uniform(0, 14)
    return map_coordinates(a, [yy + dy, xx + dx], order=1, mode="constant")

def render(text, fontname, rng):
    f = font(fontname, rng.randint(44, 64))
    tmp = Image.new("L", (1000, 140), 0)
    ImageDraw.Draw(tmp).text((20, 20), text, font=f, fill=255, stroke_width=rng.choice([0, 0, 1, 2]), stroke_fill=255)
    a = np.asarray(tmp, dtype=np.float32) / 255
    ys, xs = np.where(a > .2)
    if len(xs) == 0: a = np.zeros((20, 20), np.float32)
    else: a = np.pad(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 10)
    a = warp(a, rng)
    im = Image.fromarray((a * 255).astype(np.uint8)).rotate(rng.uniform(-4, 4), resample=Image.BILINEAR, expand=True)
    if rng.random() < .4: im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.5, 1.4)))
    bg = rng.randint(200, 255); ink = rng.randint(0, 90)
    arr = np.asarray(im, dtype=np.float32) / 255
    arr = bg - arr * (bg - ink) + np.random.randn(*arr.shape) * rng.uniform(0, 8)
    return to_model_input(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))

def sample(rng, W, script, fonts):
    s = rng.choice(["en", "hi"]) if script == "both" else script
    kind = "mix" if script == "both" and rng.random() < .15 else s
    t = rand_text(rng, W, kind)
    fs = fonts["hi"] if kind in ("hi", "mix") else fonts["en"]
    if kind == "mix": fs = [x for x in fonts["hi"] if x in fonts["en"] or True]
    return render(t, rng.choice(fs), rng), t, ("mix" if kind == "mix" else s)
