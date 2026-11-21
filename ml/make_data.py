"""Build cached datasets: python make_data.py [--real LABELS.tsv]  (synthetic by default)."""
import argparse, numpy as np, random, os
from multiprocessing import Pool
from config import *
import synth
W = None
def work(a):
    n, script, fonts, seed = a
    global W
    if W is None: W = synth.words()
    rng = random.Random(seed); np.random.seed(seed % 2**32)
    X, T, S = [], [], []
    for _ in range(n):
        x, t, s = synth.sample(rng, W, script, fonts); X.append((x * 255).astype(np.uint8)); T.append(t); S.append(s)
    return np.stack(X), T, S
def build(name, n, script, fonts, seed):
    chunks = [(n // 40, script, fonts, seed * 1000 + i) for i in range(40)]
    with Pool(2) as p: res = p.map(work, chunks)
    X = np.concatenate([r[0] for r in res]); T = sum([r[1] for r in res], []); S = sum([r[2] for r in res], [])
    np.savez_compressed(f"../data/{name}.npz", x=X, t=np.array(T), s=np.array(S)); print(name, X.shape, flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40000); a = ap.parse_args()
    os.makedirs("../data", exist_ok=True)
    build("train", a.n, "both", TRAIN_FONTS, 1)
    build("val", a.n // 10, "both", TRAIN_FONTS, 2)
    build("test_seen", a.n // 10, "both", TRAIN_FONTS, 3)       # new text, same fonts
    build("test_held", a.n // 10, "both", HELD_FONTS, 4)        # new text, new fonts
