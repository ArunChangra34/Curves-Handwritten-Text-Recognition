"""Plug-in for real datasets (IAM words, Devanagari word sets...).
Labels file: one line per image, 'relative/path.png<TAB>text<TAB>script' (script = en|hi|mix).
python real_data.py ROOT labels.tsv out_name   -> ../data/out_name.npz (same format as synthetic data)"""
import sys, os, numpy as np
from PIL import Image
from imgproc import to_model_input
root, lab, out = sys.argv[1:4]
X, T, S = [], [], []
for line in open(lab, encoding="utf-8"):
    p, t, s = line.rstrip("\n").split("\t")
    X.append((to_model_input(Image.open(os.path.join(root, p))) * 255).astype(np.uint8)); T.append(t); S.append(s)
np.savez_compressed(f"../data/{out}.npz", x=np.stack(X), t=np.array(T), s=np.array(S)); print(out, len(X))
