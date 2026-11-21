"""Evaluate every trained model on seen-font and held-out-font test sets, per script. Writes ../results/results.json"""
import json, os, time, numpy as np, torch
from charset import *
from models import CRNN, SlotCNN
from train import load, predict
from metrics import cer_wer
torch.set_num_threads(2); os.makedirs("../results", exist_ok=True)
RUNS = {"crnn_joint": ("crnn", "both"), "cnn_baseline": ("cnn", "both"), "crnn_en": ("crnn", "en"), "crnn_hi": ("crnn", "hi")}
R = {}; examples = []
for name, (kind, sc) in RUNS.items():
    p = f"../checkpoints/{name}.pt"
    if not os.path.exists(p): continue
    m = CRNN(NUM_CLASSES) if kind == "crnn" else SlotCNN(NUM_CLASSES); m.load_state_dict(torch.load(p)); m.eval()
    R[name] = {"params": sum(x.numel() for x in m.parameters())}
    for ts in ("test_seen", "test_held"):
        X, T, S = load(ts); t0 = time.time(); P = predict(m, kind, X); R[name]["ms_per_image"] = 1000 * (time.time() - t0) / len(X)
        for grp in ("all", "en", "hi", "mix"):
            k = np.ones(len(T), bool) if grp == "all" else (S == grp)
            if k.sum() == 0: continue
            c, w, e = cer_wer([P[i] for i in np.where(k)[0]], [T[i] for i in np.where(k)[0]])
            R[name][f"{ts}/{grp}"] = dict(cer=round(c, 2), wer=round(w, 2), exact=round(e, 2), n=int(k.sum()))
        if name == "crnn_joint":   # script selector: restrict decoding to the chosen script when it is known
            Pm = []
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    lg = m(X[i:i + 256])
                    for j in range(len(lg)):
                        from train import greedy
                        Pm += greedy(lg[j:j + 1], script_mask(S[i + j] if S[i + j] != "mix" else "auto"))
            c, w, e = cer_wer(Pm, T); R[name][f"{ts}/all_with_selector"] = dict(cer=round(c, 2), wer=round(w, 2), exact=round(e, 2))
            if ts == "test_held":
                wrong = [(T[i], Pm[i], S[i]) for i in range(len(T)) if T[i] != Pm[i]][:25]; examples = wrong
    print(name, json.dumps(R[name])[:300], flush=True)
json.dump({"metrics": R, "error_examples": examples}, open("../results/results.json", "w"), ensure_ascii=False, indent=1)
