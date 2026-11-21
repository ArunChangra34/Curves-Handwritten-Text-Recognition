"""python train.py --model crnn|cnn --scripts both|en|hi --epochs N --out NAME"""
import argparse, json, time, numpy as np, torch, torch.nn.functional as F
from config import *
from charset import *
from models import CRNN, SlotCNN
from metrics import cer_wer
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)

def load(name, scripts="both"):
    d = np.load(f"../data/{name}.npz"); x, t, s = d["x"], list(d["t"]), np.array(d["s"])
    keep = np.ones(len(t), bool) if scripts == "both" else (s == scripts)
    return torch.from_numpy(x[keep]).float().div(255).unsqueeze(1), [t[i] for i in np.where(keep)[0]], s[keep]

def greedy(logits, mask=None):
    if mask is not None: logits = logits.masked_fill(~torch.tensor(mask), -1e9)
    ids = logits.argmax(-1).cpu().numpy(); out = []
    for r in ids:
        prev, s = -1, []
        for i in r:
            if i != prev and i != BLANK: s.append(i)
            prev = i
        out.append(decode_ids(s))
    return out
def slot_decode(logits):
    return ["".join(CHARS[i - 1] for i in r if i > 0).rstrip() for r in logits.argmax(-1).cpu().numpy()]

def predict(model, kind, X, mask=None, bs=256):
    model.eval(); res = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg = model(X[i:i + bs]); res += greedy(lg, mask) if kind == "crnn" else slot_decode(lg)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crnn"); ap.add_argument("--scripts", default="both")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()
    Xtr, Ttr, _ = load("train", a.scripts); Xtr, Ttr = Xtr[:a.n // (1 if a.scripts == "both" else 2)], Ttr[:a.n // (1 if a.scripts == "both" else 2)]; Xva, Tva, _ = load("val", a.scripts)
    model = CRNN(NUM_CLASSES) if a.model == "crnn" else SlotCNN(NUM_CLASSES)
    print(a.out, "params", sum(p.numel() for p in model.parameters()), "train", len(Xtr), flush=True)
    steps = a.epochs * (len(Xtr) // a.bs)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.15)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    enc = [encode(t) for t in Ttr]; log = []; best = 1e9
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Xtr)); t0 = time.time(); tl = 0
        for k in range(len(Xtr) // a.bs):
            idx = perm[k * a.bs:(k + 1) * a.bs]; x = Xtr[idx]
            x = x + 0.03 * torch.randn_like(x) * (torch.rand(len(x), 1, 1, 1) < .5)       # light noise aug
            lg = model(x)
            if a.model == "crnn":
                tg = [torch.tensor(enc[i]) for i in idx.tolist()]
                lp = F.log_softmax(lg, -1).permute(1, 0, 2)
                loss = ctc(lp, torch.cat(tg), torch.full((len(idx),), lg.shape[1], dtype=torch.long), torch.tensor([len(t) for t in tg]))
            else:
                y = torch.zeros(len(idx), MAX_LEN, dtype=torch.long)
                for j, i in enumerate(idx.tolist()): y[j, :len(enc[i])] = torch.tensor(enc[i])
                loss = F.cross_entropy(lg.reshape(-1, NUM_CLASSES), y.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); sch.step(); tl += loss.item()
        cer, wer, em = cer_wer(predict(model, a.model, Xva), Tva)
        log.append(dict(epoch=ep + 1, loss=tl / (k + 1), val_cer=cer, val_wer=wer, secs=time.time() - t0))
        print(log[-1], flush=True)
        if cer < best: best = cer; torch.save(model.state_dict(), f"../checkpoints/{a.out}.pt")
    json.dump(log, open(f"../checkpoints/{a.out}.log.json", "w"))
