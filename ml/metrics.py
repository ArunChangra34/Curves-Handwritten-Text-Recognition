import numpy as np
def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]
def cer_wer(preds, refs):
    ce = sum(edit(p, r) for p, r in zip(preds, refs)); cn = sum(len(r) for r in refs)
    we = sum(edit(p.split(), r.split()) for p, r in zip(preds, refs)); wn = sum(len(r.split()) for r in refs)
    return 100 * ce / max(cn, 1), 100 * we / max(wn, 1), 100 * np.mean([p == r for p, r in zip(preds, refs)])  # CER, WER, exact-match
