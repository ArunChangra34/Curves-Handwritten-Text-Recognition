"""Unified Devanagari + English character set and script masks for CTC decoding."""
import json, string
LATIN = list(string.ascii_letters + string.digits + ".,-'")
DEVA = [chr(c) for c in list(range(0x0901, 0x0904)) + list(range(0x0905, 0x093A)) + [0x093C]
        + list(range(0x093E, 0x094E)) + list(range(0x0960, 0x0962)) + list(range(0x0966, 0x0970))]
DEVA += [chr(0x0964)]  # danda
CHARS = [" "] + LATIN + DEVA             # index 0 is CTC blank, so char i -> i+1
BLANK = 0
C2I = {c: i + 1 for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

def encode(s): return [C2I[c] for c in s if c in C2I]
def decode_ids(ids): return "".join(CHARS[i - 1] for i in ids if i > 0)
def script_mask(script):
    """Boolean mask over classes allowed for a script: 'en', 'hi' or 'auto'."""
    m = [False] * NUM_CLASSES; m[BLANK] = True; m[C2I[" "]] = True
    for c in CHARS:
        isd = "ऀ" <= c <= "ॿ"
        if script == "auto" or (script == "hi" and (isd or c.isdigit() or c in ".,-'")) or (script == "en" and not isd):
            m[C2I[c]] = True
    return m
def dump(path):
    json.dump({"chars": "".join(CHARS), "blank": BLANK, "num_classes": NUM_CLASSES,
               "masks": {s: script_mask(s) for s in ("auto", "en", "hi")}}, open(path, "w"), ensure_ascii=False)
if __name__ == "__main__": print(NUM_CLASSES, "classes;", len(DEVA), "devanagari symbols")
