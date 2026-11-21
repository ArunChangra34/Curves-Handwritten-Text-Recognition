"""python infer.py IMAGE [--script auto|en|hi]"""
import sys, torch
from PIL import Image
from charset import *
from models import CRNN
from imgproc import to_model_input
from train import greedy
img = sys.argv[1]; sc = sys.argv[sys.argv.index("--script") + 1] if "--script" in sys.argv else "auto"
m = CRNN(NUM_CLASSES); m.load_state_dict(torch.load("../checkpoints/crnn_joint.pt")); m.eval()
x = torch.from_numpy(to_model_input(Image.open(img))).view(1, 1, 32, 160)
print(greedy(m(x), script_mask(sc))[0])
