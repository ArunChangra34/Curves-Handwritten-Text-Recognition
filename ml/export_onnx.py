"""Export the joint CRNN to ONNX and copy web assets."""
import json, shutil, numpy as np, torch, onnx, onnxruntime as ort
from charset import *
from models import CRNN
from config import IMG_H, IMG_W
m = CRNN(NUM_CLASSES); m.load_state_dict(torch.load("../checkpoints/crnn_joint.pt")); m.eval()
x = torch.rand(1, 1, IMG_H, IMG_W)
torch.onnx.export(m, x, "../web/model/curves.onnx", input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
s = ort.InferenceSession("../web/model/curves.onnx"); a = s.run(None, {"image": x.numpy()})[0]; b = m(x).detach().numpy()
print("onnx max abs diff", float(np.abs(a - b).max()), a.shape)
dump("../web/model/charset.json")
try: shutil.copy("../results/results.json", "../web/model/results.json")
except FileNotFoundError: pass
