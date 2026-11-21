#!/bin/bash
# Trains every model used in the report, sequentially (2 CPU cores).
cd "$(dirname "$0")"; mkdir -p ../checkpoints
python3 train.py --model crnn --scripts both --epochs 8 --n 40000 --out crnn_joint
python3 train.py --model cnn  --scripts both --epochs 8 --n 40000 --out cnn_baseline
python3 train.py --model crnn --scripts en   --epochs 8 --n 40000 --out crnn_en
python3 train.py --model crnn --scripts hi   --epochs 8 --n 40000 --out crnn_hi
