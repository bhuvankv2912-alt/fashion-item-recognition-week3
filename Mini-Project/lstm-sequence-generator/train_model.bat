@echo off
title Train LSTM Sequence Generator
echo ===================================================
echo   Training LSTM Deep Sequence Generator
echo ===================================================
python train.py --epochs 20 --batch_size 32 --seq_length 48 --hidden_dim 256
pause
