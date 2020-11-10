#!/bin/bash

#PBS -l walltime=50:00:00,select=1:ncpus=20:ngpus=1:mem=50gb
#PBS -N Gradation
#PBS -A ex-msilfver-1-gpu
#PBS -m abe
#PBS -M msilfver@mail.ubc.ca
#PBS -o /project/ex-msilfver-1/gradation/gradation/LSTM_hidden_states/logs
#PBS -e /project/ex-msilfver-1/gradation/gradation/LSTM_hidden_states/logs
#PBS -V 

module load gcc python/3.7.3 cuda/10.0.130 py-pip/19.0.3-py3.7.3 gnuplot/5.2.5-py3.7.3 openblas/0.3.6 py-numpy/1.16.4-py3.7.3 libx11/1.6.7 pbspro/19.1.3 py-scipy/1.2.1-py3.7.3 py-scikit-learn/0.21.2-py3.7.3 git

export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=$HOME/python/libffi/lib64
export LD_RUN_PATH=/$HOME/python/libffi/lib64
export PKG_CONFIG_PATH=$HOME/python/libffi/lib/pkgconfig
 
python3 /project/ex-msilfver-1/gradation/gradation/OpenNMT-py/train.py --config /project/ex-msilfver-1/gradation/gradation/LSTM_hidden_states/config.MODEL.yaml