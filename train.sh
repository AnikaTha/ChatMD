#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

jupyter nbconvert ./notebooks/tiny_llama.ipynb --to python

python ./notebooks/tiny_llama.py > train.out