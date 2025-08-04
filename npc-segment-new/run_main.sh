#!/bin/bash

ulimit -n 100000
conda activate dev_torch2
python main.py