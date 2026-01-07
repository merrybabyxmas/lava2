#!/bin/bash

GPUS="1,2,3"
PER_GPU=2
SEEDS="10 20 30 40 50"

TASKS="sst2 cola mrpc qnli rte stsb"

bash run_parallel.sh dora $GPUS $PER_GPU "$TASKS" "$SEEDS"
