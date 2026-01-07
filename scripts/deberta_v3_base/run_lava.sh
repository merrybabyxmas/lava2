#!/bin/bash

GPUS="0,1,2,3"
PER_GPU=4
SEEDS="17 20 23 26"
TASKS="mrpc sst2 cola qnli rte stsb mnli qqp"
# TASKS="mrpc"


bash /home/dongwoo38/PiSSA/scripts/deberta_v3_base/run_parallel.sh lava $GPUS $PER_GPU "$TASKS" "$SEEDS"



