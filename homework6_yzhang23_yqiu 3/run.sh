#!/usr/bin/env bash
# Usage:
#   ./run.sh
#

LEARNING_RATE=(0.001 0.005 0.01 0.05 0.1 0.5)
NUM_HIDDEN=(30 40 50)
BATCH_SIZE=(16 32 64 128 256)
NUM_EPOCH=(50 100 200)

## First, tune LEARNING RATE and get the best lr: 0.05
LEARNING_RATE=(0.001 0.005 0.01 0.05 0.1 0.5)
## NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE, NUM_EPOCH
##for lr in ${LEARNING_RATE[@]}
##do
##    python mlp.py 40 $lr 16 100
##done



##Second, tune NUM_HIDDEN and get the best NUM_HIDDEN:
## NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE, NUM_EPOCH
##NUM_HIDDEN=(30 50)
##for nh in ${NUM_HIDDEN[@]}
##do
##    python mlp.py $nh 0.05 16 100
##done


##Third, tune BATCH_SIZE and get the best BATCH_SIZE:
## NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE, NUM_EPOCH
##BATCH_SIZE=(32 64 128 256)
##for bs in ${BATCH_SIZE[@]}
##do
##    python mlp.py 40 0.05 $bs 100
##done

##Third, tune BATCH_SIZE and get the best BATCH_SIZE:
## NUM_HIDDEN, LEARNING_RATE, BATCH_SIZE, NUM_EPOCH
NUM_EPOCH=(50 200)
for ne in ${NUM_EPOCH[@]}
do
    python mlp.py 40 0.05 16 $ne
done