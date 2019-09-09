#!/usr/bin/env bash

# Kill current training job so that there are no half-written images on disk.
# This is also convenient during development for ensuring that GPU memory is
# freed up for the next training job.
echo "Killing current training job..."
docker-machine ssh mosaic-training 'sudo kill -9 $(nvidia-smi | grep python | awk '"'"'{ print $3 }'"'"')'

TRAIN_DIR=$(docker-machine ssh mosaic-training 'ls -t train_jobs | head -1')
echo "Copying to weights/$TRAIN_DIR..."
docker-machine scp -r mosaic-training:/home/ubuntu/train_jobs/$TRAIN_DIR weights/$TRAIN_DIR
