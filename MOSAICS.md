### Process of generating a mosaic training video

To train a model, run:
```
docker-machine start mosaic-training
./train-ec2 --docker-machine mosaic-training \
  --config-file $(pwd)/models/MobileNet/config_mobilenet_aesthetic.json \  # path on local machine
  --samples-file $(pwd)/pruned_samples.json \  # path on local machine
  --image-dir /home/ubuntu/Downloads/AVA_dataset  # path on remote machine
```

Once the model has trained sufficiently, run:
```
tools/fetch_weights_from_latest_training_job.sh
cd src && python -m trainingvideo.generate_training_video ../weights/latest \
  --scale=24 \
  --grid_dimensions='(4,3)' \
  --grid_margin=4
```
