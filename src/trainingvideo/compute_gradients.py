from trainingvideo.weights_loader import WeightsLoader, TrainingJob
import numpy as np
import argparse

class GradientCalculator:
    def __init__(self, training_job):
        self._training_job = training_job

    def compute_gradients(self):
        weights = WeightsLoader(self._training_job).load_weights()
        gradients = []
        for i in range(len(weights) - 1):
            gradients.append(weights[i + 1] - weights[i])
        return gradients

if __name__ == '__main__':
    # Debug script for inspecting gradient statistics.
    parser = argparse.ArgumentParser(description='Generate video from the weights of a neural network')
    parser.add_argument('arrays_directory', type=str, help='Directory containing numbered .npy files')
    args = parser.parse_args()
    training_job=TrainingJob(args.arrays_directory)
    gradients = GradientCalculator(training_job).compute_gradients()
    for gradient in gradients:
        print(gradient.min(), gradient.max(), gradient.mean(), np.std(gradient))
