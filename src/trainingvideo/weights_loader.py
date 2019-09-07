from datetime import datetime
import os
import numpy as np

def training_job_start_time(datetime_string):
    return datetime.strptime(datetime_string, "%Y_%m_%d_%H_%M_%S") 

def get_latest_training_job(directory):
    training_directories = sorted([
        f for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
    ], key=training_job_start_time)
    return training_directories[-1]

def get_train_job_directory(parent, subdir):
    if subdir == "latest":
        latest_training_job = get_latest_training_job(parent)
        print("Generating video for %s..." % latest_training_job)
        return os.path.join(parent, latest_training_job)
    else:
        return subdir

class TrainingJob:
    def __init__(self, directory):
        # directory should be a string in one of the two forms:
        #   weights/2019_09_06_19_23_43 - a specific training run
        #   weights/latest - a special form which identifies the latest training dir
        # The directory should contain only .npy files. Each array is expected to have
        # a mean of zero and a standard deviation of 1.
        parent, subdir = os.path.split(directory)
        self.start_time = get_train_job_directory(parent, subdir)
        self.directory = os.path.join(parent, self.start_time)

def get_training_step(filename):
    return int(os.path.splitext(filename)[0])

class WeightsLoader:
    def __init__(self, training_job):
        self._training_job = training_job

    def load_weights(self):
        training_job_directory = self._training_job.directory
        files = sorted(os.listdir(training_job_directory), key=get_training_step)
        # The arrays in this list each have shape (8, 10, 3, N), where N is the
        # number of filters in the layer.
        arrays = []
        for array_file in files:
            with open(os.path.join(training_job_directory, array_file), 'rb') as f:
                weights_array = np.load(f, allow_pickle=False)
                arrays.append(weights_array)
        return arrays
