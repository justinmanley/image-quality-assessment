import os
from keras.callbacks import Callback
from utils.utils import ensure_dir_exists
import numpy as np

class LayerWeightsLogger(Callback):
    def __init__(self, job_dir, layer):
        self._weights_dir = os.path.join(job_dir, "saved_weights")
        self._layer = layer
        self._epoch = 0
        ensure_dir_exists(self._weights_dir)

    def on_batch_begin(self, batch, logs=None):
        total_batch = self._epoch * self.params['steps'] + batch
        weights_filename = os.path.join(self._weights_dir, str(total_batch)) + '.npy'
        if os.path.exists(weights_filename):
            raise ValueError("Weights file %s already exists" % weights_filename)

        weights = self._layer.get_weights()
        with open(weights_filename, 'wb') as f:
            np.save(f, weights[0], allow_pickle=False)

    def on_epoch_end(self, epoch, logs=None):
        print('epoch_end', epoch)
        self._epoch = epoch
        
