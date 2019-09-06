
import os
import argparse
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from handlers.data_generator import TrainDataGenerator, TestDataGenerator
from handlers.model_builder import Nima
from handlers.samples_loader import load_samples
from handlers.config_loader import load_config
from utils.utils import ensure_dir_exists
from utils.keras_utils import TensorBoardBatch
from callbacks.layer_weights_logger import LayerWeightsLogger
from utils.mosaics_loader import MosaicsLoader

import numpy as np


def train(base_model_name,
          n_classes,
          samples,
          image_dir,
          batch_size,
          epochs_train_dense,
          epochs_train_all,
          learning_rate_dense,
          learning_rate_all,
          dropout_rate,
          job_dir,
          img_format='jpg',
          existing_weights=None,
          multiprocessing_data_load=False,
          num_workers_data_load=2,
          decay_dense=0,
          decay_all=0,
          mosaics=None,
          **kwargs):

    # build NIMA model and load existing weights if they were provided in config
    nima = Nima(
        base_model_name,
        n_classes,
        learning_rate_dense,
        dropout_rate,
        decay=decay_dense,
        weights=existing_weights)
    nima.build()

    if existing_weights is not None:
        print('loading existing weights from %s' % existing_weights)
        nima.nima_model.load_weights(existing_weights)
        
    # override weights with mosaics.
    # it is crucial that this happen after existing weights are loaded.
    mosaics_layer = nima.base_model.layers[2]
    if mosaics:
        loader = MosaicsLoader("/src/mosaics")
        default_layer_weights = mosaics_layer.get_weights()
        mosaics_array = loader.load_and_standardize_mosaics(mosaics) 
        if mosaics_array.shape != default_layer_weights[0].shape:
            raise ValueError(
                "Shape of mosaics %s does not match the shape of the layer: %s" % (
                    str(mosaics_array.shape), str(default_layer_weights[0].shape)))
        print('Setting weights of convolutional layer to mosaics')
        mosaics_layer.set_weights([mosaics_array])

        success = np.allclose(mosaics_layer.get_weights()[0], mosaics_array)
        if not success:
            raise ValueError('weights do not match')
        else:
            print('mosaic weights match in mosaicnet.py')

    # split samples in train and validation set, and initialize data generators
    samples_train, samples_test = train_test_split(
        samples, test_size=0.05, shuffle=True, random_state=10207)

    training_generator = TrainDataGenerator(samples_train,
                                            image_dir,
                                            batch_size,
                                            n_classes,
                                            nima.preprocessing_function(),
                                            img_format=img_format)

    validation_generator = TestDataGenerator(samples_test,
                                             image_dir,
                                             batch_size,
                                             n_classes,
                                             nima.preprocessing_function(),
                                             img_format=img_format)

    # initialize callbacks TensorBoardBatch and ModelCheckpoint
    tensorboard = TensorBoardBatch(log_dir=os.path.join(job_dir, 'logs'))

    model_save_name = 'weights_'+base_model_name.lower()+'_{epoch:02d}_{val_loss:.3f}.hdf5'
    model_file_path = os.path.join(job_dir, 'weights', model_save_name)
    model_checkpointer = ModelCheckpoint(filepath=model_file_path,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True)
    weights_logger = LayerWeightsLogger(job_dir, mosaics_layer)

    # start training only dense layers
    for layer in nima.base_model.layers:
        layer.trainable = False

    nima.compile()
    nima.nima_model.summary()

    nima.nima_model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs_train_dense,
                                  verbose=1,
                                  use_multiprocessing=multiprocessing_data_load,
                                  workers=num_workers_data_load,
                                  max_q_size=30,
                                  callbacks=[tensorboard, model_checkpointer])

    # start training all layers
    for layer in nima.base_model.layers:
        layer.trainable = True

    nima.learning_rate = learning_rate_all
    nima.decay = decay_all
    nima.compile()
    nima.nima_model.summary()

    # check that weights match mosaics
    if mosaics:
        loader = MosaicsLoader("/src/mosaics")
        mosaics_array = loader.load_and_standardize_mosaics(mosaics) 
        success = np.allclose(mosaics_layer.get_weights()[0], mosaics_array)
        if not success:
            raise ValueError('weights do not match before beginning conv training')
        else:
            print('mosaic weights match')

    nima.nima_model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs_train_dense+epochs_train_all,
                                  initial_epoch=epochs_train_dense,
                                  verbose=1,
                                  use_multiprocessing=multiprocessing_data_load,
                                  workers=num_workers_data_load,
                                  max_q_size=30,
                                  callbacks=[tensorboard, model_checkpointer, weights_logger])

    K.clear_session()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job-dir', help='train job directory with samples and config file', required=True)
    parser.add_argument('-i', '--image-dir', help='directory with image files', required=True)

    args = parser.parse_args()

    image_dir = args.__dict__['image_dir']
    print('image_dir: ' + image_dir)
    job_dir = args.__dict__['job_dir']

    ensure_dir_exists(os.path.join(job_dir, 'weights'))
    ensure_dir_exists(os.path.join(job_dir, 'logs'))

    config_file = os.path.join(job_dir, 'config.json')
    config = load_config(config_file)

    samples_file = os.path.join(job_dir, 'samples.json')
    samples = load_samples(samples_file)

    train(samples=samples, job_dir=job_dir, image_dir=image_dir, **config)
