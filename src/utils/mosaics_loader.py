from imageio import imread
import os
import numpy as np
from sklearn.decomposition import PCA

class PCAWhitener:
    def __init__(self, array):
        self.pca = PCA(whiten = True)
        self.shape = array.shape
        self.pca.fit(array.reshape(self.shape[0], -1).T)
        
    def process(self, unprocessed):
        return self.pca.transform(unprocessed.reshape(self.shape[0], -1).T).T.reshape(self.shape)
    
    def restore(self, processed):
        return self.pca.inverse_transform(processed.reshape(self.shape[0], -1).T).T.reshape(self.shape)

class MosaicsLoader:
    def __init__(self, mosaics_dir):
        self._mosaics_dir = mosaics_dir

    def _load_mosaic(self, mosaic_id):
        return imread(os.path.join(self._mosaics_dir, str(mosaic_id) + '.png'))

    def load_mosaics(self, mosaic_ids):
        unscaled_mosaics = np.stack(
            [self._load_mosaic(mosaic_id) for mosaic_id in mosaic_ids],
            axis = -1,
        )
        return unscaled_mosaics.astype(np.uint8) / 255.0 

    def load_and_standardize_mosaics(self, mosaic_ids):
        mosaics = self.load_mosaics(mosaic_ids)
        whitener = PCAWhitener(mosaics)
        return whitener.process(mosaics)
