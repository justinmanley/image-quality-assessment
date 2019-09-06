import unittest
from mosaics_loader import MosaicsLoader, PCAWhitener
import numpy as np
from numpy.testing import assert_almost_equal

mosaics_list = [
    162, 163, 164, 165,
    166, 167, 168, 169,
    170, 171, 172, 176,
    177, 178, 179, 180
  ]

class MosaicsLoaderTest(unittest.TestCase):
    def testMosaicsShape(self):
        loader = MosaicsLoader("/Users/justin/projects/mosaics/architect_mosaics_processed")
        weights = loader.load_and_standardize_mosaics(mosaics_list)

        self.assertEqual(weights.shape, (8, 10, 3, 16))

    def testMosaicsStatistics(self):
        loader = MosaicsLoader("/Users/justin/projects/mosaics/architect_mosaics_processed")
        weights = loader.load_and_standardize_mosaics(mosaics_list)

        self.assertAlmostEqual(weights.mean(), 0)
        self.assertAlmostEqual(np.std(weights), 1, places=2)


class PCAWhitenerTest(unittest.TestCase):
    def testRestoreInvertsProcess(self):
        example = np.random.random_sample(size = (8, 10, 3, 16))
        whitener = PCAWhitener(example)
        processed = whitener.process(example)
        assert_almost_equal(whitener.restore(processed), example)

if __name__ == '__main__':
    unittest.main()
