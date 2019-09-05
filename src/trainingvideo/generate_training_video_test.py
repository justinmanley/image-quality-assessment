import unittest
import numpy as np
from trainingvideo.generate_training_video import FrameComposer
from numpy.testing import assert_equal


class GenerateTrainingVideoTest(unittest.TestCase):
    def testFrameComposerFrameShape(self):
        images = np.stack([
            np.full((7, 11, 3), 0.1 * i -0.3)
            for i in range(0, 10)
        ], axis=-1)
        self.assertEqual(images.shape, (7, 11, 3, 10))

        frame = FrameComposer(
            scale=1,
            grid_dimensions=(5, 2),
            grid_margin=1).compose(images)

        self.assertEqual(frame.shape, (15, 59, 3))

    def testFrameComposerFrameShapeWithScale(self):
        images = np.stack([
            np.full((7, 11, 3), 0.1 * i -0.3)
            for i in range(0, 10)
        ], axis=-1)
        self.assertEqual(images.shape, (7, 11, 3, 10))

        frame = FrameComposer(
            scale=2,
            grid_dimensions=(5, 2),
            grid_margin=1).compose(images)

        self.assertEqual(frame.shape, (30, 118, 3))

    def testFrameComposerFrameContents(self):
        shape = (2, 2, 3)
        images = np.stack([
            np.full(shape, 0.1 * i - 0.3)
            for i in range(0, 6)
        ], axis=-1)
        self.assertEqual(images.shape, (2, 2, 3, 6))

        frame = FrameComposer(
            scale=1,
            grid_dimensions=(3, 2),
            grid_margin=1).compose(images)

        self.assertEqual(frame.shape, (5, 8, 3))

        # First square (i = 0)
        self.assertEqual(frame[:2, :2].shape, (2, 2, 3))
        assert_equal(frame[0:2, 0:2], 51)

        # Second square (i = 1)
        assert_equal(frame[3:5, 0:2], 76)

        # Third square (i = 2)
        assert_equal(frame[0:2, 3:5], 102)

        # Fourth square (i = 3)
        assert_equal(frame[3:5, 3:5], 127)

        # Fifth square (i = 4)
        assert_equal(frame[0:2, 6:8], 153)

        # Sixth square (i = 5)
        assert_equal(frame[3:5, 6:8], 178)

        # Margins
        assert_equal(frame[2], 127)
        assert_equal(frame[:,2], 127)
        assert_equal(frame[:,5], 127)

    def testFrameComposerFrameContentsWithScale(self):
        shape = (2, 2, 3)
        images = np.stack([
            np.full(shape, 0.1 * i - 0.3)
            for i in range(0, 6)
        ], axis=-1)
        self.assertEqual(images.shape, (2, 2, 3, 6))

        composer = FrameComposer(
            scale=2,
            grid_dimensions=(3, 2),
            grid_margin=1)
        frame = composer.compose(images)

        self.assertEqual(frame.shape, (10, 16, 3))

        # First square (i = 0)
        self.assertEqual(frame[:2, :2].shape, (2, 2, 3))
        assert_equal(frame[0:4, 0:4], 51)

        # Second square (i = 1)
        assert_equal(frame[6:10, 0:4], 76)

        # Third square (i = 2)
        assert_equal(frame[0:4, 6:10], 102)

        # Fourth square (i = 3)
        assert_equal(frame[6:10, 6:10], 127)

        # Fifth square (i = 4)
        assert_equal(frame[0:4, 12:16], 153)

        # Sixth square (i = 5)
        assert_equal(frame[6:8, 12:16], 178)

        # Margins
        assert_equal(frame[4], 127)
        assert_equal(frame[:,4], 127)
        assert_equal(frame[:,10], 127)

    def testFrameComposerOutputSize(self):
        image_shape = (2, 2, 3)
        images = np.stack([
            np.full(image_shape, 0.1 * i - 0.3)
            for i in range(0, 6)
        ], axis=-1)
        self.assertEqual(images.shape, (2, 2, 3, 6))

        composer = FrameComposer(
            scale=1,
            grid_dimensions=(3, 2),
            grid_margin=1)
        frame = composer.compose(images)

        self.assertEqual(frame.shape, (5, 8, 3))
        self.assertEqual(composer.output_size(image_shape), (8, 5, 3))


if __name__ == '__main__':
    unittest.main()
