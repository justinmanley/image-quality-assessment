import numpy as np
import argparse 
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os
from ast import literal_eval as make_tuple
from functools import reduce
from operator import mul
from handlers.config_loader import load_config
from utils.mosaics_loader import PCAWhitener, MosaicsLoader
from trainingvideo.weights_loader import WeightsLoader, TrainingJob

from math import gcd

class IdentityImageProcessor:
    def __init__(self):
        pass

    def process(self, unprocessed):
        return unprocessed

    def restore(self, processed):
        return processed

class FrameComposer:
    def __init__(self, scale, grid_dimensions, grid_margin):
        self._scale_factor = scale
        self._grid_dimensions = grid_dimensions
        self._grid_margin = grid_margin

    def compose(self, images):
        upsampled_images = self._upsample(images)
        composition = self._compose_frame(upsampled_images, self._output_size(images.shape))
        return self._normalize(composition)

    def _output_size(self, image_shape):
        # image_dimensions is a 3-tuple, of the form (cols, rows, channels)
        # (i.e. in the format returned by ndarray.shape). For example, for
        # an 8x10 (landscape) image, this would be (8,10,3).
        cols, rows = self._grid_dimensions
        image_rows, image_cols = image_shape[0:2]
        margin = self._grid_margin
        scale = self._scale_factor
        return (
            rows * image_rows * scale + (rows - 1) * margin,
            cols * image_cols * scale + (cols - 1) * margin,
            image_shape[2],
        )

    def video_output_size(self, image_shape):
        # The first and second dimensions must be flipped for the video.
        output_size = self._output_size(image_shape)
        if output_size[0] % 2 != 0 or output_size[1] % 2 != 0:
            # See https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py#L41.
            raise ValueError(
                "If either spatial dimension of the output video is odd, "
                "the video may not display properly. The video dimensions "
                "are (%d, %d). Adjusting the --grid_margin or the --scale "
                "parameters can help ensure that the spatial output dimensions "
                "are both even." % (output_size[1], output_size[0]))
        return (
            output_size[1],
            output_size[0],
            output_size[2],
        )

    def _upsample(self, a):
        return np.repeat(
            np.repeat(a, self._scale_factor, axis=0),
            self._scale_factor,
            axis=1)

    def _normalize(self, a):
        # Input array 'a' is expected to have values in the range [0,1].
        # We clip the array to the range [0, 255] after scaling it because
        # converting floats to uint8 causes negative numbers to be wrapped
        # around (so that, for example, -1 becomes 255). The visual effect
        # of this wrapping is that colors which have any channels close to
        # the boundary of the range (e.g. bright red, blue, green, yellow,
        # magenta, black, or white), may suddenly flip to another color, so
        # red might suddenly become yellow, or white might suddenly become
        # magenta.
        #
        # Note that this clipping is only effective if the range of the array
        # remains close to [0,255]; if it begins to diverge widely so that
        # the underlying mean and/or standard deviation become significantly
        # different than that of the clipped array, then the clipped array
        # will no longer accurately represent the original, raw array.
        return np.uint8(np.clip(a * 255, 0, 255))

    def _compose_frame(self, images, output_size):
        # images are expected to be in the range [0,1].
        cols, rows = self._grid_dimensions
        image_rows, image_cols = images.shape[0:2]
        margin = self._grid_margin
        # 0.5 represents grey for images in the range [0,1].
        frame = np.full(output_size, 0.5, dtype=np.float64)

        row_start = 0
        col_start = 0
        for i in range(images.shape[3]):
            image = images[:,:,:,i]
            row_end = row_start + image_rows
            col_end = col_start + image_cols
            frame[row_start:row_end,col_start:col_end] = image
            if (i + 1) % rows == 0 and i > 0:
                row_start = 0
                col_start += image_cols + margin
            else:
                row_start += image_rows + margin

        return frame


class VideoGenerator:
    def __init__(self,
            training_job,
            scale,
            grid_dimensions,
            grid_margin):
        self._training_job = training_job
        self._scale_factor = scale
        self._grid_dimensions = grid_dimensions
        self._grid_margin = grid_margin

    def generate_video(self):
        # The arrays in this list each have shape (8, 10, 3, N), where N is the
        # number of filters in the layer.
        arrays = WeightsLoader(self._training_job).load_weights()
        all_arrays = np.array(arrays)
        if all_arrays.shape[4] != reduce(mul, self._grid_dimensions, 1):
            raise ValueError('Size of grid layout must match the number of filters.')

        composer = FrameComposer(
            scale=self._scale_factor,
            grid_dimensions=self._grid_dimensions,
            grid_margin=self._grid_margin)
        processor = self._get_array_processor()

        output_size = composer.video_output_size(arrays[0].shape)
        writer = FFMPEG_VideoWriter(self._training_job.start_time + '.mp4', output_size, fps=1.0)
        with writer:
            for filters in arrays:
                restored_filters = processor.restore(filters)
                composed_frame = composer.compose(restored_filters)
                writer.write_frame(composed_frame)

    def _get_array_processor(self):
        training_config_path = os.path.join(self._training_job.directory, "config.json")
        config = load_config(training_config_path)
        if config['mosaics']:
            loader = MosaicsLoader("/Users/justin/projects/mosaics/architect_mosaics_processed")
            mosaics = loader.load_mosaics(config['mosaics'])
            return PCAWhitener(mosaics)
        else:
            return IdentityImageProcessor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate video from the weights of a neural network')
    parser.add_argument('arrays_directory', type=str, help='Directory containing numbered .npy files')
    parser.add_argument('--scale', type=int, default=1, help='Amount by which to scale the arrays')
    parser.add_argument('--grid_dimensions', default='(1,1)', type=str, help='How to lay out weights')
    parser.add_argument('--grid_margin', default=1, type=int, help='Margin between filters')
    args = parser.parse_args()
    training_job=TrainingJob(args.arrays_directory)
    VideoGenerator(
        training_job=training_job,
        scale=args.scale,
        grid_dimensions=make_tuple(args.grid_dimensions),
        grid_margin=args.grid_margin,
    ).generate_video()
