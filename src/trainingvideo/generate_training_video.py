import numpy as np
import argparse 
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os
from ast import literal_eval as make_tuple
from functools import reduce
from operator import mul
from handlers.config_loader import load_config
from utils.mosaics_loader import PCAWhitener, MosaicsLoader

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
        composition = self._compose_frame(images)
        upsampled = self._upsample(composition)
        return self._normalize(upsampled)

    def output_size(self, image_shape):
        # image_dimensions is a 3-tuple, of the form (cols, rows, channels)
        # (i.e. in the format returned by ndarray.shape). For example, for
        # an 8x10 (landscape) image, this would be (8,10,3).
        scale = self._scale_factor
        output_shape = self._output_shape(image_shape)
        return (
            output_shape[1] * scale,
            output_shape[0] * scale,
            output_shape[2]
        )

    def _output_shape(self, image_shape):
        cols, rows = self._grid_dimensions
        image_rows, image_cols = image_shape[0:2]
        margin = self._grid_margin
        return (
            rows * image_rows + (rows - 1) * margin,
            cols * image_cols + (cols - 1) * margin,
            image_shape[2],
        )

    def _upsample(self, a):
        return np.repeat(
            np.repeat(a, self._scale_factor, axis=0),
            self._scale_factor,
            axis=1)

    def _normalize(self, a):
        return np.uint8((a + 0.5) * 255)

    def _compose_frame(self, images):
        cols, rows = self._grid_dimensions
        image_rows, image_cols = images.shape[0:2]
        margin = self._grid_margin
        frame = np.full(self._output_shape(images.shape), 0, dtype=np.float64)

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
    def __init__(self, path, scale, grid_dimensions, grid_margin, training_config_path):
        # Path to a directory of .npy files. Each array is expected to have
        # values between -0.5 and 0.5.
        self._path = path
        self._scale_factor = scale
        self._grid_dimensions = grid_dimensions
        self._grid_margin = grid_margin
        self._training_config_path = training_config_path

    def generate_video(self):
        files = sorted(os.listdir(self._path))
        # The arrays in this list each have shape (8, 10, 3, N), where N is the
        # number of filters in the layer.
        arrays = []
        for array_file in files:
            with open(os.path.join(self._path, array_file), 'rb') as f:
                weights_array = np.load(f, allow_pickle=False)
                if weights_array.shape[3] != reduce(mul, self._grid_dimensions, 1):
                    raise ValueError('Size of grid layout must match the number of filters.')
                arrays.append(weights_array)

        composer = FrameComposer(
            scale=self._scale_factor,
            grid_dimensions=self._grid_dimensions,
            grid_margin=self._grid_margin)
        processor = self._get_array_processor()

        output_size = composer.output_size(arrays[0].shape)
        writer = FFMPEG_VideoWriter(self._path + '.mp4', output_size, fps=1.0)
        with writer:
            for filters in arrays:
                restored_filters = processor.restore(filters)
                composed_frame = composer.compose(restored_filters)
                writer.write_frame(composed_frame)

    def _get_array_processor(self):
        config = load_config(self._training_config_path)
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
    parser.add_argument('--training_config_path',
        default={}, type=str, help='Path to JSON training config file')
    args = parser.parse_args()
    VideoGenerator(
        path=args.arrays_directory,
        scale=args.scale,
        grid_dimensions=make_tuple(args.grid_dimensions),
        grid_margin=args.grid_margin,
        training_config_path=args.training_config_path,
    ).generate_video()
