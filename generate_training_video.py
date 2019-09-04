import numpy as np
import argparse 
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import os
from ast import literal_eval as make_tuple
from functools import reduce
from operator import mul

def intersperse(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

class FrameComposer:
    def __init__(self, images, scale, grid_dimensions, grid_margin):
        # images is a numpy array with shape (W, H, C, N) - for example,
        # (8, 10, 3, N).
        self._images = images
        self._scale_factor = scale
        self._grid_dimensions = 1, images.shape[3] # DO NOT SUBMIT
        self._grid_margin = grid_margin

    def compose(self):
        composition = self._compose_frame()
        upsampled = self._upsample(composition)
        return self._normalize(upsampled)

    def _output_shape(self):
        cols, rows = self._grid_dimensions
        image_rows, image_cols = self._images.shape[0:2]
        margin = self._grid_margin
        return (
            rows * image_rows + (rows - 1) * margin,
            cols * image_cols + (cols - 1) * margin,
            self._images.shape[2],
        )

    def output_size(self):
        output_shape = self._output_shape()
        return tuple(dim * self._scale_factor for dim in output_shape[0:2])

    def _upsample(self, a):
        return np.repeat(
            np.repeat(a, self._scale_factor, axis=0),
            self._scale_factor,
            axis=1)

    def _normalize(self, a):
        return np.uint8((a + 0.5) * 255)

    def _compose_frame(self):
        images = self._images
        cols, rows = self._grid_dimensions
        image_rows, image_cols, image_channels = images.shape[0:3]
        margin = self._grid_margin

        
        images_with_margins = intersperse(
            [images[:,:,:,i] for i in range(images.shape[3])],
            np.full((image_rows, 1, image_channels), 0))

        return np.concatenate(list(images_with_margins), axis=1)

class VideoGenerator:
    def __init__(self, path, scale, grid_dimensions, grid_margin):
        # Path to a directory of .npy files. Each array is expected to have
        # values between -0.5 and 0.5.
        self._path = path
        self._scale_factor = scale
        self._grid_dimensions = grid_dimensions
        self._grid_margin = grid_margin

    def generate_video(self):
        files = sorted(os.listdir(self._path))
        # The arrays in this list each have shape (8, 10, 3, N), where N is the
        # number of filters in the layer.
        arrays = []
        for array_file in files:
            with open(os.path.join(self._path, array_file), 'rb') as f:
                weights_array = np.swapaxes(np.load(f, allow_pickle=False), 0, 1)
                if weights_array.shape[3] != reduce(mul, self._grid_dimensions, 1):
                    raise ValueError('Size of grid layout must match the number of filters.')
                arrays.append(weights_array)

        output_size = FrameComposer(arrays[0]).output_size()
        writer = FFMPEG_VideoWriter(self._path + '.mp4', output_size, fps=1.0)
        with writer:
            for filters in arrays:
                writer.write_frame(FrameComposer(filters).compose())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate video from the weights of a neural network')
    parser.add_argument('arrays_directory', type=str, help='Directory containing numbered .npy files')
    parser.add_argument('--scale', type=int, default=1, help='Amount by which to scale the arrays')
    parser.add_argument('--grid_dimensions', default='(1,1)', type=str, help='How to lay out weights')
    parser.add_argument('--grid_margin', default=1, type=int, help='Margin between filters')
    args = parser.parse_args()
    VideoGenerator(
        path=args.arrays_directory,
        scale=args.scale,
        grid_dimensions=make_tuple(args.grid_dimensions),
        grid_margin=args.grid_margin,
    ).generate_video()
