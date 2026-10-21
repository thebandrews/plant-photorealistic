"""Read/write lossless PNG RGB samples without truncating native 16-bit renders."""
from pathlib import Path

import numpy as np
import png
from PIL import Image


def load(path):
    path = Path(path)
    if path.suffix.lower() != '.png':
        with Image.open(path) as image:
            return np.array(image.convert('RGB')), None
    with path.open('rb') as stream:
        width, height, rows, info = png.Reader(file=stream).asDirect()
        if info['bitdepth'] not in (8, 16):
            raise ValueError('Appearance PNGs must use 8-bit or 16-bit samples')
        dtype = np.uint16 if info['bitdepth'] > 8 else np.uint8
        array = np.asarray(list(rows), dtype=dtype).reshape(height, width, info['planes'])
    channels = 1 if info['greyscale'] else 3
    rgb = array[:, :, :channels]
    if channels == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    alpha = array[:, :, channels] if info['alpha'] else None
    return rgb, alpha


def save(path, rgb, alpha=None):
    rgb = np.asarray(rgb)
    bitdepth = 16 if rgb.dtype == np.uint16 else 8
    array = rgb if alpha is None else np.dstack([rgb, alpha])
    height, width = rgb.shape[:2]
    with Path(path).open('wb') as stream:
        png.Writer(width, height, greyscale=False, alpha=alpha is not None,
                   bitdepth=bitdepth).write(stream, array.reshape(height, -1).tolist())


def scaled_like(raw, baseline):
    if raw.dtype == baseline.dtype:
        return raw
    return np.rint(raw.astype(np.float64) *
                   (np.iinfo(baseline.dtype).max / np.iinfo(raw.dtype).max)).astype(baseline.dtype)
