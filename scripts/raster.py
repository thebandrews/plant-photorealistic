"""Read/write lossless PNG RGB samples without truncating native 16-bit renders."""
from pathlib import Path

import numpy as np
import png
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def load(path):
    path = Path(path)
    if path.suffix.lower() != '.png':
        with Image.open(path) as image:
            return np.array(image.convert('RGB')), None
    # Native decoding avoids Python PNG-filter loops. IMREAD_UNCHANGED keeps
    # uint16 and RGB hidden behind zero alpha; never use Pillow's RGB16 convert.
    with path.open('rb') as stream:
        header = stream.read(29)
    if len(header) < 29 or header[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Unable to decode appearance PNG')
    # Low-bit palette images can expand to RGB8 through the original decoder.
    if cv2 is not None and header[24] in (8, 16):
        array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError('Unable to decode appearance PNG')
        if array.ndim == 2:
            return np.repeat(array[..., None], 3, axis=2), None
        rgb = array[..., [2, 1, 0]]
        return rgb, array[..., 3] if array.shape[2] == 4 else None
    with path.open('rb') as stream:
        width, height, rows, info = png.Reader(file=stream).asDirect()
        if info['bitdepth'] not in (8, 16):
            raise ValueError('Appearance PNGs must use 8-bit or 16-bit samples')
        dtype = np.uint16 if info['bitdepth'] > 8 else np.uint8
        array = np.empty((height, width * info['planes']), dtype=dtype)
        for y, row in enumerate(rows):
            array[y] = row
        array = array.reshape(height, width, info['planes'])
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
                   bitdepth=bitdepth).write(stream, (row for row in array.reshape(height, -1)))


def scaled_like(raw, baseline):
    if raw.dtype == baseline.dtype:
        return raw
    return np.rint(raw.astype(np.float64) *
                   (np.iinfo(baseline.dtype).max / np.iinfo(raw.dtype).max)).astype(baseline.dtype)
