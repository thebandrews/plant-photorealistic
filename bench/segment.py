"""Box-prompted plant segmentation for the plants gate (SAM through Hugging Face transformers, CPU is fine).

The image is cut into 1024 px windows at full resolution and each window is encoded once; every plant whose
box fits a window is decoded from that window's embedding. Plants too big for any window use one encoding of
the whole image at reduced scale. Masks come back cropped to each plant's region, so native and output masks
of the same plant share coordinates.
"""
import numpy as np

WINDOW = 1024
STRIDE = 768
MARGIN = 0.15


def plant_region(box, shape):
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * MARGIN + 4, (y1 - y0) * MARGIN + 4
    h, w = shape[:2]
    return [int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my))]


def _starts(n):
    s = list(range(0, max(n - WINDOW, 0) + 1, STRIDE))
    if s[-1] != max(n - WINDOW, 0):
        s.append(max(n - WINDOW, 0))
    return s


def seed_points(ids, plant, n=4):
    """n positive prompt points on the plant's own render pixels, spread out and away from its edges."""
    import cv2
    x0, y0, x1, y1 = plant['box']
    m = (ids[y0:y1, x0:x1] == plant['pass_index']).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    pts = []
    for _ in range(n):
        if dist.max() <= 0:
            break
        yy, xx = np.unravel_index(np.argmax(dist), dist.shape)
        pts.append([float(xx + x0), float(yy + y0)])
        rad = max(4, int(max(x1 - x0, y1 - y0) / (n + 1)))
        cv2.circle(dist, (int(xx), int(yy)), rad, 0, -1)
    pts = pts or [[(x0 + x1) / 2, (y0 + y1) / 2]]
    return (pts * n)[:n]


class SamSegmenter:
    def __init__(self, model_id='facebook/sam-vit-base', device='cpu'):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch = torch
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    def _decode(self, image, boxes, points):
        """boxes and positive points (N per box) in `image` pixel coordinates -> list of full-size bool masks."""
        torch = self.torch
        labels = [[1] * len(p) for p in points]
        inputs = self.processor(image, input_boxes=[boxes], input_points=[points], input_labels=[labels],
                                return_tensors='pt').to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_embeddings(inputs['pixel_values'])
            out = []
            for i in range(0, len(boxes), 32):
                res = self.model(image_embeddings=emb, input_boxes=inputs['input_boxes'][:, i:i + 32],
                                 input_points=inputs['input_points'][:, i:i + 32],
                                 input_labels=inputs['input_labels'][:, i:i + 32], multimask_output=False)
                masks = self.processor.image_processor.post_process_masks(
                    res.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())[0]
                out += [m[0].numpy().astype(bool) for m in masks]
        return out

    def masks(self, img, plants):
        """{plant id: (mask cropped to region, region)} for each plant. Plants carry 'points' (see seed_points)."""
        h, w = img.shape[:2]
        windows = [(x, y) for y in _starts(h) for x in _starts(w)]
        assign, whole = {}, []
        for p in plants:
            r = plant_region(p['box'], img.shape)
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            fits = [(abs(x + WINDOW / 2 - cx) + abs(y + WINDOW / 2 - cy), (x, y)) for x, y in windows
                    if r[0] >= x and r[1] >= y and r[2] <= x + WINDOW and r[3] <= y + WINDOW]
            if fits:
                assign.setdefault(min(fits)[1], []).append((p, r))
            else:
                whole.append((p, r))
        result = {}
        for (x, y), items in assign.items():
            crop = img[y:y + WINDOW, x:x + WINDOW]
            boxes = [[p['box'][0] - x, p['box'][1] - y, p['box'][2] - x, p['box'][3] - y] for p, _ in items]
            points = [[[px - x, py - y] for px, py in p['points']] for p, _ in items]
            for (p, r), m in zip(items, self._decode(crop, boxes, points)):
                full = np.zeros((r[3] - r[1], r[2] - r[0]), bool)
                full[:] = m[r[1] - y:r[3] - y, r[0] - x:r[2] - x]
                result[p['id']] = (full, r)
        if whole:
            from PIL import Image
            scale = WINDOW / max(h, w)
            small = np.asarray(Image.fromarray(img).resize((round(w * scale), round(h * scale)), Image.BILINEAR))
            boxes = [[v * scale for v in p['box']] for p, _ in whole]
            points = [[[px * scale, py * scale] for px, py in p['points']] for p, _ in whole]
            for (p, r), m in zip(whole, self._decode(small, boxes, points)):
                sx0, sy0 = int(r[0] * scale), int(r[1] * scale)
                sx1, sy1 = min(m.shape[1], int(np.ceil(r[2] * scale)) + 1), min(m.shape[0], int(np.ceil(r[3] * scale)) + 1)
                part = Image.fromarray(m[sy0:sy1, sx0:sx1])
                big = np.asarray(part.resize((round((sx1 - sx0) / scale), round((sy1 - sy0) / scale)), Image.NEAREST))
                ox, oy = round(r[0] - sx0 / scale), round(r[1] - sy0 / scale)
                full = np.zeros((r[3] - r[1], r[2] - r[0]), bool)
                piece = big[oy:oy + full.shape[0], ox:ox + full.shape[1]]
                full[:piece.shape[0], :piece.shape[1]] = piece
                result[p['id']] = (full, r)
        return result
