import json
import os
from typing import Dict

import numpy as np
import rasterio.features
import scipy.ndimage as sp_ndimage
from conerf import types
from shapely.geometry import Polygon


def load_annotation(
    path: types.PathType,
    scale: float,
    num_attributes: int,
    height: int,
    width: int,
    class_to_id_mapping: Dict[str, int],
    gaussian_splatting_filter_size: float,
) -> np.ndarray:
    mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.uint8)
    if not os.path.exists(path) or num_attributes == 0:
        return mask_labels
    with open(path) as f:
        data = json.load(f)

    for datum in data["shapes"]:
        polygon = Polygon(np.array(datum["points"]) / scale)
        cls_id = class_to_id_mapping[datum["label"]]
        mask = rasterio.features.rasterize(
            [polygon], out_shape=(height, width)
        )
        mask_labels[..., cls_id] = mask_labels[..., cls_id] | mask
    mask_labels[mask_labels.sum(axis=-1) == 0, -1] = 1
    mask_labels = mask_labels.astype(np.float32)

    if gaussian_splatting_filter_size > 0:
        blurred = [
            sp_ndimage.gaussian_filter(
                layer,
                sigma=gaussian_splatting_filter_size,
            )
            for layer in mask_labels.transpose((2, 0, 1))
        ]

        mask_labels = np.stack(blurred, axis=-1)
        mask_labels = mask_labels / mask_labels.max(axis=-1, keepdims=True)
    return mask_labels


def load_depth(path: types.PathType, near: float, far: float) -> np.ndarray:
    data = np.load(path)
    data = (data - data.min()) / (data.max() - data.min() + 1e-5)
    data = 1 - data
    data = data * (far - near) + near
    return data
