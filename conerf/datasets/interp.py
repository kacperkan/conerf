# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Casual Volumetric Capture datasets.

Note: Please benchmark before submitted changes to this module. It's very easy
to introduce data loading bottlenecks!
"""
import json
from typing import List, Tuple

import cv2
import gin
import numpy as np
import yaml
from absl import logging
from conerf import gpath, types, utils
from conerf.datasets import core
from conerf.datasets.utils import load_annotation, load_depth


def load_scene_info(
    data_dir: types.PathType,
) -> Tuple[np.ndarray, float, float, float]:
    """Loads the scene center, scale, near and far from scene.json.

    Args:
      data_dir: the path to the dataset.

    Returns:
      scene_center: the center of the scene (unscaled coordinates).
      scene_scale: the scale of the scene.
      near: the near plane of the scene (scaled coordinates).
      far: the far plane of the scene (scaled coordinates).
    """
    scene_json_path = gpath.GPath(data_dir, "scene.json")
    with scene_json_path.open("r") as f:
        scene_json = json.load(f)

    scene_center = np.array(scene_json["center"])
    scene_scale = scene_json["scale"]
    near = scene_json["near"]
    far = scene_json["far"]

    return scene_center, scene_scale, near, far


def _load_image(path: types.PathType) -> np.ndarray:
    path = gpath.GPath(path)
    with path.open("rb") as f:
        raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[
            :, :, ::-1
        ]  # BGR -> RGB
        image = np.asarray(image).astype(np.float32) / 255.0
        return image


def _load_dataset_ids(data_dir: types.PathType) -> Tuple[List[str], List[str]]:
    """Loads dataset IDs."""
    dataset_json_path = gpath.GPath(data_dir, "dataset.json")
    logging.info("*** Loading dataset IDs from %s", dataset_json_path)
    with dataset_json_path.open("r") as f:
        dataset_json = json.load(f)

    return dataset_json["ids"]


@gin.configurable
class InterpDataSource(core.DataSource):
    """Data loader for videos."""

    def __init__(
        self,
        data_dir=gin.REQUIRED,
        image_scale: int = gin.REQUIRED,
        interval: int = gin.REQUIRED,
        gaussian_splatting_filter_size: float = 0.0,
        shuffle_pixels=False,
        camera_type="json",
        test_camera_trajectory="orbit-mild",
        data_in_separate_folders: bool = False,
        **kwargs,
    ):
        self.gaussian_filter_size = gaussian_splatting_filter_size
        self.data_dir = gpath.GPath(data_dir)
        if interval < 2 or interval % 2 != 0:
            raise ValueError("interval must be a positive even number.")
        all_ids = _load_dataset_ids(self.data_dir)
        if interval > len(all_ids) - 1:
            raise ValueError("interval exceeds dataset size.")
        all_indices = np.arange(len(all_ids))
        train_indices = all_indices[::interval]
        # Take the middle frames for validation.
        val_indices = (train_indices[:-1] + train_indices[1:]) // 2

        if data_in_separate_folders:
            train_ids = all_ids
            val_ids = all_ids
        else:
            train_ids = [all_ids[i] for i in train_indices]
            val_ids = [all_ids[i] for i in val_indices]
        super().__init__(train_ids=train_ids, val_ids=val_ids, **kwargs)
        (
            self.scene_center,
            self.scene_scale,
            self._near,
            self._far,
        ) = load_scene_info(self.data_dir)
        self.test_camera_trajectory = test_camera_trajectory

        self.image_scale = image_scale
        self.shuffle_pixels = shuffle_pixels

        self.rgb_dir = gpath.GPath(data_dir, "rgb", f"{image_scale}x")

        self.annotations_dir = gpath.GPath(data_dir, "annotations")
        self.annotations_file = gpath.GPath(data_dir, "annotations.yml")

        with open(gpath.GPath(data_dir, "mapping.yml")) as f:
            self.id_to_class_mapping = yaml.safe_load(f)

        self.class_to_id_mapping = {
            value: key for key, value in self.id_to_class_mapping.items()
        }

        annotations = {}
        annotations_given = {}
        if self.num_attributes > 0:
            with open(self.annotations_file) as f:
                annotations_in_file = yaml.safe_load(f)
                for entry in annotations_in_file:
                    if entry["frame"] not in annotations:
                        annotations[entry["frame"]] = np.zeros(
                            (self.num_attributes,), dtype=np.float32
                        )
                        annotations_given[entry["frame"]] = np.zeros(
                            (self.num_attributes,), dtype=np.float32
                        )
                    annotations[entry["frame"]][entry["class"]] = entry[
                        "value"
                    ]
                    annotations_given[entry["frame"]][entry["class"]] = 1
        self.annotations = annotations
        self.annotations_given = annotations_given

        self.depth_dir = gpath.GPath(data_dir, "depth", f"{image_scale}x")
        self.camera_type = camera_type
        self.camera_dir = gpath.GPath(data_dir, "camera")

        self.train_metadata_ids = {t_id: i for i, t_id in enumerate(train_ids)}
        # The pair of metadata ids for each val id.
        self.val_metadata_ids = {
            v: (self.train_metadata_ids[l], self.train_metadata_ids[r])
            for v, l, r in zip(val_ids, train_ids[:-1], train_ids[1:])
        }
        # The pair of train ids that correspond to each val id.
        self.val_pivot_ids = {
            v: (l, r)
            for v, l, r in zip(val_ids, train_ids[:-1], train_ids[1:])
        }

        metadata_path = self.data_dir / "metadata.json"
        with metadata_path.open("r") as f:
            self.metadata_dict = json.load(f)

    @property
    def num_attributes(self) -> int:
        return len(self.class_to_id_mapping.keys())

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def camera_ext(self):
        if self.camera_type == "json":
            return ".json"

        raise ValueError(f"Unknown camera_type {self.camera_type}")

    def get_rgb_path(self, item_id):
        return self.rgb_dir / f"{item_id}.png"

    def load_rgb(self, item_id):
        return _load_image(self.rgb_dir / f"{item_id}.png")

    def load_attribute_mask(
        self, item_id: str, height: int, width: int
    ) -> np.ndarray:
        if (self.rgb_dir / f"{item_id}_segmentation.npy").exists():
            return np.load(self.rgb_dir / f"{item_id}_segmentation.npy")
        return load_annotation(
            self.annotations_dir / f"{item_id}.json",
            self.image_scale,
            self.num_attributes,
            height,
            width,
            self.class_to_id_mapping,
            gaussian_splatting_filter_size=self.gaussian_filter_size,
        )

    def load_depth(self, item_id: str) -> np.ndarray:
        return load_depth(
            self.rgb_dir / f"{item_id}_depth.npy", self.near, self.far
        )

    def load_attribute_is_given(self, item_id: str) -> np.ndarray:
        return self.annotations_given.get(
            int(item_id),
            np.zeros((self.num_attributes,), dtype=np.float32),
        )

    def load_attribute_values(self, item_id: str) -> np.ndarray:
        return self.annotations.get(
            int(item_id), np.zeros((self.num_attributes,), dtype=np.float32)
        )

    def load_camera(self, item_id, scale_factor=1.0):
        if isinstance(item_id, gpath.GPath):
            camera_path = item_id
        else:
            if self.camera_type == "proto":
                camera_path = self.camera_dir / f"{item_id}{self.camera_ext}"
            elif self.camera_type == "json":
                camera_path = self.camera_dir / f"{item_id}{self.camera_ext}"
            else:
                raise ValueError(f"Unknown camera type {self.camera_type!r}.")

        return core.load_camera(
            camera_path,
            scale_factor=scale_factor / self.image_scale,
            scene_center=self.scene_center,
            scene_scale=self.scene_scale,
        )

    def glob_cameras(self, path):
        path = gpath.GPath(path)
        return sorted(path.glob(f"*{self.camera_ext}"))

    def load_test_cameras(self, count=None):
        camera_dir = (
            self.data_dir / "camera-paths" / self.test_camera_trajectory
        )
        if not camera_dir.exists():
            logging.warning(
                "test camera path does not exist: %s", str(camera_dir)
            )
            return []
        camera_paths = sorted(camera_dir.glob(f"*{self.camera_ext}"))
        if count is not None:
            stride = max(1, len(camera_paths) // count)
            camera_paths = camera_paths[::stride]
        cameras = utils.parallel_map(self.load_camera, camera_paths)
        return cameras

    def load_points(self, shuffle=False):
        with (self.data_dir / "points.npy").open("rb") as f:
            points = np.load(f)
        points = (points - self.scene_center) * self.scene_scale
        points = points.astype(np.float32)
        if shuffle:
            logging.info("Shuffling points.")
            shuffled_inds = self.rng.permutation(len(points))
            points = points[shuffled_inds]
        logging.info("Loaded %d points.", len(points))
        return points

    def _get_metadata_id(self, item_id):
        if item_id in self.train_metadata_ids:
            return self.train_metadata_ids[item_id]
        elif item_id in self.val_metadata_ids:
            # If the metadata ID is a pair then define a linear interpolation.
            left_id, right_id = self.val_pivot_ids[item_id]
            item_ts = float(self._get_time_id_fallback(item_id))
            left_ts = float(self._get_time_id_fallback(left_id))
            right_ts = float(self._get_time_id_fallback(right_id))
            # Compute what interval the middle frame lies on between the left
            # and right frames. This should be between 0 and 1.
            progression = (item_ts - left_ts) / (right_ts - left_ts)
            assert 0.0 <= progression <= 1.0
            left_metadata = self.train_metadata_ids[left_id]
            right_metadata = self.train_metadata_ids[right_id]
            return left_metadata, right_metadata, progression
        else:
            raise RuntimeError(f"Metadata for item_id {item_id} not known")

    def get_appearance_id(self, item_id):
        return self._get_metadata_id(item_id)

    def get_camera_id(self, item_id):
        raise NotImplementedError()

    def get_warp_id(self, item_id):
        return self._get_metadata_id(item_id)

    def get_time_id(self, item_id):
        return self._get_metadata_id(item_id)

    def _get_time_id_fallback(self, item_id):
        if "time_id" in self.metadata_dict[item_id]:
            return self.metadata_dict[item_id]["time_id"]
        return self.metadata_dict[item_id]["warp_id"]
