import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import PIL.Image
import tqdm
import yaml

np.random.seed(0)


def process_matrix(matrix: np.ndarray):
    location, rotation = matrix[:3, 3], matrix[:3, :3]

    # rotation[:, 0] = -rotation[:, 0]
    rotation[:, 1] = -rotation[:, 1]
    rotation[:, 2] = -rotation[:, 2]
    rotation = rotation.T

    return location, rotation


def process_seg_map(array: np.ndarray) -> np.ndarray:
    array[array == 1] = 1
    array = array - 1
    unique_values = np.unique(array)
    one_hot_array = np.zeros(
        array.shape + (len(unique_values),), dtype=np.float32
    )
    for val in unique_values:
        one_hot_array[array == val, val] = 1

    return one_hot_array


def resize(
    image: PIL.Image.Image, width=None, height=None, mode=PIL.Image.BICUBIC
):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.size[::-1]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    # resized = image.resize(dim, resample=mode)

    # return the resized image
    return image


def remove_alpha(image: PIL.Image.Image) -> PIL.Image.Image:
    return PIL.Image.fromarray(np.array(image)[..., :-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--num_cameras", type=int, default=100)
    parser.add_argument("--use_all_annotations", action="store_true")
    parser.add_argument("--annotations", type=float, default=0.05)

    args = parser.parse_args()
    scales = 1, 2, 4, 8

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    rgb_raw_dir = output_folder / "rgb-raw"
    rgb_dir = output_folder / "rgb"

    rgb_raw_dir.mkdir(exist_ok=True, parents=True)
    rgb_dir.mkdir(exist_ok=True, parents=True)

    sorted_images = list(
        sorted(
            input_folder.glob("rgba*.png"),
            key=lambda x: int(x.with_suffix("").name.replace("rgba_", "")),
        )
    )

    example_image = PIL.Image.open(sorted_images[0])

    # for index, image in enumerate(tqdm.tqdm(sorted_images)):
    #     new_name = f"{index:06d}.png"
    #     new_depth_name = f"{index:06d}_depth.npy"
    #     new_seg_name = f"{index:06d}_segmentation.npy"
    #     shutil.copy(image, rgb_raw_dir / new_name)

    #     for scale in scales:
    #         out_dir = rgb_dir / f"{scale}x"
    #         out_dir.mkdir(parents=True, exist_ok=True)

    #         resize(
    #             PIL.Image.open(image.as_posix()),
    #             width=example_image.width // scale,
    #             height=example_image.height // scale,
    #         ).save(out_dir / new_name)
    #         np.save(
    #             out_dir / new_depth_name,
    #             np.array(
    #                 resize(
    #                     PIL.Image.open(
    #                         image.parent
    #                         / image.with_suffix(".tiff").name.replace(
    #                             "rgba_", "depth_"
    #                         )
    #                     ),
    #                     width=example_image.width // scale,
    #                     height=example_image.height // scale,
    #                 )
    #             ),
    #         )
    #         np.save(
    #             out_dir / new_seg_name,
    #             process_seg_map(
    #                 np.array(
    #                     resize(
    #                         PIL.Image.open(
    #                             image.parent
    #                             / image.name.replace("rgba_", "segmentation_")
    #                         ),
    #                         mode=PIL.Image.NEAREST,
    #                         width=example_image.width // scale,
    #                         height=example_image.height // scale,
    #                     )
    #                 )
    #             ),
    #         )

    with open(output_folder / "metadata.json", "w") as f:
        json.dump(
            {
                f"{index:06d}": {
                    "warp_id": index,
                    "appearance_id": index,
                    "camera_id": index,
                }
                for index in range(len(sorted_images))
            },
            f,
        )

    with open(output_folder / "dataset.json", "w") as f:
        ids = [f"{i:06d}" for i in range(len(sorted_images))]
        train_ids = ids[::2]
        valid_ids = ids[1::2]
        json.dump(
            {
                "count": len(sorted_images),
                "num_examplars": len(train_ids),
                "ids": ids,
                "train_ids": train_ids,
                "val_ids": valid_ids,
            },
            f,
        )

    camera_matrices = np.load(input_folder / "camera.npy")

    camera_dir = output_folder / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    for i, matrix in enumerate(camera_matrices):
        position, orientation = process_matrix(matrix)

        orientation = orientation.tolist()
        position = position.tolist()

        focal_length = 50 * example_image.width / 36
        principal_point = [example_image.width / 2, example_image.height / 2]

        with open(camera_dir / f"{i:06d}.json", "w") as f:
            json.dump(
                {
                    "orientation": orientation,
                    "position": position,
                    "focal_length": focal_length,
                    "principal_point": principal_point,
                    "skew": 0.0,
                    "pixel_aspect_ratio": 1.0,
                    "radial_distortion": [0.0, 0.0, 0.0],
                    "tangential_distortion": [0.0, 0.0, 0.0],
                    "image_size": [example_image.width, example_image.height],
                },
                f,
            )

    fake_camera_matrices = np.load(input_folder / "fake_camera.npy")
    orbit_cameras = output_folder / "camera-paths" / "orbit-mild"
    orbit_cameras.mkdir(parents=True, exist_ok=True)

    for i, matrix in enumerate(fake_camera_matrices):
        position, orientation = process_matrix(matrix)

        orientation = orientation.tolist()
        position = position.tolist()
        focal_length = 50 * example_image.width / 36
        principal_point = [example_image.width / 2, example_image.height / 2]

        with open(orbit_cameras / f"{i:06d}.json", "w") as f:
            json.dump(
                {
                    "orientation": orientation,
                    "position": position,
                    "focal_length": focal_length,
                    "principal_point": principal_point,
                    "skew": 0.0,
                    "pixel_aspect_ratio": 1.0,
                    "radial_distortion": [0.0, 0.0, 0.0],
                    "tangential_distortion": [0.0, 0.0, 0.0],
                    "image_size": [example_image.width, example_image.height],
                },
                f,
            )

    id_to_cls = {0: "bunny", 1: "suzanne", 2: "teapot"}
    cls_to_id = {kls: id for id, kls in id_to_cls.items()}

    with open(output_folder / "mapping.yml", "w") as f:
        yaml.safe_dump(id_to_cls, f)

    bbox = [[-10, -10, -10], [10.0, 10.0, 10.0]]
    scale = 1.0 / np.linalg.norm((np.array(bbox[0]) - np.array(bbox[1])))
    with open(output_folder / "scene.json", "w") as f:
        json.dump(
            {
                "scale": 1.0,
                "center": [0.0, 0.0, 0.0],
                "bbox": bbox,
                "near": 0.1,
                "far": 15.0,
            },
            f,
        )

    positions = {
        "bunny": np.load(input_folder / "bunny.npy"),
        "suzanne": np.load(input_folder / "suzanne.npy"),
        "teapot": np.load(input_folder / "teapot.npy"),
    }

    times = {
        "bunny": np.load(input_folder / "bunny_time.npy"),
        "suzanne": np.load(input_folder / "suzanne_time.npy"),
        "teapot": np.load(input_folder / "teapot_time.npy"),
    }

    times = {
        key: (time - time.min()) / (time.max() - time.min() + 1e-7) * 2 - 1
        for key, time in times.items()
    }

    annotated_frames = []

    all_indices = []

    for i, (key, time) in enumerate(times.items()):
        if not args.use_all_annotations:
            num_annotations = int(len(time) * args.annotations)
            where_min = time.argmin().item()
            where_max = time.argmax().item()
            indices = [where_min, where_max] + np.random.choice(
                list(set(range(len(time))) - set([where_min, where_max])),
                size=num_annotations - 2,
            ).tolist()
        else:
            indices = list(range(len(time)))
        all_indices += indices

        for index in indices:
            annotated_frames.append(
                {
                    "class": cls_to_id[key],
                    "frame": index,
                    "value": time[index].item(),
                }
            )
            for other_key, other_times in times.items():
                if other_key == key:
                    continue
                annotated_frames.append(
                    {
                        "class": cls_to_id[other_key],
                        "frame": index,
                        "value": other_times[index].item(),
                    }
                )

    with open(output_folder / "annotations.yml", "w") as f:
        yaml.safe_dump(annotated_frames, f)


if __name__ == "__main__":
    main()
