# @title Define imports and utility functions.
import collections
import functools
import time
from pathlib import Path

import gin
import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
import skvideo.io
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.training import checkpoints
from jax import random

from conerf import (
    configs,
    datasets,
    evaluation,
    gpath,
    image_utils,
    model_utils,
    models,
    schedules,
    utils,
)
from conerf import visualization as viz

flags.DEFINE_enum(
    "mode",
    None,
    ["jax_cpu", "jax_gpu", "jax_tpu"],
    "Distributed strategy approach.",
)

flags.DEFINE_string("base_folder", None, "where to store ckpts and logs")
flags.mark_flag_as_required("base_folder")
flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
flags.DEFINE_multi_string("gin_configs", (), "Gin config files.")
flags.DEFINE_string("betas", None, "Indices from the dataset to be used")
flags.DEFINE_bool(
    "use_lr", False, "Whether use linear regression to predict attributes"
)
FLAGS = flags.FLAGS

MIN_VALUE = 0.0
MAX_VALUE = 1.0
NUM_FRAMES = 1200
NUM_RANDOM_POINTS = 6
BEZIER_VALUES = np.array([[0.0, 0.0], [0.6, 0.0], [0.4, 1.0], [1.0, 1.0]])
NUM_CAMERA_ORBITS = 6

np.random.seed(0xCAFFE)


def interpolate_points(
    point1: np.ndarray, point2: np.ndarray, steps: int
) -> np.ndarray:

    t = np.linspace(0, 1, num=steps)[:, np.newaxis]
    alphas = (
        (1 - t) ** 3 * BEZIER_VALUES[0]
        + 3 * (1 - t) ** 2 * t * BEZIER_VALUES[1]
        + 3 * (1 - t) * t**2 * BEZIER_VALUES[2]
        + t**3 * BEZIER_VALUES[3]
    )[..., 1]
    alphas = alphas[:, np.newaxis]
    new_points = (1 - alphas) * point1[np.newaxis] + alphas * point2[
        np.newaxis
    ]
    return new_points


def generate_points(num_attributes: int):
    points = [[MIN_VALUE] * num_attributes, [MAX_VALUE] * num_attributes]
    for i in range(num_attributes - 1, -1, -1):
        new_point = points[-1][:]
        new_point[i] = MIN_VALUE
        points.append(new_point)
    for i in range(num_attributes - 1, -1, -1):
        new_point = points[-1][:]
        new_point[i] = MAX_VALUE
        points.append(new_point)
    random_points = np.random.uniform(
        MIN_VALUE, MAX_VALUE, size=(NUM_RANDOM_POINTS, num_attributes)
    )

    points.extend(random_points.tolist())
    points.append([0.5] * num_attributes)
    points_array = np.array(points)
    return points_array


def generate_dynamics(points: np.ndarray) -> np.ndarray:
    output = []
    duration_per_combination = NUM_FRAMES // (len(points) - 1)
    for i in range(0, points.shape[0] - 1):
        start_point = points[i]
        end_point = points[i + 1]
        output.append(
            interpolate_points(
                start_point, end_point, duration_per_combination
            )
        )
    return np.concatenate(output, axis=0)


def main(argv):
    jax.config.parse_flags_with_absl()
    tf.config.experimental.set_visible_devices([], "GPU")
    del argv
    logging.info("*** Starting generating video")
    # Assume G3 path for config files when running locally.
    gin_configs = FLAGS.gin_configs

    logging.info("*** Loading Gin configs from: %s", str(gin_configs))
    gin.parse_config_files_and_bindings(
        config_files=gin_configs,
        bindings=FLAGS.gin_bindings,
        skip_unknown=True,
    )
    exp_config = configs.ExperimentConfig()
    train_config = configs.TrainConfig()
    eval_config = configs.EvalConfig()
    dummy_model = models.NerfModel({}, 0, 0, 0)
    # Get directory information.
    exp_dir = gpath.GPath(FLAGS.base_folder)
    if exp_config.subname:
        exp_dir = exp_dir / exp_config.subname
    checkpoint_dir = exp_dir / "checkpoints"
    # Log and create directories if this is the main process.
    if jax.process_index() == 0:
        logging.info("exp_dir = %s", exp_dir)
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=True)

        logging.info("checkpoint_dir = %s", checkpoint_dir)
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Starting process %d. There are %d processes.",
        jax.process_index(),
        jax.process_count(),
    )
    logging.info(
        "Found %d accelerator devices: %s.",
        jax.local_device_count(),
        str(jax.local_devices()),
    )
    logging.info(
        "Found %d total devices: %s.", jax.device_count(), str(jax.devices())
    )

    datasource = exp_config.datasource_cls(
        image_scale=exp_config.image_scale,
        random_seed=exp_config.random_seed,
        # Enable metadata based on model needs.
        use_warp_id=dummy_model.use_warp,
        use_appearance_id=(
            dummy_model.nerf_embed_key == "appearance"
            or dummy_model.hyper_embed_key == "appearance"
        ),
        use_camera_id=dummy_model.nerf_embed_key == "camera",
        use_time=dummy_model.warp_embed_key == "time",
    )

    rng = random.PRNGKey(exp_config.random_seed)
    np.random.seed(exp_config.random_seed + jax.process_index())
    devices_to_use = jax.devices()

    learning_rate_sched = schedules.from_config(train_config.lr_schedule)
    nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
    warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
    hyper_alpha_sched = schedules.from_config(
        train_config.hyper_alpha_schedule
    )
    hyper_sheet_alpha_sched = schedules.from_config(
        train_config.hyper_sheet_alpha_schedule
    )

    rng, key = random.split(rng)
    params = {}
    model, params["model"] = models.construct_nerf(
        key,
        batch_size=eval_config.chunk,
        embeddings_dict=datasource.embeddings_dict,
        near=datasource.near,
        far=datasource.far,
        num_attributes=datasource.num_attributes,
    )

    optimizer_def = optim.Adam(learning_rate_sched(0))
    optimizer = optimizer_def.create(params)

    state = model_utils.TrainState(
        optimizer=optimizer,
        nerf_alpha=nerf_alpha_sched(0),
        warp_alpha=warp_alpha_sched(0),
        hyper_alpha=hyper_alpha_sched(0),
        hyper_sheet_alpha=hyper_sheet_alpha_sched(0),
    )

    optimizer_def = optim.Adam(0.0)
    if train_config.use_weight_norm:
        optimizer_def = optim.WeightNorm(optimizer_def)
    optimizer = optimizer_def.create(params)
    init_state = model_utils.TrainState(optimizer=optimizer)
    del params

    # @title Define pmapped render function.

    devices = jax.devices()

    def _model_fn(key_0, key_1, params, rays_dict, extra_params):
        out = model.apply(
            {"params": params},
            rays_dict,
            extra_params=extra_params,
            rngs={"coarse": key_0, "fine": key_1},
            mutable=False,
            metadata_encoded=FLAGS.use_lr,
        )
        return jax.lax.all_gather(out, axis_name="batch")

    pmodel_fn = jax.pmap(
        # Note rng_keys are useless in eval mode since there's no randomness.
        _model_fn,
        in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
        devices=devices_to_use,
        axis_name="batch",
    )

    render_fn = functools.partial(
        evaluation.render_image,
        model_fn=pmodel_fn,
        device_count=len(devices),
        chunk=eval_config.chunk,
        ret_keys=["rgb", "med_depth", "attribute_rgb"],
    )

    camera_path = Path("camera-paths") / "orbit-mild"

    camera_dir = Path(datasource.data_dir, camera_path)
    logging.info(f"Loading cameras from {camera_dir}")
    test_camera_paths = datasource.glob_cameras(camera_dir)
    test_cameras = utils.parallel_map(
        datasource.load_camera, test_camera_paths, show_pbar=True
    )

    rng = rng + jax.process_index()  # Make random seed separate across hosts.
    _ = random.split(rng, len(devices))
    warp_lr_params = None
    hyper_lr_params = None
    if (exp_dir / "warp_lr.npy").exists():
        warp_lr_params = jnp.load(exp_dir / "warp_lr.npy")
    if (exp_dir / "hyper_lr.npy").exists():
        hyper_lr_params = jnp.load(exp_dir / "hyper_lr.npy")

    last_step = 0

    points = generate_points(datasource.num_attributes)
    dynamics = generate_dynamics(points) * 2 - 1

    while True:
        if not checkpoint_dir.exists():
            logging.info("No checkpoints yet.")
            time.sleep(10)
            continue

        state = checkpoints.restore_checkpoint(checkpoint_dir, init_state)
        state = jax_utils.replicate(state, devices=devices_to_use)
        step = int(state.optimizer.state.step[0])
        if step <= last_step:
            logging.info("No new checkpoints (%d <= %d).", step, last_step)
            time.sleep(10)
            continue

        results = []

        renders_dir = exp_dir / "supplementary" / "frames"

        num_attributes = datasource.num_attributes
        ratio = len(test_cameras) / NUM_FRAMES * NUM_CAMERA_ORBITS
        camera_indices = (
            (np.arange(NUM_FRAMES) * ratio) % len(test_cameras)
        ).astype(np.int)

        for frame_index, attribute_set in enumerate(
            tqdm.tqdm(jnp.array(dynamics))
        ):
            current_camera_index = camera_indices[frame_index]
            frames = collections.defaultdict(list)
            out_dir = renders_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            camera = test_cameras[current_camera_index]
            batch = datasets.camera_to_rays(camera)
            batch["metadata"] = {
                "appearance": jnp.zeros_like(
                    batch["origins"][..., 0, jnp.newaxis],
                    jnp.uint32,
                ),
                "warp": jnp.zeros_like(
                    batch["origins"][..., 0, jnp.newaxis],
                    jnp.uint32,
                ),
                "hyper_embed": (
                    attribute_set[jnp.newaxis, jnp.newaxis]
                    .repeat(repeats=batch["origins"].shape[0], axis=0)
                    .repeat(repeats=batch["origins"].shape[1], axis=1)
                ),
            }

            if FLAGS.use_lr:
                encodings = {}
                if warp_lr_params is not None:
                    encodings["encoded_warp"] = (
                        batch["metadata"]["hyper_embed"] @ warp_lr_params
                    )
                if hyper_lr_params is not None:
                    encodings["encoded_hyper"] = (
                        batch["metadata"]["hyper_embed"] @ hyper_lr_params
                    )
                elif model.hyper_use_warp_embed:
                    encodings["encoded_hyper"] = (
                        batch["metadata"]["hyper_embed"] @ warp_lr_params
                    )
                batch["metadata"] = encodings

            render = render_fn(state, batch, rng=rng)
            rgb = np.array(render["rgb"])
            depth_med = np.array(render["med_depth"])
            results.append((rgb, depth_med))
            depth_viz = viz.colorize(
                depth_med.squeeze(),
                cmin=datasource.near,
                cmax=datasource.far,
                invert=True,
            )

            frames["rgb"].append(image_utils.image_to_uint8(rgb))
            frames["depth"].append(image_utils.image_to_uint8(depth_viz))
            if "attribute_rgb" in render:
                attribute = np.array(nn.sigmoid(render["attribute_rgb"]))
                mask = np.concatenate(
                    np.split(
                        attribute,
                        indices_or_sections=num_attributes,
                        axis=-1,
                    ),
                    axis=1,
                ).repeat(repeats=3, axis=-1)
                overlay = np.clip(
                    (attribute.max(axis=-1, keepdims=True) * 0.5 + rgb * 0.5),
                    a_min=0.0,
                    a_max=1.0,
                )
                frames["mask"].append(image_utils.image_to_uint8(mask))
                frames["overlay"].append(image_utils.image_to_uint8(overlay))

            for key, vals in frames.items():
                PIL.Image.fromarray(vals[-1]).save(
                    (out_dir / f"{key}_{frame_index:04d}.png").as_posix()
                )
        fps = "20"
        for key, vals in frames.items():
            skvideo.io.vwrite(
                (renders_dir.parent / f"{key}.mp4").as_posix(),
                vals,
                inputdict={"-r": fps},
                outputdict={"-r": fps},
            )
        if step >= train_config.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    app.run(main)
