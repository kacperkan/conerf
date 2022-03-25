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
flags.DEFINE_bool(
    "use_lr", False, "Whether use linear regression to predict attributes"
)
FLAGS = flags.FLAGS


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
    attribute_sheet_alpha_sched = schedules.from_config(
        train_config.attribute_sheet_alpha_schedule
    )
    masking_gamma_sched = schedules.from_config(
        train_config.masking_gamma_schedule
    )

    rng, key = random.split(rng)
    params = {}
    model, params["model"] = models.construct_nerf(
        key,
        batch_size=train_config.batch_size,
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
        attribute_sheet_alpha=attribute_sheet_alpha_sched(0),
        masking_gamma=masking_gamma_sched(0),
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
    )

    camera_path = Path("camera-paths") / "orbit-mild"

    camera_dir = Path(datasource.data_dir, camera_path)
    logging.info(f"Loading cameras from {camera_dir}")
    test_camera_paths = datasource.glob_cameras(camera_dir)
    test_cameras = utils.parallel_map(
        datasource.load_camera, test_camera_paths, show_pbar=True
    )
    camera = test_cameras[len(test_cameras) // 2]

    results = []

    num_attributes = datasource.num_attributes
    attribute_values = jnp.array(
        [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    renders_dir = exp_dir / "attributes_renders"
    logging.info(
        f"{len(attribute_values) * num_attributes* len(test_cameras)} "
        "sequences to generate"
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
        i = 0
        for attribute_index in range(num_attributes):
            frames = collections.defaultdict(list)
            for attribute_value_index, attribute_value in enumerate(
                attribute_values
            ):
                logging.info(
                    f"Rendering frame {i + 1}/"
                    f"{num_attributes * len(attribute_values)}"
                )
                out_dir = (
                    renders_dir
                    / f"attribute_{attribute_index}"
                    / f"{attribute_value_index}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                attribute_set = jnp.zeros((num_attributes,), dtype=jnp.float32)
                attribute_set = attribute_set.at[attribute_index].set(
                    attribute_value
                )

                batch = datasets.camera_to_rays(camera)
                batch["metadata"] = {
                    "appearance": jnp.zeros_like(
                        batch["origins"][..., 0, jnp.newaxis], jnp.uint32
                    ),
                    "warp": jnp.zeros_like(
                        batch["origins"][..., 0, jnp.newaxis], jnp.uint32
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
                if model.use_masking:
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
                        (
                            attribute.max(axis=-1, keepdims=True) * 0.5
                            + rgb * 0.5
                        ),
                        a_min=0.0,
                        a_max=1.0,
                    )
                    frames["mask"].append(image_utils.image_to_uint8(mask))
                    frames["overlay"].append(
                        image_utils.image_to_uint8(overlay)
                    )

                for key, vals in frames.items():
                    PIL.Image.fromarray(vals[-1]).save(
                        (out_dir / "{}.png".format(key)).as_posix()
                    )
                i += 1

            fps = str(len(attribute_values) / 3)
            for key, vals in frames.items():
                skvideo.io.vwrite(
                    (out_dir.parent / f"{key}.mp4").as_posix(),
                    vals,
                    inputdict={"-r": fps},
                    outputdict={"-r": fps},
                )
        if step >= train_config.max_steps:
            break
        last_step = step


if __name__ == "__main__":
    app.run(main)
