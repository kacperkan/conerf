import gin
import jax
import jax.numpy as jnp
import tensorflow as tf
from absl import app, flags, logging
from flax import jax_utils, optim
from flax.training import checkpoints
from jax import random
from jax.config import config

from conerf import configs, gpath, model_utils, models

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
FLAGS = flags.FLAGS

config.update("jax_log_compiles", True)


def main(argv):
    jax.config.parse_flags_with_absl()
    tf.config.experimental.set_visible_devices([], "GPU")
    del argv
    logging.info("*** Starting experiment")
    gin_configs = FLAGS.gin_configs

    logging.info("*** Loading Gin configs from: %s", str(gin_configs))
    gin.parse_config_files_and_bindings(
        config_files=gin_configs,
        bindings=FLAGS.gin_bindings,
        skip_unknown=True,
    )

    # Load configurations.
    exp_config = configs.ExperimentConfig()
    train_config = configs.TrainConfig()
    eval_config = configs.EvalConfig()

    exp_dir = gpath.GPath(FLAGS.base_folder)
    if exp_config.subname:
        exp_dir = exp_dir / exp_config.subname
    logging.info("\texp_dir = %s", exp_dir)
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = exp_dir / "checkpoints"
    logging.info("\tcheckpoint_dir = %s", checkpoint_dir)

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

    rng = random.PRNGKey(20200823)

    devices_to_use = jax.local_devices()

    logging.info("Creating datasource")
    dummy_model = models.NerfModel({}, 0, 0, 0)
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
    optimizer_def = optim.Adam(0.0)
    if train_config.use_weight_norm:
        optimizer_def = optim.WeightNorm(optimizer_def)
    optimizer = optimizer_def.create(params)
    init_state = model_utils.TrainState(optimizer=optimizer)
    del params

    state = checkpoints.restore_checkpoint(checkpoint_dir, init_state)
    state = jax_utils.replicate(state, devices=devices_to_use)

    gt_indices = list(sorted(datasource.annotations.keys()))
    frames_with_gt = jnp.array(gt_indices)

    gt_attributes = jnp.stack(
        [datasource.load_attribute_values(index) for index in gt_indices],
        axis=0,
    )
    if model.use_warp:
        gt_betas = model.apply(
            {"params": jax_utils.unreplicate(state.optimizer.target["model"])},
            {model.warp_embed_key: frames_with_gt},
            method=model.encode_warp_embed,
        )
        params = jnp.linalg.pinv(gt_attributes.T @ gt_attributes) @ (
            gt_attributes.T @ gt_betas
        )
        jnp.save(exp_dir / "warp_lr", params)
    if model.has_hyper_embed:
        if not model.hyper_use_warp_embed:
            gt_betas = model.apply(
                {
                    "params": jax_utils.unreplicate(
                        state.optimizer.target["model"]
                    )
                },
                {model.hyper_embed_key: frames_with_gt},
                method=model.encode_hyper_embed,
            )
            params = jnp.linalg.pinv(gt_attributes.T @ gt_attributes) @ (
                gt_attributes.T @ gt_betas
            )
            jnp.save(exp_dir / "hyper_lr", params)
    logging.info("Fitted linear regression")


if __name__ == "__main__":
    app.run(main)
