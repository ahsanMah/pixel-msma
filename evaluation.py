import os

import numpy as np
import tensorflow as tf
from datetime import datetime

OLD_TF = tf.__version__ < "2.4.0"

if OLD_TF:
    print("Using TF < 2.4:", tf.__version__)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
else:
    AUTOTUNE = tf.data.AUTOTUNE

import configs
import utils
from datasets.dataset_loader import load_data, preprocess
from ood_detection_helper import compute_batched_score_norms
from classifier import load_last_checkpoint
from dgmm import DGMM


def compute_and_save_score_norms(model, dataset, score_cache_filename):

    os.makedirs(score_cache_filename, exist_ok=True)

    # load dataset from tfds with test/ood data
    train, val, test = load_data(dataset, include_ood=True, supervised=False)

    input_shape = utils.get_dataset_image_size(configs.config_values.dataset)
    channels = input_shape[-1]

    # Create a dictionary of score results
    datasets = {
        "val": preprocess(dataset, val, train=False).prefetch(AUTOTUNE),
        "ood": preprocess(dataset, test, train=False).prefetch(AUTOTUNE),
    }

    train_scores = {}

    ds = preprocess(dataset, train, train=False).take(10)
    ds = ds.cache()  # Use same masks for all sigmas
    ds = ds.prefetch(AUTOTUNE)

    train_scores["scores"] = compute_batched_score_norms(model, ds, masked_input=True)

    # Record masks used for scores
    masks = []
    for x in ds:
        _, m = tf.split(x, (channels, 1), axis=-1)
        masks.append(m)

    train_scores["masks"] = np.concatenate(masks, axis=0)
    np.savez_compressed(f"{score_cache_filename}/train", **train_scores)

    scores = {}
    n_iters = 100

    ## Multiple runs of random masks
    for i in range(n_iters):
        seed = 42 + i
        tf.random.set_seed(seed)
        np.random.seed(seed)
        scores = {}
        for name, ds in datasets.items():
            scores[name] = {}

            # ds = preprocess(dataset, ds, train=False)
            # _ds = iter(ds)
            _ds = ds.cache()  # Use same masks for all sigmas

            scores[name]["scores"] = compute_batched_score_norms(
                model, _ds, masked_input=True, seed=seed
            )

            # Record masks used for scores
            masks = []
            for x in _ds:
                _, m = tf.split(x, (channels, 1), axis=-1)
                masks.append(m)
            scores[name]["masks"] = np.concatenate(masks, axis=0)

        np.savez_compressed(f"{score_cache_filename}/eval_{i}", **scores)

    # TODO: Multiple runs of *same* mask for ALL data points

    return scores


def partial_observation_eval(model, complete_model_name):
    max_marginal_ratio = configs.config_values.marginal_ratio
    min_marginal_ratio = configs.config_values.min_marginal_ratio
    dataset = configs.config_values.dataset
    score_cache_dir = os.path.join("score_cache", dataset, complete_model_name)
    os.makedirs(score_cache_dir, exist_ok=True)

    n_intervals = 1 if min_marginal_ratio == max_marginal_ratio else 5
    for ratio in np.linspace(min_marginal_ratio, max_marginal_ratio, num=n_intervals):
        print(f"Evaluating for observation ratio {ratio}")
        configs.config_values.marginal_ratio = ratio
        configs.config_values.min_marginal_ratio = ratio
        score_cache_filename = f"{score_cache_dir}/eval_mr{ratio}"
        compute_and_save_score_norms(model, dataset, score_cache_filename)

        # if not os.path.exists(score_cache_filename):
        #     compute_and_save_score_norms(model, dataset, score_cache_filename)
        # else:
        #     print(f"Score cache '{score_cache_filename}' already exists!")
    return


"""
Helper function for building datasets that can be ingested by DGMM
"""


def build_ds(s, m, batch_sz):
    ll_target = tf.data.Dataset.from_tensor_slices(tf.zeros(shape=(s.shape[0], 1)))
    m = tf.data.Dataset.from_tensor_slices(m)
    s = tf.data.Dataset.from_tensor_slices(s)
    ds = tf.data.Dataset.zip((s, m, ll_target)).cache()

    ds = ds.shuffle(1000)
    ds = ds.batch(batch_sz, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_lls(gmm, ds):
    lls = []  # np.zeros(X_train_pkc.shape[0])

    for s, m, l in ds:
        lls.append(gmm((s, m)))

    return np.concatenate(lls)


def concat_images(dataset_name, ds, masks):
    ds = preprocess(dataset_name, ds, train=False)
    ds_imgs = tf.concat([x for x in ds], axis=0).numpy()
    ds_imgs[..., 1] = masks[..., 0]
    return ds


def msma_dgmm_runner(model_name, dataset, marginal_ratio, include_images=False):

    # TODO: loop over available ratios
    # marginal_ratio = 0.3
    cache_dir = (
        os.path.join("score_cache", "knee", model_name, f"eval_mr{marginal_ratio}")
        + ".npz"
    )

    hparams = "img" if include_images else ""
    model_dir = os.path.join(
        "saved_models", "msma_dgmm", dataset, f"{marginal_ratio}", hparams
    )
    os.makedirs(model_dir, exist_ok=True)

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")

    # TODO: Make val loss compute bits/dim instead
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # an absolute change of less than min_delta, will count as no improvement
            min_delta=1e-2,
            # "no longer improving" being defined as "for at least patience epochs"
            patience=10,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "e{epoch:03d}.ckpt"),
            # Only save a model if `val_loss` has improved.
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1,
            save_freq="epoch",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.9, min_delta=1e-2, patience=1, min_lr=1e-5
        ),
        tf.keras.callbacks.TensorBoard(
            f"./logs/msma_dgmm/{dataset}/{start_time}",
            update_freq=1,
            profile_batch=0,
        ),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(model_dir, "trainlog.csv")),
    ]

    print(f"Loading from {cache_dir}")
    with np.load(cache_dir, allow_pickle=True) as data:
        X_train_dict = data["train"].item()
        X_test_dict = data["val"].item()
        X_ood_dict = data["ood"].item()
    print(
        X_train_dict["scores"].shape,
        X_test_dict["scores"].shape,
        X_ood_dict["scores"].shape,
    )
    s, m = X_train_dict["scores"][:1], X_train_dict["masks"][:1]
    input_shape = m.shape[1:]

    gmm = DGMM(img_shape=input_shape, k_mixt=5, D=10)
    opt = tf.keras.optimizers.RMSprop(
        learning_rate=0.001
    )  # tf.keras.optimizers.Adamax(learning_rate=3e-4)
    gmm.compile(optimizer=opt)

    batch_size = 128
    val_batches = 10

    ### Experimental: Including images used for calculating scores
    if include_images:
        # load dataset from tfds with test/ood data
        train, test, ood = load_data(dataset, include_ood=True, supervised=False)
        X_train_dict["masks"] = concat_images(dataset, train, X_train_dict["masks"])
        X_test_dict["masks"] = concat_images(dataset, test, X_test_dict["masks"])
        X_ood_dict["masks"] = concat_images(dataset, ood, X_ood_dict["masks"])
    #########

    X_train_ds = build_ds(
        X_train_dict["scores"], X_train_dict["masks"], batch_size
    ).skip(val_batches)
    X_val_ds = (
        build_ds(X_train_dict["scores"], X_train_dict["masks"], batch_size)
        .take(val_batches)
        .cache()
    )

    X_test_ds = build_ds(X_test_dict["scores"], X_test_dict["masks"], 128)
    X_ood_ds = build_ds(X_ood_dict["scores"], X_ood_dict["masks"], 128)

    if configs.config_values.resume:
        load_last_checkpoint(gmm, model_dir)

    gmm.fit(X_train_ds, epochs=100, validation_data=X_val_ds, callbacks=callbacks)

    ### Evaluating DGMM
    score_cache_dir = os.path.join("score_cache", dataset, "msma_dgmm")
    os.makedirs(score_cache_dir, exist_ok=True)

    score_cache_filename = f"{score_cache_dir}/ll_{marginal_ratio}"
    gmm.training = False
    gmm_lls = {}
    train_ll = get_lls(gmm, X_train_ds)
    test_ll = get_lls(gmm, X_test_ds)
    ood_ll = get_lls(gmm, X_ood_ds)

    gmm_lls[str(marginal_ratio)] = {"train": train_ll, "test": test_ll, "ood": ood_ll}
    np.savez_compressed(score_cache_filename, **gmm_lls)

    return


def main():

    tf.random.set_seed(42)
    np.random.seed(42)

    # path for saving the model(s)
    save_dir, complete_model_name = utils.get_savemodel_dir()
    # save_dir = save_dir.replace("mr0.3-0.3", "mr0.3")
    model, optimizer, step, ocnn_model, ocnn_optimizer = utils.try_load_model(
        save_dir,
        step_ckpt=configs.config_values.resume_from,
        verbose=True,
        ocnn=configs.config_values.ocnn,
    )

    configs.config_values.global_batch_size = configs.config_values.batch_size

    if configs.config_values.mode == "msma" and configs.config_values.mask_marginals:
        return msma_dgmm_runner(
            complete_model_name,
            configs.config_values.dataset,
            configs.config_values.marginal_ratio,
        )

    if configs.config_values.mask_marginals:
        return partial_observation_eval(model, complete_model_name)

    score_cache_filename = f"score_cache/{complete_model_name}"

    if not os.path.exists(score_cache_filename):
        compute_and_save_score_norms(
            model, configs.config_values.dataset, score_cache_filename
        )
