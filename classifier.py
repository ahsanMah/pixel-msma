import os
import sys
import logging
from datetime import datetime

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.compat.v1 import ConfigProto

# from tensorflow.keras.applications.resnet_v2 import preprocess_input

import utils, configs
from datasets.dataset_loader import load_data, preprocess
from ood_detection_helper import get_command_line_args, ood_metrics

config = ConfigProto()
config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

AUTOTUNE = tf.data.experimental.AUTOTUNE


def build_model(dataset, pretrained=False):
    input_shape = utils.get_dataset_image_size(dataset)

    if configs.config_values.mask_marginals:
        input_shape[-1] += 1  # Append a mask channel

    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        input_shape=(input_shape[0], input_shape[1], 3) if pretrained else input_shape,
        include_top=False if pretrained else True,
        weights="imagenet" if pretrained else None,
        classes=None if pretrained else 2,
        pooling="avg",
        classifier_activation=None,
    )

    if pretrained:
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(3, 1),  # 1x1 conv to increase channels
                base_model,
                tf.keras.layers.Dense(2),
            ]
        )
    else:
        model = base_model

    logging.info(model.summary())

    bce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=3e-4)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=optimizer,
        loss=bce,
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model


def build_and_train(dataset, n_epochs=25, pretrained=False):

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # an absolute change of less than min_delta, will count as no improvement
            min_delta=1e-3,
            # "no longer improving" being defined as "for at least patience epochs"
            patience=20,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELDIR, "e{epoch:03d}.ckpt"),
            # Only save a model if `val_loss` has improved.
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1,
            save_freq="epoch",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, min_delta=1e-3, patience=1, min_lr=1e-5
        ),
        tf.keras.callbacks.TensorBoard(
            f"./logs/classifier/{dataset}/{start_time}", update_freq=1
        ),
    ]

    model = build_model(dataset, pretrained)
    train_ds, val_ds, test_ds = load_data(dataset, include_ood=False, supervised=True)

    train_ds = preprocess("knee", train_ds, train=True)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = preprocess("knee", val_ds, train=True)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = preprocess("knee", test_ds, train=False)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=callbacks,
    )

    logging.info("====== Performance on Test Set ======")
    load_last_checkpoint(model, MODELDIR)
    test_preds = model.predict(test_ds)
    test_labels = np.concatenate([np.argmax(l, axis=1) for _, l in test_ds], axis=0)

    test_scores = test_preds[:, 1]
    inlier_test_scores = test_scores[test_labels == 0]
    outlier_test_scores = test_scores[test_labels == 1]

    metrics = ood_metrics(
        inlier_test_scores, outlier_test_scores, verbose=True, plot=True
    )
    logging.info(metrics)

    return history


def load_last_checkpoint(model, savedir):
    checkpoint = tf.train.latest_checkpoint(str(os.path.abspath(savedir)))
    logging.info(f"Found ckpt: {checkpoint}")
    print(f"Found ckpt: {checkpoint}")
    if checkpoint:
        model.load_weights(checkpoint)
    else:
        Warning(f"No checkpoint found in dir {savedir}")


if __name__ == "__main__":
    # TODO: Add evaluation functions

    parser = argparse.ArgumentParser(
        description="CLI Options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment", default="train", help="what experiment to run")

    parser.add_argument("--dataset", default="knee", help="msma available dataset")

    parser.add_argument(
        "--category",
        default="kspace",
        type=str,
        help="which class label to use for training, applies to mvtec, knees",
    )

    parser.add_argument("--batch_size", default=128, type=int, help="batch size")

    parser.add_argument(
        "--n_epochs", default=25, type=int, help="number of epochs to train"
    )

    parser.add_argument(
        "--marginal_ratio",
        default=0.0,
        type=float,
        help="ratio of marginals to keep (randomly selected bewteen min_marginal_ratio and this value)",
    )

    parser.add_argument(
        "--min_marginal_ratio",
        default=0.0,
        type=float,
        help="min ratio of marginals to keep",
    )

    parser.add_argument(
        "--pretrained",
        default=0.0,
        action="store_true",
        help="whetehr to use imagenet pretrianing",
    )

    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    category = args.category
    batch_size = args.batch_size
    marginal_ratio = args.marginal_ratio
    model_path = "saved_models"
    epochs = args.n_epochs
    pretrained = args.pretrained

    # Data configuration values
    config_args = get_command_line_args(
        [
            "--dataset=" + dataset,
            "--class_label=" + str(category),
            "--batch_size=" + str(batch_size),
            "--marginal_ratio=" + str(marginal_ratio),
            "--min_marginal_ratio=" + str(marginal_ratio),
        ]
    )
    configs.config_values = config_args
    configs.config_values.global_batch_size = batch_size

    dirname = "classifier"
    if pretrained:
        dirname += "_pretrained"

    MODELDIR = os.path.join(model_path, dirname, dataset, f"{marginal_ratio}")
    os.makedirs(MODELDIR, exist_ok=True)

    logfile = os.path.join(MODELDIR, "run.log")
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)

    build_and_train(dataset, epochs, pretrained=pretrained)
