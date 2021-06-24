import os

import numpy as np
import tensorflow as tf

OLD_TF = tf.__version__ < "2.4.0"

if OLD_TF:
    print("Using TF < 2.4:", tf.__version__)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
else:
    AUTOTUNE = tf.data.AUTOTUNE

import configs
import utils
from datasets.dataset_loader import load_data, preprocess
from ood_detection_helper import compute_weighted_scores


def compute_and_save_score_norms(model, dataset, score_cache_filename):

    # load dataset from tfds with test/ood data
    train, val, test = load_data(dataset, include_ood=True)

    # Create a dictionary of score results
    datasets = {
        "train": train,
        "val": val,
        "ood": test,
    }
    scores = {}

    for name, ds in datasets.items():
        ds = preprocess(dataset, ds, train=False)
        ds = ds.batch(configs.config_values.batch_size)
        ds = ds.prefetch(AUTOTUNE)
        scores[name] = compute_weighted_scores(model, ds).numpy()

    np.savez_compressed(score_cache_filename, **scores)

    return scores


def main():

    # path for saving the model(s)
    save_dir, complete_model_name = utils.get_savemodel_dir()

    model, optimizer, step, ocnn_model, ocnn_optimizer = utils.try_load_model(
        save_dir,
        step_ckpt=configs.config_values.resume_from,
        verbose=True,
        ocnn=configs.config_values.ocnn,
    )

    score_cache_filename = f"score_cache/{complete_model_name}"

    if not os.path.exists(score_cache_filename):
        compute_and_save_score_norms(
            model, configs.config_values.dataset, score_cache_filename
        )
