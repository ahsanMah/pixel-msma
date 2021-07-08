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
        # ds = ds.batch(configs.config_values.batch_size)
        ds = ds.prefetch(AUTOTUNE)
        scores[name] = compute_weighted_scores(model, ds).numpy()

    np.savez_compressed(score_cache_filename, **scores)

    return scores


def partial_observation_eval(model, complete_model_name):
    max_marginal_ratio = configs.config_values.marginal_ratio
    min_marginal_ratio = configs.config_values.min_marginal_ratio
    dataset = configs.config_values.dataset
    score_cache_dir = os.path.join("score_cache", dataset, complete_model_name)
    os.makedirs(score_cache_dir, exist_ok=True)

    for ratio in np.linspace(min_marginal_ratio, max_marginal_ratio, num=5):
        print(f"Evaluating for observation ratio {ratio}")
        configs.config_values.marginal_ratio = ratio
        configs.config_values.min_marginal_ratio = ratio
        score_cache_filename = f"{score_cache_dir}/eval_mr{ratio}"

        if not os.path.exists(score_cache_filename):
            compute_and_save_score_norms(model, dataset, score_cache_filename)
        else:
            print(f"Score cache '{score_cache_filename}' already exists!")
    return


def main():

    tf.random.set_seed(42)
    np.random.seed(42)

    # path for saving the model(s)
    save_dir, complete_model_name = utils.get_savemodel_dir()

    model, optimizer, step, ocnn_model, ocnn_optimizer = utils.try_load_model(
        save_dir,
        step_ckpt=configs.config_values.resume_from,
        verbose=True,
        ocnn=configs.config_values.ocnn,
    )

    configs.config_values.global_batch_size = configs.config_values.batch_size

    if configs.config_values.mask_marginals:
        return partial_observation_eval(model, complete_model_name)

    score_cache_filename = f"score_cache/{complete_model_name}"

    if not os.path.exists(score_cache_filename):
        compute_and_save_score_norms(
            model, configs.config_values.dataset, score_cache_filename
        )
