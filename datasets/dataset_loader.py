import os
import pickle
import re
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import configs
import utils
import yaml
import pathlib

from .mri_utils import complex_magnitude
from .fastmri import FastKnee, FastKneeTumor

AUTOTUNE = tf.data.experimental.AUTOTUNE
# AUTOTUNE = tf.data.AUTOTUNE

CIFAR_LABELS = [
    "background",
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

BRAIN_LABELS = [
    "background",
    "CSF",
    "gray matter",
    "white matter",
    "deep gray matter",
    "brain stem",
    "cerebellum",
]

# with open("./datasets/data_configs.yaml") as f:
#     dataconfig = yaml.full_load(f)

"""
supervised: To load data for binary classification between inliers and outliers
"""


def load_data(dataset_name, include_ood=False, supervised=False):
    # load data from tfds

    if "knee" in dataset_name:
        return load_knee_data(include_ood, supervised)

    if "mvtec" in dataset_name:
        base_path = configs.dataconfig[dataset_name]["datadir"]
        dataset_dir = f"/{base_path}/{configs.config_values.class_label}/"

        with tf.device("cpu"):
            builder = tfds.ImageFolder(dataset_dir)
            val_size = int(0.1 * builder.info.splits["train"].num_examples)
            configs.dataconfig[dataset_name]["n_samples"] = builder.info.splits[
                "train"
            ].num_examples

            ds = builder.as_dataset(split="train", shuffle_files=False)
            train_ds = ds.skip(val_size)  # ds[val_size:]
            val_ds = ds.take(val_size)  # ds[:val_size]

        return train_ds, val_ds

    # Optionally split training data
    if configs.config_values.split[0] != "100":
        split = configs.config_values.split
        split = [
            "train[:{}%]".format(split[0]),
            "train[-{}%:]".format(split[1]),
            "test",
        ]
    else:
        split = ["train", "test"]
    print("Split:", split)

    if dataset_name in ["masked_fashion", "blown_fashion", "blown_masked_fashion"]:
        dataset_name = "fashion_mnist"

    if dataset_name == "multiscale_cifar10":
        dataset_name = "cifar10"

    if dataset_name in ["masked_cifar10", "seg_cifar10"]:
        with open("data/masked_cifar10/masked_cifar10_strict.p", "rb") as f:
            data = pickle.load(f)
            train = tf.data.Dataset.from_tensor_slices(data)
        with open("data/masked_cifar10/masked_cifar10_strict_test.p", "rb") as f:
            data = pickle.load(f)
            test = tf.data.Dataset.from_tensor_slices(data)
        return train, test

    if "brain" in dataset_name:
        return load_brain_data()

    if "circles" == dataset_name:
        return load_circles()

    if "pet" in dataset_name:
        dataset = tfds.load("oxford_iiit_pet")
        train, test = dataset["train"], dataset["test"]
        train = train.concatenate(
            test.shuffle(4000, seed=2020, reshuffle_each_iteration=False).take(3000)
        )
        test = test.skip(3000)
    else:
        data_generators = tfds.load(
            name=dataset_name, batch_size=-1, shuffle_files=False, split=split
        )

        # First and last will always be train/test
        # Potentially split could include a tune set which will be used later for learning the density
        # and will be ignored by score matching
        train = tf.data.Dataset.from_tensor_slices(data_generators[0]["image"])
        test = tf.data.Dataset.from_tensor_slices(data_generators[-1]["image"])

    return train, test


def load_knee_data(include_ood=False, supervised=False):

    category = configs.config_values.class_label
    complex_input = "complex" in configs.config_values.dataset
    _key = "datadir"
    if configs.config_values.longleaf:
        _key += "_longleaf"

    max_marginal_ratio = configs.config_values.marginal_ratio
    mask_marginals = configs.config_values.mask_marginals
    datadir = configs.dataconfig["knee"][_key]
    train_dir = os.path.join(datadir, "singlecoil_train")
    val_dir = os.path.join(datadir, "singlecoil_val")
    test_dir = os.path.join(datadir, "singlecoil_test_v2")
    img_sz = configs.dataconfig[configs.config_values.dataset]["image_size"]
    img_h, img_w, c = [int(x.strip()) for x in img_sz.split(",")]

    def normalize(img, complex_input=False, quantile=0.999):

        # Complex tensors are 2D
        if complex_input:
            h = np.quantile(img.reshape(-1, 2), q=quantile, axis=0)
            # l = np.min(img.reshape(-1, 2), axis=0)
            l = np.quantile(img.reshape(-1, 2), q=(1 - quantile) / 10, axis=0)
            # print(h,l)
        else:
            h = np.quantile(img, q=quantile)
            # l = np.min(img)
            l = np.quantile(img, q=(1 - quantile) / 10)

        # Min Max normalize
        img = (img - l) / (h - l)
        img = np.clip(
            img,
            0.0,
            1.0,
        )

        # Rescale to [-1, 1] for ResNets
        if supervised:
            img = (img * 2) - 1

        return img

    def make_generator(ds, ood=False):

        if complex_input:
            # Treat input as a 3D tensor (2 channels: real + imag)
            preprocessor = lambda x: np.stack([x.real, x.imag], axis=-1)
            normalizer = lambda x: normalize(x, complex_input=True)
        else:
            preprocessor = lambda x: complex_magnitude(x).numpy()[..., np.newaxis]
            normalizer = lambda x: normalize(x)

        label = 1 if ood else 0

        # TODO: Build complex loader for img

        def tf_gen_img():
            for k, x in ds:
                img = preprocessor(x)
                img = normalizer(img)
                yield img

        def tf_gen_ksp():
            for k, x in ds:
                img = preprocessor(k)
                img = normalizer(img)

                if supervised:
                    yield img, tf.one_hot(label, depth=2)
                else:
                    yield img

        if "kspace" in category:
            print(f"Training on {'complex' if complex_input else 'image'} kspace...")
            return tf_gen_ksp

        # Default to target image as category
        print(f"Training on {'complex' if complex_input else 'image'} mri...")
        return tf_gen_img

    def build_ds(datadir, ood=False):

        if supervised:
            output_type = (tf.float32, tf.int32)
            output_shape = (
                tf.TensorShape([img_h, img_w, c]),
                tf.TensorShape([2]),
            )
        else:
            output_type = tf.float32
            output_shape = tf.TensorShape([img_h, img_w, c])

        dataset = FastKneeTumor(datadir) if ood else FastKnee(datadir)
        ds = tf.data.Dataset.from_generator(
            make_generator(dataset, ood=ood),
            output_type,
            output_shape,
            # output_signature=(tf.TensorSpec(shape=(img_h, img_w, c), dtype=tf.float32)),
        )

        return ds

    test_slices = 2000
    train_ds = build_ds(train_dir)
    val_ds = build_ds(val_dir).skip(test_slices)
    test_ds = build_ds(val_dir).take(test_slices)

    if supervised:
        ood_ds_train = build_ds(train_dir, ood=True)
        ood_ds_val = build_ds(val_dir, ood=True).skip(test_slices)
        ood_ds_test = build_ds(val_dir, ood=True).take(test_slices)

        train_ds = tf.data.experimental.sample_from_datasets([train_ds, ood_ds_train])
        val_ds = tf.data.experimental.sample_from_datasets([val_ds, ood_ds_val])
        test_ds = test_ds.concatenate(ood_ds_test)

        return train_ds, val_ds, test_ds

    # The datsets used to train MSMA
    if include_ood:
        ood_ds = build_ds(val_dir, ood=True).take(test_slices)
        train_ds = train_ds.concatenate(val_ds)
        return train_ds, test_ds, ood_ds

    return train_ds, val_ds


def load_circles():
    with open("data/circles/train_smooth_64x64.p", "rb") as f:
        data = pickle.load(f)
        train = tf.data.Dataset.from_tensor_slices(data)

    with open("data/circles/test_smooth_64x64.p", "rb") as f:
        data = pickle.load(f)
        test = tf.data.Dataset.from_tensor_slices(data)

    return train, test


def load_brain_data():

    # Change to whichever dir you download the data to
    DATA_DIR = "/home/Developer/anodetect/data/processed/images/"
    train_paths = glob.glob(DATA_DIR + "/train/*")
    test_paths = glob.glob(DATA_DIR + "/test/*")

    # Create a dictionary describing the features.
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "segmentation": tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function
    def _parse_record(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    @tf.function
    def _parse_mri(example_proto):

        # Get the record and put it into a feature dict
        image_features = _parse_record(example_proto)

        # Deserialize the mri array
        mri = tf.io.parse_tensor(image_features["image"], out_type=tf.float32)
        mask = tf.io.parse_tensor(image_features["mask"], out_type=tf.float32)
        seg = tf.io.parse_tensor(image_features["segmentation"], out_type=tf.float32)

        x = tf.concat((mri, mask, seg), axis=-1)
        return x

    train = tf.data.TFRecordDataset(train_paths).map(_parse_mri)
    test = tf.data.TFRecordDataset(test_paths).map(_parse_mri)

    return train, test


@tf.function
def concat_mask(x):
    mask = tf.cast(x > 0, dtype=tf.float32)
    return tf.concat((x, mask), axis=-1)


# @tf.autograph.experimental.do_not_convert


@tf.function
def pad(x):
    offset_h = tf.random.uniform([1], minval=0, maxval=28, dtype=tf.dtypes.int32)[0]
    offset_w = tf.random.uniform([1], minval=0, maxval=28, dtype=tf.dtypes.int32)[0]
    x = tf.image.pad_to_bounding_box(
        x,
        offset_height=offset_h,
        offset_width=offset_w,
        target_height=56,
        target_width=56,
    )
    return x


@tf.function
def preproc_pet_masks(x):
    # resize + rescale [0,255] -> [0,1]
    img = tf.image.resize(x["image"], (64, 64)) / 255
    mask = tf.image.resize(x["segmentation_mask"], (64, 64))
    # 1= background, 2,3=Foreground
    mask = tf.cast(mask != 2, dtype=tf.float32)

    return tf.concat((img, mask), axis=-1)


@tf.function
def preproc_cifar_masks(x):
    img, mask = tf.split(x, (3, 1), axis=-1)
    img = img / 255
    mask = tf.cast(mask > 0, dtype=tf.float32)  # 0= background, 1=Foreground

    return tf.concat((img, mask), axis=-1)


@tf.function
def preproc_cifar_segs(x):
    img, seg = tf.split(x, (3, 1), axis=-1)
    img = img / 255
    # 0 = background, >0 = CIFAR label
    seg = tf.one_hot(tf.squeeze(seg), depth=11)

    return tf.concat((img, seg), axis=-1)  # Shape = 32x32x(3+11)


@tf.function
def preproc_cifar_multiscale(x):
    x = x / 255
    x_small_scale = tf.image.resize(x, (8, 8), method="bilinear")
    x_small_scale = tf.image.resize(x_small_scale, (32, 32), method="nearest")

    return tf.concat((x, x_small_scale), axis=-1)  # Shape = 32x32x(3+3)


@tf.function
def get_brain_only(x):
    img, mask, seg = tf.split(x, 3, axis=-1)
    img = tf.expand_dims(img, axis=-1)
    return img


@tf.function
def get_brain_masks(x):
    img, mask, seg = tf.split(x, 3, axis=-1)
    x = tf.stack((img, mask), axis=-1)
    return x


@tf.function
def get_brain_segs(x):
    img, mask, seg = tf.split(x, 3, axis=-1)
    img = tf.expand_dims(img, axis=-1)
    seg = tf.cast(tf.squeeze(seg), dtype=tf.int32)
    seg = tf.one_hot(seg, depth=7)
    x = tf.concat((img, seg), axis=-1)
    return x


@tf.function
def mvtec_preproc(x_batch):
    x = x_batch["image"]
    shape = configs.dataconfig[configs.config_values.dataset]["downsample"]
    img_sz = int(shape.split(",")[0].strip())
    x = tf.image.resize(x, (img_sz, img_sz))
    # #     x.set_shape((96, 96, 3))
    print("Resized:", x.shape, img_sz)
    x = x / 255
    return x


@tf.function
def mvtec_aug(x):
    shape = configs.dataconfig[configs.config_values.dataset]["downsample"]
    img_sz = int(shape.split(",")[0].strip())

    shape = configs.dataconfig[configs.config_values.dataset]["shape"]
    crop_sz = int(shape.split(",")[0].strip())
    print("Crop:", crop_sz)

    translate_ratio = 0.5 * (crop_sz / img_sz)

    x = tfa.image.rotate(x, tf.random.uniform((1,), 0, np.pi / 2))
    x = tfa.image.translate(
        x,
        tf.random.uniform((1, 2), -translate_ratio * img_sz, translate_ratio * img_sz),
    )
    x = tf.image.resize_with_crop_or_pad(x, crop_sz, crop_sz)
    x = tf.image.random_hue(x, max_delta=0.05)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_brightness(x, max_delta=0.2)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


@tf.function
def knee_preproc(x, l=None):
    # img_sz = configs.dataconfig[configs.config_values.dataset]["image_size"]
    # down_sz = configs.dataconfig[configs.config_values.dataset]["downsample_size"]

    # h, w, c = [int(x.strip()) for x in down_sz.split(",")]
    # x = tf.image.resize(x, (h, w), method="lanczos5")
    print("Resized:", x.shape)

    if l is not None:
        return x, l

    return x


@tf.function
def knee_preproc_hres(x):
    return x


# TODO: Confirm these with Martin
@tf.function
def knee_aug(x):
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_brightness(x, max_delta=0.2)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def np_build_mask_fn(constant_mask=False):

    # Building mask of random columns to **keep**
    # batch_sz, img_h, img_w, c = x.shape
    img_sz = configs.dataconfig[configs.config_values.dataset]["downsample_size"]
    img_h, img_w, c = [int(x.strip()) for x in img_sz.split(",")]

    # We do *not* want to mask out the middle (low) frequencies
    # Keeping 10% of low freq is equivalent to Scenario-30L in activemri paper
    low_freq_start = int(0.45 * img_w)
    low_freq_end = img_w - int(0.45 * img_w)
    low_freq_cols = np.arange(low_freq_start, low_freq_end)

    high_freq_cols = np.concatenate(
        (np.arange(0, low_freq_start), np.arange(low_freq_end, img_w))
    )

    def apply_random_mask(x):
        np.random.shuffle(high_freq_cols)
        rand_ratio = np.random.uniform(
            low=configs.config_values.min_marginal_ratio,
            high=configs.config_values.marginal_ratio,
            size=1,
        )
        n_mask_cols = int(rand_ratio * img_w)
        rand_cols = high_freq_cols[:n_mask_cols]

        mask = np.zeros((img_h, img_w, 1), dtype=np.float32)
        mask[:, rand_cols, :] = 1.0
        mask[:, low_freq_cols, :] = 1.0

        # Applying + Appending mask
        x = x * mask
        x = np.concatenate([x, mask], axis=-1)
        return x

    # Build a single mask for the entire dataset - mainly useful for evaluation
    np.random.shuffle(high_freq_cols)
    n_mask_cols = int(configs.config_values.marginal_ratio * img_w)
    rand_cols = high_freq_cols[:n_mask_cols]

    mask = np.zeros((img_h, img_w, 1), dtype=np.float32)
    mask[:, rand_cols, :] = 1.0
    mask[:, low_freq_cols, :] = 1.0

    def apply_constant_mask(x):
        # Applying the same mask to all samples
        x = x * mask
        x = np.concatenate([x, mask], axis=-1)
        return x

    if constant_mask:
        mask_fn = apply_constant_mask
    else:
        mask_fn = apply_random_mask

    return mask_fn


def map_decorator(func):
    def wrapper(x):
        # Use a tf.py_function to prevent auto-graph from compiling the method
        return tf.py_function(func, inp=(x), Tout=(x.dtype))

    return wrapper


preproc_map = {
    "knee_complex": knee_preproc,
    "knee": knee_preproc,
    "mvtec": mvtec_preproc,
    "mvtec_lowres": mvtec_preproc,
    "brain": get_brain_only,
    "masked_brain": get_brain_masks,
    "seg_brain": get_brain_segs,
    "seg_cifar10": preproc_cifar_segs,
    "masked_cifar10": preproc_cifar_masks,
    "multiscale_cifar10": preproc_cifar_multiscale,
    "masked_pet": preproc_pet_masks,
}

aug_map = {
    # "knee": knee_aug,
    "mvtec": mvtec_aug,
    "mvtec_lowres": mvtec_aug,
}

# Datasets too big (or randomly generated )to cache
cache_blacklist = {}


def preprocess(dataset_name, data, train=True):

    if dataset_name in preproc_map:
        _fn = preproc_map[dataset_name]
        data = data.map(_fn, num_parallel_calls=AUTOTUNE)

    elif dataset_name not in ["masked_pet", "masked_cifar10", "mvtec"]:
        # rescale [0,255] -> [0,1]
        data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)

    if dataset_name == "pet":
        data = data.map(lambda x: tf.image.resize(x["image"], (64, 64)))

    # Caching offline data
    fname = f"/tmp/{dataset_name}"
    if not train:
        fname += "_val"

    # data = data.cache(fname)

    if dataset_name not in cache_blacklist:
        data = data.cache()
    else:
        data = data.cache(fname)  # should be file for really large datasets

    if train:
        data = data.shuffle(buffer_size=100, reshuffle_each_iteration=True)

    # reordered batching and masking
    if configs.config_values.mask_marginals:
        _fn = lambda x: tf.numpy_function(
            func=np_build_mask_fn(configs.config_values.constant_mask),
            inp=[x],
            Tout=tf.float32,
        )

        def mask_fn(x, l=None):
            x = _fn(x)

            if l is not None:
                return x, l

            return x

        data = data.map(mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if configs.config_values.dataset != "celeb_a":
        data = data.batch(configs.config_values.global_batch_size)

    if train and dataset_name in aug_map:
        data = data.map(aug_map[dataset_name], num_parallel_calls=2)

    # Online augmentation
    if dataset_name in ["blown_fashion", "blown_masked_fashion"]:
        data = data.map(pad)

    if dataset_name in ["masked_fashion", "blown_masked_fashion"]:
        data = data.map(concat_mask)

    if train and dataset_name in [
        "multiscale_cifar10",
        "cifar10",
        "masked_cifar10",
        "seg_cifar10",
        "pet",
        "masked_pet",
    ]:
        data = data.map(
            lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=AUTOTUNE
        )  # randomly flip along the vertical axis

    if train and "brain" in dataset_name:
        data = data.map(
            lambda x: tf.image.random_flip_up_down(
                x
            ),  # randomly flip along the direction of hemispheres
            num_parallel_calls=AUTOTUNE,
        )
    return data


def _preprocess_celeb_a(data, random_flip=True):
    # Discard labels and landmarks
    data = data.map(lambda x: x["image"], num_parallel_calls=AUTOTUNE)
    # Take a 140x140 centre crop of the image
    data = data.map(
        lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140),
        num_parallel_calls=AUTOTUNE,
    )
    # Resize to 32x32
    data = data.map(lambda x: tf.image.resize(x, (32, 32)), num_parallel_calls=AUTOTUNE)
    # Rescale
    data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)
    # Maybe cache in memory
    # data = data.cache()
    # Randomly flip
    if random_flip:
        data = data.map(
            lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=AUTOTUNE
        )
    return data


def get_celeb_a(random_flip=True):
    batch_size = configs.config_values.batch_size
    data_generators = tfds.load(
        name="celeb_a", batch_size=batch_size, data_dir="data", shuffle_files=True
    )
    train = data_generators["train"]
    test = data_generators["test"]
    train = _preprocess_celeb_a(train, random_flip=random_flip)
    test = _preprocess_celeb_a(test, random_flip=False)
    return train, test


def get_celeb_a32():
    """
    Loads the preprocessed celeb_a dataset scaled down to 32x32
    :return: tf.data.Dataset with single batch as big as the whole dataset
    """
    path = "./data/celeb_a32"
    if not os.path.exists(path):
        print(path, " does not exits")
        return None

    images = utils.get_tensor_images_from_path(path)
    data = tf.data.Dataset.from_tensor_slices(images)
    data = data.map(lambda x: tf.cast(x, tf.float32))
    data = data.batch(int(tf.data.experimental.cardinality(data)))
    return data


def get_ood_data(dataset_name):
    print("Getting OOD Dataset...")
    OOD_LABEL = 0
    data = tfds.load(name="mnist", batch_size=-1, data_dir="data", shuffle_files=False)

    mask = data["train"]["label"] != OOD_LABEL
    inlier_train = tf.data.Dataset.from_tensor_slices(data["train"]["image"][mask])

    mask = data["test"]["label"] != OOD_LABEL
    inlier_test = tf.data.Dataset.from_tensor_slices(data["test"]["image"][mask])

    mask = data["test"]["label"] == OOD_LABEL
    ood_test = tf.data.Dataset.from_tensor_slices(data["test"]["image"][mask])

    inlier_train = preprocess("mnist", inlier_train, train=True)
    inlier_test = preprocess("mnist", inlier_test, train=False)
    ood_test = preprocess("mnist", ood_test, train=False)

    return inlier_train, inlier_test, ood_test


def get_train_test_data(dataset_name):

    if dataset_name == "mnist_ood":
        train, test, _ = get_ood_data(dataset_name)
    elif dataset_name != "celeb_a":
        train, test = load_data(dataset_name)
        train = preprocess(dataset_name, train, train=True)
        test = preprocess(dataset_name, test, train=False)
    else:
        train, test = get_celeb_a()
    return train, test


def get_data_inpainting(dataset_name, n):
    if dataset_name == "celeb_a":
        data = get_celeb_a(random_flip=False)[0]
        data = next(iter(data.take(1)))[:n]
    else:
        data_generator = tfds.load(
            name=dataset_name,
            batch_size=-1,
            data_dir="data",
            split="train",
            shuffle_files=True,
        )
        data = data_generator["image"]
        data = tf.random.shuffle(data, seed=1000)
        data = data[:n] / 255
    return data


def get_data_k_nearest(dataset_name):
    data_generator = tfds.load(
        name=dataset_name,
        batch_size=-1,
        data_dir="data",
        split="train",
        shuffle_files=False,
    )
    data = tf.data.Dataset.from_tensor_slices(data_generator["image"])
    data = data.map(lambda x: tf.cast(x, dtype=tf.float32))

    return data
