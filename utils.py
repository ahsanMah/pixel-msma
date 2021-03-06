import argparse
import os
import re  # Adding more code
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

from tensorflow.keras import mixed_precision

import configs
from model.refinenet import RefineNet
from model.refinenet import RefineNetTwoResidual
from model.refinenet import RefineNetLite
from losses.losses import dsm_loss

# from model.resnet import ResNet

dict_datasets_image_size = {
    "circles": (64, 64, 1),
    "highres": (2048, 1024, 3),
    "brain": (91, 109, 1),
    "masked_brain": (91, 109, 2),
    "seg_brain": (91, 109, 8),
    "pet": (64, 64, 3),
    "masked_pet": (64, 64, 4),
    "blown_fashion": (56, 56, 1),
    "blown_masked_fashion": (56, 56, 2),
    "masked_fashion": (28, 28, 2),
    "fashion_mnist": (28, 28, 1),
    "mnist_ood": (28, 28, 1),
    "mnist": (28, 28, 1),
    "cifar10": (32, 32, 3),
    "masked_cifar10": (32, 32, 4),
    "seg_cifar10": (32, 32, 14),
    "multiscale_cifar10": (32, 32, 6),
    "celeb_a": (32, 32, 3),
    "svhn_cropped": (32, 32, 3),
}

dict_train_size = {
    "circles": 100000,
    "svhn_cropped": 73000,
    "cifar10": 60000,
    "brain": 10500,
    "masked_brain": 10500,
    "seg_brain": 10500,
    "masked_cifar10": 40000,
    "seg_cifar10": 40000,
    "multiscale_cifar10": 50000,
    "pet": 6500,
    "masked_pet": 6500,
    "blown_fashion": 60000,
    "blown_masked_fashion": 60000,
    "masked_fashion": 60000,
    "fashion_mnist": 60000,
    "mnist_ood": 60000,
    "mnist": 60000,
}

dict_splits = {
    "masked_fashion": (1, 1),
    "masked_brain": (1, 1),
    "seg_brain": (1, 7),
    "masked_cifar10": (3, 1),
    "seg_cifar10": (3, 11),
    "multiscale_cifar10": (3, 3),
}

with open("./datasets/data_configs.yaml") as f:
    configs.dataconfig = yaml.safe_load(f)


def find_k_closest(image, k, data_as_array):
    l2_distances = tf.reduce_sum(tf.square(data_as_array - image), axis=[1, 2, 3])
    _, smallest_idx = tf.math.top_k(-l2_distances, k)
    closest_k = tf.gather(data_as_array, smallest_idx[:k])
    return closest_k, smallest_idx[:k]


def get_dataset_image_size(dataset_name):
    input_shape = [
        int(x.strip())
        for x in configs.dataconfig[dataset_name]["downsample_size"].split(",")
    ]

    if configs.config_values.mask_marginals:
        input_shape[-1] += 1  # Append a mask channel

    return input_shape


def check_args_validity(args):
    assert args.model in [
        "baseline",
        "resnet",
        "refinenet_lite",
        "refinenet",
        "refinenet_twores",
        "masked_refinenet",
    ]

    args.global_batch_size = args.batch_size

    if args.max_to_keep == -1:
        args.max_to_keep = None

    args.mask_marginals = args.marginal_ratio >= 0.0
    args.y_cond = args.marginal_ratio >= 0.0

    args.split = args.split.split(",")
    args.split = list(map(lambda x: x.strip(), args.split))
    return


def _build_parser():
    parser = argparse.ArgumentParser(
        description="CLI Options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", default="train", help="what experiment to run")
    parser.add_argument(
        "--mode",
        default="score",
        help="what mode to run the experiment in (applies to evaluation only)",
    )
    parser.add_argument("--dataset", default="mnist", help="tfds name of dataset")
    parser.add_argument(
        "--model",
        default="refinenet",
        help="Model to use. Can be 'refinenet', 'resnet', 'baseline'",
    )
    parser.add_argument(
        "--filters",
        default=128,
        type=int,
        help="number of filters in the model.",
    )
    parser.add_argument(
        "--num_L",
        default=10,
        type=int,
        help="number of levels of noise to use",
    )
    parser.add_argument(
        "--sigma_low",
        default=0.01,
        type=float,
        help="lowest value for noise",
    )
    parser.add_argument(
        "--sigma_high",
        default=1.0,
        type=float,
        help="highest value for noise",
    )
    parser.add_argument(
        "--sigma_sequence",
        default="geometric",
        type=str,
        help="can be 'geometric' or 'linear'",
    )
    parser.add_argument(
        "--steps",
        default=200000,
        type=int,
        help="number of steps to train the model for",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-4,
        type=float,
        help="learning rate for the optimizer",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "--samples_dir",
        default="./samples/",
        help="folder for saving samples",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="./saved_models/",
        help="folder for saving model checkpoints (default: ./saved_models/)",
    )
    parser.add_argument(
        "--checkpoint_freq",
        default=5000,
        type=int,
        help="how often to save a model checkpoint (default: 5000 iterations)",
    )
    parser.add_argument(
        "--resume",
        action="store_false",
        help="whether to resume from latest checkpoint (default: True)",
    )
    parser.add_argument(
        "--resume_from",
        default=-1,
        type=int,
        help="Step of checkpoint where to resume the model from. (default: latest one)",
    )
    parser.add_argument(
        "--log_freq",
        default=100,
        type=int,
        help="how often to save a model checkpoint (default: 5000 iterations)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="whether to resume from latest checkpoint (default: False)",
    )

    parser.add_argument(
        "--init_samples",
        default="",
        help="Folder with images to be used as x0 for sampling with annealed langevin dynamics",
    )
    parser.add_argument(
        "--k",
        default=10,
        type=int,
        help="number of nearest neighbours to find from data (default: 10)",
    )
    parser.add_argument(
        "--eval_setting",
        default="sample",
        type=str,
        help="can be 'sample' or 'fid'",
    )
    parser.add_argument(
        "--ocnn",
        action="store_true",
        help="whether to attach an ocnn to the model",
    )
    parser.add_argument(
        "--y_cond",
        action="store_true",
        help="whether the model is conditioned on auxiallary y information",
    )
    parser.add_argument(
        "--max_to_keep",
        default=2,
        type=int,
        help="Number of checkopints to keep saved",
    )
    parser.add_argument(
        "--split",
        default="100,0",
        type=str,
        help=r"optional train/validation split percentages e.g. 0.9*train, 0.1*train (default means all train, no val set)",
    )
    parser.add_argument(
        "--T",
        default=100,
        type=int,
        help="number of iterations to sample for each sigma",
    )
    parser.add_argument(
        "--eps",
        default=2 * 1e-5,
        type=int,
        help="epsilon for generating samples (default: 2*1e-5)",
    )
    parser.add_argument(
        "--class_label",
        default="all",
        type=str,
        help="which class label to use for training, applies to mvtec",
    )

    parser.add_argument(
        "--longleaf",
        action="store_true",
        help="whether model is running on longleaf server",
    )

    parser.add_argument(
        "--marginal_ratio",
        default=-1.0,
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
        "--constant_mask",
        action="store_true",
        help="mostly useful for evaluating data with a constant mask for entire run",
    )

    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="training with mixed precision can help reduce memory footprint and increase batch size",
    )

    return parser


def get_command_line_args():
    parser = _build_parser()

    parser = parser.parse_args()

    check_args_validity(parser)

    s = "=" * 20 + "\nParameters: \n"
    for key in parser.__dict__:
        s += key + ": " + str(parser.__dict__[key]) + "\n"
    s += "=" * 20 + "\n"

    print(s)

    return parser


def get_tensorflow_device():
    device = "gpu:0" if tf.test.is_gpu_available() else "cpu"
    print("Using device {}".format(device))
    return device


def get_savemodel_dir():
    models_dir = configs.config_values.checkpoint_dir
    model_name = configs.config_values.model

    ds_name = configs.config_values.dataset
    if configs.config_values.mask_marginals:
        # ds_name = f"{ds_name}_mr{configs.config_values.marginal_ratio}"
        ds_name = f"{ds_name}_mr{configs.config_values.min_marginal_ratio}-{configs.config_values.marginal_ratio}"

    # Folder name: model_name+filters+dataset+L
    complete_model_name = "{}{}_{}-{}_L{}_SH{:.0e}_SL{:.0e}".format(
        model_name,
        configs.config_values.filters,
        ds_name,
        configs.config_values.class_label,
        configs.config_values.num_L,
        configs.config_values.sigma_high,
        configs.config_values.sigma_low,
    )
    folder_name = os.path.join(models_dir, complete_model_name) + "/"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name, complete_model_name


def evaluate_print_model_summary(model, verbose=True):
    batch = 1
    input_shape = [
        batch,
    ] + get_dataset_image_size(configs.config_values.dataset)

    # # TODO: This is very hacky, find better solution
    # if configs.config_values.class_label == "kspace_complex":
    #     input_shape[-1] += 1  # kspace is 2 channels

    print(input_shape)
    sigma_levels = get_sigma_levels()  # tf.linspace(0.0,1.0,3) #
    idx_sigmas = tf.ones(batch, dtype=tf.int32)
    #     sigmas = tf.gather(sigma_levels, idx_sigmas)
    #     sigmas = tf.cast(tf.reshape(sigmas, shape=(batch, 1, 1, 1)), dtype=tf.float32)
    x = [tf.ones(shape=input_shape), idx_sigmas]
    model(x)
    if verbose:
        print(model.summary())


def attach_ocnn(top=True, encoding=False):
    pass


def try_load_model(
    save_dir, step_ckpt=-1, return_new_model=True, verbose=True, ocnn=False
):
    """
    Tries to load a model from the provided directory, otherwise returns a new initialized model.
    :param save_dir: directory with checkpoints
    :param step_ckpt: step of checkpoint where to resume the model from
    :param verbose: true for printing the model summary
    :return:
    """
    ocnn_model = None
    ocnn_optimizer = None

    import tensorflow as tf

    tf.compat.v1.enable_v2_behavior()
    if configs.config_values.model == "baseline":
        configs.config_values.num_L = 1

    splits = False
    # if configs.config_values.y_cond:
    #     splits = dict_splits[configs.config_values.dataset]

    # initialize return values
    model_name = configs.config_values.model
    if model_name == "resnet":
        model = ResNet(filters=configs.config_values.filters, activation=tf.nn.elu)
    elif model_name in ["refinenet", "baseline"]:
        model = RefineNet(
            filters=configs.config_values.filters,
            activation=tf.nn.elu,
            y_conditioned=configs.config_values.y_cond,
            splits=splits,
        )
    elif model_name == "refinenet_lite":
        model = RefineNetLite(
            filters=configs.config_values.filters, activation=tf.nn.elu
        )
    elif model_name == "refinenet_twores":
        model = RefineNetTwoResidual(
            filters=configs.config_values.filters, activation=tf.nn.elu
        )
    elif model_name == "masked_refinenet":
        print("Using Masked RefineNet...")
        # assert configs.config_values.y_cond
        model = MaskedRefineNet(
            filters=configs.config_values.filters,
            activation=tf.nn.elu,
            splits=dict_splits[configs.config_values.dataset],
            y_conditioned=configs.config_values.y_cond,
        )

    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=configs.config_values.learning_rate
    )
    step = 0

    evaluate_print_model_summary(model, verbose)

    if ocnn:
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Flatten, Dense, AvgPool2D

        # Building OCNN on top
        print("Building OCNN...")
        Input = [
            Input(name="images", shape=(28, 28, 1)),
            Input(name="idx_sigmas", shape=(), dtype=tf.int32),
        ]

        score_logits = model(Input)
        x = Flatten()(score_logits)
        x = Dense(128, activation="linear", name="embedding")(x)
        dist = Dense(1, activation="linear", name="distance")(x)
        ocnn_model = Model(inputs=Input, outputs=dist, name="OC-NN")
        ocnn_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        evaluate_print_model_summary(ocnn_model, verbose=True)

    # if resuming training, overwrite model parameters from checkpoint
    if configs.config_values.resume:
        if step_ckpt == -1:
            print("Trying to load latest model from " + save_dir)
            checkpoint = tf.train.latest_checkpoint(str(os.path.abspath(save_dir)))
        else:
            print(
                "Trying to load checkpoint with step",
                step_ckpt,
                " model from " + save_dir,
            )
            onlyfiles = [
                f
                for f in os.listdir(save_dir)
                if os.path.isfile(os.path.join(save_dir, f))
            ]
            # r = re.compile(".*step_{}-.*".format(step_ckpt))
            r = re.compile("ckpt-{}\\..*".format(step_ckpt))

            name_all_checkpoints = sorted(list(filter(r.match, onlyfiles)))
            print(name_all_checkpoints)
            # Retrieve name of the last checkpoint with that number of steps
            name_ckpt = name_all_checkpoints[-1][:-6]
            # print(name_ckpt)
            checkpoint = save_dir + name_ckpt
        if checkpoint is None:
            print("No model found.")
            if return_new_model:
                print("Using a new model")
            else:
                print("Returning None")
                model = None
                optimizer = None
                step = None
        else:
            step = tf.Variable(0)

            if ocnn:
                ckpt = tf.train.Checkpoint(
                    step=step,
                    optimizer=optimizer,
                    model=model,
                    ocnn_model=ocnn_model,
                    ocnn_optimizer=ocnn_optimizer,
                )
            else:
                ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)

            ckpt.restore(checkpoint)
            step = int(tf.cast(step, tf.int32))
            print("Loaded model: " + checkpoint)

    return model, optimizer, step, ocnn_model, ocnn_optimizer


def get_sigma_levels():
    if configs.config_values.model == "baseline":
        sigma_levels = tf.ones(1) * configs.config_values.sigma_low
    elif configs.config_values.sigma_sequence == "linear":
        sigma_levels = tf.linspace(
            configs.config_values.sigma_high,
            configs.config_values.sigma_low,
            configs.config_values.num_L,
        )
    elif configs.config_values.sigma_sequence == "geometric":
        sigma_levels = tf.math.exp(
            tf.linspace(
                tf.math.log(configs.config_values.sigma_high),
                tf.math.log(configs.config_values.sigma_low),
                configs.config_values.num_L,
            )
        )
    elif configs.config_values.sigma_sequence == "hybrid":
        sigma_levels_geometric = tf.math.exp(
            tf.linspace(
                tf.math.log(configs.config_values.sigma_high),
                tf.math.log(configs.config_values.sigma_low),
                configs.config_values.num_L,
            )
        )
        sigma_levels_linear = tf.linspace(
            configs.config_values.sigma_high,
            configs.config_values.sigma_low,
            configs.config_values.num_L,
        )
        sigma_levels = (sigma_levels_geometric + sigma_levels_linear) / 2
    return sigma_levels


def get_init_samples():
    if configs.config_values.init_samples == "":
        return None

    path = configs.config_values.init_samples
    if not os.path.exists(path):
        raise ValueError("Path ", path, " does not exist.")

    images = get_tensor_images_from_path(path)

    images /= 255
    return images


def get_tensor_images_from_path(path, resize=True):
    images = []
    for i, filename in enumerate(os.listdir(path)):
        image = tf.io.decode_image(tf.io.read_file(path + "/" + filename))
        if resize:
            size = max(image.shape[0], image.shape[1])
            is_square = image.shape[0] == image.shape[1]
            if not is_square:
                min_size = min(image.shape[0], image.shape[1])
                image = tf.image.resize_with_crop_or_pad(image, min_size, min_size)
                size = min_size
                is_square = True
            if size != 32 and is_square:
                image = tf.image.resize(image, (32, 32))
        images.append(image)

    return tf.convert_to_tensor(images)


def manage_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices("GPU")
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


"""
Returns an optimal L value according to Technique 2 in NCSNv2 paper
A range of Cs is also result to help pick a better value
sigma_high should be largest intra-dataset Euclidean distance
"""


def suggest_optimal_L(dim=32 * 32, sigma_high=50.0, sigma_low=0.01, limit=1000):
    def calc_C(L):
        gamma = (sigma_low / sigma_high) ** (1 / (L - 1))
        D = np.sqrt(2 * dim)
        C = norm.cdf(D * (gamma - 1) + 3 * gamma) - norm.cdf(
            D * (gamma - 1) - 3 * gamma
        )
        return C

    L_range = np.arange(2, limit) * 1.0
    Cs = [calc_C(l) for l in L_range]  # C value for every L
    optimal_L = np.where(np.isclose(Cs, 0.9, rtol=1e-3, atol=1e-3))[0][0]
    optimal_C = Cs[optimal_L]
    plt.plot(L_range, Cs)
    print("Suggested Optimal: L={:d} w/ C={:.3f}".format(optimal_L, optimal_C))

    return optimal_L, Cs


def build_distributed_trainers(
    strategy, model, optimizer, ema, sigma_levels, num_replicas, loss_aggregators
):

    num_L = sigma_levels.shape[0]
    train_loss, test_loss = loss_aggregators
    input_shape = get_dataset_image_size(configs.config_values.dataset)
    channels = input_shape[-1] - 1

    @tf.function(experimental_compile=True)
    def dsm_loss(score, x_perturbed, x, sigmas):
        target = (x_perturbed - x) / (tf.square(sigmas))
        loss = (
            0.5
            * tf.reduce_sum(tf.square(score + target), axis=[1, 2, 3], keepdims=True)
            * tf.square(sigmas)
        )
        # Note: We changed reduce_mean to account for multiple GPUs
        # Necessary when mirrored strategy is used
        loss = tf.reduce_mean(loss) / num_replicas
        return loss

    @tf.function
    def train_fn(x_batch):
        def step_fn(x_batch):

            idx_sigmas = tf.random.uniform(
                [x_batch.shape[0]], minval=0, maxval=num_L, dtype=tf.dtypes.int32
            )
            sigmas = tf.gather(sigma_levels, idx_sigmas)
            sigmas = tf.reshape(sigmas, shape=(x_batch.shape[0], 1, 1, 1))

            if configs.config_values.y_cond:
                # --> Noise may only be applied to foreground
                x_batch, masks = tf.split(x_batch, (channels, 1), axis=-1)
                perturbation = tf.random.normal(shape=x_batch.shape) * sigmas
                perturbation = tf.multiply(perturbation, masks)

                # Used for calculating loss
                x_batch_perturbed = x_batch + perturbation
                # Input has conditioning information
                x_batch_input = tf.concat((x_batch_perturbed, masks), axis=-1)
            else:
                x_batch_perturbed = (
                    x_batch + tf.random.normal(shape=x_batch.shape) * sigmas
                )
                x_batch_input = x_batch_perturbed

            with tf.GradientTape() as t:
                scores = model([x_batch_input, idx_sigmas])

                # if configs.config_values.y_cond:
                #     scores = scores * masks

                current_loss = dsm_loss(scores, x_batch_perturbed, x_batch, sigmas)

                if configs.config_values.mixed_precision:
                    scaled_loss = optimizer.get_scaled_loss(current_loss)

            if configs.config_values.mixed_precision:
                gradients = t.gradient(scaled_loss, model.trainable_variables)
                gradients = optimizer.get_unscaled_gradients(gradients)
            else:
                gradients = t.gradient(current_loss, model.trainable_variables)

            opt_op = optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

            with tf.control_dependencies([opt_op]):
                # Creates the shadow variables, and add ops to maintain moving averages
                # Also creates an op that will update the moving
                # averages after each training step
                training_op = ema.apply(model.trainable_variables)

            return current_loss

        per_replica_losses = strategy.run(step_fn, args=(x_batch,))
        mean_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        train_loss(mean_loss)

        return mean_loss

    @tf.function
    def test_fn(x_batch, idx):
        def step_fn(x_batch, idx):
            idx_sigmas = idx * tf.ones([x_batch.shape[0]], dtype=tf.dtypes.int32)
            sigmas = tf.gather(sigma_levels, idx_sigmas)
            sigmas = tf.reshape(sigmas, shape=(x_batch.shape[0], 1, 1, 1))

            if configs.config_values.y_cond:
                # --> Noise may only be applied to foreground
                x_batch, masks = tf.split(x_batch, (channels, 1), axis=-1)
                perturbation = tf.random.normal(shape=x_batch.shape) * sigmas
                # perturbation = tf.multiply(perturbation, masks)

                # Used for calculating loss
                x_batch_perturbed = x_batch + perturbation
                # Input has conditioning information
                x_batch_input = tf.concat((x_batch_perturbed, masks), axis=-1)
            else:
                x_batch_perturbed = (
                    x_batch + tf.random.normal(shape=x_batch.shape) * sigmas
                )
                x_batch_input = x_batch_perturbed

            scores = model([x_batch_input, idx_sigmas])

            if configs.config_values.y_cond:
                scores = scores * masks

            loss = dsm_loss(scores, x_batch_perturbed, x_batch, sigmas)
            return loss

        per_replica_losses = strategy.run(
            step_fn,
            args=(
                x_batch,
                idx,
            ),
        )
        mean_loss = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        test_loss(mean_loss)

        return mean_loss

    return train_fn, test_fn
