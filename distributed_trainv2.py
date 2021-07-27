import csv
from datetime import datetime

import tensorflow as tf

OLD_TF = tf.__version__ < "2.4.0"

if OLD_TF:
    print("Using TF < 2.4:", tf.__version__)
    import tensorflow.keras.mixed_precision.experimental as mixed_precision

    mixed_precision.set_global_policy = mixed_precision.set_policy
    AUTOTUNE = tf.data.experimental.AUTOTUNE
else:
    from tensorflow.keras import mixed_precision

    AUTOTUNE = tf.data.AUTOTUNE

from tqdm import tqdm

import configs
import utils
from datasets.dataset_loader import get_train_test_data
from losses.losses import dsm_loss, ocnn_loss, update_radius, normalized_dsm_loss

SIGMA_LEVELS = None
LOGGER = tf.get_logger()


def main():

    policy_name = "float32"

    if configs.config_values.mixed_precision:
        policy_name = "mixed_float16"

    if tf.__version__ < "2.4.0":
        policy = tf.keras.mixed_precision.experimental.Policy(policy_name)
        tf.keras.mixed_precision.experimental.set_policy(policy)
    else:
        policy = mixed_precision.Policy(policy_name)
        mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()

    # device = utils.get_tensorflow_device()
    tf.random.set_seed(2019)
    BATCH_SIZE_PER_REPLICA = configs.config_values.batch_size
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
    LOG_FREQ = 100
    LOG_FREQ = configs.config_values.log_freq
    configs.config_values.global_batch_size = GLOBAL_BATCH_SIZE

    # array of sigma levels
    # generate geometric sequence of values between sigma_low (0.01) and sigma_high (1.0)
    SIGMA_LEVELS = utils.get_sigma_levels()
    NUM_L = configs.config_values.num_L

    # path for saving the model(s)
    save_dir, complete_model_name = utils.get_savemodel_dir()

    # Swapping to EMA weights for evaluation/checkpoints
    def swap_weights():
        if ema.in_use == False:
            ema.in_use = True
            ema.training_state = [tf.identity(x) for x in model.trainable_variables]
            for var in model.trainable_variables:
                var.assign(ema.average(var))
            LOGGER.info("Swapped to EMA...")
            return

        # Else switch back to training state
        for var, var_train_state in zip(model.trainable_variables, ema.training_state):
            var.assign(var_train_state)
        ema.in_use = False
        LOGGER.info("Swapped back to training state.")
        return

    print("GPUs in use:", NUM_REPLICAS)

    with strategy.scope():
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        ema.in_use = False

        model, optimizer, step, ocnn_model, ocnn_optimizer = utils.try_load_model(
            save_dir,
            step_ckpt=configs.config_values.resume_from,
            verbose=True,
            ocnn=configs.config_values.ocnn,
        )

        if configs.config_values.mixed_precision:
            print("Using mixed-prec optimizer...")
            if OLD_TF:
                optimizer = mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale="dynamic"
                )
            else:
                optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        # Checkpoint should also be under strategy
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0), optimizer=optimizer, model=model
        )

    manager = tf.train.CheckpointManager(
        ckpt, directory=save_dir, max_to_keep=configs.config_values.max_to_keep
    )
    step = 0

    ####### Training Steps #######
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)

    train_step, test_step = utils.build_distributed_trainers(
        strategy,
        model,
        optimizer,
        ema,
        SIGMA_LEVELS,
        NUM_REPLICAS,
        (train_loss, test_loss),
    )

    # FIXME: "test_data" needs to be a val_data = 10% of training data

    # load dataset from tfds (or use downloaded version if exists)
    train_data, test_data = get_train_test_data(configs.config_values.dataset)

    # # split data into batches
    train_data = train_data.repeat()
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    test_data = test_data.take(32).cache()

    train_data = strategy.experimental_distribute_dataset(train_data)
    test_data = strategy.experimental_distribute_dataset(test_data)

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    basename = "logs/{model}/{dataset}/{time}".format(
        model=configs.config_values.model,
        dataset=configs.config_values.dataset,
        time=start_time,
    )
    train_log_dir = basename + "/train"
    test_log_dir = basename + "/test"

    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step + 1)
    progress_bar.set_description("current loss ?")

    steps_per_epoch = (
        configs.dataconfig[configs.config_values.dataset]["n_samples"]
        // configs.config_values.batch_size
    )

    epoch = step // steps_per_epoch

    train_summary_writer = None  # tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = None  # tf.summary.create_file_writer(test_log_dir)

    if configs.config_values.profile:
        tf.profiler.experimental.start(basename + "/profile")

    avg_loss = 0

    for data_batch in progress_bar:

        if step % steps_per_epoch == 0:
            epoch += 1

        step += 1

        # train_step = None
        # if (
        #     configs.config_values.y_cond
        #     or configs.config_values.model == "masked_refinenet"
        # ):
        #     train_step, test_step = train_one_masked_step, test_step_masked
        # else:
        #     train_step, test_step = train_one_step, test_one_step

        current_loss = train_step(data_batch)
        train_loss(current_loss)

        progress_bar.set_description(
            "[epoch {:d}] | current loss {:.3f}".format(
                epoch, train_loss.result().numpy()
            )
        )

        if step % LOG_FREQ == 0:

            if train_summary_writer == None:
                train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                test_summary_writer = tf.summary.create_file_writer(test_log_dir)

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=step)

            # Swap to EMA
            swap_weights()
            for x_test in test_data:
                _loss = test_step(data_batch, NUM_L - 1)
                test_loss(_loss)

            with test_summary_writer.as_default():
                tf.summary.scalar("loss", test_loss.result(), step=step)
            swap_weights()

            # Reset metrics every epoch
            train_loss.reset_states()
            test_loss.reset_states()

        # loss_history.append([step, current_loss.numpy()])
        avg_loss += current_loss

        if step % configs.config_values.checkpoint_freq == 0:
            swap_weights()
            ckpt.step.assign(step)
            manager.save()
            swap_weights()
            # Append in csv file
            # with open(save_dir + 'loss_history.csv', mode='a', newline='') as csv_file:
            #     writer = csv.writer(csv_file, delimiter=';')
            #     writer.writerows(loss_history)

            print(
                "\nSaved checkpoint. Average loss: {:.3f}".format(
                    avg_loss / configs.config_values.checkpoint_freq
                )
            )
            avg_loss = 0

        if step == total_steps:
            if configs.config_values.profile:
                tf.profiler.experimental.stop()
            return
