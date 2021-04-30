import os
import tensorflow as tf
import configs
import evaluation
import trainv2 as train
import utils

utils.manage_gpu_memory_usage()

EXPERIMENTS = {
    "train": train.main,
    "evaluation": evaluation.main,
}

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    run = EXPERIMENTS[args.experiment]

    run()
