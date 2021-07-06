import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import configs
import evaluation
import generate

# import trainv2 as train
import distributed_trainv2 as train
import utils
import json

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

utils.manage_gpu_memory_usage()

EXPERIMENTS = {"train": train.main, "eval": evaluation.main, "gen": generate.main}

if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = utils.get_command_line_args()
    configs.config_values = args

    save_dir, complete_model_name = utils.get_savemodel_dir()
    with open(save_dir + "/params.json", "w") as f:
        json.dump(vars(args), f)

    run = EXPERIMENTS[args.experiment]

    run()
