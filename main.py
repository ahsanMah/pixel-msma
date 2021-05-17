import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
import configs
import evaluation
import trainv2 as train
import utils
import json


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
        
    save_dir, complete_model_name = utils.get_savemodel_dir()
    with open(save_dir +"/params.json", "w") as f:
        json.dump(vars(args),f)
    
    run = EXPERIMENTS[args.experiment]

    run()
