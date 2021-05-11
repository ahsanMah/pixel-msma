#! /usr/bin/python3

import os,sys,pickle
import pathlib,glob
import yaml
import argparse
# print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl

import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow_datasets as tfds
from tqdm.auto import tqdm


from dgmm import DGMM, trainer, get_patch_loaders
from ood_detection_helper import *

#TODO: Use a logger

tfb = tfp.bijectors
tfd = tfp.distributions
# tf.enable_v2_behavior()

mpl.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 18 
plt.rcParams['axes.labelsize'] = 16
sns.set(style="darkgrid")

# BASE_PATH = "/content/drive/MyDrive/"
# BASE_PATH = "./"
# WORKDIR = "/Developer/pixel-msma"
# os.chdir(BASE_PATH + WORKDIR)
# sys.path.append(os.path.join(BASE_PATH, WORKDIR))

RANDOM_SEED  = 42


parser = argparse.ArgumentParser(description='Evaluation Options')
parser.add_argument('--experiment', default='train',
                    help="what experiment to run")
parser.add_argument('--object', default='hazelnut',
                    help="name of mvtec object to evaluate")
parser.add_argument('--r_sz', default=12, type=int,
                    help="Size of receptive field for DGMM")
parser.add_argument('--img_sz', default=96, type=int,
                    help="Size of image tensors used for training")
parser.add_argument('--n_epochs', default=50, type=int,
                        help="Size of image tensors used for training")
parser.add_argument('--save_path', default='saved_models/', type=str,
                        help="Size of image tensors used for training")

#### NCSN Arguments #####
parser.add_argument('--filters', default=128, type=int,
                        help='number of filters in the model. (default: 128)')
parser.add_argument('--num_L', default=10, type=int,
                    help="number of levels of noise to use (default: 10)")
parser.add_argument('--sigma_low', default=0.01, type=float,
                    help="lowest value for noise (default: 0.01)")
parser.add_argument('--sigma_high', default=1.0, type=float,
                    help="highest value for noise (default: 1.0)")

args = parser.parse_args()
print(args)

OBJECT = args.object
EXPERIMENT = args.experiment
MODEL_PATH = args.save_path
IMG_W= args.img_sz
IMG_H= args.img_sz
BS=128
N_EPOCHS = args.n_epochs
R_SZ = args.r_sz

N_FILTERS = args.filters
SIGMA_HIGH = args.sigma_high
SIGMA_LOW  = args.sigma_low
NUM_L = args.num_L
SIGMA_LEVELS = tf.math.exp(
    tf.linspace(tf.math.log(SIGMA_HIGH),
                tf.math.log(SIGMA_LOW),
                NUM_L)
)

cache_dir = os.path.join("score_cache", "mvtec")
os.makedirs("score_cache/mvtec", exist_ok=True)
SCORE_CACHE = os.path.join(cache_dir,
                           f"{OBJECT}_f{N_FILTERS}_L{NUM_L}scores.npz")

with open("./datasets/data_configs.yaml") as f:
    dataconfig = yaml.safe_load(f)

dataset_dir = dataconfig["mvtec"]["datadir"] + f"/{OBJECT}/"
# dataset_resized_dir = os.path.join(dataconfig["mvtec"]["datadir"], "..", f"mvtec_imgs/{OBJECT}")
dataset_resized_dir = os.path.join("mvtec_imgs", f"{OBJECT}")
train_dir = pathlib.Path(f"{dataset_dir}/train/")
test_dir = pathlib.Path(f"{dataset_dir}/test/")
seg_dir = pathlib.Path(f"{dataset_dir}/ground_truth/")
print(train_dir, test_dir, seg_dir, sep="\n")

LABELS = sorted(os.listdir(f"{test_dir}"))
ANO_LABELS = [x for x in LABELS if x != "good"]
ANO_LABELS_IDX = [LABELS.index(x) for x in ANO_LABELS]
INLIER_LABEL = LABELS.index("good")
print(LABELS, INLIER_LABEL, ANO_LABELS, ANO_LABELS_IDX)

image_count = len(list(train_dir.glob('*/*.png')))
print("Training Images:", image_count)
image_count = len(list(test_dir.glob('*/*.png')))
print("Testing Images:", image_count)
image_count = len(list(seg_dir.glob('*/*.png')))
print("Anomalous Images:", image_count)

@tf.function
def preproc(x,y):
  return x/255, y

@tf.function(experimental_compile=True)
def weighted_norm(x, sigmas):
    x = tf.norm(tf.reshape(x, shape=(x.shape[0], x.shape[1], -1)),
                   axis=2, ord="euclidean", keepdims=False)
    x  = x * sigmas
    return x

def compute_scores(model, xs):
  scores = []

  for x,y in tqdm(xs):
    # x = tf.expand_dims(xs[i],0)
    per_sigma_scores = []
    for idx,sigma_val in enumerate(SIGMA_LEVELS):
        sigma = idx*tf.ones([x.shape[0]], dtype=tf.dtypes.int32)
        score = model([x, sigma]) * sigma_val
        # score = score ** 2
        per_sigma_scores.append(score)
    scores.append(tf.stack(per_sigma_scores, axis=1))

  # N x WxH x L Matrix of score norms
  scores =  tf.squeeze(tf.concat(scores, axis=0))
  return scores

def build_ds_loaders():

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
              train_dir,
              seed=RANDOM_SEED,
              shuffle=False,
              image_size=(IMG_W, IMG_H),
              batch_size=BS
  ).map(preproc)

  test_ds = tf.keras.preprocessing.image_dataset_from_directory(
              test_dir,
              class_names=LABELS,
              seed=RANDOM_SEED,
              shuffle=False,
              image_size=(IMG_H, IMG_H),
              batch_size=BS
  ).map(preproc)

  seg_ds = tf.keras.preprocessing.image_dataset_from_directory(
              seg_dir,
              class_names=ANO_LABELS,
              seed=RANDOM_SEED,
              shuffle=False,
              image_size=(IMG_H, IMG_H),
              batch_size=BS
  )

  return train_ds, test_ds, seg_ds

def load_img_data(as_tfds=False):
#   dirname = f"/content/drive/MyDrive/ML_Datasets/mvtec_imgs/{OBJECT}/"
  fname = os.path.join(dataset_resized_dir, f"{IMG_W}x{IMG_H}.npz")

  if os.path.exists(fname) and not as_tfds:
    with np.load(fname) as img_data:
      train_ds_imgs = img_data["train"]
      test_ds_imgs = img_data["test"]
      seg_ds_imgs = img_data["seg"]
      test_labels= img_data["test_labels"]
  
  else:
    train_ds, test_ds, seg_ds = build_ds_loaders()

    train_ds_imgs = tf.concat([x for x,l in train_ds], axis=0).numpy()
    test_ds_imgs = tf.concat([x for x,l in test_ds], axis=0).numpy()
    seg_ds_imgs = tf.concat([x for x,l in seg_ds], axis=0).numpy()
    test_labels = tf.concat([l for x,l in test_ds], axis=0).numpy()

    os.makedirs(dataset_resized_dir, exist_ok=True)
    np.savez_compressed(
        fname,
        train= train_ds_imgs,
        test= test_ds_imgs,
        seg= seg_ds_imgs,
        test_labels= test_labels
    )
  
  if as_tfds:
    return train_ds, test_ds, seg_ds

  return train_ds_imgs, test_ds_imgs, seg_ds_imgs, test_labels

def get_patch_LLs(rsz, epoch_num):
  
  model.load_weights(f"saved_models/dgmm/mvtec-{OBJECT}/{rsz}x{rsz}/e{epoch_num}")
  _, test_builder = get_patch_loaders(IMG_W, IMG_H, receptive_field_sz=rsz)
  inlier_train_ds = test_builder(train_ds_imgs, train_score_tensors, batch_sz=TEST_SZ)
  test_patch_ds = test_builder(test_ds_imgs, test_score_tensors, batch_sz=TEST_SZ)
  train_LL = []
  for x in inlier_train_ds:
    train_LL.append(model.log_pdf(x["score"], x["image"]))

  train_LL = np.concatenate(train_LL, axis=0)

  test_LL = []
  for x in test_patch_ds:
    test_LL.append(model.log_pdf(x["score"], x["image"]))
    # print(x["image"].shape)
  test_LL = np.concatenate(test_LL, axis=0)

  return train_LL, test_LL

'''
Runs NCSN and caches score results
'''
def ncsn_runner():
  
  if os.path.exists(SCORE_CACHE):
    print("Using score cache...")
    data = np.load(SCORE_CACHE)
    train_score_tensors = data["train"]
    test_score_tensors = data["test"]
    
    return train_score_tensors, test_score_tensors

  train_ds, test_ds, seg_ds = build_ds_loaders()
  test_ds = test_ds.cache()

  model, args = load_model(
      inlier_name="mvtec", checkpoint=-1, 
      filters=N_FILTERS, num_L=NUM_L,
      save_path=MODEL_PATH,
      class_label=OBJECT
  )
  model.trainable = False

  train_score_tensors = compute_scores(model, train_ds).numpy()
  test_score_tensors = compute_scores(model, test_ds).numpy()

#   test_labels = []
#   for x,l in test_ds:
#     test_labels.append(l.numpy())
#   test_labels = np.concatenate(test_labels, axis=0)
  
  np.savez_compressed(
    (SCORE_CACHE),
    train = train_score_tensors,
    test =  test_score_tensors,
  )

  return train_score_tensors, test_score_tensors

def msma_runner():

  train_scores, test_scores = ncsn_runner()
  
  train_ds, test_ds, seg_ds = build_ds_loaders()
  test_labels = []
  for x,l in test_ds:
    test_labels.append(l.numpy())
  test_labels = np.concatenate(test_labels, axis=0)
     
  train_norms = weighted_norm(train_scores, SIGMA_LEVELS)
  test_norms = weighted_norm(test_scores, SIGMA_LEVELS)

  metrics = auxiliary_model_analysis(
                train_norms,
                test_norms[test_labels==INLIER_LABEL],
                [test_norms[test_labels==idx] for idx in ANO_LABELS_IDX],
                labels=["Train", "Inlier"]+ANO_LABELS,
                components_range=range(1,11,1),
                flow_epochs=1
  )

  
    
  return metrics

def dgmm_trainer():

  train_ds_imgs, test_ds_imgs, seg_ds_imgs, test_labels = load_img_data()

#   data = np.load(f"{SCORE_CACHE}")
#   train_score_tensors = data["arr_0"]
#   test_score_tensors = data["arr_1"]
    
  train_score_tensors, test_score_tensors = ncsn_runner()

  TRAIN_SZ = 64
  TEST_SZ = 1

  train_builder, test_builder = get_patch_loaders(IMG_W, IMG_H, sigma_l=NUM_L, receptive_field_sz=R_SZ)
  train_patch_ds = train_builder(
                    train_ds_imgs,
                    train_score_tensors,
                    batch_sz=TRAIN_SZ
                  )
  val_patch_ds = test_builder(
                  test_ds_imgs[test_labels == INLIER_LABEL],
                  test_score_tensors[test_labels == INLIER_LABEL],
                  batch_sz=TEST_SZ
                )

  model = DGMM(img_shape=(IMG_W,IMG_H,4), latent_dim=128, k_mixt=10)
  optimizer = tf.keras.optimizers.Adamax(3e-4)
  model.save_weights("saved_models/dgmm_init")
  
  losses, val_losses = trainer(model, optimizer, train_patch_ds, val_patch_ds,
                              f"mvtec-{OBJECT}", r_sz=R_SZ, n_samples=train_ds_imgs.shape[0],
                              n_epochs=N_EPOCHS)

  df = pd.DataFrame(np.stack((losses,val_losses),axis=1), columns=["Train", "Val"])
  df.plot()
  plt.savefig(f"figs/train-r{R_SZ}-E{N_EPOCHS}-B{TRAIN_SZ}.png", bbox_inches="tight")
  return 

def dgmm_runner():
  pass

if __name__ == "__main__":

    runners = {
      "train": dgmm_trainer,
      "pixel_eval": dgmm_runner,
      "sample": msma_runner
    }

    runner = runners[EXPERIMENT]

    runner()

