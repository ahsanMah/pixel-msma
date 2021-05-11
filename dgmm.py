import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow.keras as tfk

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision

from tqdm.auto import tqdm

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
# tf.enable_v2_behavior()
tf.data.AUTOTUNE = tf.data.experimental.AUTOTUNE

EPSILON = 1e-5

class DGMM(tf.keras.Model):
  """
  Deep GMM
  
  D : # Dimension of Multivariate Normals aka Event shape
  """

  def __init__(self, img_shape, latent_dim=32, k_mixt=3, D=10):
    super(DGMM, self).__init__()

    # FIXME: Change Kernel Size?, k_mixt?

    # self.hidden = tfk.Sequential([
    #     tfk.layers.InputLayer(input_shape=img_shape),
    #     tfk.layers.Conv2D(latent_dim*2, 3, activation=tf.nn.swish),
    #     tfk.layers.GlobalAveragePooling2D(),
    #     Dense(latent_dim, activation=tf.nn.swish,
    #           kernel_regularizer=tfk.regularizers.l2(0.001),
    #           name="latent")
    # ])

    self.resnet = tfk.applications.MobileNetV3Small(
        input_shape=(img_shape[0], img_shape[1], 3),
        alpha=1.0,
        include_top=False,
        weights=None,
        pooling="avg"
    )

    self.hidden = tfk.Sequential([
          tfk.layers.InputLayer(input_shape=img_shape),
          tfk.layers.Conv2D(filters=3, kernel_size=1),
          self.resnet
        ], name="latent")

    self.alpha = tfk.Sequential([
        self.hidden,
        Dense(k_mixt, activation=None,
              kernel_initializer="he_uniform"),
        tfk.layers.Activation('linear', dtype='float32')
    ], name="alpha")

    self.mu = tfk.Sequential([
        self.hidden,
        Dense(k_mixt * D, activation=None, name="mu",
              kernel_initializer="he_uniform"),
        Reshape((k_mixt, D), dtype='float32')
    ], name="mu")

    self.sigma = tfk.Sequential([
        self.hidden,
        Dense(k_mixt * (D * (D+1) // 2),
              activation=tf.nn.softplus
              # kernel_regularizer=tfk.regularizers.l2(0.001)
              ),
        Reshape((k_mixt, (D * (D+1)) // 2)),
        tfk.layers.Lambda(self.stable_lower_triangle, dtype='float32')
    ], name='sigma')
  
  @tf.function(experimental_compile=True)
  def stable_lower_triangle(self, x):
    return tfp.math.fill_triangular(x) + 1e-5

  @tf.function
  def log_pdf_univariate(self, x, y):

    gmm = tfd.MixtureSameFamily(
      mixture_distribution=
        tfd.Categorical(
          logits= self.alpha(y)),
      components_distribution=
        tfd.Normal(
          loc= self.mu(y),
          scale= self.sigma(y))
      )
    
    return gmm.log_prob(tf.reshape(x,(-1,)))

  @tf.function(experimental_compile=True)
  def log_pdf(self, x, y):

    # sigma = tfp.math.fill_triangular(self.sigma(y)) + 1e-5
    # print(sigma.dtype)
    gmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=self.alpha(y)),
        components_distribution=tfd.MultivariateNormalTriL(
            loc=self.mu(y),
            scale_tril=self.sigma(y))
        )
    return gmm.log_prob(x)

  @tf.function(experimental_compile=True)
  def call(self, inputs):
    x,y = inputs
    # print(x.dtype, y.dtype)
    return self.log_pdf(tf.cast(x, dtype=tf.float32),
                        tf.cast(y, dtype=tf.float32))

@tf.function(experimental_compile=True)
def log_loss(model, x, y):
  log_prob = model.log_pdf(x,y)
  return -tf.reduce_mean(log_prob)

@tf.function
def train_step(model, x, y, optimizer):
  with tf.GradientTape() as tape:
    loss = log_loss(model, x, y)
    
    if mixed_precision.global_policy().name == "mixed_float16":
      loss = optimizer.get_scaled_loss(loss)

  gradients = tape.gradient(loss, model.trainable_variables)
  
  if mixed_precision.global_policy().name == "mixed_float16":
    gradients = optimizer.get_unscaled_gradients(gradients)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def build_anchors(img_w, img_h, receptive_field_sz=4, stride=None):
  stride = stride if stride else receptive_field_sz 
  start = receptive_field_sz // 2
  end_w = img_w - receptive_field_sz // 2 + 1
  end_h = img_h - receptive_field_sz // 2 + 1
    
  # List of anchors denoting midpoint of patches  
  c_x = np.arange(start, end_w, stride)
  c_y = np.arange(start, end_h, stride)
  shift_x, shift_y = np.meshgrid(c_x,c_y)
  shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
  anchors = np.column_stack((shift_x, shift_y)).astype(dtype=np.int32)
  
  return anchors

def get_overlap_map(img_sz, r_sz, anchors):
    counts = np.zeros(shape=(img_sz,img_sz))

    for x,y in anchors:
        counts[x-r_sz//2: x+r_sz//2,
               y-r_sz//2: y+r_sz//2] += 1
    
    return counts

def get_patch_loaders(img_w, img_h, nc=3, sigma_l=10, receptive_field_sz = 4):

  # List of anchors denoting midpoint of patches 
  ANCHORS = tf.constant(build_anchors(img_w, img_h, receptive_field_sz))
  print("Anchors:", ANCHORS.shape)

  @tf.function
  def get_test_patches(x_batch):
    
    offsets = tf.tile(ANCHORS, [x_batch["image"].shape[0], 1])
    offsets = ANCHORS #tf.tile(ANCHORS, [x_batch["image"].shape[0], 1])

    print("Offset shape:", offsets.shape)
    s = tf.repeat(x_batch["score"], ANCHORS.shape[0], axis=0)
    x = tf.repeat(x_batch["image"], ANCHORS.shape[0], axis=0)
    print("Repeated Batch shape:", x.shape)

    x = tfa.image.cutout(x, mask_size=(receptive_field_sz, receptive_field_sz),
                         offset=offsets, constant_values=0.0)
    
    # Generate mask for cut pixels
    mask = x[..., 0] == 0

    # Get norms of just the patches
    x_patch = tf.where(mask[..., tf.newaxis,tf.newaxis], s, tf.zeros_like(s))
    print("Patch:", x_patch.shape)
    # x_patch = tf.reshape(x_patch, shape=[-1, receptive_field_sz*receptive_field_sz*nc, sigma_l])
    x_patch = tf.reshape(x_patch, shape=[x_patch.shape[0], -1, sigma_l])
    print("Flat Patch:", x_patch.shape)
    score = tf.norm(x_patch, axis=1, ord="euclidean")

    # Append mask to provide positional encoding of patch
    x = tf.concat([x, tf.cast(mask[...,tf.newaxis], tf.float32)], axis=-1)

    # Resize for mobilenet 
    # x = tf.image.resize(x, (32,32), method="nearest")
    
    #change this to x_patch if you want to verify the selection works
    return dict(image=x, score=score) 

  @tf.function
  def get_random_patches(x,s):
    x = x + EPSILON
    x = tfa.image.random_cutout(x, mask_size=(receptive_field_sz, receptive_field_sz),
                                constant_values=0.0)
    
    # Generate mask for cut pixels
    mask = x[..., 0] == 0
    # mask = tf.expand_dims(mask, axis=-1)

    # Get score-norms of just the patches
    x_patch = tf.where(mask[..., tf.newaxis,tf.newaxis], s, tf.zeros_like(s))
    print(x_patch.shape)
    x_patch = tf.reshape(x_patch, shape=[-1, img_w*img_h*3, sigma_l])
    print(x_patch.shape)
    score = tf.norm(x_patch, axis=1, ord="euclidean")

    # Append mask to provide positional encoding of patch
    x = tf.concat([x, tf.cast(mask[...,tf.newaxis], tf.float32)], axis=-1)

    # Resize for mobilenet
    # x = tf.image.resize(x, (32,32), method="nearest")

    return dict(image=x, score=score) #change this to x_patch if you want to verify the selection works

  def build_test_ds(x, s, batch_sz):
    s = tf.transpose(s, perm=[0,2,3,4,1])
    ds = {"image": tf.cast(x, tf.float32),
          "score": tf.cast(s, tf.float32)}
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.batch(batch_sz, drop_remainder=True)
    ds = ds.map(get_test_patches).cache()
    # ds = ds.interleave(get_test_patches, cycle_length=1)
    # ds = ds.map(preproc)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

  def build_train_ds(x, s, batch_sz):
    s = tf.transpose(s, perm=[0,2,3,4,1])
    s = tf.data.Dataset.from_tensor_slices(s)
    x = tf.data.Dataset.from_tensor_slices(x)
    ds = tf.data.Dataset.zip((x, s)).cache()
    ds = ds.shuffle(100)
    ds = ds.batch(batch_sz, drop_remainder=False)
    ds = ds.map(get_random_patches, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

  return build_train_ds, build_test_ds

'''
Starts a fresh round of training for `n_epochs`
'''
def trainer(model, optimizer, train_ds, test_ds, dataset,
          r_sz, n_samples, n_epochs=20):
  

  model.load_weights("saved_models/dgmm_init")

  val_ds = test_ds.take(8).cache()

  avg_loss = tfk.metrics.Mean()
  val_loss = tfk.metrics.Mean()

  n_steps = n_epochs * n_samples
  epochs_bar = tqdm(range(n_epochs), desc="Epoch")
  losses = [0]
  val_losses = [0]

  for i in epochs_bar:
    progress_bar = tqdm(train_ds, desc=f"Loss: {losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}", leave=False)
    for j, x in enumerate(progress_bar):
      loss = train_step(model, x["score"], x["image"], optimizer)
      avg_loss(loss)

      if j % 10 == 0:
        for x_val in val_ds:
          val_loss(log_loss(model, x_val["score"], x_val["image"]))

        losses.append(avg_loss.result())
        val_losses.append(val_loss.result())

        val_loss.reset_states()
        avg_loss.reset_states()
        
        progress_bar.set_description(f"Loss: {losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
    
    model.save_weights(f"saved_models/dgmm/{dataset}/{r_sz}x{r_sz}/e{i}")
  
  return losses, val_losses
