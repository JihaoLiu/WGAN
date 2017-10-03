import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generator import Generator

REAL_LABEL = 1.0
LAMBDA = 10

class CycleGAN:
  def __init__(self,
               X_train_file='',
               Y_train_file='',
               batch_size=1,
               image_size_w=128,
               image_size_h=128,
               use_lsgan=True,
               norm='instance',
               lambda1=10.0,
               lambda2=10.0,
               beta1=0.5,
               ngf=64
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size_w = image_size_w
    self.image_size_h = image_size_h
    self.beta1 = beta1
    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file
    self.learning_rate = tf.placeholder(tf.float32)

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.G = Generator('G', self.is_training, ngf=ngf, norm=norm,
                       image_size_w=image_size_w, image_size_h=image_size_h)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.F = Generator('F', self.is_training, norm=norm,
                       image_size_w=image_size_w, image_size_h=image_size_h)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size_w, image_size_h, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size_w, image_size_h, 3])

  def model(self):
    X_reader = Reader(self.X_train_file, name='X', image_size_w=self.image_size_w,
                      image_size_h=self.image_size_h, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y', image_size_w=self.image_size_w,
                      image_size_h=self.image_size_h, batch_size=self.batch_size)

    x = X_reader.feed()
    y = Y_reader.feed()

    cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)
    W_a = self.disc_loss(self.D_Y, y, self.fake_y)
    W_b = self.disc_loss(self.D_X, x, self.fake_x)
    W = W_a + W_b

    GP_a = self.gradien_penalty(self.D_Y, y, self.fake_y)
    GP_b = self.gradien_penalty(self.D_X, x, self.fake_x)
    GP = GP_a + GP_b

    # X -> Y
    fake_y = self.G(x)
    G_gan_loss = self.gen_loss(self.D_Y, fake_y)

    # Y -> X
    fake_x = self.F(y)
    F_gan_loss = self.gen_loss(self.D_X, fake_x)

    loss_g = G_gan_loss + F_gan_loss

    G_loss = cycle_loss + loss_g
    C_loss = LAMBDA*GP + W

    # summary
    tf.summary.scalar('loss/G', G_loss)
    tf.summary.scalar('loss/C', C_loss)
    tf.summary.scalar('loss/GradientPenalty', GP)
    tf.summary.scalar('loss/cycle', cycle_loss)
    tf.summary.scalar('lr/learning_rate', self.learning_rate)

    tf.summary.image('X/generated', utils.batch_convert2int(
              tf.image.resize_images(self.G(x), (48, 160))))
    tf.summary.image('X/reconstruction', utils.batch_convert2int(
              tf.image.resize_images(self.F(self.G(x)), (48, 160))))
    tf.summary.image('Y/generated', utils.batch_convert2int(
              tf.image.resize_images(self.F(y), (48, 160))
    ))
    tf.summary.image('Y/reconstruction', utils.batch_convert2int(
              tf.image.resize_images(self.G(self.F(y)), (48, 160))
    ))

    return G_loss, C_loss, fake_y, fake_x

  def optimize(self, G_loss, C_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      beta1 = self.beta1
      tf.summary.scalar('learning_rate/{}'.format(name), self.learning_rate)
      learning_step = (
          tf.train.AdamOptimizer(self.learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, [self.G.variables, self.F.variables], name='G')
    C_optimizer = make_optimizer(C_loss, [self.D_Y.variables, self.D_X.variables], name='C')

    return G_optimizer, C_optimizer

  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss

  def cycle_consistency_loss(self, G, F, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  # wgan_gp
  def gen_loss(self, D, fake_y):
    loss = -tf.reduce_mean(D(fake_y))
    return loss


  def gradien_penalty(self, D, y, fake_y):
    alpha = tf.random_uniform(
      shape=[1, 1],
      minval=0.,
      maxval=1.
    )
    differences = fake_y - y
    interpolates = y + (alpha * differences)
    gradients = tf.gradients(D(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty

  def disc_loss(self, D, y, fake_y):
    disc_loss = tf.reduce_mean(D(fake_y)) - tf.reduce_mean(D(y))
    return disc_loss

  # 1300 414
  # 440  140
  # 160  48