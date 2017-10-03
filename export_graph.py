""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""

import tensorflow as tf
import os
from scipy.misc import imread, imresize, imsave
from tensorflow.python.tools.freeze_graph import freeze_graph
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', 'test_image', 'checkpoints directory path')
tf.flags.DEFINE_bool('XtoY', True, 'get resule with XtoY')
tf.flags.DEFINE_integer('image_size_w', '128', 'image size, default: 256')
tf.flags.DEFINE_integer('image_size_h', '128', 'image size, default: 256')
tf.flags.DEFINE_integer('output_image_w', 48, 'image width')
tf.flags.DEFINE_integer('output_image_h', 160, 'image height')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_string('output_dir', 'gen_image/', 'generated image save dir')
tf.flags.DEFINE_string('test_image', './plate_test/', 'test image dir')


def get_result(XtoY=True):
  graph = tf.Graph()
  try:
    os.mkdir(FLAGS.output_dir)
  except:
    print('dir already exist!')
  with tf.Session(graph=graph) as sess:
    cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm, image_size_w=FLAGS.image_size_w,
                         image_size_h=FLAGS.image_size_h)
    input_image = tf.placeholder(tf.float32, shape=[FLAGS.image_size_w, FLAGS.image_size_h, 3], name='input_image')
    cycle_gan.model()

    if XtoY:
      output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
    else:
      output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))
    fixed_output_img = tf.image.resize_images(output_image, (FLAGS.output_image_w, FLAGS.output_image_h))
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, latest_ckpt)
    for index, fname in enumerate(os.listdir(FLAGS.test_image)):
      img = imread(FLAGS.test_image + fname)
      img = imresize(img, (128, 128))
      feed = {input_image: img}
      gen_img = sess.run(fixed_output_img, feed_dict=feed)
      image_dir = FLAGS.output_dir + fname
      imsave(image_dir, imresize(gen_img, (36, 136)))
      if index % 25 == 0:
        print(index)


def main(unused_argv):
  get_result(XtoY=FLAGS.XtoY)

if __name__ == '__main__':
  tf.app.run()
