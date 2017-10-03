import tensorflow as tf
import random
import os
import scipy.misc
import cv2
import numpy as np
import time
try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('X_input_dir', './plate_resize_test',
                       'X input directory, default: data/apple2orange/trainA')
tf.flags.DEFINE_string('Y_input_dir', './plate_resize_real',
                       'Y input directory, default: data/apple2orange/trainB')
tf.flags.DEFINE_string('X_output_file', './plate_synthetic.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y_output_file', 'data/tfrecords/plate_real.tfrecords',
                       'Y output tfrecords file, default: data/tfrecords/orange.tfrecords')


def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  # file_name = file_path.split('/')[-1]
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_path))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error as e:
    pass

  images_num = len(file_paths)

  # dump to tfrecords file
  writer = tf.python_io.TFRecordWriter(output_file)
  with tf.Session() as sess:
    for i in range(len(file_paths)):
      file_path = file_paths[i]

      with tf.gfile.FastGFile(file_path, 'rb') as f:
        image_raw_data = f.read()
      example = _convert_to_example(file_path, image_raw_data)
      writer.write(example.SerializeToString())
      if i % 500 == 0:
        print("Processed {}/{}.".format(i, images_num))

  print("Done.")
  writer.close()


def resize_and_crop_and_flip_img(input_dir, output_file):
  try:
    os.mkdir(output_file)
  except os.error as e:
    print('dir already exist！')

  for index, item in enumerate(sorted(os.listdir(input_dir))):
    # print(item)
    start_time = time.time()
    img = cv2.imread(input_dir + '/' + item)
    resize_ = cv2.resize(img, (143, 143))
    for j in range(4):
      crop_x, crop_y = np.random.randint(0, 143-128, 2)
      img_ = resize_[crop_x:crop_x+128, crop_y:crop_y+128].copy()
      flip_1, flip_2 = np.random.rand(2)
      # dandom flip up and down
      if flip_1 < 0.5:
        img_ = cv2.flip(img_, 0)
      if flip_2 >= 0.5:
        img_ = cv2.flip(img_, 1)
      new_file_path = output_file + str(j) + '_' + str(index) + '.png'
      cv2.imwrite(new_file_path, img_)
    end_time = time.time()
    print("img index {}, time cost {}".format(index, end_time - start_time))

def main(unused_argv):
  print("Convert X data to tfrecords...")
  data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)
  # print("Convert Y data to tfrecords...")
  # data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)
  # resize_and_crop_and_flip_img('./plate_test_font', 'plate_resize_test/')

if __name__ == '__main__':
  tf.app.run()
