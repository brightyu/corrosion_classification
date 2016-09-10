from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. 
# image size of 64 x 64. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 64

# Global constants describing the corrostion data set 
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 155210 #50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 38482 #10000

def read_corrosion(filename_queue, label_queue):
    """Reads and parses examples from corrosion data files

    Args:
        filename_queue: A queue of strings with the filenames to read from.
        data_dir: a string of the data directory

    Returns:
        An object representing a single example, with the following fields:
          height: number of rows in the result (32)
          width: number of columns in the result (32)
          depth: number of color channels in the result (3)
          label: an int32 Tensor with the label in the range 0..9.
          uint8image: a [height, width, depth] uint8 Tensor with the image data
  "
    """
    class CorrosionRecord(object):
        pass

    result = CorrosionRecord()

    result.height = 64
    result.width = 64
    result.depth = 3

    # create the reader
    image_reader = tf.WholeFileReader()

    #Read whole file from the queue
    _, image_file = image_reader.read(filename_queue)

    #text reader
    label_reader = tf.TextLineReader()
    _, csv_row = label_reader.read(label_queue)

    # create a label reader
    # Match the label with the image
    """image_name = os.path.basename(image_file).strip('.jpg')
    label_file = os.path.join(data_dir, 'Labels', image_name + '.txt')
    if not tf.gfile.Exists(label_file):
        raise ValueError('Failed to find file: ' + label_file)
    label_file_open = open(label_file)
    label = int(label_file_open.readline(1))"""
    result.label = tf.string_to_number(csv_row, tf.int32)

    result.uint8image = tf.image.decode_jpeg(image_file)
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def get_corrosion_filenames(data_dir, train_or_test):
    """ Returns a list of all filenames in a training or test dataset

    Args:
        data_dir: path to the corrosion data directory
        train_or_test: string of either 'train' or 'test' to read the files

    Returns:
        filenames: a list of all the image files in the dataset
    """

    filenames = []
    label_files = []
    file_list = open(os.path.join(data_dir, 'ImageSets', train_or_test + '.txt'))
    for line in file_list:
        file_to_add = os.path.join(data_dir, 'Images', line.strip('\n') + '.jpg')
        label_to_add = os.path.join(data_dir, 'Labels', line.strip('\n') + '.txt')
        if not tf.gfile.Exists(file_to_add):
            raise ValueError('Failed to find file: ' + file_to_add)
        if not tf.gfile.Exists(label_to_add):
            raise ValueError('Failed to find file: ' + label_to_add)
        #file_to_add = line.strip('\n')
        filenames.append(file_to_add)
        label_files.append(label_to_add)
    file_list.close()

    return filenames, label_files



def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for corrosion dataset
  
  Data dirctory contains an ImageSets directory with a list of images
  to train on and a directory 'Images' with the corresponding images

  Args:
    data_dir: Path to the corrosion data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  filenames, label_files = get_corrosion_filenames(data_dir, 'train')

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
  labelname_queue = tf.train.string_input_producer(label_files, shuffle=False)

  # Read examples from files in the filename queue.
  read_input = read_corrosion(filename_queue, labelname_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d corrosion images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for corrosion evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the corrosion data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames, label_files = get_corrosion_filenames(data_dir, 'train')
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames, label_files = get_corrosion_filenames(data_dir, 'test')
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
  labelname_queue = tf.train.string_input_producer(label_files, shuffle=False)

  # Read examples from files in the filename queue.
  read_input = read_corrosion(filename_queue, labelname_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
  #                                                       width, height)
  resized_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
