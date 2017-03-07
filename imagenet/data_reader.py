
import os
import numpy as np
import tensorflow as tf
import pickle

import opts
import transform



def read_cifar10_tfrecord(filename_queue):
  class CIFAR10Record_tfrecord(object):
    pass
  result = CIFAR10Record_tfrecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = opts.FLAGS.image_height
  result.width = opts.FLAGS.image_width
  result.depth = opts.FLAGS.image_channel
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  #features = tf.parse_single_example(
  #        serialized_example, 
  #        dense_keys=['image_raw','label'],
  #        dense_types=[tf.string,tf.int64])
  
  features = tf.parse_single_example(
          serialized_example,
          features={
              'image_raw': tf.FixedLenFeature([],tf.string),
              'label'    : tf.FixedLenFeature([],tf.int64)
          })
        

  # Convert from a string to a vector of uint8 that is record_bytes long.
  image = tf.decode_raw(features['image_raw'], tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(features['label'], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(image, [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def read_image(filename_queue):
  class image_reader(object):
    pass
  result = image_reader()


  label = filename_queue[1]

  image_read = tf.read_file(filename_queue[0])
  image = tf.image.decode_jpeg(image_read,channels=3)
  


  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(label, tf.int32)

  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = image

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
  #if opts.FLAGS.test_only:
  #  num_preprocess_threads = 4 
  #else:
  num_preprocess_threads = opts.FLAGS.num_threads
  if shuffle and (not opts.FLAGS.test_only):
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
  #tf.image_summary('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def read_labeled_image_list(image_list_file):
  f = open(image_list_file, 'r')
  filenames = []
  labels = []
  for line in f:
    filename, label = line[:-1].split(' ')
    filenames.append(filename)
    #filenames.append(os.path.join(opts.FLAGS.train_dir,filename))
    labels.append(int(label))
  return filenames, labels

def augmented_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  with open('filelist.pickle','rb') as pk:
    list_dic = pickle.load(pk)

  filenames_with_label = [files for files in list_dic['filelist']]

  image_list,label_list = read_labeled_image_list('./text/filelist_0.txt')
  #image_list,label_list = read_labeled_image_list(filenames_with_label)

  
  #with open('class2idx.pickle','rb') as pk:
  #  dir2idx = pickle.load(pk)
  #for files in os.listdir(data_dir):
  #  if files.endswith(".txt"):
  #    filenames.append(os.path.join(data_dir,files))


  for f in image_list:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filenames = tf.convert_to_tensor(image_list,dtype=tf.string)
  labels = tf.convert_to_tensor(label_list,dtype=tf.int32)

  # Create a queue that produces the filenames to read.
  #filename_queue = tf.train.string_input_producer(filenames)
  filename_queue = tf.train.slice_input_producer([filenames,labels])


  # Read examples from files in the filename queue.
  read_input = read_image(filename_queue)
  #read_input = read_cifar10_tfrecord(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  augmented_image = transform.transform_train(reshaped_image) 

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  #min_queue_examples = int(opts.FLAGS.num_example_train *
  #                         min_fraction_of_examples_in_queue)
  min_queue_examples = 1000
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)



  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(augmented_image, read_input.label,
                                         min_queue_examples, opts.FLAGS.batch_size,
                                         shuffle=opts.FLAGS.shuffle)



def train_inputs():
  images,labels = augmented_inputs(data_dir=opts.FLAGS.train_dir,
                                   batch_size = opts.FLAGS.batch_size)

  if opts.FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  return images,labels


def augmented_inputs_test(data_dir, batch_size):
  filenames = []
  for files in os.listdir(data_dir):
    if files.endswith(".tfrecords"):
      filenames.append(os.path.join(data_dir,files))
  #filenames = [os.path.join(data_dir, 'data_batch_0.bin')]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10_tfrecord(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  augmented_image = transform.transform_test(reshaped_image) 

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(opts.FLAGS.num_example_test *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to test. '
         'This will take a few minutes.' % min_queue_examples)



  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(augmented_image, read_input.label,
                                         min_queue_examples, opts.FLAGS.batch_size,
                                         shuffle=False)
def test_inputs():
  images,labels = augmented_inputs_test(data_dir=opts.FLAGS.test_dir,
                                        batch_size = opts.FLAGS.batch_size)

  if opts.FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  return images,labels





def augmented_inputs_eval(data_dir, batch_size):
  filenames = []
  for files in os.listdir(data_dir):
    if files.endswith(".tfrecords"):
      filenames.append(os.path.join(data_dir,files))
  #filenames = [os.path.join(data_dir, 'data_batch_0.bin')]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10_tfrecord(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  augmented_image = transform.transform_eval(reshaped_image) 

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(opts.FLAGS.num_example_test *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to test. '
         'This will take a few minutes.' % min_queue_examples)



  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(augmented_image, read_input.label,
                                         min_queue_examples, opts.FLAGS.batch_size,
                                         shuffle=False)
def eval_inputs():
  images,labels = augmented_inputs_eval(data_dir=opts.FLAGS.eval_dir,
                                        batch_size = opts.FLAGS.batch_size)

  if opts.FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)

  return images,labels






