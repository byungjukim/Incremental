import tensorflow as tf
import os
import numpy as np
import cPickle



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images,labels,filename):
  num_examples=labels.shape[0]
  if images.shape[0] != num_examples:
    print num_examples, images.shape[0]
    raise ValueError("Size does not match")


  print('writing',filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
          'image_raw': _bytes_feature(image_raw),
          'label'    : _int64_feature(int(labels[index]))}))
    writer.write(example.SerializeToString())

