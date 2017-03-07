
import tensorflow as tf
import opts


def transform_train(original_image):
  
  reshape_size = 256
  #original image
  augmented_image = original_image
  #original_shape = tf.shape(augmented_image)
  #ratio = tf.to_float(reshape_size) / tf.to_float(tf.minimum(original_shape[-2],original_shape[-3]))

  #new_shape = tf.to_int32(ratio*original_image[-3:-1])
  #augmented_image = tf.resize_images(augmented_image,new_shape)



  augmented_image = tf.image.resize_images(augmented_image,[reshape_size,reshape_size])


  # Randomly crop
  augmented_image = tf.random_crop(augmented_image,[opts.FLAGS.crop_h, opts.FLAGS.crop_w, opts.FLAGS.crop_ch])

  # Randomly flip 
  augmented_image = tf.image.random_flip_left_right(augmented_image)

  # Standardization  ( Mean subtraction, Std division )
  #augmented_image = tf.image.per_image_whitening(augmented_image)
  augmented_image = tf.image.per_image_standardization(augmented_image)


  # Color augmentation
  #augmented_image = tf.image.random_brightness(augmented_image, max_delta=0.4)
  #augmented_image = tf.image.random_contrast(augmented_image, lower=0.6, upper=1.4)
  #augmented_image = tf.image.random_saturation(augmented_image, lower=0.6, upper=1.4)



  return augmented_image





def transform_test(original_image):
  
  #original image
  augmented_image = original_image

  # Pad images
  padded_h = opts.FLAGS.image_height + 2 * opts.FLAGS.pad_h
  padded_w = opts.FLAGS.image_width + 2 * opts.FLAGS.pad_w
  augmented_image = tf.image.resize_image_with_crop_or_pad(augmented_image,padded_h, padded_w)

  # Center crop
  augmented_image = tf.image.resize_image_with_crop_or_pad(augmented_image,opts.FLAGS.crop_h, opts.FLAGS.crop_w)

  
  # Standardization  ( Mean subtraction, Std division )
  #augmented_image = (original_image / 255.0) - 0.5
  #augmented_image = tf.image.per_image_whitening(augmented_image)
  augmented_image = tf.image.per_image_standardization(augmented_image)

  return augmented_image





def transform_eval(original_image):
  
  #original image
  augmented_image = original_image

  # Pad images
  padded_h = opts.FLAGS.image_height + 2 * opts.FLAGS.pad_h
  padded_w = opts.FLAGS.image_width + 2 * opts.FLAGS.pad_w
  augmented_image = tf.image.resize_image_with_crop_or_pad(augmented_image,padded_h, padded_w)

  # Center crop
  augmented_image = tf.image.resize_image_with_crop_or_pad(augmented_image,opts.FLAGS.crop_h, opts.FLAGS.crop_w)

  
  # Standardization  ( Mean subtraction, Std division )
  #augmented_image = (original_image / 255.0) - 0.5
  augmented_image = tf.image.per_image_whitening(augmented_image)

  return tf.concat(2,[augmented_image,original_image])
