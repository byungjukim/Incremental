
import tensorflow as tf
import math


import layers
import opts

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if opts.FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if opts.FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def test_loss(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  return cross_entropy_mean 


def _conv(name,x,kernel_size,in_filter,out_filter,stride=1,is_first=True):
  with tf.variable_scope(name,reuse=not(is_first)) as scope:
    n=kernel_size*kernel_size*out_filter
    kernel = _variable_with_weight_decay('weights',
                                         shape=[kernel_size,kernel_size,in_filter,out_filter],
                                         stddev=math.sqrt(2.0/n),
                                         wd=opts.FLAGS.weight_decay)
    return tf.nn.conv2d(x,kernel,[1,stride,stride,1],padding='SAME')
    


def residual_block_A(image,feature_in,feature_out,stride,is_training=True,active_before_residual=False):
  if active_before_residual:   #the first block. activate BEFORE residual
    with tf.variable_scope('shared_preact1') as scope:
      image = layers.batch_norm(image,feature_in,name=scope.name,is_training=is_training)
      image = tf.nn.relu(image)
      original = image
  else:
    with tf.variable_scope('preact1') as scope:
      original = image 
      image = layers.batch_norm(image,feature_in,name=scope.name,is_training=is_training)
      image = tf.nn.relu(image)


  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, feature_in, feature_out],
                                         stddev=math.sqrt(2.0/(3*3*feature_out)),
                                         wd=opts.FLAGS.weight_decay)
    image = tf.nn.conv2d(image, kernel, [1, stride, stride, 1], padding='SAME')


  with tf.variable_scope('preact2') as scope:
    image = layers.batch_norm(image,feature_out,name=scope.name,is_training=is_training)
    image = tf.nn.relu(image)
    
    

  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, feature_out, feature_out],
                                         stddev=math.sqrt(2.0/(3*3*feature_out)),
                                         wd=opts.FLAGS.weight_decay)
    image = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')


  with tf.variable_scope('add') as scope:
    if feature_in != feature_out:
      original = tf.nn.avg_pool(original,[1,stride,stride,1],[1,stride,stride,1],'VALID')
      original = tf.pad(original,
                    [[0,0],[0,0],[0,0],[(feature_out-feature_in)//2,(feature_out-feature_in)//2]])
    image += original
    return image


def residual_block_bottleneck(image,feature_in,feature_out,stride,is_training=True,active_before_residual=False):
  if active_before_residual:   #the first block. activate BEFORE residual
    with tf.variable_scope('shared_preact1') as scope:
      image = layers.batch_norm(image,feature_in,name=scope.name,is_training=is_training)
      image = tf.nn.relu(image)
      original = image
  else:
    with tf.variable_scope('preact1') as scope:
      original = image 
      image = layers.batch_norm(image,feature_in,name=scope.name,is_training=is_training)
      image = tf.nn.relu(image)


  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, feature_in, feature_out],
                                         stddev=math.sqrt(2.0/(feature_out)),
                                         wd=opts.FLAGS.weight_decay)
    image = tf.nn.conv2d(image, kernel, [1, stride, stride, 1], padding='SAME')


  with tf.variable_scope('preact2') as scope:
    image = layers.batch_norm(image,feature_out,name=scope.name,is_training=is_training)
    image = tf.nn.relu(image)
    
    

  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, feature_out, feature_out],
                                         stddev=math.sqrt(2.0/(3*3*feature_out)),
                                         wd=opts.FLAGS.weight_decay)
    image = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')


  with tf.variable_scope('preact3') as scope:
    image = layers.batch_norm(image,feature_out,name=scope.name,is_training=is_training)
    image = tf.nn.relu(image)
    
    

  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, feature_out, feature_out*4],
                                         stddev=math.sqrt(2.0/(feature_out*4)),
                                         wd=opts.FLAGS.weight_decay)
    image = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')






  with tf.variable_scope('add') as scope:
    if feature_in != 4*feature_out:
      with tf.variable_scope('shortcut') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[1, 1, feature_in, 4*feature_out],
                                             stddev=math.sqrt(2.0/(4*feature_out)),
                                             wd=opts.FLAGS.weight_decay)
        original = tf.nn.conv2d(original, kernel, [1, stride, stride, 1], padding='SAME')
    image += original
    return image


def imagenet_resnet(images,is_training=True):
  num_feature = [64,128,256,512]
  num_block = [3,4,6,3]     #50layer
  #num_block = [3,4,23,3]    #101layer

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 3, 64],
                                         stddev=math.sqrt(2.0/(7*7*64)),
                                         #stddev=0.1,
                                         wd=opts.FLAGS.weight_decay)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    pool = tf.nn.max_pool(conv,[1,3,3,1],[1,2,2,1],padding='SAME',name='pooling')

#residual_block_A(image,block_name,feature_in,feature_out,stride):
  with tf.variable_scope('conv2_0'):
    conv2 = residual_block_bottleneck(pool, 64,num_feature[0],1,is_training=is_training,
                                active_before_residual=True)
  for n in range(num_block[0]-1):
    with tf.variable_scope('conv2_%d'%(n+1)):
      conv2 = residual_block_bottleneck(conv2,4*num_feature[0],num_feature[0],1,is_training=is_training)
  

  with tf.variable_scope('conv3_0'):
    conv3 = residual_block_bottleneck(conv2,4*num_feature[0],num_feature[1],2,is_training=is_training)
  
  for n in range(num_block[1]-1):
    with tf.variable_scope('conv3_%d'%(n+1)):
      conv3 = residual_block_bottleneck(conv3,4*num_feature[1],num_feature[1],1,is_training=is_training)


  with tf.variable_scope('conv4_0'):
    conv4 = residual_block_bottleneck(conv3,4*num_feature[1],num_feature[2],2,is_training=is_training)
  
  for n in range(num_block[2]-1):
    with tf.variable_scope('conv4_%d'%(n+1)):
      conv4 = residual_block_bottleneck(conv4,4*num_feature[2],num_feature[2],1,is_training=is_training)
  
  with tf.variable_scope('conv5_0'):
    conv5 = residual_block_bottleneck(conv4,4*num_feature[2],num_feature[3],2,is_training=is_training)
  
  for n in range(num_block[3]-1):
    with tf.variable_scope('conv5_%d'%(n+1)):
      conv5 = residual_block_bottleneck(conv5,4*num_feature[3],num_feature[3],1,is_training=is_training)


  with tf.variable_scope('final_preact') as scope:
    bn = layers.batch_norm(conv5,4*num_feature[3],name=scope.name,is_training=is_training)
    bn_relu = tf.nn.relu(bn, name=scope.name)


  with tf.variable_scope('average_pool') as scope:
    avg = tf.reduce_mean(bn_relu,[1,2], name=scope.name)

  with tf.variable_scope('fc') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(avg, [opts.FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, opts.FLAGS.num_class],
                                          stddev=math.sqrt(2.0/opts.FLAGS.num_class), 
                                          #stddev=0.1,
                                          wd=opts.FLAGS.weight_decay)
    bias = _variable_on_cpu('bias', [opts.FLAGS.num_class], tf.constant_initializer(0.01))
    #softmax_linear = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name)
    softmax_linear = tf.matmul(reshape, weights) + bias


  return softmax_linear


def imagenet_resnet34(images,is_training=True,N=18):
  num_feature = [64,128,256,512]
  num_block = [3,4,6,3]


  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 3, 64],
                                         stddev=math.sqrt(2.0/(7*7*64)),
                                         #stddev=0.1,
                                         wd=opts.FLAGS.weight_decay)
    conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
    pool = tf.nn.max_pool(conv,[1,3,3,1],[1,2,2,1],padding='SAME',name='pooling')

#residual_block_A(image,block_name,feature_in,feature_out,stride):
  with tf.variable_scope('conv2_0'):
    conv2 = residual_block_A(pool,              64,num_feature[0],1,is_training=is_training,
                              active_before_residual=True)
  for n in range(num_block[0]-1):
    with tf.variable_scope('conv2_%d'%(n+1)):
      conv2 = residual_block_A(conv2,num_feature[0],num_feature[0],1,is_training=is_training)
  

  with tf.variable_scope('conv3_0'):
    conv3 = residual_block_A(conv2,num_feature[0],num_feature[1],2,is_training=is_training)
  
  for n in range(num_block[1]-1):
    with tf.variable_scope('conv3_%d'%(n+1)):
      conv3 = residual_block_A(conv3,num_feature[1],num_feature[1],1,is_training=is_training)


  with tf.variable_scope('conv4_0'):
    conv4 = residual_block_A(conv3,num_feature[1],num_feature[2],2,is_training=is_training)
  
  for n in range(num_block[2]-1):
    with tf.variable_scope('conv4_%d'%(n+1)):
      conv4 = residual_block_A(conv4,num_feature[2],num_feature[2],1,is_training=is_training)
  
  with tf.variable_scope('conv5_0'):
    conv5 = residual_block_A(conv4,num_feature[2],num_feature[3],2,is_training=is_training)
  
  for n in range(num_block[3]-1):
    with tf.variable_scope('conv5_%d'%(n+1)):
      conv5 = residual_block_A(conv5,num_feature[3],num_feature[3],1,is_training=is_training)





  with tf.variable_scope('final_preact') as scope:
    bn = layers.batch_norm(conv5,num_feature[3],name=scope.name,is_training=is_training)
    bn_relu = tf.nn.relu(bn, name=scope.name)


  with tf.variable_scope('average_pool') as scope:
    avg = tf.reduce_mean(bn_relu,[1,2], name=scope.name)

  with tf.variable_scope('fc') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(avg, [opts.FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, opts.FLAGS.num_class],
                                          stddev=math.sqrt(2.0/opts.FLAGS.num_class), 
                                          #stddev=0.1,
                                          wd=opts.FLAGS.weight_decay)
    bias = _variable_on_cpu('bias', [opts.FLAGS.num_class], tf.constant_initializer(0.0))
    #softmax_linear = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name)
    softmax_linear = tf.matmul(reshape, weights) + bias


  return softmax_linear


def _VGG(images,is_training=True):
  x = _conv('conv1a',images,3,3,64,1)
  x = layers.batch_norm(x,64,'bn1a',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv1b',x,3,64,64,1)
  x = layers.batch_norm(x,64,'bn1b',is_training=is_training)
  x = tf.nn.relu(x)

  x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

  x = _conv('conv2a',x,3,64,128,1)
  x = layers.batch_norm(x,128,'bn2a',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv2b',x,3,128,128,1)
  x = layers.batch_norm(x,128,'bn2b',is_training=is_training)
  x = tf.nn.relu(x)

  x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

  x = _conv('conv3a',x,3,128,256,1)
  x = layers.batch_norm(x,256,'bn3a',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv3b',x,3,256,256,1)
  x = layers.batch_norm(x,256,'bn3b',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv3c',x,3,256,256,1)
  x = layers.batch_norm(x,256,'bn3c',is_training=is_training)
  x = tf.nn.relu(x)

  x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

  x = _conv('conv4a',x,3,256,512,1)
  x = layers.batch_norm(x,512,'bn4a',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv4b',x,3,512,512,1)
  x = layers.batch_norm(x,512,'bn4b',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv4c',x,3,512,512,1)
  x = layers.batch_norm(x,512,'bn4c',is_training=is_training)
  x = tf.nn.relu(x)

  x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

  x = _conv('conv5a',x,3,512,512,1)
  x = layers.batch_norm(x,512,'bn5a',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv5b',x,3,512,512,1)
  x = layers.batch_norm(x,512,'bn5b',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('conv5c',x,3,512,512,1)
  x = layers.batch_norm(x,512,'bn5c',is_training=is_training)
  x = tf.nn.relu(x)

  x = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

  x = _conv('fc1',x,7,512,4096)
  x = layers.batch_norm(x,4096,'fc1',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('fc2',x,1,4096,4096)
  x = layers.batch_norm(x,4096,'fc2',is_training=is_training)
  x = tf.nn.relu(x)

  x = _conv('fc3',x,1,4096,opts.FLAGS.num_class)

  return tf.reduce_mean(x,[1,2])


def _model(images):
  n=18
  if opts.FLAGS.test_only:
    is_training=False
  elif not opts.FLAGS.train:
    is_training = False
  else:
    is_training = True

  #return imagenet_resnet34(images,is_training=is_training,N=n)
  #return _VGG(images,is_training=is_training)
  return imagenet_resnet(images,is_training=is_training)

