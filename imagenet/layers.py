
import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

import model

def batch_norm(x, n_out, name='bn',is_training=True):
  scope_name = name
  with tf.variable_scope(scope_name) as scope:
    with tf.device('/cpu:0'):
      beta  = tf.get_variable('beta' ,[n_out],tf.float32,
                        initializer=tf.constant_initializer(0.0,tf.float32))
      gamma = tf.get_variable('gamma',[n_out],tf.float32,
                        initializer=tf.constant_initializer(1.0,tf.float32))

      moving_mean = tf.get_variable('moving_mean',[n_out],
                        initializer=tf.constant_initializer(0.0), trainable=False)
      moving_var  = tf.get_variable('moving_var', [n_out],
                        initializer=tf.constant_initializer(0.0), trainable=False)



    if is_training:
      batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')

      decay = 0.9
      #decay = 1.0
      apply_op1 = moving_mean.assign( (1.0-decay)*batch_mean + decay * moving_mean )
      apply_op2 = moving_var.assign(  (1.0-decay)*batch_var  + decay * moving_var  )
       
      with tf.control_dependencies([apply_op1,apply_op2]):
        y = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,gamma,0.001)
        #y = tf.nn.batch_normalization(x,moving_mean,moving_var,beta,gamma,0.001)

    else:
      #batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
      #y = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,gamma,0.001)
      y = tf.nn.batch_normalization(x,moving_mean,moving_var,beta,gamma,0.001)

    
    return y


