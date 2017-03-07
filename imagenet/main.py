
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import os
import math

import opts
import model
import data_reader



def tower_loss(scope):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  images, labels = data_reader.train_inputs()

  # Build inference Graph.
  #logits = model.feed_forward_resnet20(images)
  logits = model._model(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = model.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')


  return total_loss

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



def multi_gpu_train():
  with tf.device('/cpu:0'):
    global_step = tf.get_variable('global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)


    num_steps_per_epoch = (opts.FLAGS.num_example_train / opts.FLAGS.batch_size)
    decay_steps = int(num_steps_per_epoch * opts.FLAGS.LR_step)
    lr = tf.train.exponential_decay(opts.FLAGS.LR,
                                    global_step,
                                    decay_steps,
                                    opts.FLAGS.LR_decay,
                                    staircase=True)

    opt = tf.train.MomentumOptimizer(lr,opts.FLAGS.momentum,use_nesterov=opts.FLAGS.nesterov)
    
    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(opts.FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('gpu_%d' % (i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss = tower_loss(scope)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)
    
    grads = average_gradients(tower_grads)
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    # Create a saver.
    #saver = tf.train.Saver(tf.all_variables())
    saver = tf.train.Saver(tf.global_variables())

    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()


    # Start running operations on the Graph.
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)
    
    ckpt = tf.train.get_checkpoint_state(opts.FLAGS.save_path)
    if opts.FLAGS.resume:
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

      else:
        print('No checkpoint file found! Initialize randomly.')
        init = tf.initialize_all_variables()
        sess.run(init)
    else:
      sess.run(init)
    
    #sess.run(init)
  
    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    max_iteration = opts.FLAGS.MAX_EPOCH * opts.FLAGS.num_example_train / opts.FLAGS.batch_size
    _display = opts.FLAGS.display
    _save = opts.FLAGS.save
    _save_path = opts.FLAGS.save_path
    save_step = math.ceil(opts.FLAGS.num_example_train / opts.FLAGS.batch_size * opts.FLAGS.save)
    for step in xrange(1,max_iteration+1):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      current_batch = (step * opts.FLAGS.batch_size / float(opts.FLAGS.num_example_train))
      if step % _display == 0:
        num_examples_per_step = opts.FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.3f sec/batch), %.2f/%d epoch')
        print (format_str % (datetime.now(), step, loss_value, sec_per_batch,current_batch,opts.FLAGS.MAX_EPOCH))
        
      # Save the model checkpoint periodically.
      current_epoch = int(math.ceil(current_batch))
      #if current_epoch % _save == 0 or (step + 1) == max_iteration:
      if step % save_step == 0 or (step) == max_iteration:
        if step > 0:
          checkpoint_path = os.path.join(_save_path, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=current_epoch)




def op_train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_steps_per_epoch = opts.FLAGS.num_example_train / opts.FLAGS.batch_size
  decay_steps = int(num_steps_per_epoch * opts.FLAGS.LR_step)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(opts.FLAGS.LR,
                                  global_step,
                                  decay_steps,
                                  opts.FLAGS.LR_decay,
                                  staircase=True)
  #tf.scalar_summary('learning_rate', lr)
  # Compute gradients.
  #opt = tf.train.GradientDescentOptimizer(lr)
  opt = tf.train.MomentumOptimizer(lr,opts.FLAGS.momentum,use_nesterov=opts.FLAGS.nesterov)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  
  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op





def train():
  global_step = tf.Variable(0, trainable=False)

  # Get images and labels for CIFAR-10.
  images, labels = data_reader.train_inputs()

  # Build a Graph that computes the logits predictions from the
  # inference model.
  #logits = model.feed_forward_basic(images)
  #logits = model.feed_forward_resnet20(images)
  logits = model._model(images)

  # Calculate loss.
  loss = model.loss(logits, labels)

  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters.
  #train_op = model.train(loss, global_step)
  train_op = op_train(loss, global_step)

  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables())

  # Start running operations on the Graph.
  sess = tf.InteractiveSession()
  sess.run(init)


  if opts.FLAGS.resume:
    ckpt = tf.train.get_checkpoint_state(opts.FLAGS.save_path)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return



  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)
  
  max_iteration = opts.FLAGS.MAX_EPOCH * opts.FLAGS.num_example_train / opts.FLAGS.batch_size
  _display = opts.FLAGS.display
  _save = opts.FLAGS.save
  _save_path = opts.FLAGS.save_path
  save_step = math.ceil(opts.FLAGS.num_example_train / opts.FLAGS.batch_size * opts.FLAGS.save)

  for step in xrange(1,max_iteration+1):
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    current_batch = (step * opts.FLAGS.batch_size / float(opts.FLAGS.num_example_train))
    if step % _display == 0:
      num_examples_per_step = opts.FLAGS.batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      
      #process = step/float(max_iteration)*100

      format_str = ('%s: step %d, loss = %.2f (%.3f sec/batch), %.2f/%d epoch')
      print (format_str % (datetime.now(), step, loss_value, sec_per_batch,current_batch,opts.FLAGS.MAX_EPOCH))
      
      #format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
      #              'sec/batch)')
      #print (format_str % (datetime.now(), step, loss_value,
      #                     examples_per_sec, sec_per_batch))

    # Save the model checkpoint periodically.
    current_epoch = int(math.ceil(current_batch))
    if step % save_step == 0 or (step) == max_iteration:
    #if current_epoch % _save == 0 or (step + 1) == max_iteration:
      if step > 0:
        checkpoint_path = os.path.join(_save_path, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=current_epoch)
        #saver.save(sess, checkpoint_path, global_step=step)




def test():
  global_step = tf.Variable(0, trainable=False)

  images, labels = data_reader.test_inputs()
  #logits = model.feed_forward_basic(images)
  #logits = model.feed_forward_resnet20(images)
  logits = model._model(images)

  loss = model.test_loss(logits, labels)

  top_1_op = tf.nn.in_top_k(logits, labels, 1)
  top_k_op = tf.nn.in_top_k(logits, labels, opts.FLAGS.Top_k)

  saver = tf.train.Saver(tf.all_variables())

  sess = tf.InteractiveSession()
  init = tf.initialize_all_variables()
  sess.run(init)


  ckpt = tf.train.get_checkpoint_state(opts.FLAGS.save_path)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
  else:
    print('No checkpoint file found')
    return

  tf.train.start_queue_runners(sess=sess)
  
  max_iteration = int(math.ceil(opts.FLAGS.num_example_test / opts.FLAGS.batch_size))
  _display = opts.FLAGS.display
  

  top_1_correct = 0
  top_k_correct = 0
  total_count = float(max_iteration * opts.FLAGS.batch_size)
  total_loss = 0.0

  for step in xrange(1,max_iteration+1):
    start_time = time.time()
    step_loss = sess.run([loss])
    duration = time.time() - start_time
    

    total_loss += step_loss[0]
    current_total = float(step*opts.FLAGS.batch_size)


    if opts.FLAGS.Top_k > 1:
      top_1_predictions = sess.run([top_1_op])
      top_k_predictions = sess.run([top_k_op])
      top_1_correct += np.sum(top_1_predictions)
      top_k_correct += np.sum(top_k_predictions)

      format_str = '%s: Top 1 Accuracy %.4f, Top %d Accuracy %.4f, Test loss %.6f, out of %d examples'
      test_log = format_str % (datetime.now(),top_1_correct/current_total, 
                               opts.FLAGS.Top_k,top_k_correct/current_total,
                               total_loss/step,step*opts.FLAGS.batch_size)


    else:
      top_1_predictions = sess.run([top_1_op])
      top_1_correct += np.sum(top_1_predictions)

      format_str = '%s: Top 1 Accuracy %.4f, Test loss %.6f, out of %d examples'
      test_log = format_str % (datetime.now(),top_1_correct/current_total,
                               total_loss/step,step*opts.FLAGS.batch_size)

    print (test_log)
  print ('')


def multi_gpu_test():
  with tf.device('/cpu:0'):
    global_step = tf.get_variable('global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images, labels = data_reader.test_inputs()
    with tf.device('/gpu:0'):
      with tf.name_scope('gpu_0') as scope:
        logits = model._model(images)
       
        loss = model.test_loss(logits, labels)
        
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_k_op = tf.nn.in_top_k(logits, labels, opts.FLAGS.Top_k)



    saver = tf.train.Saver(tf.all_variables())

    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)
    #init = tf.initialize_all_variables()
    #sess.run(init)


    ckpt = tf.train.get_checkpoint_state(opts.FLAGS.save_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      init = tf.initialize_all_variables()
      sess.run(init)
      #return

    tf.train.start_queue_runners(sess=sess)
    
    max_iteration = int(math.ceil(opts.FLAGS.num_example_test / opts.FLAGS.batch_size))
    _display = opts.FLAGS.display
    

    top_1_correct = 0
    top_k_correct = 0
    total_count = float(max_iteration * opts.FLAGS.batch_size)
    total_loss = 0.0

    for step in xrange(1,max_iteration+1):
      current_total = float(step*opts.FLAGS.batch_size)

      if opts.FLAGS.Top_k > 1:
        start_time = time.time()
        top_1_predictions,top_k_predictions,step_loss = sess.run([top_1_op,top_k_op,loss])
        duration = time.time() - start_time

        total_loss += step_loss
        top_1_correct += np.sum(top_1_predictions)
        top_k_correct += np.sum(top_k_predictions)

        format_str = '%s: Top 1 Accuracy %.4f, Top %d Accuracy %.4f, Test loss %.6f, out of %d examples'
        test_log = format_str % (datetime.now(),top_1_correct/current_total, 
                                 opts.FLAGS.Top_k,top_k_correct/current_total,
                                 total_loss/step,step*opts.FLAGS.batch_size)
        final_result = '(Accuracy)Top-1 : %.4f, Top-%d : %.4d'%(top_1_1correct/current_total,opts.FLAGS.Top_k,top_k_correct/current_total)


      else:
        top_1_predictions, step_loss = sess.run([top_1_op,loss])

        total_loss += step_loss
        top_1_correct += np.sum(top_1_predictions)

        format_str = '%s: Top 1 Accuracy %.4f, Test loss %.6f, out of %d examples'
        test_log = format_str % (datetime.now(),top_1_correct/current_total,
                                 total_loss/step,step*opts.FLAGS.batch_size)
        final_result = '(Accuracy)Top-1 : %.4f'%(top_1_correct/current_total)

      #print (test_log)
    print ('-----------Final Result-------------')
    print final_result
    print ('')


def multi_gpu_eval():
  with tf.device('/cpu:0'):
    global_step = tf.get_variable('global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images, labels = data_reader.eval_inputs()
    with tf.device('/gpu:0'):
      with tf.name_scope('gpu_0') as scope:
        feed_image,original_image = tf.split(3,2,images)
        logits = model._model(feed_image)

        prediction = tf.argmax(logits,1)
        #prediction = tf.argmin(logits,1)
        result = tf.nn.softmax(logits)
        matching_images = tf.identity(original_image)

    saver = tf.train.Saver(tf.all_variables())

    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.InteractiveSession(config=config)
    #init = tf.initialize_all_variables()
    #sess.run(init)


    ckpt = tf.train.get_checkpoint_state(opts.FLAGS.save_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      raise ValueError('No checkpoint file found')

    tf.train.start_queue_runners(sess=sess)
    
    max_iteration = int(math.ceil(opts.FLAGS.num_example_eval / opts.FLAGS.batch_size))
    _display = opts.FLAGS.display
    
    total_pixel_image = opts.FLAGS.image_height*opts.FLAGS.image_width*opts.FLAGS.image_channel
    prediction_result = np.zeros((max_iteration*opts.FLAGS.batch_size),dtype=np.uint64)
    preserve_image = np.zeros((max_iteration*opts.FLAGS.batch_size,opts.FLAGS.image_height,opts.FLAGS.image_width,opts.FLAGS.image_channel),dtype=np.uint8)
    
    cor=0
    for step in xrange(1,max_iteration+1):
      current_total = float(step*opts.FLAGS.batch_size)

      batch_prediction,batch_images,batch_result = sess.run([prediction,matching_images,result])
      batch_prediction = np.asarray(batch_prediction,dtype=np.int32)
      prob = np.array(np.amax(batch_result,1)<opts.FLAGS.eval_threshold,dtype=np.float)*10 #exclude
      prediction_result[opts.FLAGS.batch_size*(step-1):opts.FLAGS.batch_size*step] = batch_prediction+prob
      preserve_image[opts.FLAGS.batch_size*(step-1):opts.FLAGS.batch_size*step] = batch_images
    return prediction_result, preserve_image


def main(argv=None):  # pylint: disable=unused-argument
  if not opts.FLAGS.test_only:
    if opts.FLAGS.train:
      multi_gpu_train()
    else:
      #eval
      unsup_labels,preserve_image = multi_gpu_eval()
      
      #sort
      import relabel
      import tfrecorder
      

      num_pixel_image = opts.FLAGS.image_height*opts.FLAGS.image_width*opts.FLAGS.image_channel
      new_unsup = relabel.relabel(np.reshape(preserve_image,(-1,num_pixel_image)),unsup_labels, opts.FLAGS.num_class,num_pixel_image)

      #print new_unsup['image'][:20]
      #save
      save_path = os.path.join(opts.FLAGS.eval_save,'new_unsup.tfrecords')
      new_unsup['image'] = np.transpose(new_unsup['image'].reshape((-1,opts.FLAGS.image_height,opts.FLAGS.image_width,opts.FLAGS.image_channel)),(0,3,1,2))
      if new_unsup['image'].shape[0] > 0:
        tfrecorder.convert_to(np.array(new_unsup['image'],dtype=np.uint8), new_unsup['label'],save_path)
      #tfrecorder.convert_to(preserve_image.astype(np.uint8), unsup_labels.astype(np.uint64),save_path)


  #if opts.FLAGS.test_only or (opts.FLAGS.Final_test and not(opts.FLAGS.test_only)):
  else:
    multi_gpu_test()


if __name__ == '__main__':
  tf.app.run()
